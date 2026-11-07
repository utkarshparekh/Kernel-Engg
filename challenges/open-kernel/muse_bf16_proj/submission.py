# okc: problem=muse_bf16_proj
"""First Triton baseline for Muse Spark's dense BF16 projections."""

from typing import TypedDict

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from flashinfer.gemm.kernels.dense_bf16_gemm_warp_splitk import (
    default_tactic,
    run_warp_splitk_dense,
)
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


class Weights(TypedDict):
    qkv_gate: torch.Tensor
    o: torch.Tensor
    pooler: torch.Tensor
    unpooler: torch.Tensor
    router: torch.Tensor


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor
    config: Config


@triton.jit
def _router_m8_fp32(
    x_ptr,
    weight_ptr,
    output_ptr,
    M: tl.constexpr,
    N_SIZE: tl.constexpr,
    K_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute the exact-FP32 M=8 router with one program per expert."""
    output_row = tl.program_id(0)
    token_offsets = tl.arange(0, M)
    k_offsets = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((M,), dtype=tl.float32)

    for k_start in range(0, K_SIZE, BLOCK_K):
        k = k_start + k_offsets
        weight = tl.load(weight_ptr + output_row * K_SIZE + k)
        activations = tl.load(
            x_ptr + token_offsets[:, None] * K_SIZE + k[None, :]
        )
        products = activations.to(tl.float32) * weight[None, :]
        accumulator += tl.sum(products, axis=1)

    tl.store(
        output_ptr + token_offsets * N_SIZE + output_row,
        accumulator,
    )


@triton.jit
def _router_m8_hilo_partial(X, W, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
             PART_K: tl.constexpr):
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    rows = tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(0, PART_K, BK):
        logical_k = ps * PART_K + start + inner
        original_k = logical_k % K
        a = tl.load(X + rows[:, None] * K + original_k[None, :],
                    mask=rows[:, None] < M, other=0.0)
        b = tl.load(W + logical_k[:, None] * N + cols[None, :])
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
    tl.store(P + ps * M * N + rows[:, None] * N + cols[None, :], acc,
             mask=rows[:, None] < M)


@triton.jit
def _router_m8_hilo_finish(P, Y, COUNT: tl.constexpr, SPLITS: tl.constexpr,
            BLOCK: tl.constexpr):
    out = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    split = tl.arange(0, SPLITS)
    partials = tl.load(P + split[:, None] * COUNT + out[None, :],
                       mask=out[None, :] < COUNT, other=0.0)
    tl.store(Y + out, tl.sum(partials, axis=0), mask=out < COUNT)


@triton.jit
def _bf16_projection(
    x_ptr,
    w_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    W_STRIDE_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Compute out[M,N] = x[M,K] @ w[N,K].T."""
    pid = tl.program_id(0)
    programs_m = tl.cdiv(M, BLOCK_M)
    programs_n = tl.cdiv(N, BLOCK_N)

    programs_per_group = GROUP_M * programs_n
    group = pid // programs_per_group
    first_m = group * GROUP_M
    group_m = tl.minimum(programs_m - first_m, GROUP_M)
    pid_m = first_m + (pid % programs_per_group) % group_m
    pid_n = (pid % programs_per_group) // group_m

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    inner = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_indices = k_start + inner
        x = tl.load(
            x_ptr + rows[:, None] * K + k_indices[None, :],
            mask=(rows[:, None] < M) & (k_indices[None, :] < K),
            other=0.0,
        )
        # prepare() stores W.T as contiguous [K, original_N], so warp lanes
        # traverse adjacent output columns instead of striding by K.
        w = tl.load(
            w_ptr + k_indices[:, None] * W_STRIDE_N + cols[None, :],
            mask=(cols[None, :] < N) & (k_indices[:, None] < K),
            other=0.0,
        )
        accumulator += tl.dot(x, w)

    tl.store(
        out_ptr + rows[:, None] * N + cols[None, :],
        accumulator.to(tl.bfloat16),
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


def prepare(weights: Weights):
    state = {name: value.contiguous() for name, value in weights.items()}
    for name, value in weights.items():
        if name != "router":
            state[f"{name}_t"] = value.t().contiguous()
    if "unpooler" in weights:
        state["unpooler_slice_t"] = weights["unpooler"][:1024].t()
    if "router" in weights:
        router = weights["router"]
        router_hi = router.to(torch.bfloat16)
        router_lo = (router - router_hi.float()).to(torch.bfloat16)
        state["router_hi_t"] = router_hi.t().contiguous()
        state["router_lo_t"] = router_lo.t().contiguous()
        state["router_both_t"] = torch.cat(
            (state["router_hi_t"], state["router_lo_t"]), dim=1
        ).contiguous()
        # M=8 uses the same two BF16 halves along K, with X addressed twice
        # inside the kernel. Total execution weight bytes remain 8 MiB.
        state["router_hilo_k_t"] = torch.cat(
            (state["router_hi_t"], state["router_lo_t"]), dim=0
        ).contiguous()
    state["native_tactics"] = {}
    for layer, n, k in (
        ("qkv_gate", 2304, 8192),
        ("pooler", 4096, 8192),
        ("unpooler", 8192, 4096),
        ("o", 8192, 1024),
    ):
        if layer not in weights:
            continue
        for tactic_m in (1, 8, 16, 32):
            state["native_tactics"][(layer, tactic_m)] = default_tactic(
                tactic_m, n, k
            )
        if layer == "unpooler":
            for tactic_m in (1, 8):
                state["native_tactics"][(layer, tactic_m, 1024)] = default_tactic(
                    tactic_m, 1024, 4096
                )
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    config = data["config"]
    x = data["x"]
    m, n, k, layer = config["m"], config["n"], config["k"], config["layer"]

    if layer == "router":
        if m == 8 and n == 256 and k == 8192 and x.is_contiguous():
            partials = torch.empty(
                (16, 8, 256), device=x.device, dtype=torch.float32
            )
            output = torch.empty(
                (8, 256), device=x.device, dtype=torch.float32
            )
            _router_m8_hilo_partial[(8, 16)](
                x, state["router_hilo_k_t"], partials,
                8, 256, 8192, 16, 32, 128, 1024,
                num_warps=4, num_stages=2,
            )
            _router_m8_hilo_finish[(16,)](
                partials, output, 2048, 16, 128, num_warps=4
            )
            return output
        if m == 8:
            output = torch.empty(
                (m, n),
                device=x.device,
                dtype=torch.float32,
            )
            _router_m8_fp32[(n,)](
                x,
                state["router"],
                output,
                M=8,
                N_SIZE=n,
                K_SIZE=k,
                BLOCK_K=512,
                num_warps=4,
            )
            return output
        if m < 8:
            return F.linear(x.float(), state["router"][:n])
        result = torch.mm(x, state["router_both_t"], out_dtype=torch.float32)
        result[:, :256].add_(result[:, 256:])
        return result[:, :n]

    native_m = 1 if m == 1 else 8 if m <= 8 else 16 if m <= 16 else 32
    if m <= 32 and layer in ("qkv_gate", "pooler", "o"):
        out = torch.empty((m, n), device=x.device, dtype=x.dtype)
        run_warp_splitk_dense(
            x,
            state[layer].t(),
            out,
            False,
            state["native_tactics"][(layer, native_m)],
        )
        return out
    if layer == "unpooler" and m <= 8:
        out = torch.empty((m, n), device=x.device, dtype=x.dtype)
        run_warp_splitk_dense(
            x,
            state["unpooler_slice_t"][:, :n],
            out,
            False,
            state["native_tactics"][(layer, native_m, 1024)],
        )
        return out
    if layer == "unpooler" and m <= 32:
        out = torch.empty((m, n), device=x.device, dtype=x.dtype)
        run_warp_splitk_dense(
            x,
            state[layer].t(),
            out,
            False,
            state["native_tactics"][(layer, native_m)],
        )
        return out

    use_transposed_mm = (
        layer == "pooler"
        or (layer == "unpooler" and m > 8)
        or (layer == "qkv_gate" and m in (16, 128))
    )
    if not use_transposed_mm:
        return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)

    weight_t = state[f"{layer}_t"]
    # Establish a strong Python baseline first.  prepare() has already converted
    # W[N,K] into contiguous W.T[K,N], so this enters cuBLAS without a strided B.
    # The Triton kernel above remains the next optimization seam (split-K for the
    # short/wide shapes), but is not used until it beats this path in measurements.
    return torch.mm(x, weight_t[:, :n])
