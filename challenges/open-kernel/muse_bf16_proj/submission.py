# okc: problem=muse_bf16_proj
"""First Triton baseline for Muse Spark's dense BF16 projections."""

from typing import TypedDict

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
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


def _launch_config(m: int) -> tuple[int, int, int, int, int]:
    """Conservative starting configurations; tune only after measuring H200."""
    if m <= 8:
        return 16, 64, 32, 4, 8
    if m <= 32:
        return 32, 64, 32, 4, 8
    if m <= 128:
        return 64, 64, 32, 4, 8
    return 64, 128, 32, 8, 8


def prepare(weights: Weights):
    state = {name: value.contiguous() for name, value in weights.items()}
    for name, value in weights.items():
        if name != "router":
            state[f"{name}_t"] = value.t().contiguous()
    if "router" in weights:
        router = weights["router"]
        router_hi = router.to(torch.bfloat16)
        router_lo = (router - router_hi.float()).to(torch.bfloat16)
        state["router_hi_t"] = router_hi.t().contiguous()
        state["router_lo_t"] = router_lo.t().contiguous()
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    config = data["config"]
    x = data["x"]
    m, n, k, layer = config["m"], config["n"], config["k"], config["layer"]

    if layer == "router":
        if m <= 8:
            return F.linear(x.float(), state["router"][:n])
        result = torch.mm(x, state["router_hi_t"][:, :n], out_dtype=torch.float32)
        result += torch.mm(x, state["router_lo_t"][:, :n], out_dtype=torch.float32)
        return result

    # Shape-specific dispatch from paired H200 measurements:
    #   * pre-transposed torch.mm consistently wins for pooler;
    #   * it wins for the larger unpooler calls, but not its small sliced calls;
    #   * qkv_gate only showed a useful win at M=16 and M=128;
    #   * o was neutral/slower across the scored shapes.
    # Fall back to SGLang everywhere else so a local layout win cannot turn into
    # a regression for a different aspect ratio.
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
