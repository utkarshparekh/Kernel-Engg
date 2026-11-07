"""Original uncompressed M=8 router; isolated from the active submission.

W_hi and W_lo are concatenated along K. X is addressed twice inside the
partial kernel, avoiding a materialized repeated activation or a cast kernel.
All partial products and their final reduction accumulate in FP32.
"""

import torch
import triton
import triton.language as tl

from plan import M, N, K, BM, BN, BK, SPLITS, WARPS, STAGES, PART_K


@triton.jit
def _partial(X, W, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
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
def _finish(P, Y, COUNT: tl.constexpr, SPLITS: tl.constexpr,
            BLOCK: tl.constexpr):
    out = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    split = tl.arange(0, SPLITS)
    partials = tl.load(P + split[:, None] * COUNT + out[None, :],
                       mask=out[None, :] < COUNT, other=0.0)
    tl.store(Y + out, tl.sum(partials, axis=0), mask=out < COUNT)


def prepare(weight):
    if (weight.shape != (N, K) or weight.dtype != torch.float32
            or not weight.is_cuda or not weight.is_contiguous()):
        raise ValueError("expected contiguous CUDA FP32 router weight [256,8192]")
    high = weight.to(torch.bfloat16)
    low = (weight - high.float()).to(torch.bfloat16)
    # Two BF16 copies occupy exactly the original FP32 weight's byte count.
    return torch.cat((high.t(), low.t()), dim=0).contiguous()


def run(x, weight_pair, *, diagnostics=False):
    if (x.shape != (M, K) or x.dtype != torch.bfloat16 or not x.is_cuda
            or not x.is_contiguous() or weight_pair.shape != (2*K, N)
            or weight_pair.dtype != torch.bfloat16 or not weight_pair.is_contiguous()
            or weight_pair.device != x.device):
        raise ValueError("isolated candidate requires BF16 X[8,8192] and prepared router weights")
    partials = torch.empty((SPLITS, M, N), device=x.device, dtype=torch.float32)
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    main = _partial[(N//BN, SPLITS)](
        x, weight_pair, partials, M, N, K, BM, BN, BK, PART_K,
        num_warps=WARPS, num_stages=STAGES)
    finish = _finish[(triton.cdiv(M*N, 128),)](
        partials, out, M*N, SPLITS, 128, num_warps=4)
    if diagnostics:
        return out, {"partial": main, "finish": finish}
    return out
