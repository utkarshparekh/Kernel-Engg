"""Gate one calculated Triton SM90 tile for pooler M=256.

The H200 has 132 SMs.  A 64x128 output tile produces exactly
ceil(256/64) * ceil(4096/128) = 128 programs for this shape, giving nearly one
program per SM without a configuration sweep.  Each program walks K in tiles
of 64 and uses ``tl.dot`` so Hopper tensor cores perform the BF16 GEMM.

The candidate advances only after challenge-checker correctness, CUDA-graph
capture, and alternating cold-L2 timing against the current submission.
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import Callable

import torch
import triton
import triton.language as tl

import reference
from submission import kernel_fn, prepare


LAYER = "pooler"
M = 256
BLOCK_M = 64
BLOCK_N = 128
BLOCK_K = 64
NUM_WARPS = 8
NUM_STAGES = 4
REPEATS = 40
WARMUP_REPLAYS = 5
L2_FLUSH_BYTES = 64 * 1024 * 1024
MIN_SPEEDUP = 1.20
TARGET_US = 18.7


@triton.jit
def pooler_m256_kernel(
    x_ptr,
    weight_t_ptr,
    output_ptr,
    M_SIZE: tl.constexpr,
    N_SIZE: tl.constexpr,
    K_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute output[M,N] = x[M,K] @ weight_t[K,N]."""
    programs_n = tl.cdiv(N_SIZE, BLOCK_N)
    program = tl.program_id(0)
    program_m = program // programs_n
    program_n = program % programs_n

    rows = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    inner = tl.arange(0, BLOCK_K)

    x_offsets = rows[:, None] * K_SIZE + inner[None, :]
    weight_offsets = inner[:, None] * N_SIZE + columns[None, :]
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, K_SIZE, BLOCK_K):
        x = tl.load(x_ptr + x_offsets)
        weight = tl.load(weight_t_ptr + weight_offsets)
        accumulator = tl.dot(x, weight, accumulator)
        x_offsets += BLOCK_K
        weight_offsets += BLOCK_K * N_SIZE

    output_offsets = rows[:, None] * N_SIZE + columns[None, :]
    tl.store(output_ptr + output_offsets, accumulator.to(tl.bfloat16))


def make_weight(device: torch.device) -> torch.Tensor:
    n, k = reference.LAYERS[LAYER]
    generator = torch.Generator(device=device).manual_seed(
        reference.WEIGHTS_SEED + sum(map(ord, LAYER))
    )
    return (
        torch.randn(
            (n, k),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        / (k**0.5)
    ).to(torch.bfloat16)


def make_forced_probe_input(device: torch.device) -> dict:
    n, k = reference.out_features(LAYER, M), reference.LAYERS[LAYER][1]
    x = torch.zeros((M, k), device=device, dtype=torch.bfloat16)
    rows = torch.arange(M, device=device)
    probe_columns = torch.tensor(
        [0, 1, 31, 127, k // 2, k - 128, k - 2, k - 1],
        device=device,
    )
    columns = probe_columns[rows % probe_columns.numel()]
    probe_values = torch.tensor(
        [8, -16, 32, -64, 128, -8, 16, -32],
        device=device,
        dtype=torch.bfloat16,
    )
    values = probe_values[rows % probe_values.numel()]
    x[rows, columns] = values
    return {
        "x": x,
        "config": {"m": M, "n": n, "k": k, "layer": LAYER},
    }


def launch_candidate(
    x: torch.Tensor,
    weight_t: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    n, k = output.shape[1], x.shape[1]
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(n, BLOCK_N),)
    pooler_m256_kernel[grid](
        x,
        weight_t,
        output,
        M_SIZE=M,
        N_SIZE=n,
        K_SIZE=k,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return output


def capture(
    callable_: Callable[[], torch.Tensor],
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    callable_()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = callable_()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def main() -> None:
    device = torch.device("cuda")
    n, k = reference.out_features(LAYER, M), reference.LAYERS[LAYER][1]
    programs = triton.cdiv(M, BLOCK_M) * triton.cdiv(n, BLOCK_N)
    print("GPU:", torch.cuda.get_device_name(device))
    print(f"Experiment: Triton {LAYER} M={M} N={n} K={k}")
    print(
        f"Tile: {BLOCK_M}x{BLOCK_N}x{BLOCK_K}, "
        f"warps={NUM_WARPS}, stages={NUM_STAGES}, programs={programs}"
    )

    weight = make_weight(device).contiguous()
    weights = {LAYER: weight}
    state = prepare(weights)
    weight_t = state[f"{LAYER}_t"][:, :n]
    assert weight_t.shape == (k, n)
    assert weight_t.is_contiguous()
    candidate_output = torch.empty(
        (M, n), device=device, dtype=torch.bfloat16
    )

    correctness_inputs = (
        (
            "normal",
            reference.generate_input(
                M, LAYER, seed=20261030 + M, device=str(device)
            ),
        ),
        (
            "outliers",
            reference.generate_input(
                M,
                LAYER,
                seed=20261130 + M,
                dist="outliers",
                device=str(device),
            ),
        ),
        ("forced_probes", make_forced_probe_input(device)),
    )

    correctness = {}
    for distribution, data in correctness_inputs:
        output = launch_candidate(data["x"], weight_t, candidate_output)
        torch.cuda.synchronize()
        ok, message = reference.check_implementation(data, output, weights)
        correctness[distribution] = {"ok": ok, "message": message}
        print(f"Correctness {distribution}: ok={ok} {message}")

    output_path = (
        Path(os.environ["GPU_RESULTS_DIR"]) / "triton_pooler_m256.json"
    )
    if not all(item["ok"] for item in correctness.values()):
        result = {
            "candidate": "triton_64x128x64_w8_s4",
            "layer": LAYER,
            "m": M,
            "n": n,
            "k": k,
            "status": "incorrect",
            "correctness": correctness,
        }
        output_path.write_text(json.dumps(result, indent=2) + "\n")
        print("REJECT: correctness failed")
        print("Saved:", output_path)
        return

    timing_data = correctness_inputs[0][1]

    def baseline_call() -> torch.Tensor:
        return kernel_fn(timing_data, state)

    def candidate_call() -> torch.Tensor:
        return launch_candidate(
            timing_data["x"], weight_t, candidate_output
        )

    baseline_graph, baseline_output = capture(baseline_call)
    candidate_graph, captured_candidate_output = capture(candidate_call)

    baseline_ok, baseline_message = reference.check_implementation(
        timing_data, baseline_output, weights
    )
    candidate_ok, candidate_message = reference.check_implementation(
        timing_data, captured_candidate_output, weights
    )
    if not baseline_ok:
        raise RuntimeError(f"Current submission failed: {baseline_message}")
    if not candidate_ok:
        raise RuntimeError(
            f"Triton failed after CUDA graph capture: {candidate_message}"
        )

    l2_flush = torch.empty(
        L2_FLUSH_BYTES // 4, device=device, dtype=torch.float32
    )
    timed = (
        {
            "name": "current_submission",
            "graph": baseline_graph,
            "times_us": [],
            "start": torch.cuda.Event(enable_timing=True),
            "end": torch.cuda.Event(enable_timing=True),
        },
        {
            "name": "triton_64x128x64_w8_s4",
            "graph": candidate_graph,
            "times_us": [],
            "start": torch.cuda.Event(enable_timing=True),
            "end": torch.cuda.Event(enable_timing=True),
        },
    )

    for _ in range(WARMUP_REPLAYS):
        for item in timed:
            l2_flush.zero_()
            item["graph"].replay()
    torch.cuda.synchronize()

    for repetition in range(REPEATS):
        order = timed if repetition % 2 == 0 else tuple(reversed(timed))
        for item in order:
            l2_flush.zero_()
            item["start"].record()
            item["graph"].replay()
            item["end"].record()
            item["end"].synchronize()
            item["times_us"].append(
                item["start"].elapsed_time(item["end"]) * 1000.0
            )

    current_us = statistics.median(timed[0]["times_us"])
    candidate_us = statistics.median(timed[1]["times_us"])
    speedup = current_us / candidate_us
    decision = "expand" if speedup >= MIN_SPEEDUP else "reject"
    target_compatible = candidate_us <= TARGET_US
    result = {
        "candidate": "triton_64x128x64_w8_s4",
        "layer": LAYER,
        "m": M,
        "n": n,
        "k": k,
        "tile": [BLOCK_M, BLOCK_N, BLOCK_K],
        "num_warps": NUM_WARPS,
        "num_stages": NUM_STAGES,
        "programs": programs,
        "status": "pass",
        "decision": decision,
        "target_compatible": target_compatible,
        "minimum_speedup": MIN_SPEEDUP,
        "target_us": TARGET_US,
        "current_median_us": current_us,
        "candidate_median_us": candidate_us,
        "speedup": speedup,
        "correctness": correctness,
        "current_samples_us": timed[0]["times_us"],
        "candidate_samples_us": timed[1]["times_us"],
    }
    print(
        f"RESULT current={current_us:.3f} us candidate={candidate_us:.3f} us "
        f"speedup={speedup:.3f}x decision={decision} "
        f"target_compatible={target_compatible}"
    )
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
