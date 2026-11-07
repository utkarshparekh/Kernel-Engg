"""Gate one calculated QuACK SM90 tile for pooler M=256.

The 64x128 output tile creates ceil(256/64) * ceil(4096/128) = 128
output tiles, nearly filling the H200's 132 SMs in one wave.  This benchmark
compiles exactly that TMA/WGMMA kernel; it does not autotune or sweep configs.
It advances only after challenge-checker correctness, CUDA-graph capture, and
alternating cold-L2 timing against the current integrated submission.
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import Callable

import torch
from quack.gemm import gemm

import reference
from submission import kernel_fn, prepare


LAYER = "pooler"
M = 256
TILE_M = 64
TILE_N = 128
TILE_K = 64
CLUSTER_M = 1
CLUSTER_N = 1
REPEATS = 40
WARMUP_REPLAYS = 5
L2_FLUSH_BYTES = 64 * 1024 * 1024
MIN_SPEEDUP = 1.20
TARGET_US = 18.7
CANDIDATE = "quack_sm90_64x128x64_c1x1_persistent"


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
    weight: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    # QuACK's lower-level GEMM accepts A=[M,K], B=[N,K], D=[M,N].  It
    # returns a cached launch plan, so the caller returns the output buffer.
    gemm(
        x,
        weight,
        output,
        None,
        None,
        tile_M=TILE_M,
        tile_N=TILE_N,
        tile_K=TILE_K,
        cluster_M=CLUSTER_M,
        cluster_N=CLUSTER_N,
        pingpong=False,
        persistent=True,
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
    output_tiles = ((M + TILE_M - 1) // TILE_M) * (
        (n + TILE_N - 1) // TILE_N
    )
    print("GPU:", torch.cuda.get_device_name(device))
    print(f"Experiment: QuACK SM90 {LAYER} M={M} N={n} K={k}")
    print(
        f"Tile: {TILE_M}x{TILE_N}x{TILE_K}, "
        f"cluster={CLUSTER_M}x{CLUSTER_N}, output_tiles={output_tiles}"
    )

    weight = make_weight(device).contiguous()
    weights = {LAYER: weight}
    state = prepare(weights)
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

    print("Compiling one exact QuACK configuration...")
    launch_candidate(correctness_inputs[0][1]["x"], weight, candidate_output)
    torch.cuda.synchronize()

    correctness = {}
    for distribution, data in correctness_inputs:
        output = launch_candidate(data["x"], weight, candidate_output)
        torch.cuda.synchronize()
        ok, message = reference.check_implementation(data, output, weights)
        correctness[distribution] = {"ok": ok, "message": message}
        print(f"Correctness {distribution}: ok={ok} {message}")

    output_path = (
        Path(os.environ["GPU_RESULTS_DIR"]) / "quack_pooler_m256.json"
    )
    if not all(item["ok"] for item in correctness.values()):
        result = {
            "candidate": CANDIDATE,
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
        return launch_candidate(timing_data["x"], weight, candidate_output)

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
            f"QuACK failed after CUDA graph capture: {candidate_message}"
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
            "name": CANDIDATE,
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
        "candidate": CANDIDATE,
        "layer": LAYER,
        "m": M,
        "n": n,
        "k": k,
        "tile": [TILE_M, TILE_N, TILE_K],
        "cluster": [CLUSTER_M, CLUSTER_N],
        "persistent": True,
        "pingpong": False,
        "output_tiles": output_tiles,
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
