"""Gate FlashInfer cuTile on the highest-impact large-M projection.

This is deliberately one experiment, not a tactic sweep.  Pooler M=256 has
shape [256, 8192] @ [8192, 4096] and the largest remaining BF16 latency gap.
The installed cuTile backend provides a persistent SM90 GEMM with per-shape
autotuning, so the first eager call is allowed to tune before correctness,
CUDA-graph capture, and cold-L2 timing.

The candidate advances only if it is correct and at least 20% faster than the
current integrated submission.  A target-compatible result is <= 18.7 us.
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import Callable

import torch
from flashinfer.gemm import mm_bf16

import reference
from submission import kernel_fn, prepare


LAYER = "pooler"
M = 256
REPEATS = 40
WARMUP_REPLAYS = 5
L2_FLUSH_BYTES = 64 * 1024 * 1024
MIN_SPEEDUP = 1.25  # candidate <= 80% of current latency
TARGET_US = 18.7


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


def capture(callable_: Callable[[], torch.Tensor]) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
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
    print("GPU:", torch.cuda.get_device_name(device))
    print(f"Experiment: cuTile {LAYER} M={M} N={n} K={k}")

    weight = make_weight(device).contiguous()
    weights = {LAYER: weight}
    state = prepare(weights)

    # FlashInfer requires B=[K,N] in column-major layout.  Transposing the
    # contiguous challenge weight [N,K] creates exactly that view; calling
    # contiguous() here would silently change the required layout.
    candidate_weight = weight[:n].t()
    assert candidate_weight.shape == (k, n)
    assert candidate_weight.stride() == (1, k)
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

    def candidate_call(x: torch.Tensor) -> torch.Tensor:
        return mm_bf16(
            x,
            candidate_weight,
            out=candidate_output,
            out_dtype=torch.bfloat16,
            backend="cutile",
        )

    # The first call performs cuTile's per-shape exhaustive autotune.  It is
    # intentionally outside every correctness and timing measurement.
    print("Autotuning cuTile once for this exact shape...")
    candidate_call(correctness_inputs[0][1]["x"])
    torch.cuda.synchronize()

    correctness = {}
    for distribution, data in correctness_inputs:
        output = candidate_call(data["x"])
        torch.cuda.synchronize()
        ok, message = reference.check_implementation(data, output, weights)
        correctness[distribution] = {"ok": ok, "message": message}
        print(f"Correctness {distribution}: ok={ok} {message}")

    if not all(item["ok"] for item in correctness.values()):
        result = {
            "candidate": "flashinfer_cutile",
            "layer": LAYER,
            "m": M,
            "n": n,
            "k": k,
            "status": "incorrect",
            "correctness": correctness,
        }
        output_path = Path(os.environ["GPU_RESULTS_DIR"]) / "cutile_pooler_m256.json"
        output_path.write_text(json.dumps(result, indent=2) + "\n")
        print("REJECT: correctness failed")
        print("Saved:", output_path)
        return

    timing_data = correctness_inputs[0][1]

    def baseline_call() -> torch.Tensor:
        return kernel_fn(timing_data, state)

    def timed_candidate_call() -> torch.Tensor:
        return candidate_call(timing_data["x"])

    baseline_graph, baseline_output = capture(baseline_call)
    candidate_graph, captured_candidate_output = capture(timed_candidate_call)

    baseline_ok, baseline_message = reference.check_implementation(
        timing_data, baseline_output, weights
    )
    candidate_ok, candidate_message = reference.check_implementation(
        timing_data, captured_candidate_output, weights
    )
    if not baseline_ok:
        raise RuntimeError(f"Current submission failed: {baseline_message}")
    if not candidate_ok:
        raise RuntimeError(f"cuTile failed after CUDA graph capture: {candidate_message}")

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
            "name": "flashinfer_cutile",
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
        "candidate": "flashinfer_cutile",
        "layer": LAYER,
        "m": M,
        "n": n,
        "k": k,
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
    output_path = Path(os.environ["GPU_RESULTS_DIR"]) / "cutile_pooler_m256.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
