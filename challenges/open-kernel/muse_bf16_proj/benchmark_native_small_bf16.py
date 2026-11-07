"""Evaluate installed H200-native kernels for eligible small-M BF16 projections.

The candidate set is derived from each native kernel's published eligibility
rules.  There is no tactic sweep:

* M=1 uses SGLang's H200-tuned Hopper BF16 GEMV for all four projections.
* M=8/16 uses SGLang TinyGEMM only for the shapes it explicitly supports.

Every candidate is checked on normal, outlier, and forced-probe inputs before
being compared with the current submission under cold-L2 CUDA-graph replay.
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import Callable

import torch

import reference
from sglang.kernels.ops.gemm.hopper_bf16_gemv import (
    hopper_bf16_gemv,
    use_hopper_bf16_gemv,
)
from sglang.kernels.ops.gemm.tiny_gemm import (
    can_use_tiny_gemm,
    tiny_gemm_bf16,
)
from submission import kernel_fn, prepare


REPEATS = 40
WARMUP_REPLAYS = 5
L2_FLUSH_BYTES = 64 * 1024 * 1024
MIN_MEANINGFUL_SPEEDUP = 1.10

CASES = (
    ("hopper_gemv", "qkv_gate", 1),
    ("hopper_gemv", "o", 1),
    ("hopper_gemv", "pooler", 1),
    ("hopper_gemv", "unpooler", 1),
    ("tiny_gemm", "qkv_gate", 8),
    ("tiny_gemm", "pooler", 8),
    ("tiny_gemm", "unpooler", 8),
    ("tiny_gemm", "qkv_gate", 16),
    ("tiny_gemm", "pooler", 16),
)


def make_weight(layer: str, device: torch.device) -> torch.Tensor:
    n, k = reference.LAYERS[layer]
    generator = torch.Generator(device=device).manual_seed(
        reference.WEIGHTS_SEED + sum(map(ord, layer))
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


def make_forced_probe_input(
    layer: str,
    m: int,
    device: torch.device,
) -> dict:
    n, k = reference.out_features(layer, m), reference.LAYERS[layer][1]
    x = torch.zeros((m, k), device=device, dtype=torch.bfloat16)
    rows = torch.arange(m, device=device)
    candidate_columns = torch.tensor(
        [0, 1, 31, 127, k // 2, k - 128, k - 2, k - 1],
        device=device,
    )
    columns = candidate_columns[rows % len(candidate_columns)]
    candidate_values = torch.tensor(
        [8, -16, 32, -64, 128, -8, 16, -32],
        device=device,
        dtype=torch.bfloat16,
    )
    values = candidate_values[rows % len(candidate_values)]
    x[rows, columns] = values
    return {
        "x": x,
        "config": {"m": m, "n": n, "k": k, "layer": layer},
    }


def capture(callable_: Callable[[], torch.Tensor]) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    output = callable_()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = callable_()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def check_output(data: dict, output: torch.Tensor, weights: dict) -> tuple[bool, str]:
    return reference.check_implementation(data, output, weights)


def main() -> None:
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(device))
    print("Cases:", CASES)

    l2_flush = torch.empty(
        L2_FLUSH_BYTES // 4,
        device=device,
        dtype=torch.float32,
    )
    all_results: list[dict] = []

    for candidate_name, layer, m in CASES:
        weight = make_weight(layer, device).contiguous()
        weights = {layer: weight}
        state = prepare(weights)
        n, k = reference.out_features(layer, m), reference.LAYERS[layer][1]
        candidate_weight = weight[:n]

        correctness_inputs = (
            (
                "normal",
                reference.generate_input(
                    m,
                    layer,
                    seed=20261030 + m + sum(map(ord, layer)),
                    device=str(device),
                ),
            ),
            (
                "outliers",
                reference.generate_input(
                    m,
                    layer,
                    seed=20261130 + m + sum(map(ord, layer)),
                    dist="outliers",
                    device=str(device),
                ),
            ),
            ("forced_probes", make_forced_probe_input(layer, m, device)),
        )

        if candidate_name == "hopper_gemv":
            eligible = use_hopper_bf16_gemv(m, n, k)

            def candidate_call(x: torch.Tensor) -> torch.Tensor:
                return hopper_bf16_gemv(x, candidate_weight)

        else:
            eligible = m <= 16 and can_use_tiny_gemm(n, k, max_m=16)
            candidate_output = torch.empty(
                (m, n),
                device=device,
                dtype=torch.bfloat16,
            )

            def candidate_call(x: torch.Tensor) -> torch.Tensor:
                return tiny_gemm_bf16(
                    x,
                    candidate_weight,
                    out=candidate_output,
                    max_m=16,
                )

        if not eligible:
            raise RuntimeError(
                f"Research eligibility mismatch for {candidate_name} {layer} M={m}"
            )

        failures = []
        for distribution, data in correctness_inputs:
            candidate_output_for_check = candidate_call(data["x"])
            torch.cuda.synchronize()
            ok, message = check_output(data, candidate_output_for_check, weights)
            if not ok:
                failures.append(f"{distribution}: {message}")

        if failures:
            print(
                f"REJECT {candidate_name} {layer} M={m}: "
                + "; ".join(failures)
            )
            all_results.append(
                {
                    "candidate": candidate_name,
                    "layer": layer,
                    "m": m,
                    "status": "incorrect",
                    "failures": failures,
                }
            )
            continue

        timing_data = correctness_inputs[0][1]

        def baseline_call() -> torch.Tensor:
            return kernel_fn(timing_data, state)

        def timed_candidate_call() -> torch.Tensor:
            return candidate_call(timing_data["x"])

        baseline_graph, baseline_output = capture(baseline_call)
        candidate_graph, captured_candidate_output = capture(timed_candidate_call)

        baseline_ok, baseline_message = check_output(
            timing_data, baseline_output, weights
        )
        candidate_ok, candidate_message = check_output(
            timing_data, captured_candidate_output, weights
        )
        if not baseline_ok:
            raise RuntimeError(
                f"Current submission failed {layer} M={m}: {baseline_message}"
            )
        if not candidate_ok:
            raise RuntimeError(
                f"Captured candidate failed {layer} M={m}: {candidate_message}"
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
                "name": candidate_name,
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
        decision = "accept" if speedup >= MIN_MEANINGFUL_SPEEDUP else "reject"
        result = {
            "candidate": candidate_name,
            "layer": layer,
            "m": m,
            "n": n,
            "k": k,
            "status": "pass",
            "decision": decision,
            "current_median_us": current_us,
            "candidate_median_us": candidate_us,
            "speedup": speedup,
            "current_samples_us": timed[0]["times_us"],
            "candidate_samples_us": timed[1]["times_us"],
        }
        all_results.append(result)
        print(
            f"RESULT {candidate_name} {layer} M={m}: "
            f"current={current_us:.3f} us candidate={candidate_us:.3f} us "
            f"speedup={speedup:.3f}x decision={decision}"
        )

        del state, weights, weight, candidate_weight
        torch.cuda.empty_cache()

    output_path = Path(os.environ["GPU_RESULTS_DIR"]) / "native_small_bf16.json"
    output_path.write_text(json.dumps(all_results, indent=2) + "\n")
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
