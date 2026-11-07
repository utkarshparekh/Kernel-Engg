"""Compare a small, calculated set of FlashInfer QKV tactics on H200.

This is a standalone experiment. It does not import or modify submission.py.
Each candidate must pass correctness before it is CUDA-graph timed. Timed graph
replays follow an L2-sized cache flush so the comparison resembles the judge's
cold-L2 protocol.
"""

from __future__ import annotations

import dataclasses
import json
import os
import statistics
from pathlib import Path

import torch
from flashinfer.gemm.kernels.dense_bf16_gemm_warp_splitk import (
    WarpSplitKTactic,
    run_warp_splitk_dense,
)


M_VALUES = (1, 8)
N = 2304
K = 8192
REPEATS = 40
WARMUP_REPLAYS = 5
L2_FLUSH_BYTES = 64 * 1024 * 1024
BF16_TOLERANCE = 1e-2

CANDIDATES = (
    (
        "default",
        WarpSplitKTactic(
            output_tile=32,
            token_tile=8,
            k_tile=256,
            stages=11,
            b_loader_warps=2,
        ),
    ),
    (
        "more_ctas_two_resident",
        WarpSplitKTactic(
            output_tile=16,
            token_tile=8,
            k_tile=256,
            stages=9,
            b_loader_warps=2,
        ),
    ),
    (
        "one_loader_warp",
        WarpSplitKTactic(
            output_tile=16,
            token_tile=8,
            k_tile=256,
            stages=9,
            b_loader_warps=1,
        ),
    ),
    (
        "deep_pipeline_one_resident",
        WarpSplitKTactic(
            output_tile=16,
            token_tile=8,
            k_tile=256,
            stages=16,
            b_loader_warps=2,
        ),
    ),
    (
        "smaller_k_tile",
        WarpSplitKTactic(
            output_tile=16,
            token_tile=8,
            k_tile=128,
            stages=16,
            b_loader_warps=2,
        ),
    ),
)


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = actual.float() - expected.float()
    denominator = expected.float().norm().clamp(min=1e-12)
    return float(difference.norm() / denominator)


def logical_bandwidth_tb_s(m: int, elapsed_us: float) -> float:
    logical_bytes = 2 * (N * K + m * K + m * N)
    return logical_bytes / (elapsed_us * 1e-6) / 1e12


def tactic_dict(tactic: WarpSplitKTactic) -> dict[str, int]:
    return dataclasses.asdict(tactic)


def main() -> None:
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(device))
    print("Candidates:", len(CANDIDATES), "M values:", M_VALUES)

    generator = torch.Generator(device=device).manual_seed(20261030)
    weight = (
        torch.randn(
            (N, K),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        / (K**0.5)
    ).contiguous()
    weight_t = weight.t()

    # Writing a buffer larger than H200's 50 MB L2 before each measured replay
    # evicts the GEMM operands without adding that work to the event interval.
    l2_flush = torch.empty(
        L2_FLUSH_BYTES // 4,
        device=device,
        dtype=torch.float32,
    )

    all_results: list[dict] = []
    for m in M_VALUES:
        x = torch.randn(
            (m, K),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        expected = torch.mm(x.float(), weight_t.float()).to(torch.bfloat16)

        runnable = []
        for name, tactic in CANDIDATES:
            output = torch.empty((m, N), device=device, dtype=torch.bfloat16)
            try:
                # Compile and launch before capture. Compilation is never timed.
                run_warp_splitk_dense(x, weight_t, output, False, tactic)
                torch.cuda.synchronize()
                error = relative_l2(output, expected)
                if error > BF16_TOLERANCE:
                    print(
                        f"REJECT M={m} {name}: relative L2 {error:.3e} "
                        f"> {BF16_TOLERANCE:g}"
                    )
                    all_results.append(
                        {
                            "m": m,
                            "name": name,
                            "tactic": tactic_dict(tactic),
                            "status": "incorrect",
                            "relative_l2": error,
                        }
                    )
                    continue

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run_warp_splitk_dense(x, weight_t, output, False, tactic)
                graph.replay()
                torch.cuda.synchronize()

                captured_error = relative_l2(output, expected)
                if captured_error > BF16_TOLERANCE:
                    print(
                        f"REJECT M={m} {name}: captured relative L2 "
                        f"{captured_error:.3e} > {BF16_TOLERANCE:g}"
                    )
                    all_results.append(
                        {
                            "m": m,
                            "name": name,
                            "tactic": tactic_dict(tactic),
                            "status": "capture_incorrect",
                            "relative_l2": captured_error,
                        }
                    )
                    continue

                runnable.append(
                    {
                        "name": name,
                        "tactic": tactic,
                        "graph": graph,
                        "relative_l2": captured_error,
                        "times_us": [],
                        "start": torch.cuda.Event(enable_timing=True),
                        "end": torch.cuda.Event(enable_timing=True),
                    }
                )
                print(
                    f"PASS M={m} {name}: relative L2 "
                    f"{captured_error:.3e}"
                )
            except Exception as error:
                print(f"ERROR M={m} {name}: {type(error).__name__}: {error}")
                all_results.append(
                    {
                        "m": m,
                        "name": name,
                        "tactic": tactic_dict(tactic),
                        "status": "error",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )

        for _ in range(WARMUP_REPLAYS):
            for item in runnable:
                l2_flush.zero_()
                item["graph"].replay()
        torch.cuda.synchronize()

        # Alternate the candidate order to reduce first/last ordering bias.
        for repetition in range(REPEATS):
            order = runnable if repetition % 2 == 0 else tuple(reversed(runnable))
            for item in order:
                l2_flush.zero_()
                item["start"].record()
                item["graph"].replay()
                item["end"].record()
                item["end"].synchronize()
                item["times_us"].append(
                    item["start"].elapsed_time(item["end"]) * 1000.0
                )

        default_median = next(
            statistics.median(item["times_us"])
            for item in runnable
            if item["name"] == "default"
        )
        for item in runnable:
            times = item["times_us"]
            median_us = statistics.median(times)
            result = {
                "m": m,
                "name": item["name"],
                "tactic": tactic_dict(item["tactic"]),
                "status": "pass",
                "relative_l2": item["relative_l2"],
                "median_us": median_us,
                "minimum_us": min(times),
                "maximum_us": max(times),
                "speedup_vs_default": default_median / median_us,
                "logical_bandwidth_tb_s": logical_bandwidth_tb_s(m, median_us),
                "samples_us": times,
            }
            all_results.append(result)
            print(
                f"RESULT M={m} {item['name']}: median={median_us:.3f} us "
                f"min={min(times):.3f} us speedup={default_median / median_us:.3f}x "
                f"logical_BW={result['logical_bandwidth_tb_s']:.2f} TB/s"
            )

    output_path = Path(os.environ["GPU_RESULTS_DIR"]) / "qkv_tactics.json"
    output_path.write_text(json.dumps(all_results, indent=2) + "\n")
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
