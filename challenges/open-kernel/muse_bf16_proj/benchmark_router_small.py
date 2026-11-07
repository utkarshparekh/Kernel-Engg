"""Benchmark a fused exact-FP32 router kernel for M=1 and M=8 on H200.

This experiment is intentionally narrow.  The router has fixed N=256 and
K=8192, so one Triton program owns one output row, streams that FP32 weight row
once, and reuses it across all token rows.  This replaces F.linear's observed
multi-kernel path with one launch while preserving FP32 weights and FP32
accumulation.

Every candidate must pass the challenge's real correctness checker before it
is CUDA-graph timed.  Timed replays alternate with the F.linear baseline and
start after a buffer larger than H200's L2 has been written.
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl

import reference


N = 256
K = 8192
M_VALUES = (1, 8)
REPEATS = 40
WARMUP_REPLAYS = 5
L2_FLUSH_BYTES = 64 * 1024 * 1024

# These are the only materially distinct points worth testing for this fixed
# reduction.  BLOCK_K=256 is the expected balance; 128 tests more iterations
# and less register pressure, while 512 tests fewer iterations and more
# register pressure.  Eight warps are tested only where the larger reduction
# tile provides enough work to distribute.
CANDIDATES = (
    ("bk128_w4", 128, 4),
    ("bk256_w4", 256, 4),
    ("bk256_w8", 256, 8),
    ("bk512_w4", 512, 4),
    ("bk512_w8", 512, 8),
)


@triton.jit
def router_row_stream_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    M: tl.constexpr,
    N_SIZE: tl.constexpr,
    K_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One program computes every token's dot product for one expert row."""
    output_row = tl.program_id(0)
    token_offsets = tl.arange(0, M)
    k_offsets = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((M,), dtype=tl.float32)

    for k_start in range(0, K_SIZE, BLOCK_K):
        k = k_start + k_offsets
        weight = tl.load(weight_ptr + output_row * K_SIZE + k)
        activations = tl.load(x_ptr + token_offsets[:, None] * K_SIZE + k[None, :])
        products = activations.to(tl.float32) * weight[None, :]
        accumulator += tl.sum(products, axis=1)

    tl.store(output_ptr + token_offsets * N_SIZE + output_row, accumulator)


def launch_router(
    x: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    block_k: int,
    num_warps: int,
) -> None:
    router_row_stream_kernel[(N,)](
        x,
        weight,
        output,
        M=x.shape[0],
        N_SIZE=N,
        K_SIZE=K,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )


def make_forced_probe_input(m: int, device: torch.device) -> dict:
    """Make every row a checker probe, including edge columns and signs."""
    x = torch.zeros((m, K), device=device, dtype=torch.bfloat16)
    rows = torch.arange(m, device=device)
    columns = torch.tensor(
        [0, 1, 31, 127, K // 2, K - 128, K - 2, K - 1],
        device=device,
    )[:m]
    values = torch.tensor(
        [8, -16, 32, -64, 128, -8, 16, -32],
        device=device,
        dtype=torch.bfloat16,
    )[:m]
    x[rows, columns] = values
    return {
        "x": x,
        "config": {"m": m, "n": N, "k": K, "layer": "router"},
    }


def capture(callable_) -> torch.cuda.CUDAGraph:
    callable_()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        callable_()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def checked(data: dict, output: torch.Tensor, weights: dict) -> tuple[bool, str]:
    return reference.check_implementation(data, output, weights)


def main() -> None:
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(device))
    print("Candidates:", CANDIDATES)

    weights = reference.generate_weights(reference.WEIGHTS_SEED, device=str(device))
    router_weight = weights["router"].contiguous()
    l2_flush = torch.empty(
        L2_FLUSH_BYTES // 4,
        device=device,
        dtype=torch.float32,
    )
    all_results: list[dict] = []

    for m in M_VALUES:
        correctness_inputs = (
            (
                "normal",
                reference.generate_input(
                    m, "router", seed=20261030 + m, device=str(device)
                ),
            ),
            (
                "outliers",
                reference.generate_input(
                    m,
                    "router",
                    seed=20261130 + m,
                    dist="outliers",
                    device=str(device),
                ),
            ),
            ("forced_probes", make_forced_probe_input(m, device)),
        )
        timing_data = correctness_inputs[0][1]
        x = timing_data["x"]

        baseline_output = torch.empty((m, N), device=device, dtype=torch.float32)

        def baseline_call() -> None:
            torch.mm(x.float(), router_weight.t(), out=baseline_output)

        baseline_graph = capture(baseline_call)
        baseline_ok, baseline_message = checked(
            timing_data, baseline_output, weights
        )
        if not baseline_ok:
            raise RuntimeError(f"F.linear-equivalent baseline failed: {baseline_message}")

        runnable = []
        for name, block_k, num_warps in CANDIDATES:
            output = torch.empty((m, N), device=device, dtype=torch.float32)
            failures = []
            for distribution, data in correctness_inputs:
                launch_router(
                    data["x"], router_weight, output, block_k, num_warps
                )
                torch.cuda.synchronize()
                ok, message = checked(data, output, weights)
                if not ok:
                    failures.append(f"{distribution}: {message}")

            if failures:
                print(f"REJECT M={m} {name}: {'; '.join(failures)}")
                all_results.append(
                    {
                        "m": m,
                        "name": name,
                        "block_k": block_k,
                        "num_warps": num_warps,
                        "status": "incorrect",
                        "failures": failures,
                    }
                )
                continue

            def candidate_call(
                output=output,
                block_k=block_k,
                num_warps=num_warps,
            ) -> None:
                launch_router(x, router_weight, output, block_k, num_warps)

            graph = capture(candidate_call)
            captured_ok, captured_message = checked(
                timing_data, output, weights
            )
            if not captured_ok:
                print(f"REJECT M={m} {name} after capture: {captured_message}")
                all_results.append(
                    {
                        "m": m,
                        "name": name,
                        "block_k": block_k,
                        "num_warps": num_warps,
                        "status": "capture_incorrect",
                        "failure": captured_message,
                    }
                )
                continue

            print(f"PASS M={m} {name}: {captured_message}")
            runnable.append(
                {
                    "name": name,
                    "block_k": block_k,
                    "num_warps": num_warps,
                    "graph": graph,
                    "times_us": [],
                }
            )

        timed = [
            {
                "name": "baseline_torch_mm",
                "graph": baseline_graph,
                "times_us": [],
            },
            *runnable,
        ]
        for item in timed:
            item["start"] = torch.cuda.Event(enable_timing=True)
            item["end"] = torch.cuda.Event(enable_timing=True)

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

        baseline_median = statistics.median(timed[0]["times_us"])
        for item in timed:
            times = item["times_us"]
            median_us = statistics.median(times)
            result = {
                "m": m,
                "name": item["name"],
                "status": "pass",
                "median_us": median_us,
                "minimum_us": min(times),
                "maximum_us": max(times),
                "speedup_vs_baseline": baseline_median / median_us,
                "samples_us": times,
            }
            if "block_k" in item:
                result["block_k"] = item["block_k"]
                result["num_warps"] = item["num_warps"]
            all_results.append(result)
            print(
                f"RESULT M={m} {item['name']}: median={median_us:.3f} us "
                f"min={min(times):.3f} us "
                f"speedup={baseline_median / median_us:.3f}x"
            )

    output_path = Path(os.environ["GPU_RESULTS_DIR"]) / "router_small.json"
    output_path.write_text(json.dumps(all_results, indent=2) + "\n")
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
