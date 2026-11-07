"""Portable plan/math for the approved single M=8 router experiment."""

import math
import random
import statistics
import struct

M, N, K = 8, 256, 8192
BM, BN, BK = 16, 32, 128
SPLITS, WARPS, STAGES = 16, 4, 2
LOGICAL_K = 2 * K
PART_K = LOGICAL_K // SPLITS
BASELINE_SHA256 = "2f8f50f019238f3916b1f2cd3105840751f57cf359db89e9772c5ab7b689c3ae"


def geometry():
    return {"m": M, "n": N, "k": K, "logical_k": LOGICAL_K,
            "block_m": BM, "block_n": BN, "block_k": BK,
            "splits": SPLITS, "warps": WARPS, "stages": STAGES,
            "partial_programs": (N // BN) * SPLITS,
            "reduction_programs": M * N // 128,
            "k_iterations_per_program": PART_K // BK,
            "original_weight_bytes": N * K * 4,
            "execution_weight_bytes": LOGICAL_K * N * 2,
            "scratch_bytes": SPLITS * M * N * 4,
            "unique_activation_bytes": M * K * 2,
            "logical_activation_read_bytes": 2 * (N // BN) * M * K * 2,
            "scratch_read_plus_write_bytes": 2 * SPLITS * M * N * 4,
            "output_bytes": M * N * 4,
            "useful_hilo_flops": 4 * M * N * K,
            "requested_tile_flops": 4 * BM * N * K,
            "measured_dram_bytes": None}


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bf16(x):
    """Scalar round-to-nearest-even oracle for finite, nonoverflowing inputs."""
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def hilo(w):
    w = f32(w)
    high = bf16(w)
    return high, bf16(f32(w - high))


def paired_stats(samples, cycles):
    """Cluster bootstrap of AB+BA cycles, not independent timing selection."""
    blocks = [statistics.mean(math.log(s["current"] / s["candidate"])
                              for s in samples if s["cycle"] == i)
              for i in range(cycles)]
    rng = random.Random(862)
    boots = sorted(math.exp(statistics.mean(rng.choices(blocks, k=cycles)))
                   for _ in range(2000))
    point = math.exp(statistics.mean(blocks))
    return {"geometric_raw_speedup": point,
            "bootstrap_ci95": [boots[49], boots[1949]],
            "ci95_with_uncertainty_floor": [min(boots[49], point/1.015),
                                             max(boots[1949], point*1.015)],
            "minimum_relative_uncertainty": 0.015}
