# Pooler M=16: one controlled lossless-compression experiment

Status: local CPU tests pass; CUDA compilation, GPU correctness and performance
are **unverified**. This directory is isolated from the active submission.

## Question and acceptance gate

For `X[16,8192] @ W[4096,8192].T`, can fewer weight bytes outweigh decoding cost?
Run exactly one configuration, with three arms:

1. `packed`: the new register-fed BF16 Tensor Core kernel with exact packing.
2. `raw`: the **same kernel binary**, with every weight record stored as raw BF16.
3. `current`: an exact snapshot of the active FlashInfer implementation.

The candidate must preserve every weight bit and pass normal, outlier and forced
single-channel probes before timing. Local memory in the compute kernel stops
the run before timing. A technical win requires packed time below **both** other
arms, with the paired 95% interval above 1x (a conservative 1.5% minimum
uncertainty is applied). A 20% time reduction against current is reported as a
separate stretch result. A small technical win does not automatically justify
integration. There is no automatic integration, official check or submission.

This tests one technique on one shape. It cannot establish 1.55–1.57x aggregate
performance; expansion requires evidence that each expensive shape benefits.

## Format: exact values, explicit fallback cost

Each warp owns eight output features. One record contains their next 64 weights
per feature: **512 BF16 values**. Arrange them in the register fragments required
by `mma.sync.m16n8k16`, then choose one of these representations:

| Representation | Contents | Bytes/record |
|---|---|---:|
| Packed | 512 exact low bytes + 512 four-bit sign/exponent codes | 768 |
| Raw fallback | All 512 original BF16 bit patterns | 1,024 |
| Descriptor, both arms | Offset, exponent base, raw flag | 4 |

The low byte retains the lowest exponent bit and all seven mantissa bits.
For the upper seven exponent bits `e=(bits>>8)&127`, packing is legal only if
`max(e)-min(e)<=7`. Three code bits store `e-base`; the fourth stores the sign.
Otherwise the **whole record** remains raw BF16. Thus all finite values,
subnormals, zeros, infinities and NaN payloads have an exact representation.
The compute path reconstructs original BF16 bits before MMA. No quantization,
correction GEMM or approximate weight substitution is used.

Descriptors contain `(byte_offset/16)` in bits 0–23, exponent base in bits 24–30,
and the raw flag in bit 31. Blocks are 16-byte aligned. Payload records are
ordered by K unit, output CTA, warp; descriptors are CTA-major and copied to
shared memory once at kernel entry. The GPU packer is checked against an
independent scalar byte-format oracle on deliberately mixed records.

## Compute configuration

```text
128 CTAs × 128 threads (4 warps)
       |
       +-- CTA output: 16 tokens × 32 features
       +-- warp output: 16 tokens × 8 features
       +-- 8 records prefetched into registers per warp
       +-- X: cp.async → two swizzled shared buffers → ldmatrix → registers
       +-- W: global payload → registers → exact decode (or raw bypass)
       +-- mma.sync.m16n8k16 BF16 inputs, FP32 accumulators → BF16 output
```

K advances 512 columns per outer iteration, with four MMA steps per 64-column
record. The raw and packed arms share activation staging, descriptor access,
tiling, accumulator mapping and output stores. Only weight loading and decoding
differ. Static shared storage is 32 KiB for X plus 2 KiB for descriptors.
Compiler-reported registers, local memory and occupancy are saved by the run.
No WGMMA/TMA shared-weight expansion, split-K scratch or correction buffer is
introduced. This specifically avoids repeating the earlier wide-M shared-weight
expansion experiment; it does not assume those results predict this kernel.

## Byte and compute accounting

- Work: `2 × 16 × 4096 × 8192 = 1,073,741,824 FLOPs`.
- Original weight payload: **64 MiB**.
- If every record packs: **48 MiB** payload, plus **256 KiB** descriptors.
- Raw fallback adds 256 bytes per fallback record. Actual counts are measured
  during preparation; 48 MiB is not assumed to be total traffic.
- Unique X: 256 KiB; output: 128 KiB.
- Logical X global reads across 128 CTAs: **32 MiB** in both arms. These may hit
  L2. Logical shared writes are 32 MiB and ldmatrix reads 128 MiB.
- No correction table or temporary output buffer. Packing allocations are
  untimed preparation costs and are released; payload/metadata are retained.

With no raw fallbacks, the payload reduction is 16 MiB. At 4.8 TB/s that is
**3.495 microseconds of ideal bandwidth time**, not a runtime prediction.
Decoding, address instructions, occupancy and operand preparation can erase it.
The JSON distinguishes compulsory-byte lower bounds, logical global requests
and **unmeasured** actual DRAM traffic. Actual HBM counters require a separately
approved profiler run; this benchmark does not claim to collect them.

## Validation and timing protocol

Local standard-library tests cover all 65,536 BF16 bit patterns, byte-permutation
decoding, raw fallback, mixed offsets, fragment coverage, swizzle bijection,
ldmatrix address mapping, output ownership mapping and statistics direction.
These are not a replacement for CUDA validation.

The GPU script will:

1. Compile once and reject nonzero compute-kernel local-memory usage.
2. Check mixed-format GPU packing against the scalar oracle, and restore both
   full weight representations bit for bit.
3. Generate the official seeded weights. Check all three implementations against
   FP64 references for normal, outlier and forced-probe inputs.
4. Check retained output ownership and replay correctness after changing X.
5. Capture each arm as a CUDA graph. Before **every** timed replay, read a 128 MiB
   flush buffer outside the event interval. Use all six arm orders in each of ten
   balanced cycles: 60 samples per arm, one configuration, no tuning sweep.
6. Bootstrap complete balanced cycles for paired speedup intervals, with a 1.5%
   uncertainty floor. Save samples, byte accounting, source hashes, resource
   usage, failures and gate results to `GPU_RESULTS_DIR/pooler_m16_ab.json`.

Reported times are **raw graph event times**. The official judge subtracts a
fixed graph cost; do not equate these raw results with official shape times or
an aggregate leaderboard score. This local test consumes no official quota.

## Commands

Local validation, already run:

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/utkarshparekh/Kernel-Engg/.venv/bin/python \
  /Users/utkarshparekh/Kernel-Engg/challenges/open-kernel/muse_bf16_proj/candidates/pooler_m16_ab/benchmark.py --cpu-test
```

**Pending separate approval:** one H200 sandbox using the existing `just run`
utility, its challenge image, artifact collection and automatic termination:

```bash
cd /Users/utkarshparekh/Kernel-Engg
GPU_PROFILE=0 MODAL_BASE_IMAGE=lmsysorg/sglang:dev-muse-spark-open \
  just provider=modal gpu=H200 run \
  challenges/open-kernel/muse_bf16_proj/candidates/pooler_m16_ab/benchmark.py
```

The existing runner limits its execution command to 240 seconds and sandbox to
300 seconds. Compilation may fail or exhaust that budget; it is not a measured
win. No additional GPU calls are scheduled automatically.

## Sources and provenance

The implementation is independently written after studying public approaches;
it is **not a clean-room claim**. Public work supplied the direction that exact
BF16 packing with a small-M Tensor Core path is worth investigating. This
experiment uses a per-record raw fallback instead of a sparse correction path.

- [NathenFernandes/muse-bf16-proj, solution branch](https://github.com/NathenFernandes/muse-bf16-proj/tree/solution)
  — public reference for direction; no competitor kernel is copied into this candidate.
- [NVIDIA PTX ISA: m16n8k16 fragments](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float)
  — A/B/C register mapping, BF16 MMA, ldmatrix and asynchronous copy semantics.
- Local official `TASK.md` permits lossless weight compression in `prepare()`.
  The candidate imports no reference/harness module. Only the standalone
  benchmark imports local helper copies to generate inputs and validate outputs.

`baseline_snapshot.py`, `reference.py`, and `checks.py` are exact snapshots of
the user's existing files for this self-contained benchmark upload. Baseline
SHA-256: `2f8f50f019238f3916b1f2cd3105840751f57cf359db89e9772c5ab7b689c3ae`.
The active `submission.py` is unchanged.
