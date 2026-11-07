# Uncompressed router M=8 — isolated candidate

One original Triton implementation, one configuration, one shape.

## Latest measured result and integration status — 2026-10-31

The approved H200 run passed all 19 correctness checks, verified Tensor Core
instructions, and reported zero spilling. Cold-L2 raw graph medians were
16.928 us current versus 13.376 us candidate. Paired speedup was 1.28925x,
95% interval [1.26519, 1.32765], over 60 pairs. The sandbox terminated normally.

[Measured JSON](/Users/utkarshparekh/Kernel-Engg/results/20261031T194826Z-78b85f33/artifacts/router_m8_hilo.json)
and saved PTX are the authoritative experiment evidence. Do not rerun this
configuration merely because earlier sections describe the original protocol.

The user approved integration. Active `submission.py` now contains the exact
tested kernels under unique names, the same launch settings, a prepared K-axis
hi/lo view, and a route guarded to contiguous M=8/N=256/K=8192 router inputs.
All other dispatch code and the original FP32 fallback remain unchanged.
An AST comparison verified exact kernel/launch equivalence and that removing
the new route and prepared view restores the previous source AST. All seven
CPU checks pass.

Integrated source SHA-256:
`1cf94ccd746ae6a802766b69fcedc6a855064ecbbeeff26444dad9ccfc11bb9a`.
The approved integrated official check completed as job `f4437a5ffbdf4d09`:
**1.20591791331x**, 95% interval [1.187829, 1.224007], 9/9 tests passed, no
flags. Reported step time is 4,005.911 us. M=8 router adjusted time was
9.7015 us, compared with 13.2403 us in the preceding check.

[Official profile](/Users/utkarshparekh/Kernel-Engg/challenges/open-kernel/muse_bf16_proj/profiles/f4437a5ffbdf4d09.json).
The aggregate point estimate rose from 1.18706x to 1.20592x; the cross-run
intervals overlap. The local paired M8 comparison is the stronger evidence for
this kernel's improvement. No leaderboard submission was made. This source is
the new verified checkpoint; 1.55x still requires about 22.2% less aggregate time.

## Evidence and question

The service identified the exact active source as previously checked in job
`6750c2608ef6433a`: 1.18706445196x, 9/9 passed, no flags. M=8 router time was
13.240 us adjusted, 17.312 us raw. Its gap to the saved published router result
was the largest individual score opportunity. Normalizing that published result
to our run's SGLang baseline screens at approximately 3.14% aggregate gain for
this one replacement. That is historical cross-run screening, **not a forecast**.

Question: can Tensor Core BF16 hi/lo computation with enough split-K programs
beat the existing FP32 streaming/reduction kernel, including scratch reduction?
This candidate changes no BF16 projection. It does not establish 1.55x overall.

## Representation and computation

The task explicitly permits FP32 router weights as a BF16 hi/lo pair:

```text
H = BF16(W)
L = BF16(W - FP32(H))
P = concatenate(H.T, L.T) along K       # [16384,256], BF16
Y ≈ [X, X] @ P                        # FP32 accumulation/output
```

`[X,X]` is addressed inside the kernel using `logical_k % 8192`; it is never
materialized. The weight pair has the **same 8 MiB** as the original FP32 W.
There is no compression, codec, exception table, or weight-byte reduction.
Hi/lo is the task's permitted approximation, not a claim of bit-exact FP32
reconstruction. Plain BF16 weights alone are not used.

| Fixed parameter | Value |
|---|---:|
| Input / output | BF16 [8,8192] / FP32 [8,256] |
| Requested output tile | 16x32, with eight masked rows |
| K tile / logical K | 128 / 16384 |
| K splits / K per split | 16 / 1024 |
| Partial programs | 8 feature tiles x 16 splits = 128 |
| Warps / pipeline stages | 4 / 2 |
| Reduction programs | 16 |
| GPU kernel launches | 2 |
| FP32 partial workspace | 16x8x256x4 = 128 KiB |

Each partial slot has one writer and is fully initialized on every invocation.
A second kernel sums all 16 FP32 partials. No atomics, cross-CTA semaphores,
output caching or activation-derived preparation is used. Output and partial
storage are allocated per invocation and captured by CUDA graphs.

## Cost accounting and acceptance

- Weight payload: 8 MiB, unchanged; unique X: 128 KiB; output: 8 KiB.
- Logical activation reads: 2 MiB across feature tiles and hi/lo planes.
- Scratch writes plus reads: 256 KiB, potentially served through cache.
- Useful hi/lo work: 67,108,864 FLOPs. The requested padded tile entails
  134,217,728 FLOPs; compiler lowering may add further padding.
- Actual HBM traffic and stall causes are unmeasured. Logical requests are
  not reported as physical DRAM transfers. Two launches and scratch reduction
  are part of the timed call and may erase any compute advantage.

The benchmark uses an exact snapshot of current code as the control, identical
weights/inputs, 128 MiB read sweeps before each graph replay, and balanced AB/BA
order. Thirty cycles produce 60 pairs. Complete cycles are bootstrapped for a
95% interval with a conservative 1.5% uncertainty floor. Raw graph times are
not directly equated with the judge's adjusted score.

Before timing, require official reference correctness on normal, outlier and
forced-probe inputs, a second weight seed, retained-output ownership, and
graph replay correctness after changing inputs. Save PTX and resource counts;
require Tensor Core MMA instructions and zero reported spills.

Advance only if the significant paired win also cuts raw time by at least 20%.
There is no automatic configuration sweep, expansion, integration, official
check or submission. A failure is retained as evidence before further work.

## Local checks and proposed execution

Seven standard-library checks cover byte budgets, K coverage, scratch address
ownership, the necessary low residual, scalar hi/lo split-K math, statistics
direction and source/snapshot integrity. They cannot validate CUDA execution.

Local command, already run in the repo venv:

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/utkarshparekh/Kernel-Engg/.venv/bin/python \
  /Users/utkarshparekh/Kernel-Engg/challenges/open-kernel/muse_bf16_proj/candidates/router_m8_hilo/benchmark.py --cpu-test
```

Previously approved and completed H200 command (not authorization to repeat):

```bash
cd /Users/utkarshparekh/Kernel-Engg
GPU_PROFILE=0 MODAL_BASE_IMAGE=lmsysorg/sglang:dev-muse-spark-open \
  just provider=modal gpu=H200 run \
  challenges/open-kernel/muse_bf16_proj/candidates/router_m8_hilo/benchmark.py
```

The runner uploads this directory, limits execution to 240 seconds, collects
artifacts under the repo's `results`, and terminates its sandbox. The JSON is
`artifacts/router_m8_hilo.json`, with `partial.ptx` and `finish.ptx` beside it.

The following integrated check was approved and completed. Do not repeat it
for unchanged source; retrieve saved job `f4437a5ffbdf4d09` instead:

```bash
cd /Users/utkarshparekh/Kernel-Engg
just okc-check
```

## References and boundaries

- Local official `TASK.md`: FP32 router precision, permitted hi/lo, graph timing.
- [Triton matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
  and [tl.dot API](https://triton-lang.org/main/python-api/generated/triton.language.dot.html):
  standard tiled multiplication, BF16 inputs and FP32 accumulation.
- [NathenFernandes public solution](https://github.com/NathenFernandes/muse-bf16-proj/tree/solution):
  reference evidence for hi/lo Tensor Core routers and split-K. No competitor
  kernel is copied into this candidate; this is not a clean-room claim.
- [Pierre-Louis Delcroix's post](https://www.linkedin.com/posts/activity-7521877887277408256-qHvb/):
  reports 1.55x dense projections and #29 overall across problems, and emphasizes
  fusion, shape-specific choices and measurement. Its short link leads to the
  leaderboard, not source code; it provides no new configuration to reproduce.

The displayed dense-projection board checked on 2026-10-31 showed 1.59x at #20
and 1.58x at #50. These are rounded, provisional values. The user's working
milestone remains 1.55x; no rank is promised for that score.

`baseline_snapshot.py`, `reference.py` and `checks.py` are exact copies of the
user's existing files. Only the standalone benchmark imports reference checks.
Baseline SHA-256: `2f8f50f019238f3916b1f2cd3105840751f57cf359db89e9772c5ab7b689c3ae`.
