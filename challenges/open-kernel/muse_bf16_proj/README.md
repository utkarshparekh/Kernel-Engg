# Muse BF16 projection challenge

This directory is initialized from `okc init muse_bf16_proj`. `submission.py`
must remain self-contained; local helpers are not uploaded by the judge.

## Development loop

```bash
just okc-smoke   # Modal H200, official SGLang image
just okc-check   # held-out correctness and timing, unscored
just okc-submit  # leaderboard submission
```

The first measured version uses pre-transposed weights with `torch.mm` as a
strong cuBLAS baseline. The retained Triton kernel is the next optimization seam:
specialize it with split-K only after it beats the measured baseline.
