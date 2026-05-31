# Progressive Resolution Benchmarking Guide

## Purpose
This document explains the benchmark design, how to reproduce results, and how
to interpret them.  Keep it updated as new experiments are added.

---

## Quick Start
```bash
# Full suite (all groups, 50 steps, seed 42)
bash scratch/test_progressive_benchmark.sh

# Single group
bash scratch/test_progressive_benchmark.sh --group A

# Custom steps / seed
bash scratch/test_progressive_benchmark.sh --steps 20 --seed 0

# Legacy sweep (various progressive configs, original script)
bash scratch/test_progressive_gen.sh --steps 50
```

---

## Experimental Design

### Controlled Variables (same for ALL runs)
| Variable | Value |
|---|---|
| Model | FLUX.1-dev (`/miele/brian/modelscope/black-forest-labs/FLUX.1-dev`) |
| Resolution | 1024×1024 |
| Attention backend | `torch_sdpa` |
| Steps | 50 (default) |
| Seed | 42 (default) |
| DIT offload | `false` — transformer GPU-resident for all runs |
| Text/VAE offload | layerwise (auto, small overhead, not measured) |

**Why disable DIT CPU offload?**  
With offload enabled, every step pays ~0.41 s to PCIe-transfer the 22 GB
transformer regardless of sequence length.  This constant overhead dilutes the
quadratic attention savings that make progressive generation fast, depressing
measured speedup from ~1.6× to ~1.35×.  The paper measures pure compute; we
match that by disabling offload.

### Benchmark Groups

| Group | Config | Purpose |
|---|---|---|
| **A** | Fullres + Progressive, GPU-resident, no opts | Core comparison; mirrors paper setup |
| **B** | Fullres + TeaCache | Best-effort fullres; sets the competitive bar |
| **C** | Progressive + TeaCache | Compatibility test; may fail if TeaCache cache state is not reset at stage transitions |

### Configs Within Group A

| ID | Config | Stage split (50 steps) | Expected speedup |
|---|---|---|---|
| A1 | fullres | 50 @ 128² | 1.00× (baseline) |
| A2 | dct_rewind L1 δ=0.01 | 18@64² + 32@128² | ~1.37× (token-step) |
| A3 | dct_rewind L1 δ=0.05 | 28@64² + 22@128² | ~1.72× (token-step) |
| A4 | dct_rewind L2 δ=0.01 | 10@32² + 8@64² + 32@128² | ~1.45× (token-step) |

**Stage split** = how many denoising steps run at each latent resolution.
Transition points are computed from the Bayes-optimal spectrum criterion
(paper Eq. 142-146) applied to FLUX's actual sigma schedule (dynamic shift
μ=1.15 for 1024×1024).

**Token-step speedup** = fullres_token_steps / progressive_token_steps (linear
model of compute).  Actual speedup may differ due to fixed-overhead per step
and quadratic attention for long sequences.

---

## How to Read the Results

### Speedup formula
```
speedup = A1_total_s / run_total_s
```

### Key comparisons
- **A1 vs A2/A3/A4**: Does progressive outperform fullres without any optimization?
- **B1 vs A2**: Does fullres + TeaCache beat progressive without TeaCache?
  - If B1 < A2: combine both (Group C)
  - If B1 > A2: progressive alone is competitive
- **C1/C2 vs A2/A3**: Does adding TeaCache to progressive help or break things?

### Compatibility checks
- If Group C produces wrong images (black, artifacts) or errors: TeaCache is
  **incompatible** with progressive.  Fix: reset TeaCache state in
  `_on_resolution_change` hook.
- If Group C produces correct images: compare timing to confirm additive benefit.

---

## Adding New Experiments

1. Add a new `run_gen` call in the appropriate Group in `test_progressive_benchmark.sh`.
2. Or add a new Group section (copy Group C pattern).
3. Results auto-appear in the timing TSV and speedup summary.

Common additions:
```bash
# Flash Attention backend (if available)
run_gen "A_fa_fullres" --attention-backend fa

# Different progressive delta
run_gen "A_prog_L1_d0.02" \
    --progressive-mode dct_rewind --progressive-levels 1 --progressive-delta 0.02

# dct (no rewind) vs dct_rewind comparison
run_gen "A_prog_L1_dct_plain" \
    --progressive-mode dct --progressive-levels 1 --progressive-delta 0.01
```

---

## Known Optimization Compatibility

| Optimization | Fullres | Progressive | Notes |
|---|---|---|---|
| TeaCache | ✅ | ⚠️ Needs test | Must reset TeaCache state at stage transition |
| Cache-Dit | ✅ | ❌ Broken | Caches per step count; sequence-length change invalidates cache |
| STA (Sliding Tile Attn) | ✅ | ❌ Broken | Tile config computed for initial sequence length |
| torch.compile | ✅ | ❌ Broken | Fixed sequence length in compiled kernel |
| Layerwise offload | ✅ | ✅ | Component-level, unaffected |
| LoRA | ✅ | ✅ | Weight-level, unaffected |

---

## Result History

| Date | Steps | A1 (s) | A2 speedup | A3 speedup | A4 speedup | B1 speedup | Notes |
|---|---|---|---|---|---|---|---|
| 2026-05-30 | 20 | 24.73 | 1.28× | 1.37× | 1.26× | — | with DIT offload (biased) |
| 2026-05-31 | 50 | 57.82 | 1.35× | 1.56× | 1.41× | — | with DIT offload (biased) |
| 2026-05-31 | 50 | TBD | TBD | TBD | TBD | TBD | GPU-resident (this run) |

---

## DCT Correctness Reference

GPU implementation (torch.fft Makhoul) vs scipy:
- Relative error: **1.7e-7** (when using same noise)
- All intermediate computation in **float32**; only output cast to bfloat16
- RNG: PyTorch vs numpy generates different numbers for same seed (statistically
  equivalent; bit-identical reproduction requires same RNG)

---

## Files
```
scratch/
  test_progressive_benchmark.sh   # Main benchmark script (this guide's companion)
  test_progressive_gen.sh         # Legacy sweep script (varies δ/levels)
  benchmark_guide.md              # This file
  results/bench_YYYYMMDD_HHMMSS/  # Per-run outputs
    *.png                         # Generated images
    *.log                         # Full sglang output per run
    timing.tsv                    # run_id / total_s / denoise_s / avg_step_s

python/sglang/multimodal_gen/
  runtime/pipelines_core/stages/progressive_resolution/
    spectral_ops.py     # GPU DCT-II / IDCT-II (torch.fft)
    scheduler_utils.py  # Stage transition math
    upsample.py         # dct_upsample_2d, apply_upsample
    denoising.py        # ProgressiveDenoisingStage base class
  runtime/pipelines/
    flux_progressive.py # FluxProgressiveDenoisingStage + freqs_cis update
    flux.py             # Modified to use FluxProgressiveDenoisingStage
  configs/sample/sampling_params.py  # +progressive_mode/levels/delta
```
