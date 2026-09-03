---
name: sglang-diffusion-cache-dit
description: "Workflow for upgrading/integrating cache-dit in SGLang diffusion (multimodal_gen): DBCache, DMD calibrator, TaylorSeer, SVDQuant DQ; porting upstream PRs and resolving conflicts against the per-request knob system; adding new cache knobs; building the sglang generate CLI test matrix; precision validation (PSNR / log evidence); troubleshooting environment issues (wheel ABI, svdq extension, flashinfer conflicts). Use when upgrading or integrating cache-dit in sglang diffusion, porting cache-dit PRs with conflicts, adding cache knobs, running the sglang generate CLI test matrix, or validating precision (PSNR) for DBCache/DMD/SVDQuant paths."
user-invocable: true
---

# SGLang Diffusion × Cache-DiT Integration

Path placeholders used throughout: `<sglang_dir>` = sglang repo root, `<cache_dit_dir>` = local cache-dit repo root (if not present locally, clone it: `git clone https://github.com/vipshop/cache-dit`), `<flux_model_dir>` = the DiT model weights actually under test (**if the user has not specified the model, ask for the model name and checkpoint path first — the workflow is not FLUX-specific; FLUX.1-dev is only the reference run**), `<cuda_home>` = CUDA toolkit root (typically `/usr/local/cuda`), `<gpu_id>` = a free GPU index.

## GATE CHECK (confirm before starting)

```
STOP — are all of the following confirmed?
  1. All work happens in a dedicated env for sglang diffusion testing (e.g. `conda activate sgl`),
     fully isolated from cache-dit/ffpa dev envs: never touch other envs, and never let sglang
     dependencies leak into them. If the env does not exist, create a new dedicated one first
     (conda/venv both fine, see §3) — never reuse an existing env just to save effort. Install
     missing dependencies directly into the dedicated env.
  2. GPU: run sglang jobs on the GPUs allocated for them (e.g. CUDA_VISIBLE_DEVICES=<gpu_id>);
     other GPUs may be busy with other jobs.
  3. The test artifact directory <sglang_dir>/.tmp/{task}/ exists (.tmp/ is gitignored; never
     write test outputs into the repo root).
  4. CLI args have been verified once via `sglang generate --help | grep -- <every arg you use>`
     (the old --warmup from earlier docs/PRs is gone; it is now
     --warmup-mode {off,request,server} + --warmup-steps N).
  5. Plan-time alignment — align ALL of the following with the user while drafting the plan
     (before any run), not afterwards:
     - Model under test: model name + weights path. Do NOT default to FLUX.1-dev; adapt the
       case-name prefix and the PSNR baseline table to the actual model.
     - Generation settings: resolution, step count, prompt/seed (reference run used
       1024x1024 / 28 steps — follow the user's actual setup instead).
     - Local cache-dit checkout: does `<cache_dit_dir>` exist? The yaml configs in §4 come from
       `<cache_dit_dir>/examples/configs/`; if absent, agree with the user whether to
       `git clone https://github.com/vipshop/cache-dit` or obtain the configs another way.
     Anything unknown → ASK the user first.
  NO → fix these before touching anything.
```

**Hard rules**
- Conflict resolution: **keep the target branch's refactored structure** (the knob system in §2). If the upstream PR's direct-write style targets code that has since been refactored, re-inject it following the new pattern; never revert the target branch's refactor.
- After touching cache-related modules, run **all** `test/unit/test_cache_dit*.py` (not just one file — a skipped stub test once left a 9/9 ImportError that only surfaced at CLI stage).

## 1. Integration map (4 files on the sglang side)

| File | Responsibility |
|------|----------------|
| `python/pyproject.toml` | `cache-dit==x.y.z` version pin in the diffusion extra |
| `python/sglang/multimodal_gen/envs.py` | env vars: annotation section + lazy getters + `_CACHE_DIT_SECONDARY_CONFIGS` + special bool getters |
| `python/sglang/multimodal_gen/runtime/cache/cache_dit_integration.py` | `CacheDitConfig`, `enable_cache_on_transformer` (single/dual transformer), custom BlockAdapter, per-request knob validation set, calibrator construction |
| `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py` | `_build_cache_dit_config()` + `_cache_dit_knob()` — **the single injection point for new cache parameters** |

Data flow: `env (with secondary fallback) → knob (request > env) → CacheDitConfig fields → mutual-exclusion guard → cache-dit Config (DBCacheConfig / DMDCalibratorConfig / TaylorSeerCalibratorConfig)`

Key facts:
- Knob priority: request override (`sampling_params.cache_dit_params`) > env; a secondary knob first inherits the request-level primary value, then falls back to `SGLANG_CACHE_DIT_SECONDARY_*`, then to the primary env default.
- A single change in `_build_cache_dit_config` automatically covers both the primary/secondary call sites **and** the minimax_h3 subclass (its override calls `super()`).
- Dual-transformer models (wan2.2 etc.) can reuse the primary config wholesale for the secondary transformer (`_cache_dit_secondary_uses_primary_config()`); new fields are inherited for free.
- The calibrator is a single slot: DMD × TaylorSeer are mutually exclusive; `_assert_calibrator_exclusive(config, label)` guards both enable entry points (enabling both raises ValueError).

## 2. Standard six steps to add a cache knob (DMD as the example)

1. **envs.py annotation section**: add the primary + secondary variables (e.g. `SGLANG_CACHE_DIT_DMD: bool = False`).
2. **envs.py getter**: non-bool → add a `(SUFFIX, type, default)` tuple to `_CACHE_DIT_SECONDARY_CONFIGS` (auto-generates the secondary fallback); bool → write a dedicated `_secondary_xxx_getter` (`get_bool_env_var(SECONDARY_X, default=os.getenv(PRIMARY_X, "false"))`) and register it.
3. **cache_dit_integration.py**: add the field to `CacheDitConfig` (with docstring and default).
4. **cache_dit_integration.py**: add the key to `CACHE_DIT_REQUEST_KNOB_KEYS` (symmetric with taylorseer; automatically joins the request/secondary validation sets and the remount detection `cache_dit_overrides_key`).
5. **denoising.py**: append a knob() entry after taylorseer_order inside `_build_cache_dit_config`:
   `enable_dmd=knob("enable_dmd", envs.SGLANG_CACHE_DIT_DMD, envs.SGLANG_CACHE_DIT_SECONDARY_DMD, secondary=secondary)`
6. **Calibrator/special logic**: construct the corresponding cache-dit Config inside `enable_cache_on_transformer` and `enable_cache_on_dual_transformer` (DMD: `if enable_dmd: DMDCalibratorConfig(...) elif enable_taylorseer: ...`), and add the field to the log format string (placeholder and argument counts must stay aligned).

Conflict forecast when porting an upstream PR: `denoising.py` always conflicts (the PR's direct `envs.XXX` → Config construction was refactored away on the target branch) → apply only step 5; `cache_dit_integration.py` conflicts locally (the per-request knob system added on the target branch sits right next to the PR's insertion points) → keep the target branch's frozenset and append the new keys.

## 3. Environment setup (dedicated env)

**Environment principle (mandatory, top priority)**: everything — pip installs/uninstalls, source builds, unit tests and CLI runs — happens inside the dedicated env (e.g. `conda activate sgl`). It is fully isolated from the cache-dit/ffpa dev envs: never install/uninstall packages in other envs, and never let sglang dependency changes (torch, cache-dit, flashinfer, ...) leak into them. If the env does not exist yet, create a dedicated one before continuing — do not reuse any existing env:

```bash
conda create -n sgl python=3.12
conda activate sgl
cd <sglang_dir>
pip install -e ".[diffusion]" --no-build-isolation
```

(Any other virtualenv mechanism works as well; what matters is **dedicated and isolated**. After switching envs, confirm versions with `pip show sglang cache-dit` before testing.)

```bash
conda activate sgl
# Pure Python (no SVDQuant):
pip install cache-dit==<ver>            # or cache-dit-cu13==<ver> --no-deps
# SVDQuant (PTQ nvfp4) requires a working CUDA extension. The PyPI wheel can be ABI-incompatible
# with a newer torch (undefined symbol: materialize_cow_storage → the wheel was built against an
# older torch). In that case, build from the local cache-dit source tree
# (no local checkout? git clone https://github.com/vipshop/cache-dit first):
cd <cache_dit_dir>
export CUDA_HOME=<cuda_home>
pip install setuptools-scm              # missing → metadata-generation-failed
CACHE_DIT_BUILD_SVDQUANT=1 pip install ".[quantization]" --no-build-isolation
# Extension self-check (pinpoints the load error in one step):
python -c "from cache_dit.quantization.svdquant import svdq_is_available, svdq_get_load_error as e; print(svdq_is_available(), e())"
```

Other environment pitfalls:
- A stale `flashinfer-cubin` that mismatches the flashinfer main package → blocks every sglang import; if the installed flashinfer has no matching cubin release (e.g. 0.6.18), `pip uninstall flashinfer-cubin` directly.
- The dedicated env may lack pytest → the test files use `unittest.main()` style; run `python <test_file.py>` directly.
- `test_cache_dit_integration.py` **replaces the cache_dit module with a stub** (no real install required): after changing top-level imports in `cache_dit_integration.py`, sync the stub's top-level symbols in `_install_cache_dit_stub()` (missing BlockAdapterRegister/Parallelism*/DMDCalibratorConfig once caused 9/9 ImportError).

## 4. CLI test matrix (reference run: PRO 5000, FLUX.1-dev, 1024×1024, 28 steps)

Nine-case design (backend × acceleration feature combos):

| # | backend | feature | driven by |
|---|---------|---------|-----------|
| 1-3 | SGLD (default) | baseline / DBCache / +DMD | env: `SGLANG_CACHE_DIT_ENABLED=true` (+`SGLANG_CACHE_DIT_DMD=true`) |
| 4-6 | diffusers | baseline / DBCache / +DMD | yaml: `cache.yaml` / `cache_dmd.yaml` (<cache_dit_dir>/examples/configs/) |
| 7-9 | diffusers | SVDQ nvfp4 / +compile / +compile+DBCache+DMD | yaml: `blackwell/quantize_svdq.yaml` / `blackwell/cache_dmd_svdq.yaml` + `--enable-torch-compile` |

Common args: `--model-path=$FLUX_DIR --log-level=info --prompt='...' --width=1024 --height=1024 --num-inference-steps=28 --warmup-mode request --warmup-steps 1 --dit-cpu-offload false --text-encoder-cpu-offload false --save-output --output-path .tmp/{task}/outputs`; compile cases add `--warmup-steps 28`; the blackwell yaml is already `svdq_nvfp4_r128_dq` (nvfp4, not int4) — grep to confirm before running.

**run_case script pattern** (saved as `.tmp/{task}/run_matrix.sh`):
- `timeout <1800~3600>` guard + stdout redirect to `logs/{name}.log` + `summary.log` records `PASS/FAIL` (dual criteria: rc + png existence) + a failure does not abort the remaining cases.
- When several cases fail with a common root cause, fix it and **re-run only the failed segment** via a small sub-script, not the whole matrix.
- Submit in the background; then do a **one-shot** health check (`sleep 45-60 && tail -3 logs/<first_case>.log && nvidia-smi -i <gpu_id>`) to confirm the model is loading without CLI errors, then stop and wait for the completion notification — **never poll**.
- Save long verification scripts (PSNR / perf extraction) as `.py` files; avoid long `python -c` one-liners (nested f-strings are error-prone).

**Fixed failure-diagnosis order**:
1. Check `summary.log` and whether the output files were produced (`EOFError` / `worker did not terminate gracefully, forcing` / leaked-semaphore lines in the tail are multiprocess shutdown noise, not the failure itself);
2. `grep -iE 'error|assert|exception|raise' logs/{case}.log | head` to find the **first** traceback (the real cause is often mid-log, e.g. `AssertionError: Quantization backend ... not supported`);
3. Use package-level diagnostics when available (`svdq_is_available()/svdq_get_load_error()`).

### 4.1 Full command reference (battle-tested during the 1.5.1 upgrade; adapt and reuse)

**Unit tests + negatives (mandatory after touching cache modules)**:
```bash
cd <sglang_dir>/.tmp/{task}
conda activate sgl
# Run ALL cache-dit-related tests via glob (unittest style, direct run; the env may lack pytest)
for t in ../../python/sglang/multimodal_gen/test/unit/test_cache_dit*.py; do echo "== $t"; python $t 2>&1 | tail -3; done
# Mutual-exclusion negatives already exist as unit tests
# (test_both_calibrators_raise_on_{,dual_}transformer) — no ad-hoc script needed
```

**CLI matrix driver script** (`.tmp/{task}/run_matrix.sh`; run `bash run_matrix.sh` in the background):
```bash
#!/bin/bash
set -u
BASE=<sglang_dir>/.tmp/{task}
FLUX_DIR=<flux_model_dir>   # model actually under test — ask the user if not specified
CFG=<cache_dit_dir>/examples/configs
OUT=$BASE/outputs; LOGS=$BASE/logs; mkdir -p "$OUT" "$LOGS"
export CUDA_VISIBLE_DEVICES=<gpu_id>
PROMPT='A fantasy landscape with mountains and a river, detailed, vibrant colors'
COMMON=(
  --model-path="$FLUX_DIR" --log-level=info --prompt="$PROMPT"
  --width=1024 --height=1024 --num-inference-steps=28
  --warmup-mode request --warmup-steps 1          # old --warmup is gone; verify via --help first
  --dit-cpu-offload false --text-encoder-cpu-offload false
  --save-output --output-path "$OUT"
)
run_case() {  # $1=name $2=timeout_s; remaining args are case-specific
  local name=$1; shift; local timeout_s=$1; shift
  echo "[$(date '+%H:%M:%S')] START $name" >> "$LOGS/summary.log"
  timeout "$timeout_s" sglang generate "${COMMON[@]}" "$@" \
    --output-file-name "$name.png" > "$LOGS/$name.log" 2>&1
  local rc=$?
  [[ $rc -eq 0 && -f "$OUT/$name.png" ]] && st=PASS || st=FAIL
  echo "[$(date '+%H:%M:%S')] $st $name (rc=$rc, png=$([[ -f $OUT/$name.png ]] && echo yes || echo no))" >> "$LOGS/summary.log"
}
# SGLD triple (env-driven)
run_case flux_sgld 1800
SGLANG_CACHE_DIT_ENABLED=true run_case flux_cache_sgld 1800
SGLANG_CACHE_DIT_ENABLED=true SGLANG_CACHE_DIT_DMD=true run_case flux_cache_dmd_sgld 1800
# diffusers triple (yaml-driven)
run_case flux_diffusers 1800 --backend diffusers
run_case flux_cache_diffusers 1800 --backend diffusers --cache-dit-config "$CFG/cache.yaml"
run_case flux_cache_dmd_diffusers 1800 --backend diffusers --cache-dit-config "$CFG/cache_dmd.yaml"
# SVDQ nvfp4 triple (requires the svdq extension)
run_case flux_svdq_nvfp4_diffusers 2400 --backend diffusers --cache-dit-config "$CFG/blackwell/quantize_svdq.yaml"
run_case flux_svdq_nvfp4_compile_diffusers 3600 --backend diffusers --warmup-steps 28 \
  --enable-torch-compile --cache-dit-config "$CFG/blackwell/quantize_svdq.yaml"
run_case flux_cache_dmd_svdq_nvfp4_compile_diffusers 3600 --backend diffusers --warmup-steps 28 \
  --enable-torch-compile --cache-dit-config "$CFG/blackwell/cache_dmd_svdq.yaml"
echo "[$(date '+%H:%M:%S')] MATRIX DONE" >> "$LOGS/summary.log"
```
When several cases fail with a common cause, copy the script keeping only the failed segment and re-run (the run_svdq.sh pattern).

**One-shot health check after startup** (then wait for completion; do not poll):
```bash
sleep 60 && tail -3 logs/flux_sgld.log | cut -c1-160 && nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i <gpu_id>
# Expect: model loading / inferring + tens of GB of VRAM in use;
# `ambiguous option` / Traceback → kill immediately and fix the args
```

**Feature-activation verification (grep the logs)**:
```bash
grep -E 'DMD=True|Calibrator Config: DMD' logs/flux_cache_dmd_sgld.log          # DMD active
grep -E 'Match Blocks|Collected Context Config' logs/flux_cache_sgld.log        # DBCache active
grep -E 'SVDQuant.*Type: svdq_nvfp4_r128_dq' logs/flux_svdq_nvfp4_diffusers.log # quantization active
```

**Perf extraction**:
```bash
for f in logs/*.log; do echo "$f: $(grep -oE 'finished in [0-9.]+ seconds' $f | head -1) $(grep -oE '[0-9.]+it/s' $f | tail -1)"; done
```

**Quantitative PSNR/SSIM comparison** — prefer the `cache-dit-metrics` CLI (ships with cache-dit; methodology reference: the cache-dit-model-integration skill's `references/testing.md`):
```bash
# Compare each accelerated result against the same-backend baseline
cache-dit-metrics psnr ssim -i1 outputs/flux_sgld.png -i2 outputs/flux_cache_sgld.png
cache-dit-metrics psnr ssim -i1 outputs/flux_sgld.png -i2 outputs/flux_cache_dmd_sgld.png
cache-dit-metrics psnr ssim -i1 outputs/flux_diffusers.png -i2 outputs/flux_svdq_nvfp4_diffusers.png
cache-dit-metrics psnr ssim -i1 outputs/flux_svdq_nvfp4_compile_diffusers.png \
  -i2 outputs/flux_cache_dmd_svdq_nvfp4_compile_diffusers.png
```
Fallback when the CLI is unavailable (save as `.tmp/{task}/psnr.py` and run; avoid long python -c) — PSNR only, no SSIM:
```python
import numpy as np, torch
from PIL import Image
def load(p): return torch.from_numpy(np.array(Image.open(p))).float() / 255.0
def psnr(a, b):
    mse = ((a - b) ** 2).mean().item()
    return float('inf') if mse == 0 else 10 * np.log10(1.0 / mse)
pairs = [  # (label, baseline, accelerated output) — fill per actual cases
    ('sgld: cache vs base', 'flux_sgld.png', 'flux_cache_sgld.png'),
    ('sgld: cache+dmd vs base', 'flux_sgld.png', 'flux_cache_dmd_sgld.png'),
    ('diff: svdq vs base', 'flux_diffusers.png', 'flux_svdq_nvfp4_diffusers.png'),
    ('svdq: cache+dmd+compile vs svdq', 'flux_svdq_nvfp4_compile_diffusers.png',
     'flux_cache_dmd_svdq_nvfp4_compile_diffusers.png'),
]
import os; os.chdir(os.path.dirname(__file__) + '/outputs')
for name, a, b in pairs: print(f'{name:38s} PSNR = {psnr(load(a), load(b)):6.2f} dB')
```

### 4.2 Command-by-command edition (requirements-doc style, for single-case debugging / manual runs; use the 4.1 script for batch regression)

Environment setup (run once before all commands):

```bash
conda activate sgl
cd <sglang_dir>
export FLUX_DIR=<flux_model_dir>   # model actually under test — ask the user if not specified
export CUDA_VISIBLE_DEVICES=<gpu_id>
mkdir -p .tmp/{task}/outputs
# SVDQuant cases require: pip install cache-dit-cu13==<ver> --no-deps;
# fall back to a source build if the wheel is torch-ABI-incompatible (see §3);
# uninstall flashinfer-cubin if a stale copy reports a version mismatch
```

SGLD backend (env-driven):

```bash
# baseline
sglang generate --model-path=$FLUX_DIR --log-level=info \
  --prompt='A fantasy landscape with mountains and a river, detailed, vibrant colors' \
  --width=1024 --height=1024 --num-inference-steps=28 \
  --warmup-mode request --warmup-steps 1 \
  --dit-cpu-offload false --text-encoder-cpu-offload false \
  --save-output --output-path .tmp/{task}/outputs --output-file-name flux_sgld.png

# DBCache
SGLANG_CACHE_DIT_ENABLED=true \
  sglang generate --model-path=$FLUX_DIR --log-level=info \
  --prompt='A fantasy landscape with mountains and a river, detailed, vibrant colors' \
  --width=1024 --height=1024 --num-inference-steps=28 \
  --warmup-mode request --warmup-steps 1 \
  --dit-cpu-offload false --text-encoder-cpu-offload false \
  --save-output --output-path .tmp/{task}/outputs --output-file-name flux_cache_sgld.png

# DBCache + DMD Calibrator
SGLANG_CACHE_DIT_ENABLED=true SGLANG_CACHE_DIT_DMD=true \
  sglang generate --model-path=$FLUX_DIR --log-level=info \
  --prompt='A fantasy landscape with mountains and a river, detailed, vibrant colors' \
  --width=1024 --height=1024 --num-inference-steps=28 \
  --warmup-mode request --warmup-steps 1 \
  --dit-cpu-offload false --text-encoder-cpu-offload false \
  --save-output --output-path .tmp/{task}/outputs --output-file-name flux_cache_dmd_sgld.png
```

Diffusers backend (yaml-driven, `CFG=<cache_dit_dir>/examples/configs`):

```bash
# baseline
sglang generate --model-path=$FLUX_DIR --backend diffusers --log-level=info \
  --prompt='A fantasy landscape with mountains and a river, detailed, vibrant colors' \
  --width=1024 --height=1024 --num-inference-steps=28 \
  --warmup-mode request --warmup-steps 1 \
  --dit-cpu-offload false --text-encoder-cpu-offload false \
  --save-output --output-path .tmp/{task}/outputs --output-file-name flux_diffusers.png

# DBCache
sglang generate --model-path=$FLUX_DIR --backend diffusers --log-level=info \
  --prompt='A fantasy landscape with mountains and a river, detailed, vibrant colors' \
  --width=1024 --height=1024 --num-inference-steps=28 \
  --warmup-mode request --warmup-steps 1 \
  --dit-cpu-offload false --text-encoder-cpu-offload false \
  --cache-dit-config $CFG/cache.yaml \
  --save-output --output-path .tmp/{task}/outputs --output-file-name flux_cache_diffusers.png

# DBCache + DMD Calibrator
sglang generate --model-path=$FLUX_DIR --backend diffusers --log-level=info \
  --prompt='A fantasy landscape with mountains and a river, detailed, vibrant colors' \
  --width=1024 --height=1024 --num-inference-steps=28 \
  --warmup-mode request --warmup-steps 1 \
  --dit-cpu-offload false --text-encoder-cpu-offload false \
  --cache-dit-config $CFG/cache_dmd.yaml \
  --save-output --output-path .tmp/{task}/outputs --output-file-name flux_cache_dmd_diffusers.png
```

SVDQuant W4A4 NVFP4 (requires the svdq extension; the blackwell yaml is already nvfp4 — grep to confirm before running):

```bash
# SVDQuant W4A4 NVFP4
sglang generate --model-path=$FLUX_DIR --backend diffusers --log-level=info \
  --prompt='A fantasy landscape with mountains and a river, detailed, vibrant colors' \
  --width=1024 --height=1024 --num-inference-steps=28 \
  --warmup-mode request --warmup-steps 1 \
  --dit-cpu-offload false --text-encoder-cpu-offload false \
  --cache-dit-config $CFG/blackwell/quantize_svdq.yaml \
  --save-output --output-path .tmp/{task}/outputs --output-file-name flux_svdq_nvfp4_diffusers.png

# + compile
sglang generate --model-path=$FLUX_DIR --backend diffusers --log-level=info \
  --prompt='A fantasy landscape with mountains and a river, detailed, vibrant colors' \
  --width=1024 --height=1024 --num-inference-steps=28 \
  --warmup-mode request --warmup-steps 28 \
  --dit-cpu-offload false --text-encoder-cpu-offload false \
  --enable-torch-compile \
  --cache-dit-config $CFG/blackwell/quantize_svdq.yaml \
  --save-output --output-path .tmp/{task}/outputs --output-file-name flux_svdq_nvfp4_compile_diffusers.png

# + compile + DBCache + DMD Calibrator
sglang generate --model-path=$FLUX_DIR --backend diffusers --log-level=info \
  --prompt='A fantasy landscape with mountains and a river, detailed, vibrant colors' \
  --width=1024 --height=1024 --num-inference-steps=28 \
  --warmup-mode request --warmup-steps 28 \
  --dit-cpu-offload false --text-encoder-cpu-offload false \
  --enable-torch-compile \
  --cache-dit-config $CFG/blackwell/cache_dmd_svdq.yaml \
  --save-output --output-path .tmp/{task}/outputs --output-file-name flux_cache_dmd_svdq_nvfp4_compile_diffusers.png
```

## 5. Precision and feature verification

**Hard log evidence (check this before PSNR)**:
```
Enabling cache-dit ... DMD=True (history=6, rank=0, svd=medium), TaylorSeer=False ..., steps=28
[Cache-DiT] Collected Context Config: DBCache_F1B0_W4I1M0MC3_R0.24_N28_CFG0, Calibrator Config: DMD_H(6, medium)
[Cache-DiT] Match Blocks: CachedBlocks_Pattern_0_1_2, for transformer_blocks ...
[Cache-DiT] SVDQuant Type: svdq_nvfp4_r128_dq, Rank: 128
```
Every accelerated case must show its corresponding line in the log before the feature counts as genuinely active.

**PSNR reference baselines** (FLUX.1-dev on a PRO 5000, vs the same backend without acceleration; sglang's default DBCache R=0.24 is aggressive, so magnitudes differ from the cache-dit-side PSNR>30 standard — don't misjudge):

| comparison | typical PSNR |
|------------|--------------|
| SGLD: cache vs base | ≈23.5 dB |
| SGLD: cache+DMD vs base | ≈24.5 dB (**DMD should slightly beat pure cache**; if worse, investigate) |
| diffusers: cache vs base | ≈31 dB |
| diffusers: cache+DMD vs base | ≈29 dB |
| diffusers: svdq nvfp4 vs fp16 | ≈23.4 dB (normal W4A4 quantization loss) |
| within svdq stack: cache+DMD vs plain svdq | ≈28 dB |

**SSIM matters as much as PSNR** — always compute both (`cache-dit-metrics psnr ssim`). PSNR alone cannot detect structural corruption: a garbled image can still show PSNR > 20 dB, while SSIM collapses (< 0.5). If PSNR looks reasonable but SSIM is low, treat the output as corrupted and investigate — do not accept it.

Visually inspect 2-3 key PNGs first (baseline / DMD / full stack), then compute PSNR. Performance reference (single GPU): fp16 diffusers 17.15s → svdq+compile+DBCache+DMD 4.01s (≈4.3x); extract with grep `'finished in [0-9.]+ seconds'` and `it/s`.

## 6. Wrap-up

- Write the PR commit message in English to `.tmp/{task}/commit_msg.txt` (title `[Diffusion] Cache-DiT x.y.z: ...` + per-file changes + validation data); squash with `git commit -F`.
- Version bumps and cherry-picks keep the upstream commits (attribution); do not push without review.
- Record important findings and pitfalls in the repo-level knowledge base.

## 7. Pitfall quick reference

| pitfall | symptom | fix |
|---------|---------|-----|
| cache-dit wheel ABI incompatibility | `undefined symbol: _ZN3c104impl3cow...` (materialize_cow_storage), svdq_is_available()=False | source build (§3): CUDA_HOME + setuptools-scm |
| stale flashinfer-cubin | every import raises RuntimeError: version mismatch | `pip uninstall flashinfer-cubin` |
| stub tests out of sync | test_cache_dit_integration 9/9 ImportError | add the new top-level symbols to the stub |
| CLI arg drift | `ambiguous option: --warmup` | verify via `--help` first; use `--warmup-mode request --warmup-steps N` |
| running only one test file | missed regressions | run all `test_cache_dit*.py` via glob |
| mistaking shutdown noise for failure | `forcing`/`EOFError` in the log tail | check summary.log + png first; grep the first traceback |
| no pytest in the env | No module named pytest | run `python <test>.py` directly (unittest style) |

## Current Code Areas

| File | Role |
| --- | --- |
| `python/pyproject.toml` | pins the `cache-dit==x.y.z` version in the diffusion extra |
| `python/sglang/multimodal_gen/envs.py` | primary/secondary cache-dit env vars and lazy getters, including `_CACHE_DIT_SECONDARY_CONFIGS` |
| `runtime/cache/cache_dit_integration.py` | `CacheDitConfig`, single/dual-transformer enable paths, custom BlockAdapter, per-request knob validation, calibrator construction and mutual-exclusion guard |
| `runtime/pipelines_core/stages/denoising.py` | `_build_cache_dit_config()` + `_cache_dit_knob()`, the single injection point for new cache knobs |
| `python/sglang/multimodal_gen/test/unit/test_cache_dit_integration.py` | stub-based unit tests (no real cache-dit install needed); calibrator selection and exclusivity |
| `python/sglang/multimodal_gen/test/unit/test_cache_dit_per_request.py` | per-request knob reachability and secondary-inherits-primary coverage |
| `<cache_dit_dir>/examples/configs/` | yaml configs driving the diffusers-backend CLI cases (`cache.yaml`, `cache_dmd.yaml`, `blackwell/quantize_svdq.yaml`, `blackwell/cache_dmd_svdq.yaml`) |

## References

Authoritative usage references for cache in SGLang Diffusion; read them before changing user-facing cache behavior or docs:

- **`references/block_adapter.md`** — **the reference for custom BlockAdapter usage** (ideas only; all sglang adapter code stays in the sglang repo, and PatchFunctor is not recommended): `ForwardPattern` I/O contracts, `BlockAdapter` parameters (`has_separate_cfg`, `check_forward_pattern`), construction templates, third-party (non-diffusers) adapter rules, and cache interception pitfalls. Read it before writing or extending a custom BlockAdapter in `runtime/cache/cache_dit_integration.py`.
- `<cache_dit_dir>/.github/skills/cache-dit-model-integration/references/testing.md` — correctness-verification methodology behind the `cache-dit-metrics` CLI (PSNR+SSIM both mandatory, acceptance criteria, garbled-image red flags)
- `<sglang_dir>/docs/docs/sglang-diffusion/cache_dit.mdx` — Cache-DiT usage guide (env vars, per-request knobs, yaml configs)
- `<sglang_dir>/docs/docs/sglang-diffusion/caching-acceleration.mdx` — caching-acceleration guide (DBCache/DMD/TaylorSeer combinations, acceleration matrix)
- `<sglang_dir>/docs/docs/sglang-diffusion/` — the whole SGLang Diffusion docs directory; auxiliary reference for related topics (quantization, parallelism, installation, performance)
