# Execution Plan: AsymGEMM Chunked-Scatter + CUDA-Graph Update

## Overview

Two changes are being deployed:

| Change | File | Env / Flag |
|--------|------|------------|
| Chunked scatter (Option A) | `asym_gemm.py` | `SGLANG_MASKED_GEMM_CHUNK_SIZE=8` |
| CUDA-graph safe masked GEMM | AsymGEMM C++ library | remove `--disable-cuda-graph` |

Expected peak intermediate memory (Qwen-72B scale, 256 experts, decode):
`~4 GB → ~125 MB` when both changes are active.

---

## Prerequisites

- The updated AsymGEMM C++ library must be compiled and installed before starting
  the server.  The Python-level changes in `asym_gemm.py` / `entrypoint.py` rely
  on the new `(lhs, rhs, out, masked_m, expected_m)` kernel signature.
- If the binary has **not** been recompiled yet, add `--disable-cuda-graph` back
  to `server.sh` and unset `SGLANG_MASKED_GEMM_CHUNK_SIZE` until it is.

---

## Step 1 — Build and Install the Updated AsymGEMM Library

```bash
cd /workspace/AsymGEMM_main

# Build the Python extension
pip install -e . --no-build-isolation   # or your project's build command

# Smoke-test: verify the new masked API is present
python -c "
import asym_gemm, inspect
sig = inspect.signature(asym_gemm.m_grouped_fp8_asym_gemm_nt_masked)
print('FP8 masked sig:', sig)
# Expected: (lhs, rhs, out, masked_m, expected_m, ...)  — NO offsets/experts/list_size
"
```

If the signature still shows `offsets, experts, list_size`, the old binary is loaded.
Stop here and rebuild.

---

## Step 2 — Verify the Python Layer

```bash
cd /workspace/sglang

# Quick import check — should complete with no errors
python -c "
from sglang.srt.layers.asym_gemm_wrapper.entrypoint import (
    grouped_gemm_nt_f8f8bf16_masked,
    grouped_gemm_nt_fp4fp4bf16_masked,
    grouped_gemm_nt_bf16bf16bf16_masked,
)
import inspect
for fn in [grouped_gemm_nt_f8f8bf16_masked,
           grouped_gemm_nt_fp4fp4bf16_masked,
           grouped_gemm_nt_bf16bf16bf16_masked]:
    sig = inspect.signature(fn)
    params = list(sig.parameters)
    assert 'offsets' not in params, f'{fn.__name__} still has offsets param'
    assert 'masked_m' in params, f'{fn.__name__} missing masked_m param'
    print(fn.__name__, '— OK:', params)
"
```

---

## Step 3 — Correctness Test (Chunked vs Non-Chunked)

Run a unit-level correctness check before the full server:

```bash
cd /workspace/sglang

python - <<'EOF'
import os, torch
# Force the chunked path
os.environ["SGLANG_MASKED_GEMM_CHUNK_SIZE"] = "8"

# Import after setting env var so the module-level constant picks it up
from sglang.srt.layers.moe.moe_runner.asym_gemm import _MASKED_GEMM_CHUNK_SIZE
print(f"Chunk size active: {_MASKED_GEMM_CHUNK_SIZE}")
assert _MASKED_GEMM_CHUNK_SIZE == 8, "Env var not picked up"
print("OK — chunked path will be used when src2dst is in running_state")
EOF
```

For a full end-to-end numerical comparison, run:

```bash
# Reference: chunked scatter disabled
SGLANG_MASKED_GEMM_CHUNK_SIZE=0 python -m pytest \
    python/sglang/test/ -k "asym_gemm" -v 2>&1 | tee /tmp/ref_test.log

# With chunked scatter enabled
SGLANG_MASKED_GEMM_CHUNK_SIZE=8 python -m pytest \
    python/sglang/test/ -k "asym_gemm" -v 2>&1 | tee /tmp/chunked_test.log

diff /tmp/ref_test.log /tmp/chunked_test.log
```

---

## Step 4 — Start the Server

```bash
cd /workspace/sglang
bash server.sh
```

The server will start with:
- `SGLANG_MASKED_GEMM_CHUNK_SIZE=8` — chunked scatter active
- CUDA graph **enabled** (no `--disable-cuda-graph`)
- `--mem-fraction-static 0.85` (raised from 0.25 now that the ~4 GB spike is gone)

Wait for the log line:
```
INFO:     Application startup complete.
```

**Fallback:** If the server crashes on the first decode with a CUDA error, the
most likely cause is the old binary being loaded.  Restart with:
```bash
# Revert to safe mode
sed -i 's/--mem-fraction-static 0.85/--mem-fraction-static 0.25/' server.sh
SGLANG_MASKED_GEMM_CHUNK_SIZE=8 \
python -m sglang.launch_server \
  --model-path /workspace/models/Qwen3.5-397B-A17B-FP8 \
  --disable-radix-cache \
  --moe-runner-backend asym_gemm \
  --mem-fraction-static 0.25 \
  --disable-cuda-graph
```

---

## Step 5 — Run Benchmark (Throughput)

In a second terminal:

```bash
cd /workspace/sglang
bash client.sh
```

Expected metrics to watch:
- `output_throughput` (tokens/s)
- `median_e2e_latency_ms`

---

## Step 6 — Peak Memory Measurement

```bash
# 1. Measure BEFORE chunked scatter (restart server with chunk_size=0)
SGLANG_MASKED_GEMM_CHUNK_SIZE=0 bash server.sh &
sleep 60
bash client.sh --profile

# Check GPU memory in a separate terminal while client runs:
watch -n 0.5 "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"

# 2. Measure WITH chunked scatter (default server.sh)
bash server.sh &
sleep 60
bash client.sh --profile
```

Record `nvidia-smi` peak `memory.used` for both runs.  The chunked scatter
should eliminate the ~4 GB spike seen during the masked GEMM forward pass.

Alternatively, instrument directly:

```bash
python - <<'EOF'
import torch, os

os.environ["SGLANG_MASKED_GEMM_CHUNK_SIZE"] = "8"
torch.cuda.reset_peak_memory_stats()

# ... run one forward pass ...

peak = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak memory allocated: {peak:.2f} GB")
EOF
```

---

## Step 7 — Tuning `SGLANG_MASKED_GEMM_CHUNK_SIZE`

`chunk_size` must satisfy:

1. **Divides `num_groups` evenly** (or the last partial chunk is handled correctly —
   the code handles it, but even division avoids wasted kernel launches).
   For Qwen-MoE: `num_groups = 256`; good values: 4, 8, 16, 32, 64.

2. **Matches the `kNumGroups` JIT constant** — once a chunk size is used, the
   AsymGEMM JIT compiles a specialisation for `num_groups = chunk_size`.  The
   same `chunk_size` must be used across all runs (no per-step variation).

3. **Balances buffer size vs kernel launches**:
   - Smaller chunk → less peak memory, more kernel launches per forward pass.
   - Larger chunk → fewer launches, more memory.

| `chunk_size` | gateup peak | launches per layer |
|---|---|---|
| 4  | ~39 MB  | 64 |
| 8  | ~78 MB  | 32 |
| 16 | ~156 MB | 16 |
| 32 | ~312 MB | 8  |

Recommended starting point: **8**.

---

## Environment Variables Summary

| Variable | Default | Purpose |
|----------|---------|---------|
| `SGLANG_MASKED_GEMM_CHUNK_SIZE` | `0` (disabled) | Chunk size for chunked scatter |
| `SGLANG_MASKED_GEMM_FAST_ACT` | `0` (disabled) | Fused SiLU+FP8-quant activation |
| `SGLANG_ASYMGEMM_SANITY_CHECK` | `0` (disabled) | Validate FP8 scale rounding |

---

## Rollback

To revert all Python changes cleanly:

```bash
cd /workspace/sglang
git diff --name-only  # list changed files
git checkout -- python/sglang/srt/layers/asym_gemm_wrapper/entrypoint.py
git checkout -- python/sglang/srt/layers/moe/moe_runner/asym_gemm.py
git checkout -- python/sglang/srt/layers/moe/moe_runner/asym_gemm_bf16.py
git checkout -- python/sglang/srt/layers/moe/moe_runner/asym_gemm_fp4.py
git checkout -- server.sh client.sh
```
