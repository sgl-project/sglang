# Step5 B300 decode optimizations

These opt-in FlashInfer patches accompany the SGLang Step5 attention changes.
They start from the **ABI-repaired baseline**: FlashInfer 0.7.1 with the base
patches 01/02/03/06/07 already applied. Follow
[`base/README.md`](base/README.md) to prepare that baseline first.
The version string alone is insufficient; `manifest.json` pins the exact input,
output, and patch hashes. The main Dockerfiles do not apply these patches.

## Changes

- `04-flashinfer-bf16-swiglu-step.patch`: preserve BF16 expert weights with
  BF16 FC1/FC2 and Step5 post-SiLU clamp. See [BF16-EXPERTS.md](BF16-EXPERTS.md)
  for the MegaMoE/TRTLLM mixed configuration and validation.

- `01-flashinfer-moe.patch`: skip nonlocal EP routes and use a
  warp-local activation/scale reduction for the measured BF16 FC1 configuration
  (inner dimension 3072 or 768, top-k 8, four tokens per CTA, post-SiLU
  clamp). The 768 case covers TP4 intermediate size 384. Other configurations retain
  the existing activation kernel, including its PDL handling.
- `02-flashinfer-routing-prefix.patch`: reuse the integer CTA prefix to compute
  padded expert row offsets, removing a second BlockScan and its barrier.
- `03-flashinfer-tp4-rmsnorm-order.patch`: preserve the TRTLLM FP32
  multiplication order in MNNVL RMSNorm for non-quantized BF16 TP4, hidden
  size 4096 and zero weight bias at every row count. Export a capability
  (version 2) from the rebuilt binary so SGLang can identify this arithmetic
  fix. Other shapes retain their existing arithmetic.

The MoE patch also copies aligned FP8 rows in four-byte vectors for hidden
size 4096, top-k 8 and 128–256 rows, and in sixteen-byte vectors above 256 rows
(prefill). Route indices and FP32 scales retain their original layout and
values; unsupported shapes retain the scalar copy. The warp-local activation
reads four consecutive BF16 values per lane and stores four packed FP8 values;
results are unchanged.

## Isolated build

Use the intended CUDA Python environment and absolute paths. The parent package
and its AOT cache must remain available: unchanged files are hard-linked where
possible and unchanged AOT modules are referenced by symlink. Patched files get
new inodes before modification. Output and cache directories must be fresh and
disjoint from the source directories.

```bash
python docker/patches/step5/build.py \
  --parent-package /path/to/site-packages/flashinfer \
  --parent-aot-package /path/to/site-packages/flashinfer_jit_cache \
  --output-root /path/to/flashinfer-step5 \
  --cache-root /path/to/flashinfer-step5-cache
```

The builder checks source hashes, applies patches without fuzz, and verifies
that the parent stayed unchanged. It builds complete changed modules in a fresh
release JIT cache, checks the 46-argument MoE ABI and the MNNVL arithmetic
capability, and creates a private AOT
overlay so an old installed binary cannot override the patched source. Library
hashes and compiler flags are written to `build-manifest.json`.

`--prepare-only` performs source preparation without importing CUDA packages.
It is a packaging check, not a CUDA build or numerical test. Use different output
and cache paths for the subsequent full build.
A changed input hash or patch failure requires resolving the package version
mismatch; do not force a patch or reuse a partially built output directory.

## Server configuration

For the current Step5 internal-routing setup, complete patch order, AOT
requirements and TP4/EP1 launch command, use the
[Step5 FlashInfer guide](base/README.md). The configuration and validation
notes below describe this optimization package and its historical qualification.

The measured B300 (SM103) configuration selects the SGLang optimizations
automatically, without optimization environment variables:

- BF16 Indexer with hidden size 4096, head dimension 256, four local index
  heads, and TP4 or TP8: fold score scaling and fuse the Q FP8 cast. Pack QKZ
  for TP4 auto and TP8 at 4/256 rows, using the Blackwell
  TGV dispatcher for the TP4 four-row case. Preserve the original
  parameters/loaders and fall back if a
  module transform separates their packed storage. The FP8 overflow policy is
  probed once per device before CUDA graph capture.
- Serialized FP8 block-128 MoE with hidden size 4096, intermediate size 1536,
  top-k 8, TP4/EP1 or TP8/EP8 and no attention DP: combine shared output with
  deferred finalization for BF16 inputs. BF16 checkpoint expert layers now remain BF16 by default and bypass
  this FP8 path. Explicit load-time quantization remains available through
  `SGLANG_STEP5_QUANTIZE_BF16_MOE=1`. The three MTP layers are dense SWA layers, not MoE.
- Pure TP4 BF16 small-row communication: combine MoE finalization, shared
  output, custom allreduce v2, residual addition and RMSNorm. Attention output
  uses the same collective via an identity route. The TP4 path covers 1–32
  rows, including draining batches, with hidden size 4096, the supported local
  group/workspace and the parent fusion contract. Unsupported shapes retain
  the existing backend.
- TP4 contiguous BF16 residual/RMSNorm communication above the small-row path
  and up to the verify batch (max running requests times the widest draft,
  capped by the fusion limit) at hidden size 4096 uses a separate MNNVL
  workspace sized to that batch, initialized before graph capture. All ranks
  must report the patched binary capability through the actual CPU group;
  otherwise all retain the original path. Admission also requires the
  small-row path's architecture/group/collective capability checks. This is a
  shape guard: eligible prefill rows in the band can use it as well as VERIFY.
- Sparse attention uses the supported N8/N16 SM10x path and the native FP32
  warp maximum reduction; the softmax sum reduction order stays unchanged.
  M256 DSA side-stream overlap covers TP4 QKV width 3456 and TP8 width 1920.
- Eligible SM103 TP4 M4 DSA split-KV merge includes rounded head gating
  and enables PDL for the existing output projection.
- Eligible TP4 M4 Indexer W uses the existing tiny BF16 GEMV
  with live original weights and hook/layout/backend fallbacks. Together with
  packed QKZ, this reduces four projection launches to two.
- SM103 BF16 head gates with M1–32/N16/K4096 use the existing Blackwell
  tcgen05 TGV implementation. See validation for rounding and acceptance results.
- Full Step5 top-1 MTP chains use the existing split argmax on SM10x, omitting
  unused softmax probabilities. Rejection sampling and incomplete chains keep
  the original probability path; draft position updates are unchanged.
- SWA verification uses native FA4 with its existing head-gate epilogue. The
  experimental split-attention replacement is removed to preserve reference
  arithmetic. DSA merge and small-batch projections also retain the reference
  dispatch and rounding behavior.
- Small SM103 Indexer verification uses the existing Triton scorer with a wider
  grid. The small-batch SM100 scorer override and its metadata-hoisting path are
  removed. General scoring and prefill retain upstream dispatch.
- The custom CUDA register TopK and its JIT dispatch have been removed. Decode
  retains the original CuTe stable selector. The existing DeepSelect prefill
  path fuses integer history-length preparation and region/address packing;
  scores and DeepSelect selection policy are unchanged. Decode DeepSelect
  integration is maintained in PR #40.
- The TP4 Indexer normalization/compression pair uses PDL for one-to-four
  four-row requests. It reads the previous ring and compression weights before
  the new normalized key is ready; each request retains its own ring state.
- CUDA graph capture records side work near its join while retaining the
  original fork, preventing a new main replay stream at each layer.

The existing dev environment overrides remain compatible. The historical
optimizations below need no additional environment overrides; BF16 expert
backend selection is documented in [BF16-EXPERTS.md](BF16-EXPERTS.md). Sparse decode and the FlashInfer patches use
kernel/backend capability and shape checks. The measured command uses TRTLLM
as the requested communication backend. The guarded 256-row Step5 path
automatically uses MNNVL when the matching patched binary is present; small rows
use custom allreduce v2. Other paths retain the requested backend.

The command below uses the isolated overlay built above, with real MTP.
For in-place patching without AOT, use the
[launch guide](base/README.md#launch-step5-4x-b300-tp4ep1-real-mtp).
From the repository root, substitute the checkpoint and completed build paths.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="/path/to/flashinfer-step5:$PWD/python"
export FLASHINFER_WORKSPACE_BASE=/path/to/flashinfer-step5-cache
export FLASHINFER_JIT_DEBUG=0
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD SGLANG_SIMULATE_ACC_TOKEN_MODE

python -m sglang.launch_server \
  --model-path /path/to/fp8_vit_mtp3 --tp 4 \
  --mem-fraction-static 0.75 --attention-backend fa4 --page-size 64 \
  --reasoning-parser step5 \
  --moe-runner-backend flashinfer_trtllm \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --enable-multi-layer-eagle --max-running-requests 64 \
  --flashinfer-allreduce-fusion-backend trtllm --random-seed 42 --port 31114
```

This is pure TP4: EP remains 1. Both the backend selection and the matching
patched package are part of the measured configuration; checking out this
branch alone does not reproduce the reported throughput.

## Focused SGLang checks

Run on a supported CUDA host after selecting the intended package and source:

```bash
python -m pytest -q test/registered/unit/spec/test_multi_layer_draft_batch.py
python -m pytest -q test/registered/kernels/ops/speculative/test_spec_topk1.py
```

These existing tests exercise MTP
batch isolation, embedding, graph replay and top-1 selection; they do not replace targeted FP8 boundary checks,
four-rank communication checks, full GSM8K, or real MTP acceptance checks.
