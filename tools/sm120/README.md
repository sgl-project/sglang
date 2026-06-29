# SM120 tilelang paged-MQA-logits validator

Validates the SM120 (RTX PRO 6000 / Blackwell desktop) fix for
`tilelang_fp8_paged_mqa_logits` by compiling the tilelang kernel and comparing
its output against the pure-torch reference `fp8_paged_mqa_logits_torch_sm120`
(ground truth) on GLM-5.2-shaped inputs.

## Branches

- `sm120-glm5.2-fp8` — known-good. DSA indexer paged logits default to the
  **torch** fallback on SM120 (tilelang kernel disabled). Use this to serve.
- `sm120-tilelang-experiment` — candidate **tilelang kernel fix** (2-D K tile in
  `fp8_paged_mqa_logits_kernel`). Validate it here before promoting.

## Files

- `Dockerfile` — bases on an SGLang runtime image, checks out the chosen fork
  branch, runs the validator.
- `validate_tilelang_paged_logits.py` — the validator (PASS = lowers + matches
  torch within fp8 tolerance).
- `build_and_validate.sh` — one-command build + run.

## Quick start (on the SM120 box)

```bash
git clone -b sm120-tilelang-experiment https://github.com/shivajid/sglang.git
cd sglang/tools/sm120
chmod +x build_and_validate.sh
# pass the base image that already works on your box:
./build_and_validate.sh sm120-tilelang-experiment <your-working-sglang-image>
```

Or without Docker (fastest iteration loop):

```bash
cd /sgl-workspace/sglang
git fetch https://github.com/shivajid/sglang.git sm120-tilelang-experiment
git checkout -B sm120-tilelang-experiment FETCH_HEAD
python tools/sm120/validate_tilelang_paged_logits.py
```

## Build on one box, run on another

```bash
# build host (no GPU needed):
NO_RUN=1 ./build_and_validate.sh sm120-tilelang-experiment <base-image>
docker tag sm120-validator:sm120-tilelang-experiment <registry>/sm120-validator:exp
docker push <registry>/sm120-validator:exp
# GPU host:
docker run --rm --gpus all <registry>/sm120-validator:exp
```

## Interpreting results

- `PASS  max abs diff (valid region) = <small>` → kernel lowers AND is correct.
  Promote: cherry-pick the kernel commit into `sm120-glm5.2-fp8` and flip the
  SM120 default back to tilelang (set `SGLANG_OPT_USE_TILELANG_INDEXER=1` or
  revert the server_args torch default).
- `TILELANG COMPILE/RUN FAILED: ... should be at least 2` (or another tvm
  InternalError) → the kernel still doesn't lower; capture the full error.
  Likely next step: make `k_s_smem_u8` 2-D the same way as `k_smem_u8`.
- `FAIL  max abs diff = <large>` → it lowers but is numerically wrong; the
  byte/layout mapping in the 2-D copy needs adjustment.
