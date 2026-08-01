# DeepSeek-V4 RTX 5090 Cookbook Recipe Design

## Goal

Add an RTX 5090 hardware choice to the DeepSeek-V4 deployment cookbook and expose the launch command validated on an eight-GPU RTX 5090 devbox for the Flash Official (0731), FP4, low-latency, single-node combination.

## Scope

The change is limited to `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`:

- Add `rtx5090` to `supportedHardware`.
- Add a model-specific hardware entry labeled `RTX 5090`, with `32GB` VRAM and the `blackwell` vendor group.
- Add one deployment cell matching `rtx5090 | flash-official | fp4 | low-latency | single`.
- Keep the cell unverified and omit benchmark-card data.

The shared hardware catalog, deployment engine, benchmark config, and cookbook prose remain unchanged.

## Generated Command

The new cell emits the command that successfully started and served a BS=1, ISL=100000, OSL=1000 request on eight RTX 5090 GPUs:

```bash
sglang serve \
  --trust-remote-code \
  --model-path deepseek-ai/DeepSeek-V4-Flash-0731 \
  --tp 8 \
  --moe-runner-backend flashinfer_mxfp4 \
  --mem-fraction-static 0.90 \
  --cuda-graph-max-bs-decode 32 \
  --host 0.0.0.0 \
  --port 30000
```

The config stores the host, port, and model as cookbook placeholders. It does not enable HiCache: the tested TP4 + HiCache configuration failed during model-weight allocation before HiCache initialized.

## Validation

- Run the repository's cookbook/config validation for `docs_new`.
- Verify the new tuple resolves to exactly one deployment cell.
- Verify the generated command contains TP8, FlashInfer MXFP4, a 0.90 static-memory fraction, and decode CUDA graph batch size 32.
- Confirm the cell renders as Not Verified and no RTX 5090 benchmark entry is added.

## Failure Handling

If documentation tooling is unavailable locally, run the focused static/config checks that are available and report the missing dependency in the pull request. No runtime fallback command will be added because TP4 was shown not to fit the model on 32GB GPUs.
