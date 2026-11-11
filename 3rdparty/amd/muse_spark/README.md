# Muse Spark 1.2 on MI355X

This directory carries the aiter-side patches for the
ROCm implementation under `sglang.srt.rocm_muse_spark`. On HIP,
`sglang.srt.models.muse_spark` dispatches to that implementation; CUDA keeps
the repository's existing FlashInfer/JIT model unchanged.

The HIP boundary is intentional. A surgical Quark/FlyDSL splice into the
fork's v12 model loaded successfully but generated repetitive invalid text
with both Triton and FlyDSL experts. The same checkpoint and aiter stack
generated correctly through the isolated ROCm implementation. Do not merge
the two model paths until an A/B logits test identifies and validates the
v12 numerical difference; startup and operator parity alone did not catch it.

Validated revisions:

- SGLang baseline: `dc2843801d02f9130feff04635803353561eb604`
- aiter: `4ad99832823dde2315b361cbd3b54b1c5c12acd5`
- composable-kernel: `af9e1d1f1ae347c22feeb08fd2d42645075e0c5d`

The aiter revision is the `AITER_COMMIT_DEFAULT` pinned by
`docker/rocm.Dockerfile`; composable-kernel follows from it as a submodule.

Apply them in a container built from `docker/rocm.Dockerfile`:

```bash
AITER_ROOT=/sgl-workspace/aiter \
  3rdparty/amd/muse_spark/apply_aiter_patches.sh
```

Both patches are optional: without them the aiter backend keeps aiter's own
unified-attention tiles and tuned GEMM tables.

`MUSE_MOE_BACKEND=triton` is the default on HIP. Set it to `flydsl` for the
optimized FlyDSL experts (`sglang.kernels.ops.moe.muse_moe_gemm` /
`muse_moe_finalize`, no aiter patch needed), or `native` for a correctness
fallback.

Launch commands for MI355X are in the Muse Spark cookbook page
(`docs/src/snippets/configs/meta-models/muse-spark.jsx`).
