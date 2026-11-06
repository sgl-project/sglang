# Muse Spark on MI355X: aiter patches

This directory carries the aiter-side patches the ROCm implementation under
`sglang.srt.rocm_muse_spark` needs. On HIP, `sglang.srt.models.muse_spark`
dispatches to that implementation; CUDA keeps the repository's FlashInfer/JIT
model unchanged.

The patches apply to aiter `4ad99832823dde2315b361cbd3b54b1c5c12acd5`, the
`AITER_COMMIT_DEFAULT` pinned by `docker/rocm.Dockerfile`; composable-kernel
follows from it as a submodule.

| Patch | What it does |
|---|---|
| `01-aiter-fix-page-scatter-vm-fault.patch` | Fixes a gfx950 VM fault in the CK batch-prefill page gather (composable_kernel submodule). |
| `03-plan-b-aiter-unified-attention-3d.patch` | CK paged prefill plus the unified-attention 3D decode kernel (aiter half of the patch). |
| `04-aiter-muse-moe-flydsl.patch` | FlyDSL Muse MoE kernels, required by `MUSE_MOE_BACKEND=flydsl`. |
| `05-aiter-unified-attention-muse.patch` | Opt-in verify / long-prefill tile arguments of aiter's Triton `unified_attention()`. |
| `06-aiter-muse-spark-bf16-tuned-gemm.patch` | Muse Spark hipBLASLt solutions in aiter's BF16 tuned-GEMM table (CSV only). |
| `experimental/aiter-muse-flydsl-swa-decode.patch` | FlyDSL split-K paged GQA decode for the short-window layers; opt-in. |

`docker/muse_spark/muse-spark-rocm.Dockerfile` applies the patches at build
time, so an image built from it needs no runtime patching. In a container built
from the plain ROCm image, apply them by hand:

```bash
AITER_ROOT=/sgl-workspace/aiter \
  3rdparty/amd/muse_spark/apply_aiter_patches.sh

# Also install the experimental short-window decode kernel:
ENABLE_PLAN_C=1 AITER_ROOT=/sgl-workspace/aiter \
  3rdparty/amd/muse_spark/apply_aiter_patches.sh
```

`MUSE_MOE_BACKEND` selects the expert path on HIP: `auto` (default, the fused
Triton experts), `flydsl` for the FlyDSL experts, or `native` for a correctness
fallback. The experimental short-window decode stays off unless
`SGLANG_MUSE_FLYDSL_DECODE=1`.

For launch commands, see the MI355X section of the Muse Spark Open cookbook
(`docs/cookbook/autoregressive/Meta/MuseSparkOpen.mdx`).
