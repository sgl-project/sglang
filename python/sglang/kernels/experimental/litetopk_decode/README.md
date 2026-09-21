# Experimental LiteTopK B200 decode kernels

This source-only package contains the final exact-FP32 decode kernels qualified on
NVIDIA B200 for batch sizes 1, 2, 4, 8, and 16 at 128K through approximately 1M
tokens. It is opt-in and does not change SGLang's default top-k path.

The capture-static dispatch is:

| Batch | Histogram | Selector unroll | Short-row policy |
|---:|---:|---:|---|
| 1 | 2048 bins | 1 | 128 active CTAs at <=128K |
| 2 | 1024 bins | 1 | default |
| 4 | 1024 bins | 4 | default |
| 8 | 1024 bins | 4 | default |
| 16 | 1024 bins | 4 | default |

Each direct call consists of two kernels: one batched DeepGEMM-derived producer and
one CuTe selector with physical page mapping at final write sites. The selector
retains full FP32 boundary refinement; reducing the certificate histogram to 1024
bins does not reduce final selection precision.

## Build

The producer requires the DeepGEMM headers shipped by the compatible vLLM
environment. Override their location with `LITETOPK_NATIVE_DEEPGEMM_INCLUDE`.

```bash
CUDA_VISIBLE_DEVICES='' \
MAX_JOBS=1 \
CUTE_DSL_ARCH=sm_100a \
python python/sglang/kernels/experimental/litetopk_decode/build.py --all
```

Artifacts and a SHA256 manifest are written below `build/`. The build is CPU-only;
GPU execution is intentionally separate.

The checked-in selector is the qualified 2048-bin/B1 source. The build derives the
1024-bin B2/B4/B8/B16 source with checked, single-match substitutions and records its
SHA256 in the manifest.

## Scope

- Target: SM100a/B200.
- Shape: decode Q=1, 32 heads, head dimension 128, top-k 2048.
- Input scores and final selection semantics: exact FP32.
- Supported capture-static batches: 1, 2, 4, 8, 16.
- Direct two-launch use requires the producer and selector length tensors to alias.
- Unsupported shapes and non-aliased length buffers must remain on SGLang's native
  path.

This patch carries the kernel and reproducible AOT build entry only. It deliberately
does not change SGLang's default runtime dispatch because the production installer
must be enabled before CUDA graph warmup and guarded for model-specific DSA metadata.

Qualification receipts for the source snapshot:

- 125/125 cases, 775/775 rows exact.
- Two independent 5x5 paired benchmark runs: 25/25 wins over SGLang.
- Alias graph: two kernel nodes.
- Non-alias safety path in the qualified external installer: three kernel nodes.

The generated selectors retain NVIDIA's Apache-2.0 header. The producer includes
DeepGEMM-derived MIT code.
