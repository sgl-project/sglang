# Fused MQA-logits + top-k for DSA prefill (optimized LiteTopK)

`vendor/` is an optimized LiteTopK kernel: a single-launch fused FP8
MQA-logits + per-row top-k prefill kernel.
[LiteTopK](https://github.com/Heisenberg-Yin/LiteTopK)
([paper](https://arxiv.org/abs/2607.11976)) first proposed the fused indexer
top-k kernel and its sample-filter-select framework: sample each row's scores
to set a gate, keep only the keys that pass it, and select the top-k among
them. [vLLM PR #60618](https://github.com/vllm-project/vllm/pull/60618) (head
`ca75fe8043`, by Roberto L. Castro) improved it into one persistent launch
with the gate computed in the kernel, extended it to multi-request chunks, and
added the exact dense rescan; Ziyi Xu (DarkSharpness) independently came up
with the persistent-kernel idea. This kernel is adapted from #60618 with the
changes listed below. The scoring core is adapted from DeepSeek
[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM).

One persistent CTA per SM claims 8-row blocks, scores every visible key on the
tensor cores (32 heads x 128, FP8), keeps only the keys that pass a per-row
gate set from a sampled quantile of that row's scores, and selects the top-k
among them; a block whose candidate count leaves the exact range is rescanned
densely, so the result is the exact top-k up to ties at the k-th score. The
`[rows, keys]` logits are never written.

This build covers GLM-5.2's indexer only: 32 index heads x 128, FP8 index K,
top-2048.

`entry.cuh` is the SGLang side: tensor validation and the TVM-FFI host entry
the JIT exports. `python/sglang/kernels/ops/attention/dsa/fused_mqa_topk.py`
compiles it, owns the per-(device, stream) workspace and is the public entry.

## What is vendored

| Path under `vendor/` | Origin | License |
| --- | --- | --- |
| `fused_mqa_topk/*.cuh`, `persistent_epoch.inc` (all but `launch.cuh`) | `csrc/fused_mqa_topk/` of PR #60618 at `ca75fe8043`, with the changes below; `output_map.cuh` is new | per-file SPDX: MIT |
| `fused_mqa_topk/launch.cuh` | the PR's host launch code (`fused_mqa_topk.cu`) factored into a framework-agnostic launcher: `FusedMqaTopkParams` + `fused_mqa_topk::launch` | Apache-2.0 (vLLM) |

Not vendored: the PR's libtorch binding (`fused_mqa_topk.cu`, replaced by
`entry.cuh`), its vLLM Python wrapper and tests, and the DeepGEMM headers the
kernel includes (`deep_gemm/{common,mma,ptx}/`, MIT): they come from the
installed sgl-deep-gemm (see below).

Changes against the PR, all measured exact (tie-aware) on B200 with real
GLM-5.2 indexer data:

- Top-2048 only: three instantiations (`<4,2>`, `<8,2>`, `<4,3>` at
  `fused_persist::kTopK`); the launcher rejects any other top-k.
- Gate sample binned relative to a per-row reference (the row's largest sample
  in its first sample tile, 4096 fp16 bins) instead of absolute fp16 bins. Rows
  whose scores sit in a narrow band far from zero (GLM-5.2 layer 1) otherwise
  get a gate that passes far too many keys and fall back to the exact rescan.
- Dispatch and gate constants re-tuned on B200 (shared-memory candidate path up
  to 20K keys, slab gate targets by length, 16-48 sample tiles).
- Ballot-compacted candidate staging; tiles fully visible to all rows of a
  warpgroup skip the per-key bounds; the last interval is stored by the math
  warps.
- Output transforms in the selection epilogue (`output_map.cuh`): row-relative
  indices, plus an optional per-row offset (RAGGED) or a page_size=1 page table
  (PAGED, `page_table[row_batch[r], key - start]`).
- The exact rescan and the resident (<= 4K keys) path schedule one block, so
  launches with more than 32768 rows no longer score a block twice.

## Build inputs

`vendor/` resolves `<fused_mqa_topk/...>`. The registered `cutlass` JIT
dependency provides CUTLASS (the copy bundled with FlashInfer) and
`<deep_gemm/...>` from the installed sgl-deep-gemm (`deep_gemm/include`). The
PR builds against vllm-project/DeepGEMM `6ce1c5ec`; 6 of the 13 headers the
kernel includes differ in sgl-deep-gemm 0.2.3.post2, and the SASS of all four
kernels is byte-identical to a build against the `6ce1c5ec` headers. Beyond
the JIT defaults the kernel needs `--expt-extended-lambda`. The JIT builds for
the local GPU's `sm_100a` / `sm_103a`; vLLM builds the PR for `sm_100f` against
CUTLASS v4.7.1, which gives the same SASS apart from the target suffix.

## Re-syncing with upstream

Port upstream kernel changes onto `vendor/fused_mqa_topk/`, keeping
`launch.cuh`'s `FusedMqaTopkParams` / `launch` interface that `entry.cuh`
calls. When sgl-deep-gemm is bumped, rebuild and compare the kernels' SASS;
a header the kernel needs that changed incompatibly gets a shim in
`deep_gemm_compat.cuh`. `vendor/.clang-format` disables formatting so the
vendored files stay byte-comparable with their source.
