# Fused MQA-logits + top-k for DSA prefill (optimized LiteTopK)

A single-launch fused FP8 MQA-logits + per-row top-k prefill kernel for the DSA
indexer. One persistent CTA per SM claims 8-row blocks, scores every visible key
on the tensor cores (32 heads x 128, FP8), keeps only the keys that pass a
per-row gate predicted from a sampled quantile of that row's scores, and
selects the top-k among them; a block whose candidate count leaves the exact
range is rescanned densely, so the result is the exact top-k up to ties at the
k-th score. The `[rows, keys]` logits are never written.

It covers GLM-5.2's indexer only: 32 index heads x 128, FP8 index K, top-2048,
20481 to 2^20 keys per call, on SM100 (B200). The gates and the dispatch are
tuned and validated on B200 only.

[LiteTopK](https://github.com/Heisenberg-Yin/LiteTopK)
([paper](https://arxiv.org/abs/2607.11976)) first proposed the fused indexer
top-k kernel and its sample-filter-select framework.
[vLLM PR #60618](https://github.com/vllm-project/vllm/pull/60618) (head
`ca75fe8043`, by Roberto L. Castro) improved it into one persistent launch, and
Ziyi Xu (DarkSharpness) independently came up with the persistent-kernel idea.
This kernel is adapted from #60618; its scoring core is adapted from DeepSeek
[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM). The files derived from
them keep their SPDX headers.

`../fused_mqa_topk.cuh` is the TVM-FFI host entry the JIT exports (tensor
validation). `python/sglang/kernels/ops/attention/dsa/fused_mqa_topk.py`
compiles it, owns the per-(device, stream) workspace and is the public entry.

## Layout

| File | Contents |
| --- | --- |
| `constants.h` | Shapes and workspace layout shared by the kernel, the host entry and the Python wrapper |
| `common.cuh` | Device helpers: warp roles, register split, named barriers, warp scan, ordered codes, histogram bins, TMA loads |
| `candidates.cuh` | Candidate record format (fp24 score code + key index) |
| `sample.cuh` | Gate sample histogram and the gate pick |
| `persistent_kernel.cuh` | The persistent kernel: shared-memory layout, linear term, candidate staging and drain |
| `persistent_epoch.inc` | One epoch of the role loops, included twice by the kernel |
| `select.cuh` | Per-row streaming selection from the candidate slab and its exact radix fallback |
| `dense_scan.cuh` | Dense scorer of one block (DeepGEMM's MQA-logits pipeline), used by the exact rescan |
| `rescan.cuh` | Exact rescan: segment merge into per-row retained sets |
| `output.cuh` | Output index transforms (row-relative, per-row offset, page table) |
| `launch.cuh` | Host launch: tensor maps, gate sample, shared memory, launch |

## Build

The kernel includes CUTLASS / CuTe and DeepGEMM's `deep_gemm/{common,mma,ptx}`
headers through the registered `cutlass` JIT dependency (CUTLASS bundled with
FlashInfer; `deep_gemm/include` of the installed sgl-deep-gemm). Beyond the JIT
defaults it needs `--expt-extended-lambda`.

The module is built for `sm_100f` (nvcc >= 12.9), as vLLM builds #60618, not
for the JIT default `sm_100a`: on 70 measured GLM-5.2 chunks that SGLang
dispatches, the `sm_100a` build is slower on 49 (by 0.4-2.2%) and faster on 8
(by 0.6-1.6%).

The kernel's speed is sensitive to code placement in its scan loops: compare
the kernel's SASS (`cuobjdump -sass`) before and after a change, and re-measure
when it differs.
