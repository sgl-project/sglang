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
top-2048, 20481 to 2^20 keys per call, on SM100 (B200). SM103 (B300) is not
enabled: the gates and the dispatch are tuned and validated on B200 only.

`entry.cuh` is the SGLang side: tensor validation and the TVM-FFI host entry
the JIT exports. `python/sglang/kernels/ops/attention/dsa/fused_mqa_topk.py`
compiles it, owns the per-(device, stream) workspace and is the public entry.

## What is vendored

| Path under `vendor/` | Origin | License |
| --- | --- | --- |
| `fused_mqa_topk/*.cuh`, `persistent_epoch.inc` (all but `launch.cuh`) | `csrc/fused_mqa_topk/` of PR #60618 at `ca75fe8043`, with the changes below; `output_map.cuh` is new | MIT (per-file SPDX; `persistent_epoch.inc` and `sample.cuh` have none, as upstream) |
| `fused_mqa_topk/launch.cuh` | the PR's host launch code (`fused_mqa_topk.cu`) factored into a framework-agnostic launcher: `FusedMqaTopkParams` + `fused_mqa_topk::launch` | Apache-2.0 (vLLM) |

Not vendored: the PR's resident kernel and shared-memory candidate path
(`KVS = 2`) for short calls, as SGLang's indexer takes the fused path only from
24576 keys on (the op declines 20480 keys or fewer, which take the dense path;
the slab gates are tuned above that); the PR's libtorch binding
(`fused_mqa_topk.cu`, replaced by `entry.cuh`), its vLLM Python wrapper and
tests; and the DeepGEMM headers the kernel includes
(`deep_gemm/{common,mma,ptx}/`, MIT), which come from the installed
sgl-deep-gemm (see below).

Changes against the PR, all measured exact (tie-aware) on B200 with real
GLM-5.2 indexer data:

- Top-2048 only: two instantiations at `fused_persist::kTopK` (`<4,3>` with
  the streaming selection up to 196608 keys, `<4,3>` above); the launcher
  rejects any other top-k and fewer than `fused_persist::kMinKeys` (20481)
  keys.
- Gate sample binned relative to a per-row reference (the row's largest sample
  in its first sample tile, 4096 fp16 bins) instead of absolute fp16 bins. Rows
  whose scores sit in a narrow band far from zero (GLM-5.2 layer 1) otherwise
  get a gate that passes far too many keys and fall back to the exact rescan.
- Dispatch and gate constants re-tuned on B200 (slab gate targets by length,
  16-48 sample tiles).
- Ballot-compacted candidate staging; tiles fully visible to all rows of a
  warpgroup skip the per-key bounds; the last interval is stored by the math
  warps.
- Output transforms in the selection epilogue (`output_map.cuh`): row-relative
  indices, plus an optional per-row offset (RAGGED) or a page_size=1 page table
  (PAGED, `page_table[row_batch[r], key - start]`).
- The exact rescan schedules one block, so launches with more than 32768 rows
  no longer score a block twice.
- UMMA issue: one elected lane issues a block's whole tile loop and waits for
  the next tile's KV stage before the current tile's last MMA, which removes a
  tensor-core bubble per tile (MMA-only time per tile ~1350 -> ~1155 cycles;
  the MMA bound is 1152).
- Candidate drain: per-row candidate counters in shared memory; one shared
  atomic per row and interval reserves a contiguous slab range, and the stores
  are branch-free and predicated (no shuffle or atomic in the loop).
- Rows that see at most 2048 keys, the padding rows of a partial last block
  included, get a +inf gate: they are written directly and take no
  candidates, and they no longer shrink the block's range of fully visible
  tiles.
- Streaming selection for launches of up to 196608 keys (`kStream`, a
  separate instantiation, so the long-context instantiation is not affected by
  it): a bracket of the K-th score from clustered samples (relative bins, a
  second level when an outlier stretches the range), one coalesced pass over
  the row's slab, exact levels within the bracket; a missed bracket retries
  wider, then falls back to an exact radix select over the slab. In this
  instantiation the KV producer also loads the next block's first sample
  scales before waiting for the current block's selection, and the math warps
  load a sample tile's next row from TMEM while binning the current one.

## Build inputs

`vendor/` resolves `<fused_mqa_topk/...>`. The registered `cutlass` JIT
dependency provides CUTLASS (the copy bundled with FlashInfer) and
`<deep_gemm/...>` from the installed sgl-deep-gemm (`deep_gemm/include`). The
PR builds against vllm-project/DeepGEMM `6ce1c5ec`; 6 of the 13 headers the
kernel includes differ in sgl-deep-gemm 0.2.3.post2. Beyond the JIT defaults
the kernel needs `--expt-extended-lambda`.

The module is built for `sm_100f` (nvcc >= 12.9), as vLLM builds the PR, not
for the JIT default `sm_100a`. On single-request GLM-5.2 chunks (B200), the
`sm_100a` build of the long-context instantiation (> 196608 keys) keeps
local-memory loads and stores on its scan loop's hot path and is 0.6-1.4%
slower, while the streaming one is 0.6-2.5% faster. Chunks whose rows mostly
see only a few thousand keys (many short requests, sliding windows) run up to
~12% faster with `sm_100a`; the mixed chunks SGLang dispatches (one long
request plus short ones) differ by about 1% either way. Estimated from
per-chunk times, a 1M-key prefill is ~1.2% faster with `sm_100f`, a 256K one
about even, and up to ~192K keys `sm_100a` is ~1% faster. The `sm_100f` SASS
of both kernels is byte-identical to a build against CUTLASS v4.7.1 and the
`6ce1c5ec` headers.

## Re-syncing with upstream

Port upstream kernel changes onto `vendor/fused_mqa_topk/`, keeping
`launch.cuh`'s `FusedMqaTopkParams` / `launch` interface that `entry.cuh`
calls; upstream changes to the resident kernel or the shared-memory candidate
path do not apply. When sgl-deep-gemm is bumped, rebuild and compare the
kernels' SASS; a header the kernel needs that changed incompatibly gets a shim
in `deep_gemm_compat.cuh`. `vendor/.clang-format` disables formatting so the
vendored files stay diff-comparable with their source.
