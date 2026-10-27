# Fused MQA-logits + top-k for DSA prefill (optimized LiteTopK)

A single-launch fused FP8 MQA-logits + per-row top-k prefill kernel for the DSA
indexer. One persistent CTA per SM claims 8-row blocks, scores every visible key
on the tensor cores (32 heads x 128, FP8), keeps only the keys that pass a
per-row gate set from a sampled quantile of that row's scores, and selects the
top-k among them; a block whose candidate count leaves the exact range is
rescanned densely, so the result is the exact top-k up to ties at the k-th
score. The `[rows, keys]` logits are never written.

It covers GLM-5.2's indexer only: 32 index heads x 128, FP8 index K, top-2048,
20481 to 2^20 keys per call, on SM100 (B200). SM103 (B300) is not enabled: the
gates and the dispatch are tuned and validated on B200 only.

`../fused_mqa_topk.cuh` is the TVM-FFI host entry the JIT exports (tensor
validation). `python/sglang/kernels/ops/attention/dsa/fused_mqa_topk.py`
compiles it, owns the per-(device, stream) workspace and is the public entry.

## Credit

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

Files that derive from #60618 keep its SPDX headers (MIT for the kernel,
Apache-2.0 for the host launch code); the DeepGEMM-derived code (the dense
scorer, the TMA loads, the candidate record format) keeps DeepSeek's MIT
notice.

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

## Changes against #60618

All measured exact (tie-aware) on B200 with real GLM-5.2 indexer data:

- Top-2048 only and only the global candidate slab path: one kernel for every
  launch; the launcher rejects any other top-k and fewer than `kMinKeys`
  (20481) keys. #60618's resident kernel (at most 4096 keys) and
  shared-memory candidate path (at most 16384 keys) are not carried over, as
  SGLang's indexer takes the fused path only from 20481 keys on.
- Gate sample binned relative to a per-row reference (the row's largest sample
  in its first sample tile, 4096 fp16 bins) instead of absolute fp16 bins. Rows
  whose scores sit in a narrow band far from zero (GLM-5.2 layer 1) otherwise
  get a gate that passes far too many keys and fall back to the exact rescan.
- Dispatch and gate constants re-tuned on B200 (slab gate targets by length;
  above 128K keys 16-48 sample tiles instead of 16-32).
- Ballot-compacted candidate staging; tiles fully visible to all rows of a
  warpgroup skip the per-key bounds.
- Output transforms in the selection epilogue (`output.cuh`): besides
  row-relative indices, a page_size=1 page table
  (`page_table[row_batch[r], key - start]`, the PAGED transform GLM-5.2 takes
  on B200) or a per-row offset.
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
  candidates, and they no longer narrow their warpgroup's range of fully
  visible tiles.
- Streaming selection instead of #60618's staged one: a bracket of the k-th
  score from clustered samples (relative bins, a second level when an outlier
  stretches the range), one coalesced pass over the row's slab in global
  memory, exact levels within the bracket; a missed bracket retries wider,
  then falls back to an exact radix select over the slab. Only the bracket and
  the keys above it are staged in the per-warp scratch, and an overflow takes
  the wider retry and then the radix select, so a block no longer falls back to
  the exact rescan for overflowing a selection scratch. The KV producer loads
  the next block's first sample scales before waiting for the current block's
  selection, and the math warps load a sample tile's next row from TMEM while
  binning the current one.
- Math-warp scan pipeline: a tile's KV stage (its scales) is awaited after the
  first pair of TMEM loads instead of at the tile top, each half loads its
  scales after its first TMEM loads, and the second half releases the stage
  before its last loads; a half's TMEM slot addresses are formed before its
  `full_tmem` probe; the first half test-waits the second half's slot without
  blocking, and on success the second half skips its blocking probe. Barrier,
  scale-stage and staging addresses and the lane tables are read once from
  shared memory instead of being recomputed in the loop.
- Emission: two warp votes per half (rows 0-1 right after their scores, rows
  2-3 at the end of the half); per-row staging pointers, so a record is stored
  at the row's next free slot and counts come from the list ends; staged
  records keep the raw fp32 score bits and the drains form the ordered code
  (the slab contents are unchanged); the drain loop takes two records per
  iteration.
- MMA issue: an A descriptor is the stage-0 descriptor plus a start-address
  offset instead of one descriptor build per MMA.

## Build

The kernel includes CUTLASS / CuTe and DeepGEMM's `deep_gemm/{common,mma,ptx}`
headers through the registered `cutlass` JIT dependency (CUTLASS bundled with
FlashInfer; `deep_gemm/include` of the installed sgl-deep-gemm). Beyond the JIT
defaults it needs `--expt-extended-lambda`.

The module is built for `sm_100f` (nvcc >= 12.9), as vLLM builds #60618, not
for the JIT default `sm_100a`. On 70 measured GLM-5.2 chunks that SGLang
dispatches (B200; single requests and mixed chunks of 32K-1M keys), the
`sm_100a` build is slower on 49 by 0.4-2.2% and faster on 8 by 0.6-1.6%
(layer-1/6 chunks of about 32K keys, and one long request plus 40 short
ones); estimated from per-chunk times, a 1M-key single-request prefill is
1.3-2.0% faster with `sm_100f`. Chunks of only short requests (no row sees more
than 8K keys), which SGLang keeps on the dense path, take 0.86-0.94x the time
with `sm_100a`.

The kernel's speed is sensitive to code placement in its scan loops: compare
the kernel's SASS (`cuobjdump -sass`) before and after a change, and re-measure
when it differs.
