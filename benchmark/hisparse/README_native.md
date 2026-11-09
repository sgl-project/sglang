# Native LIM and CopySFA validation

With `SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD=1`, set
`SGLANG_NPU_ENABLE_FUSED_LIM_COPYSFA=1` to use native LIM and combined-cache
CopySFA for target decode and target verify. The default value `0` selects the
original four-stream offload implementation. The draft model keeps its
independent native HBM KV pool. Ordinary prefill retains vendor SFA.

The indexer selector, CopySFA enable flag, full-visible MTP enable flag, and
pre/post metadata enable flags have been removed. The fused path does not use
legacy decode dispatch, per-query attention loops, or host callbacks.

## Geometry and residency

The maximum query width is derived from `speculative_num_draft_tokens` (one
without speculation). Each forward uses its actual `spec_info.num_tokens_per_req`;
ordinary decode uses one query. The native LIM limit is 14 queries per request.
The matching kernel wheel exposes `lim_full_visible_max_query_width()` so an
incompatible wheel fails before capture.

`SGLANG_NPU_SPARSE_KV_CACHE_SIZE=0` selects `max_query_width * 2048` resident
slots. Explicit C must cover that union, be page aligned, and be at most 32640.
`SGLANG_NPU_SPARSE_KV_HBM_BUDGET_MB` caps persistent manager allocations.
A3 uses full-visible selection; the separate HBM-tail experiment is not enabled.

In full-visible mode, L is the last actual query position plus one.
For L < C, CopySFA receives `num_cache_tokens=0`, the actual L, and the row's
real HBM page table. It attends all visible KV, including when K < L < C.
For L >= C, it consumes native LIM's resident TopK plan. This threshold is per
request on device; a short neighbor never changes the batch's graph admission.

New KV is published to paged host storage and the short row's continuous HBM
range. Per-layer valid extents prevent repeated history copies. Missing history
is restored through the allocator's physical token mapping, including radix
prefix hits. Request reuse invalidates the extents; rollback from sparse to
dense rehydrates the missing dense range. Long full-visible rows retain the
resident-refresh CopySFA optimization.

## Ascend PD split K/V

Set the following on the Ascend Prefill and Decode workers when using
`--disaggregation-transfer-backend ascend`:

```bash
export SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD=1
export SGLANG_NPU_ENABLE_FUSED_LIM_COPYSFA=1
export SGLANG_NPU_ENABLE_PD_SPLIT_KV=1
```

Prefill allocates native HBM KV. Decode registers separate MemFabric DRAM
CKV and KPE buffers in the same page order and keeps indexer and draft KV in
HBM. Use BF16 KV, `--page-size 128`, and one TP group per worker with PP=1
and no data or context parallelism. Use the same model on both workers.
Decode requires the split-host CopySFA support from
[sgl-kernel-npu #909](https://github.com/sgl-project/sgl-kernel-npu/pull/909)
and a MemFabric installation with shared AccOffload and Ascend transfer
support. The default flag value is `0`.

## Component checks

Initialize Ascend/ATB and select the matching framework and kernel packages,
then run from the framework repository root:

```bash
python3 -m pytest -q \
  test/manual/npu/test_lim_short_dynamic.py \
  test/manual/npu/test_fused_host_kv.py \
  test/manual/npu/test_lim_mtp_copy_metadata.py \
  test/manual/npu/test_lim_pre_metadata_graph.py \
  test/manual/npu/test_lim_mtp_manager.py
```

The mixed-row fixture checks FP32 attention, both KV layers, request/TP padding,
L=1, page/K/C boundaries, sparse-to-dense rollback, radix-prefix attachment,
ragged chunk prefill and exact persistent HBM accounting. Graph capture starts
with all padding and then replays cold real requests.
The separate prefill test checks vendor SFA against FP32 attention after
page-permuted prefix reuse and checks continuous HBM KV bitwise.

```bash
python3 benchmark/hisparse/bench_lim_framework.py \
  --widths 1,4 --graph both --timing --output /tmp/native-lim.json
```

Timing covers one indexer owner and two layer consumers, including current-KV
publication, residency maintenance, CopySFA and output clones. It uses fixed
resident inputs; it excludes model projections, scheduler and sampling.
The `check_lim_*` and `bench_lim_copy_sfa.py` entry points use this same fixture.
Historical measurements in this directory remain tied to their recorded SHAs.

## Original vLLM operator comparison

Build/install the original CopySFA OPP from the chosen vLLM Ascend revision and
source its private vendor `set_env.bash`. Keep its operator sources unchanged.
The helper binds the original Torch adapter rather than replacing its device
implementation:

```bash
python3 benchmark/hisparse/build_copy_sfa_reference.py \
  --vllm-csrc /path/to/vllm-ascend/csrc --build-dir /tmp/vllm-copy-reference
python3 benchmark/hisparse/compare_copy_sfa_short_reference.py \
  --reference-library /tmp/vllm-copy-reference/vllm_copy_reference.so \
  --reference-commit 4f3648abd258a5cbdd1a9ae34165a85f5e44beaf \
  --output /tmp/copy-short-reference.json
```

Both operators consume the same frozen query/KV/page tensors, with an
independent FP32 oracle. The comparison covers C=0 short-row attention only;
it does not establish equivalence of the long-row full-visible and vLLM
prefix/tail policies. Graph timings unroll ten operator calls and exclude LIM,
KV publication and residency staging. Hardware acceptance is separate: A3
results do not establish A5 execution or whole-model accuracy.

## Rebased validation status

The current integration preserves the existing four-stream implementation in
`manager.py` and `attention.py`. Fused execution uses `fused_manager.py`,
`fused_attention.py`, and `fused_lim.py`. The paired kernel is PR #909 at
`e82667615fb69eea3d918dbd8a2a6dda81f6a2d7`. Runtime validation of this rebased
framework commit is pending; earlier A3 results belong to their recorded source
versions. A5 runtime validation is also pending.
