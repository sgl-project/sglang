# Capture GPU Attribution

The raw traces and their SHA256 values are recorded in
[`capture-serving-performance.json`](capture-serving-performance.json).
One H100 runs Qwen3-0.6B BF16 with ordinary overlap scheduling, decode CUDA
graphs, local TCP Mooncake and a test Catalog. Each probe has five batches of
eight requests after ten warmup batches. All 40 requests reach READY in each
enabled workload. This profiler run does not independently read back tensors;
the separate uninstrumented serving benchmark validates 422 snapshots.

CPU scopes come from the test-only `training_capture_profile_server` entrypoint.
`summarize_capture_trace.py` follows the CUDA launch correlation rather than
assuming that GPU work fits within the CPU scope. It counts only CPU
`user_annotation` events; PyTorch also exports same-name GPU annotations.
The regression fails before this filter and passes after it. All observed D2H
events contain a byte count. Kernel durations below are summed work, not latency.

## Kernel Table

Shares use all kernel time in the corresponding enabled trace: 77.330ms for the
128-to-1 workload and 249.826ms for the 1-to-32 workload. DMA is reported separately.
The teacher row aggregates all top-k, conversion and LSE kernels attributed to
that scope; it does not attribute unrelated target model kernels.

| Workload | Kernel / Group | GPU Time | Kernel Share | Launches | Python Source / CPU Op |
| --- | --- | ---: | ---: | ---: | --- |
| 128-to-1 | KV gather and copy kernels | 0.874ms | 1.13% | 540 | `training_capture/kv_exporter.py`, `index_select` / `copy_` |
| 128-to-1 | Teacher top-128 and LSE | 3.744ms | 4.84% | 1,395 | `training_capture/teacher.py`, `topk` / `logsumexp` |
| 1-to-32 | `indexSelectSmallIndex` | 15.133ms | 6.06% | 7,920 | `training_capture/kv_exporter.py`, `index_select` |
| 1-to-32 | `direct_copy_kernel_cuda` | 10.530ms | 4.21% | 7,920 | `training_capture/kv_exporter.py`, `copy_` |
| 1-to-32 | Teacher top-128 and LSE | 17.190ms | 6.88% | 5,115 | `training_capture/teacher.py`, `topk` / `logsumexp` |

The 128-to-1 trace contains 45 capture forwards: 40 eager and five decode-graph
forwards issued ahead of terminal result processing. KV DMA totals 62,976,000
bytes in 270 copies (2.020ms), teacher DMA 46,260 bytes in 135 copies (0.322ms),
and position DMA 41,000 bytes in 45 copies (0.124ms). These include five extra
token rows beyond the nominal 40-request prompt/teacher payload.

The 1-to-32 trace contains 165 capture forwards: five eager and 160 decode-graph
forwards. Each of the eight requests contributes 33 staged rows, including a
one-ahead row that can be trimmed before publication. KV DMA totals 16,220,160
bytes in 7,920 copies (20.752ms), teacher DMA 1,356,960 bytes in 3,960 copies
(10.043ms), and position DMA 10,560 bytes in 1,320 copies (3.505ms). Together this
is 13,200 copies, 17,587,680 bytes and 34.300ms of DMA work. The 1,320 CPU KV calls
take 779.516ms under instrumentation; that is neither kernel time nor normal
serving latency. All D2H in this trace, including non-capture traffic, takes
34.727ms. Capture-off traces have no attributed capture groups.

## Overlap Opportunity Table

| Priority | Verdict | Scope | Evidence | Dependency | Next Measurement |
| --- | --- | --- | --- | --- | --- |
| 1 | Not established | KV and teacher D2H | Many small copies; the generic triage reports no qualifying formal overlap row | Pool reads must precede slot reuse; Host storage must remain reserved until completion | Compare a bounded batching implementation with identical Store readback, abort and overlap tests, then repeat the serving benchmark |
| 2 | Not established | Teacher top-k/LSE | 17.190ms device work over 165 decode-workload calls | Raw scores must be captured before sampling mutation or graph-buffer reuse | Measure launch reduction without changing raw logits, vocabulary IDs or full-vocabulary normalization |

Neither the single-trace generic triage nor summed durations prove removable
critical-path time. No concurrency/fusion speedup is claimed here.

## Fuse Pattern Table

| Pattern | Verdict | Source Evidence | Constraint |
| --- | --- | --- | --- |
| Batch selected-layer KV gathers/copies | Candidate, unimplemented | `SelectedLayerKVExporter.export` loops over six K/V buffers per request; the decode trace records 7,920 gather kernels and 7,920 copy kernels | Preserve per-request ordering, noncontiguous physical slots, exact tensor bits and safe lifetime through cancellation |
| Pack teacher fields before D2H | Candidate, unimplemented | `RequestCaptureContext.record_teacher_range` copies IDs, logits and LSE separately, 3,960 transfers for 1,320 rows | Preserve int32 vocab IDs, FP32 raw logits/LSE and manifest layout without delaying slot release indefinitely |
| FP8 scaled-MM replacement inferred from `nvjet` name | Rejected for this run | Generic skill triage emits this row, but the actual model is BF16 unquantized; its source maps to `quantization/unquant.py` | Kernel-name presence does not establish FP8 semantics or applicability of an upstream change |
| Existing fused norm / QK norm / RoPE paths | Presence only | Generic triage maps these target-model kernels to their existing source paths | Their presence does not prove an additional fusion is missing or relevant to capture overhead |

The generic three-table command was run on both enabled traces:

```bash
python .claude/skills/llm-torch-profiler-analysis/scripts/analyze_llm_torch_profile.py \
  --framework sglang --input /path/to/on/decode \
  --kernel-table-limit 15 --overlap-table-limit 10
```

This report corrects the inapplicable FP8 heuristic and keeps only source-backed
capture observations. The profiler probes use 100% admission and wait for spare
reservations; they cannot quantify the 10% sampling throughput reduction seen
in the separate normal-serving run or certify a production SLO.
