# op_trace: GPU-only per-op tracing (experimental spike)

Goal: attribute every GPU kernel of an e2e decode/prefill step to a semantic op (module path such as `model.layers.3.self_attn.qkv_proj`, op kind `self_attn.qkv_proj`) while the runtime only pays for GPU-side kernel timestamps. Attribution is computed once, at CUDA-graph capture time; replay stays untouched. Kernel symbol, grid and graph node are attributes, never the identity.

All numbers below: Qwen3-8B BF16, TP1, one GB300 (driver 580.159.04, CUDA 13.0, torch 2.14.1+cu130 / bundled kineto), default cookbook config (`trtllm_mha` attention, full decode CUDA graphs) except `--disable-prefill-cuda-graph` where noted, driven through `sglang.benchmark.one_batch` internals (input 512, decode bs 1 and 64; prefill 1 x 4096).

## How it works

- `capture.py` (gated by `SGLANG_ENABLE_OP_TRACE_MAP`, off by default; zero cost when off): when the full CUDA-graph backend starts a capture session it installs forward pre/post hooks on every submodule of the model. During a capture, each hook reads the node count of the graph under capture (`cudaStreamGetCaptureInfo` + `cudaGraphGetNodes(numNodes=0)`), so every node is assigned the innermost module whose forward created it. Modules shared by several parents (the single rotary embedding of Qwen3) take their path from the calling parent. Before the capture ends the graph's nodes are snapshotted (type, kernel symbol via `cuFuncGetName`, grid, block) together with a per-process capture ordinal and the `CUgraph` handle of the graph under capture (torch frees the captured graph after instantiation, so handles can repeat across captures). The capture path deliberately makes no CUPTI call: the first CUPTI call in a process attaches CUPTI and slows every later graph replay (see B3). The map is written as `op_map-TP<rank>-pid<pid>.json` into `SGLANG_TORCH_PROFILER_DIR`.
- Outside a capture, the same hooks emit `op::<path>` spans (`RecordFunctionFast`) on eager forwards, only while a torch profiler is active.
- `analyze.py` joins a kineto trace with the map: replayed kernels carry `graph id` (the instantiated exec) and `graph node id` = `(exec id << 32) | node creation index`. The exec is matched to a captured graph exactly when the trace carries the execs in creation order with their source graph handles (native collector, `opTraceExecs`): execs and captures are walked in order and paired when the handle agrees, with the fingerprint as a check; else by fingerprint alone (node index range, per-node grid, symbol compared after demangling). Each bucket reports which (`graph_join`: exact / fingerprint / fingerprint-ambiguous / eager); then `node_op[node index]` gives the op. Eager kernels join through their launch correlation id to the innermost `op::` span on the launching thread (needs CPU activity). Steps come from the CPU `step[...]` span when present (kernels joined by launch time), else one step per graph replay on the GPU timeline. For eager prefill the bench takes one CPU+GPU step as the reference and times a GPU-only pass of N steps: the timing trace is cut into N steps at its largest idle gaps (host synchronize between steps), each step is aligned to the reference symbol sequence (difflib), and unmatched or missing kernels count as step mismatches. Op maps are only joined with traces of the same pid. A bucket with any symbol mismatch or less than 95% of kernel time attributed exports `status: "error"` records.
- CLI `python -m sglang.srt.observability.op_trace summarize TRACE` prints one table per step bucket (op kind, module class, layers, kernels/step, us/step p50, share, us/layer p50, top kernel); `--json` writes a `TraceSummary` (msgspec), `--records` keeps `StepRecord`/`KernelRecord`. `diff BASE NEW` prints the op-level delta sorted by absolute us/step change; `--depth N` rolls op kinds up to N components.

## Q1. What does kineto record?

- Replay (CUDA activity only): every kernel and graph memcpy/memset event has `graph id` and `graph node id` in its args. `graph id` is the id of the instantiated exec, not of the captured graph (probe: captured graph 1, exec 2; e2e: captured 1 -> exec 2, captured 4 -> exec 5). The low 32 bits of `graph node id` are the node's creation index, identical to `cudaGraphGetNodes` order and to the node count our hooks read during capture. Eager kernels have `graph id` 0.
- Capture (CPU+CUDA): kineto records the launch runtime calls (`cudaLaunchKernelExC`, `cuLaunchKernelEx`, ...) with correlation ids and the CPU op stack, but no node id and no kernel name, and no kernel events (nothing executes). So capture-time kineto alone cannot produce the map; reading the graph under capture can.
- `cudaGraphNodeGetToolsId`, `cudaGraphNodeGetLocalId`, `cudaGraphGetId`, `cudaGraphExecGetId` exist in cuda-bindings 13.4 but return `cudaErrorCallRequiresNewerDriver` on driver 580 (CUDA 13.0).

## Q2. Overhead

Graph replay GPU time (CUDA events around `graph.replay()`, p50 of 40 steps, same KV range; "off clean" is before any profiler session in the process):

| decode | nodes | off clean | GPU-only | CPU+GPU | off after a session |
|---|---:|---:|---:|---:|---:|
| bs=1 | 366 | 2962 us | 3253 us (+9.8%, 0.80 us/node) | 3251 us | 3041 us (+79 us) |
| bs=64 | 438 | 4285 us | 4585 us (+7.0%, 0.69 us/node) | 4597 us | n/a (KV growth confounds) |

Per-step wall time (sync per step, bench loop incl. CPU prep; run-to-run noise ~5%): bs=1 3.84 / 4.16 / 4.38 ms, bs=64 5.30 / 5.63 / 5.84 ms for off / GPU-only / CPU+GPU.

Microbenchmark (one graph of 400 x 1.25 us kernels): replay 409 us off, 703-739 us profiled (0.73-0.82 us/node), 478-480 us after `prof.stop()` and still after 2 s (+0.17 us/node persists for the rest of the process).

- The profiler cost on graph replay is GPU-side and per node (~0.7-0.8 us/node with kineto's default CUPTI path); CPU activity adds nothing to replay GPU time, only CPU/wall time.
- Kernel durations: GPU-only vs CPU+GPU per-kernel durations are identical (graph busy 3215 vs 3216 us at bs=1, 4671 vs 4663 us at bs=64). But the traced per-kernel sum (3215 us) exceeds the unprofiled whole replay (2962 us): traced durations are inflated by ~0.7 us/kernel, which matters for 2-4 us kernels (norms, activations, rope). Diffs between two traces taken the same way mostly cancel it; absolute shares of small ops are biased up.
- After one kineto session, graph replay stays ~0.2 us/node slower for the rest of the process. `TEARDOWN_CUPTI=1` removed this in the single-session microbenchmark, but in the e2e process the next profiler session then recorded zero GPU activity and the penalty remained, so it is not a fix.
- Enabling CUPTI hardware-event tracing (`cuptiActivityEnableHWTrace(1)`, returns success) before kineto starts hung the profiled replays; kineto exposes no option for it.

Eager prefill 1 x 4096 (`--disable-prefill-cuda-graph`), wall p50:

| config | off | GPU-only | CPU+GPU |
|---|---:|---:|---:|
| no op-trace hooks | 39.9 ms | 41.0 ms (+2.6%) | 45.9 ms (+15%) |
| hooks, `RecordFunctionFast` spans | 40.3 ms | 44.1 ms (+10%) | 51.4 ms (+27%) |
| hooks, `torch.profiler.record_function` spans | 39.9 ms | 51.9 ms (+30%) | 62.4 ms (+56%) |

Idle hooks (profiler off) are within noise here because this prefill is GPU-bound. Torch has no query for the active profiler's activities, so `capture.set_eager_spans(enabled)` lets the code that starts the profiler say so; sglang's `_ProfilerTorch` sets it from its activity list (spans only with CPU activity) and resets it on stop, while external profilers keep the old behaviour (`_profiler_enabled()`). Measured: GPU-only prefill 44.6 ms with spans -> 42.5 ms without (off 39.9 ms); the remaining ~1.5 ms over the hook-free 41.0 ms is the hooks' own Python calls.

## Q3. Capture-time op map and coverage

Implemented option 1 (exact join by graph node id), since Q1 shows it is available. Option 2 (launch ordinal within one replay) would also have worked here: the decode graphs are single-stream and replay order equals node order (0 inversions), but with PDL kernels already overlap (bs=64 busy sum 4671 us > span 4523 us), and any multi-stream graph would break a time-ordered join, while the node index is order-independent.

Decode, GPU-only trace, 40 steps:

| bucket | kernel time attributed | unattributed | symbol mismatches |
|---|---:|---|---:|
| DECODE graph bs=64 | 99.4% | 12.7 eager kernels/step, 28 us (input prep + sampling outside the graph) | 0 / 437 |
| DECODE graph bs=1 | 99.2% | 12.7 eager kernels/step, 27 us | 0 / 365 |

`(graph-glue)` (1 kernel, `update_trtllm_mha_graph_metadata_kernel`, 2.7 us, plus one event-record node) is inside the graph but outside the model forward. With CPU+GPU activity and the CPU step span, decode is 100% attributed (sampling falls outside the step span and is dropped).

Op identity follows module structure, not kernels: at bs=1 the FlashInfer dual GEMM is launched by `mlp` itself (gate_up + act fused), at bs=64 by `mlp.gate_up_proj` and `mlp.act_fn`; q/k norm is launched from `self_attn` (`fused_qknorm`), not from `q_norm`/`k_norm`. `--depth 1` rolls these up for diffs.

Sample `summarize` (decode bs=64, GPU-only, top 15):

| op_kind | module | layers | kernels/step | us/step p50 | share | us/layer p50 | top kernel |
|---|---|---:|---:|---:|---:|---:|---|
| mlp.gate_up_proj | MergedColumnParallelLinear | 36 | 36.0 | 1215.8 | 25.4% | 33.79 | nvjet_sm103_tst_192x64_64x6_4x1_v_bz_TNT |
| self_attn.attn | RadixAttention | 36 | 72.0 | 1168.2 | 24.0% | 31.97 | fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvCausalP64... |
| mlp.down_proj | RowParallelLinear | 36 | 72.0 | 819.1 | 18.0% | 22.69 | nvjet_sm103_tst_64x64_64x16_2x1_2cta_v_bz_splitK_TNT |
| self_attn.qkv_proj | QKVParallelLinear | 36 | 36.0 | 411.4 | 8.6% | 11.36 | nvjet_sm103_tst_64x64_64x13_4x1_v_bz_TNT |
| self_attn.o_proj | RowParallelLinear | 36 | 36.0 | 303.7 | 6.4% | 8.38 | kernel_cutlass_kernel_TgvGemmCuteExtKernel_cta64x32x128_2... |
| logits_processor | LogitsProcessor | - | 2.0 | 213.0 | 4.5% | - | nvjet_sm103_tst_512x64_64x3_2x1_2cta_v_bz_TNT |
| mlp.act_fn | SiluAndMul | 36 | 36.0 | 142.3 | 3.0% | 3.84 | sglang::act_and_mul_kernel<__nv_bfloat16, |
| post_attention_layernorm | RMSNorm | 36 | 36.0 | 141.9 | 3.0% | 3.94 | sglang::rmsnorm_kernel<...> |
| input_layernorm | RMSNorm | 36 | 36.0 | 123.4 | 2.6% | 3.46 | sglang::rmsnorm_kernel<...> |
| self_attn | Qwen3Attention | 36 | 36.0 | 102.9 | 2.2% | 2.85 | sglang::fused_qknorm_half_warp<true, __nv_bfloat16> |
| self_attn.rotary_emb | RotaryEmbedding | 36 | 36.0 | 81.0 | 1.7% | 2.24 | sglang::fused_rope_kernel<...> |
| (eager) | | - | 12.7 | 28.3 | 0.6% | - | at::native::vectorized_elementwise_kernel<...> |
| norm | RMSNorm | - | 1.0 | 3.4 | 0.1% | - | sglang::rmsnorm_kernel<...> |
| (graph-glue) | | - | 1.0 | 2.7 | 0.1% | - | update_trtllm_mha_graph_metadata_kernel |
| embed_tokens | VocabParallelEmbedding | - | 1.0 | 2.1 | 0.0% | - | at::native::vectorized_gather_kernel<16, long> |

Capture cost: capturing the default 12 decode sizes (bs 1-64) took 8.94 s with the op map vs 8.68 s without (+0.26 s, one-off); the map file is ~68 KB for 2 graphs. All 12 graphs have distinct (symbol, grid) fingerprints.

## Q4. Eager prefill via CPU-op correlation

With CPU+GPU activity and the `op::` spans, 100% of the 1 x 4096 prefill kernel time (37.4 ms, 449 kernels) is attributed (kernel -> correlation -> launching runtime call -> innermost `op::` span on that thread); the only `(eager)` rows are 8 small kernels (15 us) between ops. Cost: +27% wall vs off (51.4 vs 40.3 ms), of which ~15 points are plain kineto CPU tracing and ~12 points the spans. This is fine for a sampled step, not for continuous use. Note the default prefill path for this model is the breakable CUDA graph (`backend=breakable`), which this prototype does not map (below).

## Limitations

- Only `FullCudaGraphBackend` captures are mapped. Breakable (default prefill here) and piecewise/torch.compile graphs replay as `(unmapped-graph)`. Breakable segments can be mapped with the same node counter per segment graph, but segments are deduplicated across shapes/layers, so the layer must come from the replay sequence rather than the exec id.
- GPU-only steps know the padded graph size only, not the real batch size or KV length; eager kernels outside the graph (input prep, sampling) stay `(eager)`.
- Without the native collector's exec map, exec identity uses grids + symbol base names; unique for the 12 decode sizes checked here, but sizes that differ in no grid would be ambiguous (first match wins). In the spike runs the CUPTI ids also followed `exec = captured + 1`; that hint was dropped together with the capture-time CUPTI call.
- Untested: TP > 1 (one map per rank/pid), speculative decoding runners, LoRA variants, models whose forward bypasses `nn.Module.__call__`.
- torch.compile: the op map is unsupported and unvalidated with torch.compile capture backends (dynamo traces through the module hooks); the full-graph backend skips registering the hooks when `enable_torch_compile` is set.

## B3: minimal native CUPTI collector

`sglang/kernels/jit/csrc/profiling/cupti_collector.cpp` (a non-header-only tvm-ffi JIT module, built by `sglang/kernels/jit/debug/cupti_collector.py` through `load_jit` against a chosen CUPTI root) enables `CONCURRENT_KERNEL` (+ memcpy/memset), uses 8 MB activity buffers, reduces records to int64 rows (start, end, graph node id, graph id, correlation, interned name, stream, kind, grid) in memory, and subscribes to `CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATED` / `GRAPH_CLONED` to record (exec id, source `CUgraph` handle) in creation order. `op_trace/native.py` `write_kineto_like_trace` turns the records into the JSON the analyzer already reads (timestamps relative to the first record), plus `opTraceExecs` and `opTracePid`. `init` refuses a runtime CUPTI older than the compile-time headers (record layouts differ). CUPTI 13.4.1 (`nvidia-cuda-cupti` 13.4 wheel, `LD_PRELOAD`ed so torch and the collector share one copy; headers `cupti_activity.h`/`cupti_callbacks.h` only, since `cupti.h` needs 13.4 runtime headers) on driver 580 (CUDA 13.0); torch's bundled CUPTI is 13.0.1, which returns `CUPTI_ERROR_NOT_SUPPORTED` for HES. No kineto session ran in these processes.

Graph replay GPU time (CUDA events around `replay()`, p50 of 40 steps, rep 1; bs=64 drifts with KV length across reps, so its on/off pairs are compared within a rep):

| mode | microbench 400 x ~1 us | decode bs=1 (366 nodes) | decode bs=64 (438 nodes) |
|---|---:|---:|---:|
| a. clean process (no CUPTI) | 410 us | 2964 us | 4204 us |
| b. SW instrumentation, idle (CUPTI attached, no kinds enabled) | 487-505 us | 3103 us (+4.7%) | 4358 us |
| b. SW instrumentation, tracing | 678-764 us (+0.7-0.9 us/node) | 3334-3343 us (+12.6%) | 4668 us (+11%) |
| c. HES, idle | 470-476 us (+0.15 us/node) | 3035 us (+2.4%) | 4276 us (+1.7%) |
| c. HES, tracing | 471-475 us (+0) | 3030-3036 us (+0 vs idle) | 4336 us (+1.4% vs idle) |
| d. after stop() | = idle (SW 486-506, HES 470) | SW 3103, HES 3040 | SW 4429, HES 4345 |
| d. after cuptiFinalize() | SW: next replay faults (`unspecified launch failure`); HES: 420 us, but a re-init then records nothing | | |

e. Kernel durations: the HES median kernel in the microbench is 0.86 us vs 1.25-1.28 us under SW (kineto and native alike); the HES per-replay kernel sum is 349 us < 410 us clean replay, SW's is 485-494 us. In decode, the graph kernel sum per step is 3073 us (HES) vs 3195 us (SW native) vs 3215 us (kineto) against a 2964 us clean replay (sums exceed the span because PDL overlaps kernels), so SW inflates by ~0.35-0.4 us/kernel and HES by much less. Per-op values from HES are 5-15% lower than kineto's for small ops (rotary 70 vs 81 us/step, input norm 115 vs 123 us/step at bs=64).

Toggling: start/stop cycles keep working indefinitely as long as CUPTI is never finalized (5 cycles x 4000 records each, both modes; start 0.01 ms SW / ~1 ms HES, stop + forced flush 0.6-0.8 ms SW / ~4-5 ms HES). The penalty that persists after stop() is CUPTI being attached (SW 0.2 us/node, HES 0.15 us/node), not tracing. Attaching late (after the graphs exist) gives the same idle penalty. The first CUPTI call of any kind attaches it: the prototype's capture-time `cuptiGetGraphId` call alone raised clean bs=1 replay from 2964 to 3101 us, so the capture path now records the `CUgraph` handle instead (2965 us with the op map on, same as clean).

Join: native records join to the capture-time op map exactly like kineto's: decode bs=64 99.5% / bs=1 99.2% of kernel time attributed, 0 symbol mismatches, unattributed = the 12.7 eager kernels/step outside the graph. `GRAPHEXEC_CREATED` gives the execs in creation order with their source graph handle (exec 2 -> the bs=64 graph, exec 5 -> bs=1; no clone in between); pairing them with captures in capture order replaces the fingerprint match (handle alone is not unique), and both picked the same graphs here. CUPTI kernel names are mangled, so reports show mangled symbols (demangle at analysis time if wanted).

## Recommendation (native collector)

Go, with HES. HES makes tracing itself free on graph replay (measured +0 to +1.4%), keeps kernel durations close to unperturbed, and start/stop cycles are cheap and repeatable. The cost is a constant ~0.15 us/node (+1.7-2.4% decode replay here) from the moment CUPTI is attached, which HES needs before the CUDA context exists, so it is paid by every process that wants to be traceable. Kineto and SW-mode CUPTI cost 0.7-0.9 us/node while tracing, ~0.2 us/node after the first session, and `cuptiFinalize` either faults graph replays (SW) or disables CUPTI for good (HES). kern's ~0.1 us/node is in reach only as the HES idle cost; on this driver tracing adds nothing on top.

Minimal production shape:

- A small C++ library (`cupti_collector.cpp` plus a ring of preallocated buffers and a drain thread), built with the existing JIT/extension path, linked against one libcupti shared with torch. Opt-in flag read before CUDA init; when off, CUPTI is never touched.
- Attach at process start (HES + `GRAPHEXEC_CREATED` callback); enable/disable `CONCURRENT_KERNEL` (+ memcpy/memset) per sampled window; never finalize; never run a kineto session in the same process (one CUPTI subscriber).
- Exact exec -> captured graph identity from the callback (creation order plus graph handle), joined with the collector-independent capture-time op map; fingerprint matching only as the kineto fallback.
- Eager kernels: push an external correlation id per op scope (`cuptiActivityPushExternalCorrelationId`) instead of CPU op tracing; step boundaries and real batch metadata as a per-step CPU record.
- Open: confirm HES on B200/B300 and newer drivers, MPS/virtualized environments (HES unsupported there), and that the idle 0.15 us/node is acceptable for always-traceable serving; otherwise attach only in processes started for profiling.

The capture-time op map is collector-independent: a native collector consumes the same `op_map-*.json`.
