# Overlap Catalog

This catalog is the overlap-only companion to
`references/fuse-overlap-catalog.md`.

This revision is intentionally kernel-scoped. Keep rows here only when the
overlap is visible in a profiler as GPU kernels, collective kernels, or
streamed kernel families. Host-only scheduler, event-loop, executor, offload,
and load-path overlaps are intentionally excluded.

Use it like this:

1. Start from the `overlap-opportunity table`.
2. Match visible kernel windows, collective windows, or stream-level overlap
   against the rows below.
3. If a match exists in the mainline sections, report it as an existing
   overlap family that is missing, disabled, regressed, or unsupported on the
   current backend.
4. If a match exists only in the `PR-backed / in-flight` section, report it as
   an upstream overlap pattern, not a novel idea.
5. Only call an overlap opportunity "new" when no row in this file or
   `fuse-overlap-catalog.md` fits.

The `vLLM-origin` sections below are comparative references. They are not
necessarily present in the checked-out `sglang` tree, but they should still be
treated as upstream or analogous kernel-overlap families before labeling an
overlap opportunity as novel.

## 1. LLM / SRT kernel-overlap families

| Pattern | Trace keywords | Primary code | Existing path | Skill should conclude |
| --- | --- | --- | --- | --- |
| Single-batch overlap (SBO) | MoE combine, down-gemm, shared-expert work in nearby two-stream windows | `python/sglang/srt/batch_overlap/single_batch_overlap.py` | combine vs down-gemm overlap, combine vs shared-expert overlap, one-stream dispatch+shared overlap, explicit SM partitioning and events | If exposed MoE combine sits near neighboring compute, classify it against SBO before calling it new overlap. |
| Q and K normalization on different streams | Q-side norm and K-side norm on different streams | `python/sglang/srt/models/utils.py::apply_qk_norm`<br>`python/sglang/srt/models/qwen3.py`<br>`python/sglang/srt/models/qwen3_next.py`<br>`python/sglang/srt/models/qwen3_5.py` | Q stays on current stream, K can run on `alt_stream` in capture mode | Treat split Q / K norm as an existing overlap family when `alt_stream` is already wired. |
| DeepSeek shared-expert / routed-expert overlap | shared-expert GEMMs near DeepEP dispatch / combine | `python/sglang/srt/models/deepseek_v2.py`<br>`python/sglang/srt/batch_overlap/single_batch_overlap.py` | shared experts on `alt_stream`, overlap with dispatch / combine and down-gemm, Blackwell-specific env gating | This is an established routed-vs-shared branch overlap pattern, not a novel idea. |
| Llama4 shared branch vs routed branch overlap | shared expert branch plus routed MoE branch as adjacent windows | `python/sglang/srt/models/llama4.py` | shared expert on current stream, router + topk + routed experts on `alt_stream` | Use Llama4 as the first precedent for branch-level overlap in similar sparse models. |
| ExaoneMoE shared experts vs router experts overlap | shared expert output and router-expert output form a two-branch window | `python/sglang/srt/models/exaone_moe.py::forward_normal_dual_stream` | shared experts on current stream, router + routed experts on `alt_stream`, explicit join before combine | This is an existing dual-stream MoE overlap family. |
| Grok residual-MoE branch overlap | dense MLP and block-sparse MoE branches in parallel | `python/sglang/srt/models/grok.py::moe_with_rmoe` | dense MLP on current stream, MoE on `alt_stream`, fused dual residual RMSNorm around boundaries | Treat exposed Grok branch overlap as an existing pattern. |
| NSA dual-stream overlap | Q-proj, K-proj, RoPE, cache-store, quantization in tight two-stream windows | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` | Q / K projection split, RoPE split, cache-store vs quantization overlap | NSA already contains several dual-stream overlap precedents. |
| MoriEP async dispatch / combine comm stream | `MoriEP`<br>`_comm_stream`<br>`dispatch`<br>`combine`<br>`done_event` | `python/sglang/srt/layers/moe/token_dispatcher/moriep.py` | MoriEP can submit dispatch and combine onto a dedicated communication stream and synchronize only through events | Treat MoriEP comm / compute interleave as an existing MoE overlap family. |
| Generic `alt_stream` overlap families | `alt_stream` plus explicit `wait_stream` / `with torch.cuda.stream(...)` | `qwen2_moe.py`<br>`qwen3_moe.py`<br>`glm4_moe.py`<br>`bailing_moe.py`<br>`llada2.py`<br>`grok.py`<br>`olmo2.py`<br>`step3p5.py`<br>`longcat_flash.py`<br>`falcon_h1.py` | model-specific overlap on attention prep, MoE branches, or cache-store | Search these families before designing a new overlap scheme from scratch. |

## 2. Staging / communication kernel-overlap families

| Pattern | Trace keywords | Primary code | Existing path | Skill should conclude |
| --- | --- | --- | --- | --- |
| Decode scatter on dedicated `scatter_stream` | `scatter_stream`<br>`_scatter_stream` | `python/sglang/srt/disaggregation/common/staging_handler.py` | staging scatter kernels are submitted to a dedicated stream so the decode thread does not block on the main forward stream | Treat decode-side staging scatter windows as an existing overlap pattern. |
| Staging-buffer fused gather / scatter kernels | `_fused_gather_to_staging_kernel`<br>`_fused_scatter_from_staging_kernel` | `python/sglang/srt/disaggregation/common/staging_buffer.py` | Triton kernels gather KV slices into contiguous staging memory and scatter them back to KV cache | If heterogeneous-TP staging shows many small copy kernels, compare against this existing fused-plus-overlap family first. |

## 3. VLM / diffusion kernel-overlap families

| Pattern | Trace keywords | Primary code | Existing path | Skill should conclude |
| --- | --- | --- | --- | --- |
| Vision QK norm with aux stream | vision-side QK norm or norm-like kernels before attention | `python/sglang/srt/layers/attention/vision.py` | vision QK normalization can call shared `apply_qk_norm(...)`, with K-side work on `aux_stream` | If vision QK prep is split, first check this existing aux-stream path. |
| ViT CUDA graph disables vision aux stream | expected vision overlap is absent under ViT graph | `python/sglang/srt/models/internvl.py`<br>`python/sglang/srt/layers/attention/vision.py`<br>`python/sglang/srt/environ.py::SGLANG_VIT_ENABLE_CUDA_GRAPH` | vision `aux_stream` is intentionally disabled when ViT CUDA graph is on | Missing vision overlap may be intentional, not a regression. |
| Ulysses sequence-parallel attention | exposed `all_to_all` around attention blocks | `python/sglang/multimodal_gen/runtime/layers/attention/layer.py`<br>`python/sglang/multimodal_gen/runtime/distributed/communication_op.py` | head / sequence redistribution before and after attention | Treat sequence-parallel all-to-all as an existing distributed attention family. |
| USP attention with all-to-all and ring attention | `all_to_all`, ring-attention comm, head / sequence reshards | `python/sglang/multimodal_gen/runtime/layers/attention/layer.py` | `_usp_input_all_to_all(...)`, `_usp_output_all_to_all(...)`, `ring_attn(...)` | This is the primary existing overlap / comm family for many diffusion models. |
| Turbo-layer async all-to-all pipelining | pipelined A2A windows with explicit waits on a comm stream | `python/sglang/multimodal_gen/runtime/layers/attention/turbo_layer.py` | looped `all_to_all_single(..., async_op=True)` plus staged postprocess on a comm stream | Treat exposed turbo A2A windows as an existing pipelined overlap pattern. |
| TorchInductor compute / communication reorder | compiled traces with compute and comm partially interleaved | `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py`<br>`python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/mova.py` | `torch._inductor.config.reorder_for_compute_comm_overlap = True` | Existing compile-time reordering may already explain partial overlap in diffusion traces. |
| Dual-stream diffusion models | two nearby compute branches inside one DiT / UNet block | `python/sglang/multimodal_gen/runtime/models/dits/hunyuan3d.py` | `use_dual_stream = True` | Treat dual-branch diffusion execution as an existing overlap family. |

## 4. PR-backed / in-flight kernel-overlap families

| Pattern | Trace keywords | Primary code | Existing path | Skill should conclude |
| --- | --- | --- | --- | --- |
| PR `#21877` fused down-GEMM + combine superseding SBO | `enable_fused_grouped_gemm_combine`<br>`combine`<br>`down_gemm` | `PR #21877`<br>`python/sglang/srt/server_args.py`<br>`python/sglang/srt/layers/moe/token_dispatcher/deepep.py` | Fused combine eliminates the standalone combine window, so SBO is intentionally disabled when this path is on | If the trace discussion is about combine overlap, first classify it as this upstream fused-overlap family. |
| PR `#22410` hiSparse H2D transfer overlap with hit-attention | `transfer_stream`<br>`execute_h2d_async`<br>`hit-attention`<br>`merge_state` | `PR #22410`<br>`python/sglang/srt/layers/attention/nsa_backend.py`<br>`python/sglang/srt/hisparse/hisparse_coordinator.py` | hiSparse decode overlaps host-to-device KV transfer on a transfer stream with hit-attention on the compute stream before running miss-attention and merge | Treat hit-attention vs H2D KV transfer windows as an in-flight SGLang overlap family first. |

## 5. FlashInfer kernel-overlap families

These rows are comparative references from `flashinfer`. Use them when a trace
looks like an upstream FlashInfer overlap family even if the current `sglang`
checkout only calls part of that implementation.

| Pattern | Trace keywords | Primary code | Existing path | Skill should conclude |
| --- | --- | --- | --- | --- |
| FlashInfer PDL launch-overlap family | `enable_pdl`<br>`launch_with_pdl`<br>`cudaGridDependencySynchronize`<br>`cudaTriggerProgrammaticLaunchCompletion`<br>`trigger_completion_at_end=False`<br>`allreduce_fusion` | `flashinfer/norm/__init__.py`<br>`flashinfer/activation.py`<br>`flashinfer/rope.py`<br>`flashinfer/comm/allreduce.py`<br>`flashinfer/comm/trtllm_ar.py` | FlashInfer uses Programmatic Dependent Launch broadly, and the allreduce path can further advance completion so the next PDL-aware kernel overlaps on the same stream | Treat tight same-stream dependent windows and allreduce-followed-by-kernel windows as one existing FlashInfer launch-overlap family first. |
| FlashInfer CuTeDSL MoE aux-stream async-memset overlap | `aux_stream`<br>`main_event`<br>`memset_event`<br>`use_async_memset` | `flashinfer/fused_moe/cute_dsl/fused_moe.py` | Preallocated MoE output is zeroed on an auxiliary CUDA stream while GEMM1 runs on the main stream, then both streams join before finalize | Treat GEMM1 vs output-zero windows as an existing FlashInfer multi-stream overlap family. |
| FlashInfer green-context SM partition overlap | `split_device_green_ctx`<br>`split_device_green_ctx_by_sm_count`<br>`green_ctx` | `flashinfer/green_ctx.py` | CUDA green contexts partition SMs and create dedicated streams for concurrent kernel families on separate SM slices | Treat SM-partitioned concurrency as an existing FlashInfer overlap mechanism, not a novel scheduler idea. |

## 6. FlashInfer PR-backed / in-flight kernel-overlap families

| Pattern | Trace keywords | Primary code | Existing path | Skill should conclude |
| --- | --- | --- | --- | --- |
| PR `#2840` CuTeDSL MoE aux-stream overlap race fix | `aux_stream`<br>`use_prealloc`<br>`use_cuda_graph` | `PR #2840`<br>`flashinfer/fused_moe/cute_dsl/fused_moe.py` | Clarifies that async memset overlap is only safe for the preallocated / CUDA-graph case; non-graph mode falls back to main-stream zeroing to avoid races | Treat missing aux-stream overlap in non-graph traces as an intentional safety rule, not a novel opportunity. |
| PR `#2720` PDL runtime-API migration | `cudaGridDependencySynchronize`<br>`cudaTriggerProgrammaticLaunchCompletion`<br>`inline PTX` | `PR #2720`<br>`include/flashinfer/comm/trtllm_allreduce_fusion.cuh`<br>`include/flashinfer/pos_enc.cuh` | Repo-wide migration preserves the existing PDL overlap family while replacing inline PTX with CUDA runtime APIs across norm, RoPE, attention, and MoE codepaths | Treat PDL-looking launch groups as an upstream FlashInfer overlap family even when implementation details differ across revisions. |

## 7. TensorRT-LLM-origin kernel-overlap families

These rows are comparative references from `TensorRT-LLM`. Current mainline
TensorRT-LLM overlap rows are mostly explicit auxiliary-stream rewrites in
AutoDeploy rather than same-stream PDL windows.

| Pattern | Trace keywords | Primary code | Existing path | Skill should conclude |
| --- | --- | --- | --- | --- |
| TensorRT-LLM multi-stream MLA attention | `multi_stream_mla_attn`<br>`record_event_passthrough`<br>`_aux`<br>`wait_event` | `tensorrt_llm/_torch/auto_deploy/transform/library/multi_stream_attn.py`<br>`tensorrt_llm/_torch/auto_deploy/utils/multi_stream_utils.py` | AutoDeploy rewrites MLA Q/KV forks so the KV projection runs on an auxiliary stream while the Q path stays on the caller stream | Treat exposed Q-branch vs KV-branch overlap as an existing TensorRT-LLM multi-stream family first. |
| TensorRT-LLM multi-stream MoE shared-vs-routed overlap | `multi_stream_moe`<br>`begin_aux_stream_passthrough`<br>`end_aux_stream_passthrough`<br>`wait_aux_stream_passthrough` | `tensorrt_llm/_torch/auto_deploy/transform/library/multi_stream_moe.py`<br>`tensorrt_llm/_torch/auto_deploy/utils/multi_stream_utils.py` | Shared-expert work is moved to an auxiliary stream while routed-expert MoE work remains on the main stream and rejoins at the merge node | Treat shared-expert vs routed-expert windows as an existing TensorRT-LLM branch-overlap family. |
| TensorRT-LLM multi-stream FP8 GEMM fork parallelism | `multi_stream_gemm`<br>`trtllm_finegrained_fp8_linear`<br>`record_event_passthrough`<br>`_aux` | `tensorrt_llm/_torch/auto_deploy/transform/library/multi_stream_gemm.py`<br>`tensorrt_llm/_torch/auto_deploy/utils/multi_stream_utils.py` | Compiler pass identifies fork points with multiple FP8 linears and moves the largest GEMM to the auxiliary stream so sibling GEMMs overlap | Treat sibling FP8 linear branches as an existing TensorRT-LLM overlap family before designing a new stream split. |

## 8. TensorRT-LLM-origin PR-backed / in-flight kernel-overlap families

| Pattern | Trace keywords | Primary code | Existing path | Skill should conclude |
| --- | --- | --- | --- | --- |
| PR `#12847` `multi_stream_moe` sync fix for MLIR and piecewise cudagraphs | `multi_stream_moe`<br>`mlir_elementwise_fusion`<br>`piecewise cudagraph`<br>`caller_stream.synchronize()` | `PR #12847`<br>`tensorrt_llm/_torch/auto_deploy/transform/library/multi_stream_moe.py`<br>`tensorrt_llm/_torch/auto_deploy/utils/multi_stream_utils.py` | Open PR preserves the existing multi-stream MoE overlap family while tightening synchronization when MLIR-fused kernels or piecewise cudagraph replay are present | Treat missing or altered `multi_stream_moe` overlap under MLIR / piecewise graph modes as an in-flight TensorRT-LLM rule first. |

## 9. vLLM-origin kernel-overlap families

| Pattern | Trace keywords | Primary code | Existing path | Skill should conclude |
| --- | --- | --- | --- | --- |
| vLLM-origin AsyncTP GEMM + collective overlap | `fuse_gemm_comms`<br>`fused_matmul_reduce_scatter`<br>`fused_all_gather_matmul` | `vllm/compilation/passes/fusion/collective_fusion.py`<br>`docs/design/fusions.md` | AsyncTP overlaps GEMM with reduce-scatter / all-gather via symmetric-memory collectives | Treat GEMM+comm windows as a clear vLLM-origin overlap precedent first. |
| vLLM-origin Sequence Parallelism staging | `enable_sp`<br>`ReduceScatter`<br>`AllGather`<br>`SequenceParallelismPass` | `vllm/compilation/passes/fusion/sequence_parallelism.py`<br>`docs/design/fusions.md` | Sequence-parallel rewrites all-reduce into RS -> local norm -> AG so later passes can overlap comm and compute | Treat RS / AG staging around norm blocks as an upstream overlap-enabling family. |
| vLLM-origin shared-expert aux-stream overlap | `aux_stream`<br>`shared_experts_stream`<br>shared expert near router | `vllm/utils/torch_utils.py`<br>`vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py` | MoE shared experts can run on a dedicated aux stream and overlap with router-side work | Treat shared-expert vs router overlap as an existing upstream sparse-model family. |
| vLLM-origin DCP async all-to-all overlap | `dcp_alltoall`<br>`all_to_all_single`<br>`async_op=True` | `vllm/v1/attention/ops/dcp_alltoall.py` | Output / LSE exchange uses async all-to-all handles instead of serializing collective completion on the main path | Treat DCP all-to-all windows as an upstream async-collective family. |

## 10. vLLM-origin PR-backed / in-flight kernel-overlap families

| Pattern | Trace keywords | Primary code | Existing path | Skill should conclude |
| --- | --- | --- | --- | --- |
| PR `#35968` DSV3.2 multi-stream indexer overlap | `weights_proj`<br>`wk`<br>`k_norm`<br>`aux_stream` | `PR #35968`<br>`vllm/model_executor/models/deepseek_v2.py`<br>`vllm/utils/torch_utils.py` | Open PR overlaps the small `weights_proj` GEMM with `wk + k_norm` on a secondary CUDA stream for decode batches instead of serializing both on the default stream | Treat this as a concrete upstream decode-time kernel-overlap family when traces show underutilized projection overlap opportunities. |
| PR `#39301` GLM5 router GEMM with PDL overlap | `TRTLLM_ENABLE_PDL`<br>`router_gemm`<br>`GLM5`<br>`FI AR RMS fusion` | `PR #39301`<br>`vllm/model_executor/layers/fused_moe/router/gate_linear.py`<br>`vllm/csrc/moe/dsv3_router_gemm_utils.h` | The GLM5 router GEMM path explicitly uses PDL so the router kernel can overlap with the preceding fused allreduce-plus-RMS block on supported GPUs | Treat router-GEMM launch overlap on GLM5-like traces as an in-flight upstream family first. |

## 11. Important toggles and caveats

| Toggle / env | Location | Effect on trace interpretation |
| --- | --- | --- |
| `enable_single_batch_overlap` | `python/sglang/srt/server_args.py` | Enables the SBO family. |
| `SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO` | `python/sglang/srt/environ.py` | Alters how DeepSeek-style shared-expert overlap behaves on Blackwell. |
| `SGLANG_DISAGG_STAGING_BUFFER` | `python/sglang/srt/environ.py` | Enables the heterogeneous-TP staging-buffer family and its overlap windows. |
| `SGLANG_STAGING_USE_TORCH` | `python/sglang/srt/disaggregation/common/staging_buffer.py` | Forces torch fallback for staging gather / scatter, so Triton staging kernels may disappear by design. |
| `SGLANG_VIT_ENABLE_CUDA_GRAPH` | `python/sglang/srt/environ.py` | Can intentionally disable vision `aux_stream` overlap. |
| `enable_pdl` / `launch_with_pdl` | `flashinfer/norm/__init__.py`<br>`flashinfer/activation.py`<br>`flashinfer/rope.py`<br>`flashinfer/fused_moe/core.py`<br>`flashinfer/comm/allreduce.py` | Enables FlashInfer PDL across many kernels; launch grouping and same-stream overlap can change substantially when it is on. |
| `trigger_completion_at_end` | `flashinfer/comm/allreduce.py` | `False` enables downstream PDL-aware overlap after FlashInfer allreduce fusion; `True` delays completion to kernel end and removes that overlap window. |
| `use_cuda_graph` | `flashinfer/fused_moe/cute_dsl/fused_moe.py` | Enables the preallocated-buffer path and the safe aux-stream async-memset overlap in FlashInfer CuTeDSL MoE. |
| `split_device_green_ctx*` | `flashinfer/green_ctx.py` | Changes trace shape by partitioning SMs into separate green contexts instead of overlapping full-device streams on the default context. |
| `multi_stream_moe` | `tensorrt_llm/_torch/auto_deploy/config/default.yaml` | Enables the TensorRT-LLM shared-expert vs routed-expert overlap family. |
| `multi_stream_mla_attn` | `tensorrt_llm/_torch/auto_deploy/config/default.yaml` | Enables the TensorRT-LLM MLA Q-vs-KV branch overlap family. |
| `multi_stream_gemm` | `tensorrt_llm/_torch/auto_deploy/config/default.yaml` | Enables generalized FP8 GEMM fork overlap in TensorRT-LLM AutoDeploy. |
| `mlir_elementwise_fusion` | `tensorrt_llm/_torch/auto_deploy/config/default.yaml` | Can absorb merge adds into larger fused kernels, so missing explicit merge nodes in TensorRT-LLM multi-stream traces may be intentional. |
| `enable_torch_compile` | `python/sglang/srt/server_args.py`<br>`python/sglang/multimodal_gen/runtime/server_args.py` | Compiler-generated reordering can hide or rename overlap windows. |
| `enable_fused_grouped_gemm_combine` | `PR #21877` | In-flight path that intentionally disables SBO because combine is folded into down-GEMM. |
| `PassConfig.enable_sp` | `vllm/config/compilation.py` | Enables vLLM's sequence-parallel staging family that creates RS / AG overlap opportunities. |
| `PassConfig.fuse_gemm_comms` | `vllm/config/compilation.py` | Enables AsyncTP GEMM + collective overlap and auto-enables `enable_sp` when valid. |

## 12. Suggested refresh commands

These commands are only for maintainers refreshing this catalog by rescanning
the local source trees. They are not used by the triage scripts at runtime.

```bash
# Optional sibling checkouts used for comparative scanning:
FLASHINFER_REPO=${FLASHINFER_REPO:-../flashinfer}
TRTLLM_REPO=${TRTLLM_REPO:-../TensorRT-LLM}
VLLM_REPO=${VLLM_REPO:-../vllm}

rg -n "single_batch_overlap|alt_stream|shared_expert|scatter_stream|_fused_gather_to_staging_kernel|_fused_scatter_from_staging_kernel|async_op=True" python/sglang
rg -n "apply_qk_norm|vision.py|ring_attn|all_to_all_single|reorder_for_compute_comm_overlap|use_dual_stream" python/sglang/multimodal_gen python/sglang/srt
git log --all --format='%h %s' | rg -i 'fused|fusion|overlap|combine|all_to_all|ring attn|stream|triton|cutedsl|cuda'
rg -n "enable_pdl|launch_with_pdl|trigger_completion_at_end|aux_stream|use_async_memset|split_device_green_ctx|split_device_green_ctx_by_sm_count" "$FLASHINFER_REPO/flashinfer" "$FLASHINFER_REPO/include"
git -C "$FLASHINFER_REPO" log --all --format='%h %s' | rg -i 'fused|fusion|overlap|pdl|stream|rope|kv|quant|topk|moe'
rg -n "multi_stream_moe|multi_stream_mla_attn|multi_stream_gemm|record_event_passthrough|begin_aux_stream_passthrough|end_aux_stream_passthrough|wait_aux_stream_passthrough" "$TRTLLM_REPO/tensorrt_llm/_torch"
rg -n "mlir_elementwise_fusion|piecewise|cudagraph|caller_stream.synchronize" "$TRTLLM_REPO/tensorrt_llm/_torch"
git -C "$TRTLLM_REPO" log --all --format='%h %s' | rg -i 'overlap|multi-stream|aux stream|cudagraph|mlir|stream|flashinfer|moe|mla'
rg -n "fuse_gemm_comms|enable_sp|fused_matmul_reduce_scatter|fused_all_gather_matmul|shared_experts_stream|dcp_alltoall|async_op=True|aux_stream|maybe_execute_in_parallel" "$VLLM_REPO/vllm" "$VLLM_REPO/docs/design/fusions.md"
git -C "$VLLM_REPO" log --all --format='%h %s' | rg -i 'fused|fusion|overlap|allreduce|reduce-scatter|all-gather|all_to_all|stream|multi-stream|triton|cuda|router'
# GitHub PR scan terms for the connector or web UI:
#   "fused OR overlap repo:sgl-project/sglang"
#   "triton OR cutedsl OR cuda overlap repo:sgl-project/sglang"
#   "fused OR overlap repo:flashinfer-ai/flashinfer"
#   "pdl OR aux_stream OR green_ctx repo:flashinfer-ai/flashinfer"
#   "fused OR overlap repo:NVIDIA/TensorRT-LLM"
#   "multi-stream OR aux_stream OR cudagraph repo:NVIDIA/TensorRT-LLM"
#   "mlir OR piecewise OR flashinfer repo:NVIDIA/TensorRT-LLM"
#   "fused OR overlap repo:vllm-project/vllm"
#   "triton OR cuda overlap repo:vllm-project/vllm"
#   "multi-stream OR aux_stream overlap repo:vllm-project/vllm"
```
