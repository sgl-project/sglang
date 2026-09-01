from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Optional

import msgspec

from sglang.srt.configs.model_config import ModelImpl
from sglang.srt.distributed import get_world_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    prealloc_symmetric_memory_pool,
)
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner
from sglang.srt.hardware_backend.xpu.graph_runner.xpu_graph_runner import XPUGraphRunner
from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    get_server_return_hidden_states_mode,
)
from sglang.srt.model_executor.graph_memory_usage import (
    merge_graph_memory_usage,
    merge_graph_time_usage,
)
from sglang.srt.model_executor.graph_shared_output import GraphSharedOutput
from sglang.srt.model_executor.hook_manager import register_forward_hooks
from sglang.srt.model_executor.model_runner_components.layer_setup import (
    compute_attention_and_moe_layers,
)
from sglang.srt.model_executor.runner import (
    EagerRunner,
    PrefillCudaGraphRunner,
    get_batch_sizes_to_capture,
)
from sglang.srt.model_loader.utils import resolve_language_model
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_flags,
    get_parallel,
    get_schedule,
    get_spec,
)
from sglang.srt.utils import get_available_gpu_memory, log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner.base_runner import BaseRunner

logger = logging.getLogger(__name__)


def _align_pipeline_layers(layers: list, layer_model) -> list:
    has_start_layer = hasattr(layer_model, "start_layer")
    has_end_layer = hasattr(layer_model, "end_layer")
    assert (
        has_start_layer == has_end_layer
    ), "pipeline layer ranges must define start_layer and end_layer together"
    start_layer = layer_model.start_layer if has_start_layer else 0
    end_layer = layer_model.end_layer if has_end_layer else len(layer_model.layers)
    assert isinstance(start_layer, int) and isinstance(
        end_layer, int
    ), "pipeline layer ranges must define integer start_layer and end_layer"
    assert 0 <= start_layer <= end_layer <= len(layer_model.layers), (
        f"invalid pipeline layer range [{start_layer}, {end_layer}) for "
        f"{len(layer_model.layers)} layers"
    )
    if len(layers) == len(layer_model.layers):
        return layers
    assert (
        len(layers) <= end_layer - start_layer
    ), f"found {len(layers)} layers in PP range [{start_layer}, {end_layer})"
    return (
        [None] * start_layer + layers + [None] * (len(layer_model.layers) - end_layer)
    )


def has_standard_gqa_for_all_local_layers(
    *, attention_layer_count: int, start_layer: int, end_layer: int
) -> bool:
    """Check the layers materialized on this pipeline rank, not the full model."""
    return attention_layer_count >= end_layer - start_layer


def index_attention_layers_by_global_id(
    attention_layers: list[Any],
    mha_companion_layers: list[Any],
    layer_model=None,
) -> tuple[list[Any], list[Any]]:
    """Pad PP-local attention metadata so global layer_id remains a valid index."""
    if len(attention_layers) != len(mha_companion_layers):
        raise ValueError("attention and MHA companion metadata must be parallel")
    populated = [layer for layer in attention_layers if layer is not None]
    if not populated or any(not hasattr(layer, "layer_id") for layer in populated):
        if layer_model is not None:
            return (
                _align_pipeline_layers(attention_layers, layer_model),
                _align_pipeline_layers(mha_companion_layers, layer_model),
            )
        return attention_layers, mha_companion_layers
    max_layer_id = max(int(layer.layer_id) for layer in populated)
    indexed_attention = [None] * (max_layer_id + 1)
    indexed_companions = [None] * (max_layer_id + 1)
    for attention, companion in zip(attention_layers, mha_companion_layers):
        if attention is None:
            if companion is not None:
                raise ValueError("MHA companion has no primary attention layer")
            continue
        layer_id = int(attention.layer_id)
        if layer_id < 0 or indexed_attention[layer_id] is not None:
            raise ValueError(f"invalid or duplicate attention layer_id: {layer_id}")
        indexed_attention[layer_id] = attention
        indexed_companions[layer_id] = companion
    return indexed_attention, indexed_companions


class GraphCapture(msgspec.Struct, frozen=True, kw_only=True):
    runner: Optional[BaseRunner]
    memory_phase: str
    memory_usage_gb: float
    capture_time: float

    @property
    def memory_usage(self) -> dict[str, float]:
        return {self.memory_phase: self.memory_usage_gb}

    @property
    def time_usage(self) -> dict[str, float]:
        return {self.memory_phase: self.capture_time}


class CudaGraphsCapture(msgspec.Struct, frozen=True, kw_only=True):
    eager_runner: EagerRunner
    prefill: GraphCapture
    decode: GraphCapture

    @property
    def memory_usage(self) -> dict[str, float]:
        return merge_graph_memory_usage(
            self.prefill.memory_usage,
            self.decode.memory_usage,
        )

    @property
    def time_usage(self) -> dict[str, float]:
        return merge_graph_time_usage(
            self.prefill.time_usage,
            self.decode.time_usage,
        )


def capture_cuda_graphs(
    *, model_runner: ModelRunner, capture_decode_cuda_graph: bool = True
) -> CudaGraphsCapture:
    """Capture cuda graphs. Requires init_attention_backends() to have run.

    Spec draft runners pass capture_decode_cuda_graph=False
    because they capture their own decode-style graphs separately.

    """

    model_runner.graph_shared_output = GraphSharedOutput.create_for_model_runner(
        model_runner
    )

    # The eager (no-cuda-graph) phase runner, built AFTER the attention
    # backend so its __init__ can warm up kernels (run-once) and allocate the
    # fixed-max static buffer — both before the cuda-graph runners, so that
    # buffer is canonical in the shared pool and the cg runners coalesce onto
    # it. Always built: it serves both the fully-disabled case (decode/prefill
    # runners point at it) and the eager fallback when a cg runner can't run a
    # batch.
    eager_runner = EagerRunner(model_runner)

    if model_runner.is_draft_worker:
        moe_runner_backend = (
            get_spec().speculative_moe_runner_backend
            or get_exec().moe.moe_runner_backend
        )
        moe_a2a_backend = (
            get_spec().speculative_moe_a2a_backend or get_exec().moe.moe_a2a_backend
        )
    else:
        moe_runner_backend = get_exec().moe.moe_runner_backend
        moe_a2a_backend = get_exec().moe.moe_a2a_backend

    uses_deep_gemm_moe_runner = moe_runner_backend == "deep_gemm"
    if moe_runner_backend == "auto" and model_runner.model_config.quantization in (
        "fp8",
        "mxfp8",
    ):
        from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
        from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

        uses_deep_gemm_moe_runner = Fp8MoEMethod.is_deepgemm_moe_runner_backend_enabled(
            MoeRunnerBackend(moe_runner_backend),
            MoeA2ABackend(moe_a2a_backend),
        )

    if (
        model_runner.device == "cuda"
        and envs.SGLANG_DEEPGEMM_STANDARD_LAYOUT.get().lower() == "auto"
        and uses_deep_gemm_moe_runner
    ):
        from sglang.srt.layers.moe.moe_runner.deep_gemm import (
            set_masked_standard_layout_memory_budget,
        )

        world_group = get_world_group()
        available_memory_gb = get_available_gpu_memory(
            model_runner.device,
            model_runner.gpu_id,
            distributed=world_group.world_size > 1,
            cpu_group=world_group.cpu_group,
        )
        budget_bytes = set_masked_standard_layout_memory_budget(
            int(available_memory_gb * (1 << 30))
        )
        logger.info(
            "DeepGEMM masked layout budget: %.2f GiB from %.2f GiB free.",
            budget_bytes / (1 << 30),
            available_memory_gb,
        )

    # cuda-graph capture: prefill before decode, so both coalesce onto the
    # eager buffer allocated above. (capture_prefill_graph routes prefill
    # to the eager runner when the prefill graph is disabled.)
    prefill = capture_prefill_graph(
        model_runner=model_runner, eager_runner=eager_runner
    )

    decode_phase = "draft_decode" if model_runner.is_draft_worker else "decode"
    decode = GraphCapture(
        runner=None,
        memory_phase=decode_phase,
        memory_usage_gb=0,
        capture_time=0,
    )
    if capture_decode_cuda_graph:
        if model_runner.device in ("cuda", "musa", "cpu", "npu", "xpu"):
            decode = capture_decode_graph(model_runner=model_runner)
        elif (
            current_platform.is_out_of_tree() and current_platform.support_cuda_graph()
        ):
            decode = capture_decode_graph(model_runner=model_runner)
    else:
        decode = GraphCapture(
            runner=eager_runner,
            memory_phase=decode_phase,
            memory_usage_gb=0,
            capture_time=0,
        )

    # Register forward hooks AFTER cuda-graph capture so their tensor ops are
    # not traced into any captured graph — capture stays hook-free and hooks
    # fire only on the eager forward path (capture replay never runs Python
    # hooks anyway).
    if model_runner.server_args.forward_hooks:
        register_forward_hooks(
            model_runner.model, model_runner.server_args.forward_hooks
        )

    prealloc_symmetric_memory_pool(
        is_draft_worker=model_runner.is_draft_worker,
        enable_symm_mem=get_exec().comm.enable_symm_mem,
        device=model_runner.device,
        forward_stream=model_runner.forward_stream,
    )

    if model_runner.canary_manager is not None and not model_runner.is_draft_worker:
        model_runner.canary_manager.mark_init_finished()

    return CudaGraphsCapture(eager_runner=eager_runner, prefill=prefill, decode=decode)


def capture_prefill_graph(
    *,
    model_runner: ModelRunner,
    eager_runner: EagerRunner,
    force_for_draft_worker: bool = False,
) -> GraphCapture:
    """Initialize a prefill graph and return its startup resource usage."""

    memory_phase = "draft_prefill" if model_runner.is_draft_worker else "prefill"

    def result(
        runner: Optional[BaseRunner],
        memory_usage_gb: float = 0,
        capture_time: float = 0,
    ) -> GraphCapture:
        return GraphCapture(
            runner=runner,
            memory_phase=memory_phase,
            memory_usage_gb=memory_usage_gb,
            capture_time=capture_time,
        )

    if check_cuda_graph_backend(Phase.PREFILL, Backend.DISABLED):
        logger.info(
            "Disable prefill CUDA graph because cuda_graph_config "
            "resolved prefill.backend='disabled' (e.g. via "
            "--cuda-graph-backend-prefill=disabled or auto-disable rules)."
        )
        # Prefill cuda graph disabled: route eager prefill through the
        # EagerRunner (its can_run_graph returns False, so _forward_raw's
        # extend branch falls through to the eager path).
        if not model_runner.is_draft_worker:
            return result(eager_runner)
        return result(None)

    # Draft models skip here during __init__; the eagle worker calls
    # this method explicitly (force_for_draft_worker=True) after
    # init_lm_head so graphs capture the final embedding weights.
    if model_runner.is_draft_worker and not force_for_draft_worker:
        return result(None)

    # Skip prefill CG for EAGLE target on tc_piecewise when the fixed server
    # capture ceiling is below FULL. EAGLE target prefill requests FULL, so a
    # NULL or LAST graph is dead; capturing it can perturb FP4/TRTLLM-MoE
    # state and corrupt decode replay (see #28386 and #28870). BCG and FullCG
    # capture FULL for EAGLE targets in PrefillCudaGraphRunner.__init__, so
    # they do not need this skip.
    if (
        model_runner.spec_algorithm.is_eagle()
        and not model_runner.is_draft_worker
        and get_server_return_hidden_states_mode() < CaptureHiddenMode.FULL
        and check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)
    ):
        logger.info(
            "Disable prefill CUDA graph for EAGLE target on tc_piecewise "
            "to avoid FP4/MoE decode-replay corruption (#28386)."
        )
        return result(eager_runner)

    if (
        model_runner.lora_manager is not None
        and not model_runner.lora_manager.supports_prefill_cuda_graph
    ):
        logger.warning(
            "Disable prefill CUDA graph because the current LoRA "
            "configuration does not support it (unsupported LoRA backend, "
            "MoE LoRA, or DP attention)."
        )
        return result(eager_runner)

    # Resolve the decoder once. Some VLM wrappers (for example Kimi-VL)
    # expose it as ``language_model`` rather than ``model``.
    try:
        language_model = resolve_language_model(model_runner.model)
    except AttributeError:
        logger.warning(
            "Disable prefill CUDA graph because the model is not a language model"
        )
        return result(None)

    # Disable prefill CUDA graph for non capture size
    if not get_exec().graph.cuda_graph_config.prefill.bs:
        logger.warning("Disable prefill CUDA graph because the capture size is not set")
        return result(None)

    prefill_config = get_exec().graph.cuda_graph_config.prefill
    prefill_backend = prefill_config.backend
    parallel = get_parallel()
    if (
        prefill_backend == Backend.BREAKABLE
        and parallel.enable_prefill_cp
        and parallel.pp_size > 1
    ):
        logger.warning(
            "Disable prefill CUDA graph because pipeline parallelism combined "
            "with prefill context parallelism is not validated."
        )
        return result(eager_runner)
    context_length = model_runner.model_config.context_len
    if prefill_backend == Backend.FULL:
        max_capture_requests = prefill_config.full_prefill_max_req
        if max_capture_requests is None:
            max_capture_requests = max(get_schedule().chunked_prefill_size // 512, 1)
        max_capture_requests = min(
            max_capture_requests, model_runner.req_to_token_pool.size
        )
        # Resolve Full's fixed request-axis shape once, just like bs below.
        prefill_config.full_prefill_max_req = max_capture_requests
    else:
        max_capture_requests = model_runner.req_to_token_pool.size
    # The capture dummy batch has at most max_capture_requests rows, and
    # each row can contain at most context_length tokens. Their product is
    # therefore the largest aggregate-token bucket capture can represent.
    max_capture_tokens = max_capture_requests * context_length
    capture_num_tokens = sorted(
        num_tokens
        for num_tokens in prefill_config.bs
        if num_tokens <= max_capture_tokens
    )
    # Resolve the context- and request-capacity-bounded buckets once before
    # constructing the runner so every backend consumes the same config.
    prefill_config.bs = capture_num_tokens
    if not capture_num_tokens:
        logger.warning(
            "Disable prefill CUDA graph capture because no configured "
            "capture size fits backend=%s with max_capture_tokens=%s "
            "(max_capture_requests=%s, context_length=%s, request-pool size=%s).",
            prefill_backend,
            max_capture_tokens,
            max_capture_requests,
            context_length,
            model_runner.req_to_token_pool.size,
        )
        return eager_runner

    # Collect attention layers and moe layers from the model. Keep a VLM
    # wrapper that exposes ``language_model`` unchanged: assigning it to
    # ``model`` would register a duplicate module alias and duplicate the
    # model's state-dict namespace.
    if hasattr(model_runner.model, "model"):
        model_runner.model.model = language_model

    # Find the module that owns the decoder `layers`. Models wrap it at
    # varying depths: a direct text model exposes `.layers`, a CausalLM
    # wraps it as `.model.layers`, and some multimodal models add another
    # level (e.g. DeepSeek-OCR: OCR wrapper -> Deepseek*ForCausalLM ->
    # text model -> `.layers`). Descend the `.model` chain until we find it.
    layer_model = language_model
    while not hasattr(layer_model, "layers") and hasattr(layer_model, "model"):
        layer_model = layer_model.model

    if not hasattr(layer_model, "layers"):
        logger.warning(
            "Disable prefill CUDA graph because the model does not have a 'layers' attribute"
        )
        return result(None)

    (
        model_runner.attention_layers,
        model_runner.moe_layers,
        model_runner.moe_fusions,
        model_runner.dsa_indexers,
        model_runner.mha_companion_layers,
    ) = compute_attention_and_moe_layers(layer_model)
    (
        model_runner.attention_layers,
        model_runner.mha_companion_layers,
    ) = index_attention_layers_by_global_id(
        model_runner.attention_layers,
        model_runner.mha_companion_layers,
        layer_model,
    )

    if not has_standard_gqa_for_all_local_layers(
        attention_layer_count=sum(
            layer is not None for layer in model_runner.attention_layers
        ),
        start_layer=model_runner.layer_info.start_layer,
        end_layer=model_runner.layer_info.end_layer,
    ):
        # TODO(yuwei): support Non-Standard GQA
        log_info_on_rank0(
            logger,
            "Disable prefill CUDA graph because some layers do not apply Standard GQA",
        )
        return result(None)

    tic = time.perf_counter()
    before_mem = get_available_gpu_memory(model_runner.device, model_runner.gpu_id)
    role = "draft" if model_runner.is_draft_worker else "target"
    capture_name = f"{role} prefill"
    logger.info(
        f"Capture {capture_name} CUDA graph begin. "
        f"backend={prefill_backend}, num_tokens={capture_num_tokens}, "
        f"avail mem={before_mem:.2f} GB"
    )

    prefill_runner = PrefillCudaGraphRunner(model_runner)

    after_mem = get_available_gpu_memory(model_runner.device, model_runner.gpu_id)
    mem_usage = before_mem - after_mem
    capture_time = time.perf_counter() - tic
    logger.info(
        f"Capture {capture_name} CUDA graph end. "
        f"elapsed={capture_time:.2f} s, "
        f"mem usage={mem_usage:.2f} GB, avail mem={after_mem:.2f} GB."
    )
    return result(prefill_runner, mem_usage, capture_time)


def capture_decode_graph(*, model_runner: ModelRunner) -> GraphCapture:
    """Capture device graphs."""
    if model_runner.is_draft_worker:
        memory_phase = "draft_decode"
    elif model_runner.spec_algorithm.is_speculative():
        memory_phase = "target_verify"
    else:
        memory_phase = "decode"
    no_capture = GraphCapture(
        runner=None,
        memory_phase=memory_phase,
        memory_usage_gb=0,
        capture_time=0,
    )

    # A PD prefill server never replays the target-verify graph, and its pool
    # is built without the spec-verify scratch the capture would need.
    if (
        model_runner.spec_algorithm.is_speculative()
        and not model_runner.is_draft_worker
        and get_disagg().disaggregation_mode == "prefill"
    ):
        return no_capture
    if not model_runner.is_generation:
        # TODO: Currently, cuda graph only captures decode steps, which only exists for generation models
        return no_capture
    if model_runner.server_args.model_impl.lower() == ModelImpl.MINDSPORE:
        return no_capture
    if model_runner.device != "cpu" and check_cuda_graph_backend(
        Phase.DECODE, Backend.DISABLED
    ):
        return no_capture
    if model_runner.device == "cpu" and not get_flags().capture.enable_torch_compile:
        return no_capture

    tic = time.perf_counter()
    before_mem = get_available_gpu_memory(model_runner.device, model_runner.gpu_id)
    graph_backend = defaultdict(
        lambda: f"{current_platform.device_name} graph",
        {
            "cuda": "CUDA graph",
            "musa": "CUDA graph",
            "cpu": "CPU graph",
            "npu": "NPU graph",
            "xpu": "XPU graph",
        },
    )
    role = "draft" if model_runner.is_draft_worker else "target"
    if model_runner.spec_algorithm.is_speculative():
        capture_name = f"{role} verify"
        num_tokens_per_req = model_runner.decode_num_tokens_per_req()
    else:
        capture_name = f"{role} decode"
        num_tokens_per_req = 1
    capture_bs, _ = get_batch_sizes_to_capture(model_runner, num_tokens_per_req)
    decode_backend = get_exec().graph.cuda_graph_config.decode.backend
    logger.info(
        f"Capture {capture_name} {graph_backend[model_runner.device]} begin. "
        f"backend={decode_backend}, num_tokens_per_req={num_tokens_per_req}, "
        f"bs={capture_bs}, avail mem={before_mem:.2f} GB"
    )

    if current_platform.is_out_of_tree():
        GraphRunnerCls = current_platform.get_graph_runner_cls()
        runner = GraphRunnerCls(model_runner)
    else:
        graph_runners = defaultdict(
            model_runner._decode_cuda_graph_runner_cls,
            {
                "cpu": CPUGraphRunner,
                "npu": NPUGraphRunner,
                "xpu": XPUGraphRunner,
            },
        )
        runner = graph_runners[model_runner.device](model_runner)

    after_mem = get_available_gpu_memory(model_runner.device, model_runner.gpu_id)
    memory_usage_gb = before_mem - after_mem
    capture_time = time.perf_counter() - tic
    logger.info(
        f"Capture {capture_name} {graph_backend[model_runner.device]} end. "
        f"elapsed={capture_time:.2f} s, "
        f"mem usage={memory_usage_gb:.2f} GB, avail mem={after_mem:.2f} GB."
    )
    return GraphCapture(
        runner=runner,
        memory_phase=memory_phase,
        memory_usage_gb=memory_usage_gb,
        capture_time=capture_time,
    )
