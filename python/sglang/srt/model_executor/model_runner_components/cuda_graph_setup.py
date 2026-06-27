from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from sglang.srt.configs.model_config import ModelImpl
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    prealloc_symmetric_memory_pool,
)
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner
from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
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
from sglang.srt.utils import get_available_gpu_memory, log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner.base_runner import BaseRunner

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class DecodeGraphCapture:
    runner: Optional[BaseRunner]
    graph_mem_usage: float


@dataclass(frozen=True, slots=True, kw_only=True)
class CudaGraphsCapture:
    eager_runner: EagerRunner
    prefill_runner: Optional[BaseRunner]
    decode: DecodeGraphCapture


def capture_cuda_graphs(
    *, model_runner: ModelRunner, capture_decode_cuda_graph: bool = True
) -> CudaGraphsCapture:
    """Capture cuda graphs. Requires init_attention_backends() to have run.

    Spec draft runners pass capture_decode_cuda_graph=False
    because they capture their own decode-style graphs separately.

    Read-only over ModelRunner (PR #28978 e46665e84e): reads fields off it and
    returns a CudaGraphsCapture for ModelRunner.init_cuda_graphs to assign. The
    only ModelRunner field written back is the prefill layer set, inside
    capture_prefill_graph. Takes the whole runner as the documented leaf
    exception (every graph runner constructor eats it and keeps a live reference).
    """

    # The eager (no-cuda-graph) phase runner, built AFTER the attention
    # backend so its __init__ can warm up kernels (run-once) and allocate the
    # fixed-max static buffer — both before the cuda-graph runners, so that
    # buffer is canonical in the shared pool and the cg runners coalesce onto
    # it. Always built: it serves both the fully-disabled case (decode/prefill
    # runners point at it) and the eager fallback when a cg runner can't run a
    # batch.
    eager_runner = EagerRunner(model_runner)

    # cuda-graph capture: prefill before decode, so both coalesce onto the
    # eager buffer allocated above. (capture_prefill_graph routes prefill
    # to the eager runner when the prefill graph is disabled.)
    prefill_runner = capture_prefill_graph(
        model_runner=model_runner, eager_runner=eager_runner
    )

    decode = DecodeGraphCapture(runner=None, graph_mem_usage=0)
    if capture_decode_cuda_graph:
        if model_runner.device in ("cuda", "musa", "cpu", "npu"):
            decode = capture_decode_graph(model_runner=model_runner)
        elif (
            current_platform.is_out_of_tree() and current_platform.support_cuda_graph()
        ):
            decode = capture_decode_graph(model_runner=model_runner)
    else:
        decode = DecodeGraphCapture(runner=eager_runner, graph_mem_usage=0)

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
        enable_symm_mem=model_runner.server_args.enable_symm_mem,
        device=model_runner.device,
        forward_stream=model_runner.forward_stream,
    )

    if model_runner.canary_manager is not None and not model_runner.is_draft_worker:
        model_runner.canary_manager.mark_init_finished()

    return CudaGraphsCapture(
        eager_runner=eager_runner, prefill_runner=prefill_runner, decode=decode
    )


def capture_prefill_graph(
    *,
    model_runner: ModelRunner,
    eager_runner: EagerRunner,
    force_for_draft_worker: bool = False,
) -> Optional[BaseRunner]:
    """Build the prefill CUDA graph runner (or route to eager). Returns the runner.

    Writes model_runner.{attention_layers,moe_layers,moe_fusions,dsa_indexers}
    before constructing PrefillCudaGraphRunner, because its __init__ reads those
    off the runner (prefill_cuda_graph_runner.py) and the EagerRunner forward
    path reads them too -- the one sanctioned write-back under the
    read-only-god-object rule (PR #28978 e46665e84e); the single-arg constructor
    contract leaves no other way to hand them over.
    """

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
            return eager_runner
        return None

    # Draft models skip here during __init__; the eagle worker calls
    # this method explicitly (force_for_draft_worker=True) after
    # init_lm_head so graphs capture the final embedding weights.
    if model_runner.is_draft_worker and not force_for_draft_worker:
        return None

    # Skip prefill CG for EAGLE target on tc_piecewise: that backend
    # captures CaptureHiddenMode.NULL while runtime requests FULL, so
    # the captured graph is dead, and capturing it perturbs FP4 /
    # TRTLLM-MoE state and corrupts decode replay (see #28386). BCG
    # captures FULL for EAGLE target in PrefillCudaGraphRunner.__init__
    # (restored from #25795), so it does NOT need this skip.
    if (
        model_runner.spec_algorithm.is_eagle()
        and not model_runner.is_draft_worker
        and not model_runner.server_args.enable_return_hidden_states
        and not check_cuda_graph_backend(Phase.PREFILL, Backend.BREAKABLE)
    ):
        logger.info(
            "Disable prefill CUDA graph for EAGLE target on tc_piecewise "
            "to avoid FP4/MoE decode-replay corruption (#28386)."
        )
        return eager_runner

    # Disable prefill CUDA graph for non-language models
    if not hasattr(model_runner.model, "model"):
        logger.warning(
            "Disable prefill CUDA graph because the model is not a language model"
        )
        return None

    # Disable prefill CUDA graph for non capture size
    if not model_runner.server_args.cuda_graph_config.prefill.bs:
        logger.warning("Disable prefill CUDA graph because the capture size is not set")
        return None

    # Collect attention layers and moe layers from the model
    model_runner.model.model = resolve_language_model(model_runner.model)
    language_model = getattr(model_runner.model, "language_model", model_runner.model)

    # Resolve model with layers: handle CausalLM wrapper (.model.layers) and direct TextModel (.layers)
    if hasattr(language_model, "model") and hasattr(language_model.model, "layers"):
        layer_model = language_model.model
    elif hasattr(language_model, "layers"):
        layer_model = language_model
    else:
        logger.warning(
            "Disable prefill CUDA graph because the model does not have a 'layers' attribute"
        )
        return None

    (
        model_runner.attention_layers,
        model_runner.moe_layers,
        model_runner.moe_fusions,
        model_runner.dsa_indexers,
    ) = compute_attention_and_moe_layers(layer_model)

    if len(model_runner.attention_layers) < model_runner.model_config.num_hidden_layers:
        # TODO(yuwei): support Non-Standard GQA
        log_info_on_rank0(
            logger,
            "Disable prefill CUDA graph because some layers do not apply Standard GQA",
        )
        return None

    tic = time.perf_counter()
    before_mem = get_available_gpu_memory(model_runner.device, model_runner.gpu_id)
    prefill_backend = model_runner.server_args.cuda_graph_config.prefill.backend
    role = "draft" if model_runner.is_draft_worker else "target"
    capture_name = f"{role} prefill"
    capture_num_tokens = sorted(model_runner.server_args.cuda_graph_config.prefill.bs)
    logger.info(
        f"Capture {capture_name} CUDA graph begin. "
        f"backend={prefill_backend}, num_tokens={capture_num_tokens}, "
        f"avail mem={before_mem:.2f} GB"
    )

    prefill_runner = PrefillCudaGraphRunner(model_runner)

    after_mem = get_available_gpu_memory(model_runner.device, model_runner.gpu_id)
    mem_usage = before_mem - after_mem
    logger.info(
        f"Capture {capture_name} CUDA graph end. "
        f"elapsed={time.perf_counter() - tic:.2f} s, "
        f"mem usage={mem_usage:.2f} GB, avail mem={after_mem:.2f} GB."
    )
    return prefill_runner


def capture_decode_graph(*, model_runner: ModelRunner) -> DecodeGraphCapture:
    """Capture device decode graphs. Returns the runner + measured graph mem."""
    no_capture = DecodeGraphCapture(runner=None, graph_mem_usage=0)

    if not model_runner.is_generation:
        # TODO: Currently, cuda graph only captures decode steps, which only exists for generation models
        return no_capture
    if model_runner.server_args.model_impl.lower() == ModelImpl.MINDSPORE:
        return no_capture
    if model_runner.device != "cpu" and check_cuda_graph_backend(
        Phase.DECODE, Backend.DISABLED
    ):
        return no_capture
    if (
        model_runner.device == "cpu"
        and not model_runner.server_args.enable_torch_compile
    ):
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
        },
    )
    role = "draft" if model_runner.is_draft_worker else "target"
    if model_runner.spec_algorithm.is_speculative():
        capture_name = f"{role} verify"
        num_tokens_per_bs = (
            model_runner.spec_algorithm.get_num_tokens_per_bs_for_target_verify(
                model_runner.server_args.speculative_num_draft_tokens,
                model_runner.is_draft_worker,
            )
        )
    else:
        capture_name = f"{role} decode"
        num_tokens_per_bs = 1
    capture_bs, _ = get_batch_sizes_to_capture(model_runner, num_tokens_per_bs)
    decode_backend = model_runner.server_args.cuda_graph_config.decode.backend
    logger.info(
        f"Capture {capture_name} {graph_backend[model_runner.device]} begin. "
        f"backend={decode_backend}, num_tokens_per_bs={num_tokens_per_bs}, "
        f"bs={capture_bs}, avail mem={before_mem:.2f} GB"
    )

    if current_platform.is_out_of_tree():
        GraphRunnerCls = current_platform.get_graph_runner_cls()
        runner = GraphRunnerCls(model_runner)
    else:
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        graph_runners = defaultdict(
            lambda: DecodeCudaGraphRunner,
            {
                "cpu": CPUGraphRunner,
                "npu": NPUGraphRunner,
            },
        )
        runner = graph_runners[model_runner.device](model_runner)

    after_mem = get_available_gpu_memory(model_runner.device, model_runner.gpu_id)
    graph_mem_usage = before_mem - after_mem
    logger.info(
        f"Capture {capture_name} {graph_backend[model_runner.device]} end. "
        f"elapsed={time.perf_counter() - tic:.2f} s, "
        f"mem usage={graph_mem_usage:.2f} GB, avail mem={after_mem:.2f} GB."
    )
    return DecodeGraphCapture(runner=runner, graph_mem_usage=graph_mem_usage)
