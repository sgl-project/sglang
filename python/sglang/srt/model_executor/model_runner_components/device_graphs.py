from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING

from sglang.srt.configs.model_config import ModelImpl
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import (
    NPUGraphRunner,
)
from sglang.srt.model_executor.breakable_cuda_graph_runner import (
    BreakableCudaGraphRunner,
)
from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.piecewise_cuda_graph_runner import (
    PiecewiseCudaGraphRunner,
)
from sglang.srt.platforms import current_platform
from sglang.srt.utils import get_available_gpu_memory, log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def create_device_graphs(model_runner: "ModelRunner") -> tuple[object, float]:
    """Capture device graphs."""
    graph_runner = None
    graph_mem_usage = 0

    if not model_runner.is_generation:
        # TODO: Currently, cuda graph only captures decode steps, which only exists for generation models
        return None, 0

    if model_runner.server_args.model_impl.lower() == ModelImpl.MINDSPORE:
        return None, 0

    if model_runner.device != "cpu" and model_runner.server_args.disable_cuda_graph:
        return None, 0

    if (
        model_runner.device == "cpu"
        and not model_runner.server_args.enable_torch_compile
    ):
        return None, 0

    tic = time.perf_counter()
    before_mem = get_available_gpu_memory(model_runner.device, model_runner.gpu_id)
    graph_backend = defaultdict(
        lambda: f"{current_platform.device_name} graph",
        {
            "cuda": "cuda graph",
            "musa": "cuda graph",
            "cpu": "cpu graph",
            "npu": "npu graph",
        },
    )
    logger.info(
        f"Capture {graph_backend[model_runner.device]} begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
    )
    if current_platform.is_out_of_tree():
        GraphRunnerCls = current_platform.get_graph_runner_cls()
        graph_runner = GraphRunnerCls(model_runner)
    else:
        graph_runners = defaultdict(
            lambda: CudaGraphRunner,
            {
                "cpu": CPUGraphRunner,
                "npu": NPUGraphRunner,
            },
        )
        graph_runner = graph_runners[model_runner.device](model_runner)

    after_mem = get_available_gpu_memory(model_runner.device, model_runner.gpu_id)
    graph_mem_usage = before_mem - after_mem
    logger.info(
        f"Capture {graph_backend[model_runner.device]} end. Time elapsed: {time.perf_counter() - tic:.2f} s. "
        f"mem usage={graph_mem_usage:.2f} GB. avail mem={after_mem:.2f} GB."
    )
    return graph_runner, graph_mem_usage


def create_piecewise_cuda_graphs(model_runner: "ModelRunner"):
    """Initialize piecewise CUDA graph runner."""
    piecewise_cuda_graph_runner = None

    if model_runner.server_args.disable_piecewise_cuda_graph:
        logger.info(
            "Disable piecewise CUDA graph because --disable-piecewise-cuda-graph is set"
        )
        return None

    # Draft models use decode CUDA graphs, not PCG
    if model_runner.is_draft_worker:
        return None

    # Disable piecewise CUDA graph for non-language models
    if not hasattr(model_runner.model, "model"):
        logger.warning(
            "Disable piecewise CUDA graph because the model is not a language model"
        )
        return None

    # Disable piecewise CUDA graph for non capture size
    if not model_runner.server_args.piecewise_cuda_graph_tokens:
        logger.warning(
            "Disable piecewise CUDA graph because the capture size is not set"
        )
        return None

    # Collect attention layers and moe layers from the model
    from sglang.srt.model_executor.model_runner import resolve_language_model

    model_runner.model.model = resolve_language_model(model_runner.model)
    language_model = getattr(model_runner.model, "language_model", model_runner.model)

    # Resolve model with layers: handle CausalLM wrapper (.model.layers) and direct TextModel (.layers)
    if hasattr(language_model, "model") and hasattr(language_model.model, "layers"):
        layer_model = language_model.model
    elif hasattr(language_model, "layers"):
        layer_model = language_model
    else:
        logger.warning(
            "Disable piecewise CUDA graph because the model does not have a 'layers' attribute"
        )
        return None

    model_runner.attention_layers = []
    model_runner.moe_layers = []
    model_runner.moe_fusions = []
    for layer in layer_model.layers:
        attn_layer = None
        if hasattr(layer, "self_attn"):
            if hasattr(layer.self_attn, "attn"):
                attn_layer = layer.self_attn.attn
            elif hasattr(layer.self_attn, "attn_mqa"):
                # For DeepSeek model
                attn_layer = layer.self_attn.attn_mqa
        # For hybrid model
        elif hasattr(layer, "attn"):
            attn_layer = layer.attn
        elif hasattr(layer, "linear_attn"):
            if hasattr(layer.linear_attn, "attn"):
                attn_layer = layer.linear_attn.attn
            else:
                attn_layer = layer.linear_attn
        # For InternVL model
        elif hasattr(layer, "attention"):
            if hasattr(layer.attention, "attn"):
                attn_layer = layer.attention.attn
        # For NemotronH and similar hybrid models using 'mixer' attribute
        elif hasattr(layer, "mixer"):
            if hasattr(layer.mixer, "attn"):
                attn_layer = layer.mixer.attn
            elif hasattr(layer, "_forward_mamba"):
                # Mamba layer with split op support - store the layer itself
                attn_layer = layer

        if attn_layer is not None:
            model_runner.attention_layers.append(attn_layer)
        elif hasattr(layer, "mixer"):
            model_runner.attention_layers.append(None)

        moe_block = None
        moe_fusion = None
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            moe_block = layer.mlp.experts
            moe_fusion = layer.mlp
        if hasattr(layer, "block_sparse_moe") and hasattr(
            layer.block_sparse_moe, "experts"
        ):
            moe_block = layer.block_sparse_moe.experts
            moe_fusion = layer.block_sparse_moe
        if hasattr(layer, "moe") and hasattr(layer.moe, "experts"):
            moe_block = layer.moe.experts
            moe_fusion = layer.moe
        # For NemotronH MoE layers using 'mixer' attribute
        if hasattr(layer, "mixer") and hasattr(layer.mixer, "experts"):
            moe_block = layer.mixer.experts
            moe_fusion = layer.mixer
        model_runner.moe_layers.append(moe_block)
        model_runner.moe_fusions.append(moe_fusion)

    if len(model_runner.attention_layers) < model_runner.model_config.num_hidden_layers:
        # TODO(yuwei): support Non-Standard GQA
        log_info_on_rank0(
            logger,
            "Disable piecewise CUDA graph because some layers do not apply Standard GQA",
        )
        return None

    tic = time.perf_counter()
    before_mem = get_available_gpu_memory(model_runner.device, model_runner.gpu_id)
    logger.info(f"Capture piecewise CUDA graph begin. avail mem={before_mem:.2f} GB")

    if model_runner.server_args.enable_breakable_cuda_graph:
        # Experimental feature
        piecewise_cuda_graph_runner = BreakableCudaGraphRunner(model_runner)
    else:
        piecewise_cuda_graph_runner = PiecewiseCudaGraphRunner(model_runner)

    after_mem = get_available_gpu_memory(model_runner.device, model_runner.gpu_id)
    mem_usage = before_mem - after_mem
    logger.info(
        f"Capture piecewise CUDA graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. "
        f"mem usage={mem_usage:.2f} GB. avail mem={after_mem:.2f} GB."
    )
    return piecewise_cuda_graph_runner
