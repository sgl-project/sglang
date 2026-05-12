from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING

from sglang.srt.configs.model_config import ModelImpl
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import (
    NPUGraphRunner,
)
from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.platforms import current_platform
from sglang.srt.utils import get_available_gpu_memory

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
