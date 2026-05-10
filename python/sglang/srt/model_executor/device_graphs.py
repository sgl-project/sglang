from __future__ import annotations

import logging
import time
from collections import defaultdict

from sglang.srt.configs.model_config import ModelImpl
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner
from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.platforms import current_platform
from sglang.srt.utils import get_available_gpu_memory

logger = logging.getLogger(__name__)


def init_device_graphs(*, model_runner_ref):
    """Capture device graphs."""
    model_runner_ref.graph_runner = None
    model_runner_ref.graph_mem_usage = 0

    if not model_runner_ref.is_generation:
        # TODO: Currently, cuda graph only captures decode steps, which only exists for generation models
        return

    if model_runner_ref.server_args.model_impl.lower() == ModelImpl.MINDSPORE:
        return

    if (
        model_runner_ref.device != "cpu"
        and model_runner_ref.server_args.disable_cuda_graph
    ):
        return

    if (
        model_runner_ref.device == "cpu"
        and not model_runner_ref.server_args.enable_torch_compile
    ):
        return

    tic = time.perf_counter()
    before_mem = get_available_gpu_memory(
        model_runner_ref.device, model_runner_ref.gpu_id
    )
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
        f"Capture {graph_backend[model_runner_ref.device]} begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
    )
    if current_platform.is_out_of_tree():
        GraphRunnerCls = current_platform.get_graph_runner_cls()
        model_runner_ref.graph_runner = GraphRunnerCls(model_runner_ref)
    else:
        graph_runners = defaultdict(
            lambda: CudaGraphRunner,
            {
                "cpu": CPUGraphRunner,
                "npu": NPUGraphRunner,
            },
        )
        model_runner_ref.graph_runner = graph_runners[model_runner_ref.device](
            model_runner_ref
        )

    after_mem = get_available_gpu_memory(
        model_runner_ref.device, model_runner_ref.gpu_id
    )
    model_runner_ref.graph_mem_usage = before_mem - after_mem
    logger.info(
        f"Capture {graph_backend[model_runner_ref.device]} end. Time elapsed: {time.perf_counter() - tic:.2f} s. "
        f"mem usage={model_runner_ref.graph_mem_usage:.2f} GB. avail mem={after_mem:.2f} GB."
    )
