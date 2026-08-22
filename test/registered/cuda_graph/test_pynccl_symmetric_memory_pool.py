"""Feature test for symmetric-memory CUDA graph capture."""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=90, stage="base-b", runner_config="2-gpu-large")


def test_cuda_graph_collective_with_capture_only_allocations() -> None:
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        use_symmetric_memory,
    )
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
        set_custom_all_reduce,
    )
    from sglang.srt.model_executor.runner.shape_key import ShapeKey
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
        FullCudaGraphBackend,
    )
    from sglang.srt.runtime_context import get_context

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    config_override = get_context().override_server_args(enable_symm_mem=True)
    config_override.install()
    set_custom_all_reduce(False)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=world_size,
        enable_symm_mem=True,
    )
    tp_group = get_tp_group()

    static_input = torch.full(
        (64 * 1024 * 1024,), rank + 1, dtype=torch.uint8, device="cuda"
    )

    def forward():
        count = 4 if torch.cuda.is_current_stream_capturing() else 1
        with use_symmetric_memory(tp_group):
            outputs = [torch.empty_like(static_input) for _ in range(count)]
        for output in outputs:
            output.copy_(static_input)
            tp_group.all_reduce(output)
        return outputs

    runner = SimpleNamespace(
        device_module=torch.cuda,
        model_runner=SimpleNamespace(tp_group=tp_group),
        enable_profile_cuda_graph=False,
    )
    backend = FullCudaGraphBackend(runner)
    stream = torch.cuda.Stream()
    with tp_group.graph_capture(stream=stream), backend.capture_session(stream):
        backend.capture_one(ShapeKey(size=1), forward)

    outputs = backend.replay(ShapeKey(size=1), None)
    torch.cuda.synchronize()
    expected = world_size * (world_size + 1) // 2
    if not all(torch.all(output == expected) for output in outputs):
        raise AssertionError("symmetric-memory CUDA graph all-reduce mismatch")

    backend.cleanup()
    destroy_model_parallel()
    destroy_distributed_environment()
    config_override.restore()


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(2,), timeout=90)
