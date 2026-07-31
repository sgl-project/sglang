# SPDX-License-Identifier: Apache-2.0
"""Regression tests for per-rank accelerator device binding."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.multimodal_gen.runtime import platforms
from sglang.multimodal_gen.runtime.distributed import parallel_state
from sglang.multimodal_gen.runtime.managers import gpu_worker


class _FakePlatform:
    def __init__(self, *, cuda_alike: bool):
        self._cuda_alike = cuda_alike

    def get_torch_distributed_backend_str(self) -> str:
        return "nccl" if self._cuda_alike else "xccl"

    def is_cuda_alike(self) -> bool:
        return self._cuda_alike

    def is_npu(self) -> bool:
        return False


@pytest.mark.parametrize(
    ("device_type", "cuda_alike"),
    (("cuda", True), ("xpu", False), ("musa", True)),
)
def test_distributed_init_binds_rank_before_creating_groups(
    monkeypatch, device_type, cuda_alike
):
    """A rank must be bound before communicator setup on every accelerator."""
    calls = []
    device = SimpleNamespace(type=device_type)
    device_module = SimpleNamespace(
        set_device=lambda local_rank: calls.append(("set_device", local_rank))
    )

    monkeypatch.setattr(
        platforms, "current_platform", _FakePlatform(cuda_alike=cuda_alike)
    )
    monkeypatch.setattr(parallel_state, "_WORLD", None)
    monkeypatch.setattr(parallel_state, "get_local_torch_device", lambda: device)
    monkeypatch.setattr(
        parallel_state.torch,
        "get_device_module",
        lambda selected_device: device_module,
    )
    monkeypatch.setattr(
        parallel_state.torch.cuda,
        "set_device",
        lambda selected_device: calls.append(("set_device", selected_device)),
    )
    monkeypatch.setattr(
        parallel_state,
        "init_distributed_environment",
        lambda **kwargs: calls.append(("init_distributed", kwargs["local_rank"])),
    )
    monkeypatch.setattr(
        parallel_state,
        "initialize_model_parallel",
        lambda **kwargs: calls.append(("init_model_parallel", None)),
    )
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")

    parallel_state.maybe_init_distributed_environment_and_model_parallel(
        tp_size=1,
        sp_size=1,
    )

    assert calls == [
        ("set_device", 3),
        ("init_distributed", 3),
        ("init_model_parallel", None),
    ]


def test_gpu_worker_allows_device_modules_without_set_device(monkeypatch):
    """MPS workers must reach distributed init although torch.mps has no set_device."""
    worker = object.__new__(gpu_worker.GPUWorker)
    worker.local_rank = 0
    worker.rank = 0
    worker.master_port = 30000
    worker.server_args = SimpleNamespace(
        cfg_parallel_degree=None,
        dist_timeout=None,
        dp_size=1,
        layerwise_offload_components=[],
        num_gpus=1,
        ring_degree=1,
        sp_degree=1,
        tp_size=1,
        ulysses_degree=1,
    )
    worker._configure_persistent_torch_compile_cache = Mock()
    distributed_init = Mock()
    pipeline = object()

    monkeypatch.setattr(
        gpu_worker.torch, "get_device_module", lambda: SimpleNamespace()
    )
    monkeypatch.setattr(
        gpu_worker,
        "maybe_init_distributed_environment_and_model_parallel",
        distributed_init,
    )
    monkeypatch.setattr(gpu_worker, "model_parallel_is_initialized", lambda: False)
    monkeypatch.setattr(gpu_worker, "setproctitle", lambda title: None)
    monkeypatch.setattr(gpu_worker, "build_pipeline", lambda server_args: pipeline)
    for name in ("MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK", "RANK", "WORLD_SIZE"):
        monkeypatch.setenv(name, "original")

    worker.init_device_and_model()

    distributed_init.assert_called_once()
    assert worker.pipeline is pipeline
