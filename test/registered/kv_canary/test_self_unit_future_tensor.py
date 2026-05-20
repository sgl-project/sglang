from __future__ import annotations

import torch

from sglang.srt.kv_canary.runner.future_tensor import FutureTensor, stage_d2h_future
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="extra-a", runner_config="1-gpu-large")


def test_cpu_fallback_returns_host_copy_immediately() -> None:
    src = torch.tensor([7], dtype=torch.int32)
    future = stage_d2h_future(src_device=src, stream=None)
    assert isinstance(future, FutureTensor)
    result = future.wait()
    assert int(result.item()) == 7
    assert result.dtype == torch.int32
    assert tuple(result.shape) == (1,)


def test_cpu_fallback_does_not_allocate_pinned() -> None:
    src = torch.tensor([3], dtype=torch.int32)
    future = stage_d2h_future(src_device=src, stream=None)
    assert future._tensor.is_pinned() is False


def test_cpu_fallback_each_call_allocates_fresh_host() -> None:
    src_a = torch.tensor([11], dtype=torch.int32)
    src_b = torch.tensor([22], dtype=torch.int32)
    future_a = stage_d2h_future(src_device=src_a, stream=None)
    future_b = stage_d2h_future(src_device=src_b, stream=None)
    assert future_a._tensor.data_ptr() != future_b._tensor.data_ptr()
    assert int(future_a.wait().item()) == 11
    assert int(future_b.wait().item()) == 22


def test_cuda_stage_then_wait_returns_host_copy() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    alt_stream = torch.cuda.Stream(device=device)
    default_stream = torch.cuda.current_stream(device)
    assert alt_stream.stream_id != default_stream.stream_id

    src_first = torch.tensor([41], dtype=torch.int32, device=device)
    future_first = stage_d2h_future(src_device=src_first, stream=alt_stream)
    result_first = future_first.wait()
    assert int(result_first.item()) == 41

    src_second = torch.tensor([97], dtype=torch.int32, device=device)
    future_second = stage_d2h_future(src_device=src_second, stream=alt_stream)
    result_second = future_second.wait()
    assert int(result_second.item()) == 97


def test_cuda_pinned_when_stream_is_provided() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    alt_stream = torch.cuda.Stream(device=device)
    src = torch.tensor([5], dtype=torch.int32, device=device)
    future = stage_d2h_future(src_device=src, stream=alt_stream)
    assert future._tensor.is_pinned() is True
    assert int(future.wait().item()) == 5


def test_cuda_each_call_allocates_fresh_host() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    alt_stream = torch.cuda.Stream(device=device)
    src_a = torch.tensor([13], dtype=torch.int32, device=device)
    src_b = torch.tensor([29], dtype=torch.int32, device=device)
    future_a = stage_d2h_future(src_device=src_a, stream=alt_stream)
    future_b = stage_d2h_future(src_device=src_b, stream=alt_stream)
    assert future_a._tensor.data_ptr() != future_b._tensor.data_ptr()
    assert int(future_a.wait().item()) == 13
    assert int(future_b.wait().item()) == 29
