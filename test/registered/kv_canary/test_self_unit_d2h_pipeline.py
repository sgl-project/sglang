from __future__ import annotations

import torch

from sglang.srt.kv_canary.runner.d2h_pipeline import CanaryD2HPipeline
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="extra-a", runner_config="1-gpu-large")


def test_cpu_fallback_copies_synchronously() -> None:
    device = torch.device("cpu")
    pipeline = CanaryD2HPipeline(device=device)
    src = torch.tensor([7], dtype=torch.int32)
    dst = torch.zeros(1, dtype=torch.int32)
    event = pipeline.stage(dst_host=dst, src_device=src)
    assert event is None
    assert int(dst.item()) == 7
    CanaryD2HPipeline.wait(event)


def test_cpu_fallback_overwrites_destination_each_stage() -> None:
    device = torch.device("cpu")
    pipeline = CanaryD2HPipeline(device=device)
    dst = torch.zeros(1, dtype=torch.int32)
    src_a = torch.tensor([3], dtype=torch.int32)
    src_b = torch.tensor([11], dtype=torch.int32)
    pipeline.stage(dst_host=dst, src_device=src_a)
    assert int(dst.item()) == 3
    pipeline.stage(dst_host=dst, src_device=src_b)
    assert int(dst.item()) == 11


def test_wait_none_is_noop() -> None:
    CanaryD2HPipeline.wait(None)


def test_cuda_stage_uses_alt_stream_and_one_step_delay() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    pipeline = CanaryD2HPipeline(device=device)
    default_stream = torch.cuda.current_stream(device)
    assert pipeline._stream is not None
    assert pipeline._stream.stream_id != default_stream.stream_id

    dst_host = torch.zeros(1, dtype=torch.int32, pin_memory=True)
    src_device_first = torch.tensor([41], dtype=torch.int32, device=device)
    src_device_second = torch.tensor([97], dtype=torch.int32, device=device)

    event_first = pipeline.stage(dst_host=dst_host, src_device=src_device_first)
    assert event_first is not None
    CanaryD2HPipeline.wait(event_first)
    assert int(dst_host.item()) == 41

    event_second = pipeline.stage(dst_host=dst_host, src_device=src_device_second)
    assert event_second is not None
    CanaryD2HPipeline.wait(event_second)
    assert int(dst_host.item()) == 97
