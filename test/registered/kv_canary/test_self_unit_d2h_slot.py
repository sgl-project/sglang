from __future__ import annotations

import torch

from sglang.srt.kv_canary.runner.d2h_slot import DelayedD2HReadSlot
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="extra-a", runner_config="1-gpu-large")


def test_cpu_fallback_first_pop_returns_none() -> None:
    host = torch.zeros(1, dtype=torch.int32)
    slot = DelayedD2HReadSlot(host=host, stream=None)
    assert slot.pop() is None


def test_cpu_fallback_pop_returns_staged_value() -> None:
    host = torch.zeros(1, dtype=torch.int32)
    slot = DelayedD2HReadSlot(host=host, stream=None)
    src = torch.tensor([7], dtype=torch.int32)
    slot.stage(src_device=src)
    result = slot.pop()
    assert result is not None
    assert int(result.item()) == 7


def test_cpu_fallback_pop_idempotent_after_no_new_stage() -> None:
    host = torch.zeros(1, dtype=torch.int32)
    slot = DelayedD2HReadSlot(host=host, stream=None)
    slot.stage(src_device=torch.tensor([3], dtype=torch.int32))
    assert slot.pop() is not None
    assert slot.pop() is None


def test_cuda_stage_uses_injected_stream_and_one_step_delay() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    alt_stream = torch.cuda.Stream(device=device)
    default_stream = torch.cuda.current_stream(device)
    assert alt_stream.stream_id != default_stream.stream_id

    host = torch.zeros(1, dtype=torch.int32, pin_memory=True)
    slot = DelayedD2HReadSlot(host=host, stream=alt_stream)

    assert slot.pop() is None

    src_first = torch.tensor([41], dtype=torch.int32, device=device)
    slot.stage(src_device=src_first)
    result_first = slot.pop()
    assert result_first is not None
    assert int(result_first.item()) == 41

    src_second = torch.tensor([97], dtype=torch.int32, device=device)
    slot.stage(src_device=src_second)
    result_second = slot.pop()
    assert result_second is not None
    assert int(result_second.item()) == 97
