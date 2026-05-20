from __future__ import annotations

import pytest
import torch

from sglang.jit_kernel.kv_canary.verify import VIOLATION_FIELDS
from sglang.srt.kv_canary.violation_state import ViolationLog
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="extra-a", runner_config="1-gpu-large")


def test_violation_log_allocate_zeroed(device):
    log = ViolationLog.allocate(ring_capacity=8, device=device)
    assert log.violation_ring.shape == (8, VIOLATION_FIELDS)
    assert log.violation_ring.dtype == torch.int64
    assert int(log.violation_ring.abs().sum()) == 0
    assert log.violation_write_index.shape == (1,)
    assert log.violation_write_index.dtype == torch.int32
    assert int(log.violation_write_index.item()) == 0


def test_clear_resets_all(device):
    log = ViolationLog.allocate(ring_capacity=4, device=device)
    log.violation_ring[0, 0] = 7
    log.violation_ring[2, 3] = 11
    log.violation_write_index[0] = 5
    log.clear()
    assert int(log.violation_ring.abs().sum()) == 0
    assert int(log.violation_write_index.item()) == 0


@pytest.mark.xfail(
    strict=False,
    reason="runner._raise_with_first_violation builds a string message; "
    "structured-dict raise (exc.args[0] is dict) not yet implemented",
)
def test_raise_message_includes_idx_expected_actual(device):
    from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
    from sglang.srt.kv_canary.config import CanaryConfig, CanaryMode
    from sglang.srt.kv_canary.runner.canary_runner import CanaryRunner

    config = CanaryConfig(mode=CanaryMode.RAISE)
    k_head = torch.zeros(4, 32, dtype=torch.uint8, device=device)
    k_tail = torch.zeros(4, 32, dtype=torch.uint8, device=device)
    group = CanaryBufferGroup(
        kind=PoolKind.FULL,
        k_head=k_head,
        k_tail=k_tail,
        v_head=None,
        v_tail=None,
        real_kv_sources_k=(),
        real_kv_sources_v=(),
        swa_index_lut=None,
    )
    runner = CanaryRunner(
        config=config,
        buffer_group=group,
        device=device,
        per_forward_verify_capacity=1,
        per_forward_write_req_capacity=1,
        running_sweep_verify_capacity=1,
        radix_sweep_verify_capacity=1,
        radix_sweep_extras_capacity=1,
    )
    runner.violation_log.violation_ring[0] = torch.tensor(
        [1, 42, 5, 100, 200, 0xDEAD, 0xBEEF, 1], dtype=torch.int64, device=device
    )
    runner.violation_log.violation_write_index[0] = 1

    with pytest.raises(RuntimeError) as exc_info:
        runner._raise_with_first_violation()

    payload = exc_info.value.args[0]
    assert isinstance(payload, dict)
    for key in (
        "slot_idx",
        "position",
        "expected",
        "actual",
        "fail_reason",
        "kernel_kind",
    ):
        assert key in payload
        assert payload[key] is not None
