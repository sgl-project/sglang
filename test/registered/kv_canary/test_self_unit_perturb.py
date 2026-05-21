from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from sglang.jit_kernel.kv_canary.verify import CANARY_SLOT_BYTES, RealKvSource
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.perturb.config import (
    TargetGroupKind,
    _parse_target_group_kind,
)
from sglang.srt.kv_canary.perturb.utils import pick_target_group
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="extra-a", runner_config="1-gpu-large")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("full", TargetGroupKind.FULL),
        ("FULL", TargetGroupKind.FULL),
        (" swa ", TargetGroupKind.SWA),
        ("Any", TargetGroupKind.ANY),
        (None, TargetGroupKind.ANY),
    ],
)
def test_parse_target_group_kind_accepts_valid_values_case_insensitively(
    raw: str | None, expected: TargetGroupKind
) -> None:
    assert _parse_target_group_kind(raw) == expected


def test_parse_target_group_kind_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        _parse_target_group_kind("prefix")


def test_pick_target_group_filters_full() -> None:
    full_group = _make_group(kind=PoolKind.FULL, has_real_kv=True)
    swa_group = _make_group(kind=PoolKind.SWA, has_real_kv=True)

    group = pick_target_group(
        buffer_groups=(full_group, swa_group),
        target_kind=TargetGroupKind.FULL,
    )

    assert group is full_group


def test_pick_target_group_filters_swa() -> None:
    full_group = _make_group(kind=PoolKind.FULL, has_real_kv=True)
    swa_group = _make_group(kind=PoolKind.SWA, has_real_kv=True)

    group = pick_target_group(
        buffer_groups=(full_group, swa_group),
        target_kind=TargetGroupKind.SWA,
    )

    assert group is swa_group


def test_pick_target_group_any_uses_all_eligible_groups() -> None:
    full_group = _make_group(kind=PoolKind.FULL, has_real_kv=True)
    swa_group = _make_group(kind=PoolKind.SWA, has_real_kv=True)

    with patch.object(torch, "randint", return_value=torch.tensor(1)):
        group = pick_target_group(
            buffer_groups=(full_group, swa_group),
            target_kind=TargetGroupKind.ANY,
        )

    assert group is swa_group


def test_pick_target_group_ignores_groups_without_real_kv_sources() -> None:
    full_group = _make_group(kind=PoolKind.FULL, has_real_kv=False)
    swa_group = _make_group(kind=PoolKind.SWA, has_real_kv=True)

    group = pick_target_group(
        buffer_groups=(full_group, swa_group),
        target_kind=TargetGroupKind.FULL,
    )

    assert group is None


def _make_group(*, kind: PoolKind, has_real_kv: bool) -> CanaryBufferGroup:
    source = RealKvSource(
        tensor=torch.zeros(4, 16, dtype=torch.uint8),
        page_size=1,
        num_bytes_per_token=16,
        read_bytes=16,
    )
    real_kv_sources = (source,) if has_real_kv else ()
    return CanaryBufferGroup(
        kind=kind,
        k_head=torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8),
        k_tail=torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8),
        v_head=torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8),
        v_tail=torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8),
        real_kv_sources_k=real_kv_sources,
        real_kv_sources_v=real_kv_sources,
        swa_index_lut=None,
    )
