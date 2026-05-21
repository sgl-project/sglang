from __future__ import annotations

from typing import cast
from unittest.mock import patch

import pytest
import torch

from kv_canary_runner_unit_utils import make_forward_batch, make_pool
from sglang.jit_kernel.kv_canary.verify import CANARY_SLOT_BYTES, RealKvSource
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.perturb.config import (
    PerturbConfig,
    TargetGroupKind,
    _parse_target_group_kind,
)
from sglang.srt.kv_canary.perturb.manager import PerturbManager
from sglang.srt.kv_canary.perturb.slot_picker import collect_active_slots
from sglang.srt.kv_canary.perturb.utils import pick_target_group
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.fixtures import DEFAULT_DEVICE

register_cuda_ci(est_time=10, stage="extra-a", runner_config="1-gpu-large")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("full", TargetGroupKind.FULL),
        ("FULL", TargetGroupKind.FULL),
        (" swa ", TargetGroupKind.SWA),
    ],
)
def test_parse_target_group_kind_accepts_valid_values_case_insensitively(
    raw: str | None, expected: TargetGroupKind
) -> None:
    assert _parse_target_group_kind(raw) == expected


def test_parse_target_group_kind_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        _parse_target_group_kind("prefix")


@pytest.mark.parametrize("raw", [None, "", "any", " Any "])
def test_parse_target_group_kind_rejects_missing_or_any(raw: str | None) -> None:
    with pytest.raises(ValueError, match="SGLANG_KV_CANARY_PERTURB_TARGET_GROUP"):
        _parse_target_group_kind(raw)


@pytest.mark.parametrize(
    ("target_kind", "expected_kind"),
    [
        (TargetGroupKind.FULL, PoolKind.FULL),
        (TargetGroupKind.SWA, PoolKind.SWA),
    ],
)
def test_pick_target_group_filters_exact_kind(
    target_kind: TargetGroupKind, expected_kind: PoolKind
) -> None:
    full_group = _make_group(kind=PoolKind.FULL, has_real_kv=True)
    swa_group = _make_group(kind=PoolKind.SWA, has_real_kv=True)

    group = pick_target_group(
        buffer_groups=(full_group, swa_group),
        target_kind=target_kind,
    )

    assert group is not None
    assert group.kind == expected_kind


def test_pick_target_group_rejects_unsupported_kind() -> None:
    full_group = _make_group(kind=PoolKind.FULL, has_real_kv=True)

    with pytest.raises(ValueError, match="Unsupported target_group_kind"):
        pick_target_group(
            buffer_groups=(full_group,),
            target_kind=cast(TargetGroupKind, 2),
        )


def test_pick_target_group_ignores_groups_without_real_kv_sources() -> None:
    full_group = _make_group(kind=PoolKind.FULL, has_real_kv=False)
    swa_group = _make_group(kind=PoolKind.SWA, has_real_kv=True)

    group = pick_target_group(
        buffer_groups=(full_group, swa_group),
        target_kind=TargetGroupKind.FULL,
    )

    assert group is None


def test_perturb_manager_perturb_dispatches_all_points() -> None:
    """Verify perturb() runs each perturb point in order."""
    device = DEFAULT_DEVICE
    manager = PerturbManager(
        config=PerturbConfig(
            req_to_token_prob=0.0,
            real_kv_used_prob=0.0,
            real_kv_unused_cache_prob=0.0,
            target_group_kind=TargetGroupKind.FULL,
            warmup_steps=0,
        ),
        req_to_token_pool=make_pool(device),
        buffer_groups=(),
        step_counter_getter=lambda: 10,
    )
    forward_batch = make_forward_batch(device)
    calls: list[str] = []

    with patch.object(
        manager,
        "perturb_req_to_token",
        lambda batch: calls.append("req_to_token"),
    ), patch.object(
        manager,
        "perturb_real_kv_used",
        lambda batch: calls.append("real_kv_used"),
    ), patch.object(
        manager,
        "perturb_real_kv_unused_cache",
        lambda batch: calls.append("real_kv_unused_cache"),
    ):
        manager.perturb(forward_batch)

    assert calls == ["req_to_token", "real_kv_used", "real_kv_unused_cache"]


def test_req_to_token_perturb_uses_live_slot_as_replacement() -> None:
    """Verify req_to_token perturbation replaces a slot with another live slot."""
    device = DEFAULT_DEVICE
    pool = make_pool(device, max_reqs=4, max_seq=8)
    pool.req_to_token[1, :3] = torch.tensor([11, 22, 33], dtype=torch.int32, device=device)
    pool.req_to_token[2, :3] = torch.tensor([44, 55, 66], dtype=torch.int32, device=device)
    manager = PerturbManager(
        config=PerturbConfig(
            req_to_token_prob=1.0,
            real_kv_used_prob=0.0,
            real_kv_unused_cache_prob=0.0,
            target_group_kind=TargetGroupKind.FULL,
            warmup_steps=0,
        ),
        req_to_token_pool=pool,
        buffer_groups=(),
        step_counter_getter=lambda: 10,
    )
    forward_batch = make_forward_batch(device, bs=2, seq_lens_list=(3, 3))
    forward_batch.out_cache_loc = torch.tensor([11], dtype=torch.int32, device=device)

    snapshot = pool.req_to_token.clone()
    with patch.object(torch, "rand", return_value=torch.tensor(0.0)):
        manager.perturb_req_to_token(forward_batch)

    diff = pool.req_to_token != snapshot
    assert int(diff.sum().item()) == 1
    rows, cols = torch.nonzero(diff, as_tuple=True)
    row, col = int(rows[0].item()), int(cols[0].item())
    original = int(snapshot[row, col].item())
    replacement = int(pool.req_to_token[row, col].item())
    live_slots = {11, 22, 33, 44, 55, 66}
    assert original in live_slots
    assert replacement in live_slots
    assert replacement != original
    assert not bool(diff[1, 0].item())


def test_collect_active_slots_ignores_padded_out_cache_loc() -> None:
    """Verify out_cache_loc padding does not exclude a live slot."""
    device = DEFAULT_DEVICE
    pool = make_pool(device, max_reqs=4, max_seq=8)
    pool.req_to_token[1, :2] = torch.tensor([0, 7], dtype=torch.int32, device=device)
    forward_batch = make_forward_batch(device, bs=1, seq_lens_list=(2,))
    forward_batch.out_cache_loc = torch.tensor([7, 0, 0], dtype=torch.int32, device=device)
    forward_batch.num_token_non_padded_cpu = 1

    targets = collect_active_slots(
        forward_batch=forward_batch,
        req_to_token_pool=pool,
    )

    assert [target.value for target in targets] == [0]


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
