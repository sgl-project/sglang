from __future__ import annotations

import unittest
from typing import cast
from unittest.mock import patch

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
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="extra-a", runner_config="1-gpu-large")


class TestParseTargetGroupKind(CustomTestCase):
    def test_parse_target_group_kind_accepts_valid_values_case_insensitively(
        self,
    ) -> None:
        """Verify target group kind parsing accepts valid names case-insensitively."""
        cases = [
            ("full", TargetGroupKind.FULL),
            ("FULL", TargetGroupKind.FULL),
            (" swa ", TargetGroupKind.SWA),
        ]

        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(_parse_target_group_kind(raw), expected)

    def test_parse_target_group_kind_rejects_invalid_value(self) -> None:
        """Verify target group kind parsing rejects unknown names."""
        with self.assertRaisesRegex(ValueError, "must be one of"):
            _parse_target_group_kind("prefix")

    def test_parse_target_group_kind_rejects_missing_or_any(self) -> None:
        """Verify target group kind parsing requires an explicit concrete group."""
        for raw in [None, "", "any", " Any "]:
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(
                    ValueError, "SGLANG_KV_CANARY_PERTURB_TARGET_GROUP"
                ):
                    _parse_target_group_kind(raw)


class TestPickTargetGroup(CustomTestCase):
    def test_pick_target_group_filters_exact_kind(self) -> None:
        """Verify target group selection returns only the requested pool kind."""
        cases = [
            (TargetGroupKind.FULL, PoolKind.FULL),
            (TargetGroupKind.SWA, PoolKind.SWA),
        ]

        for target_kind, expected_kind in cases:
            with self.subTest(target_kind=target_kind):
                full_group = _make_group(kind=PoolKind.FULL, has_real_kv=True)
                swa_group = _make_group(kind=PoolKind.SWA, has_real_kv=True)

                group = pick_target_group(
                    buffer_groups=(full_group, swa_group),
                    target_kind=target_kind,
                )

                self.assertIsNotNone(group)
                self.assertEqual(group.kind, expected_kind)

    def test_pick_target_group_rejects_unsupported_kind(self) -> None:
        """Verify target group selection rejects unsupported enum values."""
        full_group = _make_group(kind=PoolKind.FULL, has_real_kv=True)

        with self.assertRaisesRegex(ValueError, "Unsupported target_group_kind"):
            pick_target_group(
                buffer_groups=(full_group,),
                target_kind=cast(TargetGroupKind, 2),
            )

    def test_pick_target_group_ignores_groups_without_real_kv_sources(self) -> None:
        """Verify target group selection skips groups without real KV sources."""
        full_group = _make_group(kind=PoolKind.FULL, has_real_kv=False)
        swa_group = _make_group(kind=PoolKind.SWA, has_real_kv=True)

        group = pick_target_group(
            buffer_groups=(full_group, swa_group),
            target_kind=TargetGroupKind.FULL,
        )

        self.assertIsNone(group)


class TestPerturbManager(CustomTestCase):
    def test_perturb_manager_perturb_dispatches_all_points(self) -> None:
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

        self.assertEqual(
            calls, ["req_to_token", "real_kv_used", "real_kv_unused_cache"]
        )


class TestReqToTokenPerturb(CustomTestCase):
    def test_req_to_token_perturb_uses_live_slot_as_replacement(self) -> None:
        """Verify req_to_token perturbation replaces a slot with another live slot."""
        device = DEFAULT_DEVICE
        pool = make_pool(device, max_reqs=4, max_seq=8)
        pool.req_to_token[1, :3] = torch.tensor(
            [11, 22, 33], dtype=torch.int32, device=device
        )
        pool.req_to_token[2, :3] = torch.tensor(
            [44, 55, 66], dtype=torch.int32, device=device
        )
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
        forward_batch.out_cache_loc = torch.tensor(
            [11], dtype=torch.int32, device=device
        )

        snapshot = pool.req_to_token.clone()
        with patch.object(torch, "rand", return_value=torch.tensor(0.0)):
            manager.perturb_req_to_token(forward_batch)

        diff = pool.req_to_token != snapshot
        self.assertEqual(int(diff.sum().item()), 1)
        rows, cols = torch.nonzero(diff, as_tuple=True)
        row, col = int(rows[0].item()), int(cols[0].item())
        original = int(snapshot[row, col].item())
        replacement = int(pool.req_to_token[row, col].item())
        live_slots = {11, 22, 33, 44, 55, 66}
        self.assertIn(original, live_slots)
        self.assertIn(replacement, live_slots)
        self.assertNotEqual(replacement, original)
        self.assertFalse(bool(diff[1, 0].item()))


class TestCollectActiveSlots(CustomTestCase):
    def test_collect_active_slots_ignores_padded_out_cache_loc(self) -> None:
        """Verify out_cache_loc padding does not exclude a live slot."""
        device = DEFAULT_DEVICE
        pool = make_pool(device, max_reqs=4, max_seq=8)
        pool.req_to_token[1, :2] = torch.tensor(
            [0, 7], dtype=torch.int32, device=device
        )
        forward_batch = make_forward_batch(device, bs=1, seq_lens_list=(2,))
        forward_batch.out_cache_loc = torch.tensor(
            [7, 0, 0], dtype=torch.int32, device=device
        )
        forward_batch.num_token_non_padded_cpu = 1

        targets = collect_active_slots(
            forward_batch=forward_batch,
            req_to_token_pool=pool,
        )

        self.assertEqual([target.value for target in targets], [0])


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


if __name__ == "__main__":
    unittest.main()
