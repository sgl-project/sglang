from __future__ import annotations

import unittest
from typing import cast
from unittest.mock import patch

import torch
from kv_canary_runner_unit_utils import make_forward_batch, make_pool

from sglang.jit_kernel.kv_canary.verify import CANARY_SLOT_BYTES, RealKvSource
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.perturb import (
    real_kv_post_forward,
)
from sglang.srt.kv_canary.perturb import (
    real_kv_unused_cache as real_kv_unused_cache_module,
)
from sglang.srt.kv_canary.perturb.config import (
    PerturbConfig,
    TargetGroupKind,
    _parse_target_group_kind,
)
from sglang.srt.kv_canary.perturb.manager import PerturbManager
from sglang.srt.kv_canary.perturb.slot_picker import collect_active_slots
from sglang.srt.kv_canary.perturb.utils import (
    WarmupGate,
    flip_first_byte_in_source,
    pick_target_group,
)
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
    def test_perturb_manager_end_of_forward_dispatches_real_kv_post_forward(
        self,
    ) -> None:
        """Verify end_of_forward() routes only to perturb_real_kv_post_forward."""
        device = DEFAULT_DEVICE
        manager = PerturbManager(
            config=PerturbConfig(
                req_to_token_prob=0.0,
                real_kv_used_prob=0.0,
                real_kv_unused_cache_prob=0.0,
                real_kv_post_forward_prob=0.0,
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
            "perturb_real_kv_post_forward",
            lambda batch: calls.append("real_kv_post_forward"),
        ), patch.object(
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
            manager.end_of_forward(forward_batch)

        self.assertEqual(calls, ["real_kv_post_forward"])


class TestRealKvPostForwardPerturb(CustomTestCase):
    def test_real_kv_post_forward_flips_a_byte_in_out_cache_loc_slot(self) -> None:
        """Verify post-forward perturbation flips one real-KV byte and leaves canary buffers
        untouched."""
        device = DEFAULT_DEVICE
        source = RealKvSource(
            tensor=torch.zeros(4, 16, dtype=torch.uint8, device=device),
            page_size=1,
            num_bytes_per_token=16,
            read_bytes=16,
        )
        group = CanaryBufferGroup(
            kind=PoolKind.FULL,
            k_head=torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device),
            k_tail=torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device),
            v_head=torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device),
            v_tail=torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device),
            real_kv_sources_k=(source,),
            real_kv_sources_v=(source,),
            swa_index_lut=None,
        )
        config = PerturbConfig(
            req_to_token_prob=0.0,
            real_kv_used_prob=0.0,
            real_kv_unused_cache_prob=0.0,
            real_kv_post_forward_prob=1.0,
            target_group_kind=TargetGroupKind.FULL,
            warmup_steps=0,
        )
        warmup_gate = WarmupGate(config=config, step_counter_getter=lambda: 10)

        forward_batch = make_forward_batch(device, bs=1, seq_lens_list=(1,))
        forward_batch.out_cache_loc = torch.tensor(
            [2], dtype=torch.int32, device=device
        )
        forward_batch.num_token_non_padded_cpu = 1

        head_snapshot = group.k_head.clone()
        v_head_snapshot = group.v_head.clone()
        k_tail_snapshot = group.k_tail.clone()
        v_tail_snapshot = group.v_tail.clone()
        source_snapshot = source.tensor.clone()

        with patch.object(torch, "rand", return_value=torch.tensor(0.0)):
            real_kv_post_forward.run(
                forward_batch=forward_batch,
                config=config,
                buffer_groups=(group,),
                warmup_gate=warmup_gate,
            )

        diff = source.tensor != source_snapshot
        self.assertEqual(int(diff.sum().item()), 1)
        self.assertTrue(bool(diff[2, 0].item()))
        self.assertEqual(int(source.tensor[2, 0].item()), 0 ^ 0xFF)
        self.assertTrue(torch.equal(group.k_head, head_snapshot))
        self.assertTrue(torch.equal(group.v_head, v_head_snapshot))
        self.assertTrue(torch.equal(group.k_tail, k_tail_snapshot))
        self.assertTrue(torch.equal(group.v_tail, v_tail_snapshot))


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
                real_kv_post_forward_prob=0.0,
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


class TestRealKvUsedPerturb(CustomTestCase):
    def test_real_kv_used_flips_first_real_kv_byte_for_active_full_slot(
        self,
    ) -> None:
        """Verify real_kv_used flips only the first real KV byte for an active FULL slot."""
        device = DEFAULT_DEVICE
        pool = make_pool(device, max_reqs=4, max_seq=8)
        pool.req_to_token.fill_(-1)
        pool.req_to_token[1, 0] = 2
        group = _make_group(kind=PoolKind.FULL, has_real_kv=True)
        source = group.real_kv_sources_k[0]
        source.tensor.copy_(
            torch.arange(source.tensor.numel(), dtype=torch.uint8).view_as(
                source.tensor
            )
        )
        manager = PerturbManager(
            config=PerturbConfig(
                req_to_token_prob=0.0,
                real_kv_used_prob=1.0,
                real_kv_unused_cache_prob=0.0,
                real_kv_post_forward_prob=0.0,
                target_group_kind=TargetGroupKind.FULL,
                warmup_steps=0,
            ),
            req_to_token_pool=pool,
            buffer_groups=(group,),
            step_counter_getter=lambda: 10,
        )
        forward_batch = make_forward_batch(device, bs=1, seq_lens_list=(1,))
        forward_batch.out_cache_loc = torch.tensor(
            [99], dtype=torch.int32, device=device
        )

        snapshot = source.tensor.clone()
        with patch.object(torch, "rand", return_value=torch.tensor(0.0)):
            manager.perturb_real_kv_used(forward_batch)

        expected = snapshot.clone()
        expected[2, 0] = int(snapshot[2, 0].item()) ^ 0xFF
        self.assertTrue(torch.equal(source.tensor, expected))

    def test_flip_first_byte_in_source_maps_swa_logical_slot_through_lut(
        self,
    ) -> None:
        """Verify SWA groups map logical slots through swa_index_lut before flipping bytes."""
        source = RealKvSource(
            tensor=torch.arange(32, dtype=torch.uint8).view(2, 16),
            page_size=2,
            num_bytes_per_token=8,
            read_bytes=8,
        )
        group = _make_group(
            kind=PoolKind.SWA,
            has_real_kv=True,
            source=source,
            swa_index_lut=torch.tensor([0, 3], dtype=torch.int32),
        )

        snapshot = source.tensor.clone()
        result = flip_first_byte_in_source(group=group, source=source, slot_idx=1)

        self.assertEqual(result, (1, 8, int(snapshot[1, 8].item())))
        expected = snapshot.clone()
        expected[1, 8] = int(snapshot[1, 8].item()) ^ 0xFF
        self.assertTrue(torch.equal(source.tensor, expected))

    def test_warmup_gate_prevents_perturbation_when_probabilities_are_one(self) -> None:
        """Verify warmup prevents all perturbations even when every probability is one."""
        device = DEFAULT_DEVICE
        pool = make_pool(device, max_reqs=4, max_seq=8)
        pool.req_to_token.fill_(-1)
        pool.req_to_token[1, 0] = 2
        group = _make_group(kind=PoolKind.FULL, has_real_kv=True)
        source = group.real_kv_sources_k[0]
        source.tensor.copy_(
            torch.arange(source.tensor.numel(), dtype=torch.uint8).view_as(
                source.tensor
            )
        )
        manager = PerturbManager(
            config=PerturbConfig(
                req_to_token_prob=1.0,
                real_kv_used_prob=1.0,
                real_kv_unused_cache_prob=1.0,
                real_kv_post_forward_prob=0.0,
                target_group_kind=TargetGroupKind.FULL,
                warmup_steps=20,
            ),
            req_to_token_pool=pool,
            buffer_groups=(group,),
            step_counter_getter=lambda: 10,
        )
        manager.attach_radix_cache(cast("BasePrefixCache", object()))
        forward_batch = make_forward_batch(device, bs=1, seq_lens_list=(1,))
        forward_batch.out_cache_loc = torch.tensor(
            [99], dtype=torch.int32, device=device
        )

        pool_snapshot = pool.req_to_token.clone()
        source_snapshot = source.tensor.clone()
        with patch.object(torch, "rand", return_value=torch.tensor(0.0)), patch.object(
            real_kv_unused_cache_module,
            "pick_orphan_slot",
            return_value=3,
        ):
            manager.perturb(forward_batch)

        self.assertTrue(torch.equal(pool.req_to_token, pool_snapshot))
        self.assertTrue(torch.equal(source.tensor, source_snapshot))


class TestRealKvUnusedCachePerturb(CustomTestCase):
    def test_real_kv_unused_cache_flips_first_real_kv_byte_for_orphan_slot(
        self,
    ) -> None:
        """Verify real_kv_unused_cache flips only the first real KV byte for an orphan slot."""
        device = DEFAULT_DEVICE
        pool = make_pool(device, max_reqs=4, max_seq=8)
        group = _make_group(kind=PoolKind.FULL, has_real_kv=True)
        source = group.real_kv_sources_k[0]
        source.tensor.copy_(
            torch.arange(source.tensor.numel(), dtype=torch.uint8).view_as(
                source.tensor
            )
        )
        manager = PerturbManager(
            config=PerturbConfig(
                req_to_token_prob=0.0,
                real_kv_used_prob=0.0,
                real_kv_unused_cache_prob=1.0,
                real_kv_post_forward_prob=0.0,
                target_group_kind=TargetGroupKind.FULL,
                warmup_steps=0,
            ),
            req_to_token_pool=pool,
            buffer_groups=(group,),
            step_counter_getter=lambda: 10,
        )
        manager.attach_radix_cache(cast("BasePrefixCache", object()))

        snapshot = source.tensor.clone()
        with patch.object(torch, "rand", return_value=torch.tensor(0.0)), patch.object(
            real_kv_unused_cache_module,
            "pick_orphan_slot",
            return_value=3,
        ):
            manager.perturb_real_kv_unused_cache(None)

        expected = snapshot.clone()
        expected[3, 0] = int(snapshot[3, 0].item()) ^ 0xFF
        self.assertTrue(torch.equal(source.tensor, expected))

    def test_real_kv_unused_cache_skips_without_radix_cache_when_forward_batch_is_none(
        self,
    ) -> None:
        """Verify unused-cache perturbation accepts no forward batch but skips without radix_cache."""
        device = DEFAULT_DEVICE
        group = _make_group(kind=PoolKind.FULL, has_real_kv=True)
        source = group.real_kv_sources_k[0]
        source.tensor.copy_(
            torch.arange(source.tensor.numel(), dtype=torch.uint8).view_as(
                source.tensor
            )
        )
        manager = PerturbManager(
            config=PerturbConfig(
                req_to_token_prob=0.0,
                real_kv_used_prob=0.0,
                real_kv_unused_cache_prob=1.0,
                real_kv_post_forward_prob=0.0,
                target_group_kind=TargetGroupKind.FULL,
                warmup_steps=0,
            ),
            req_to_token_pool=make_pool(device),
            buffer_groups=(group,),
            step_counter_getter=lambda: 10,
        )

        snapshot = source.tensor.clone()
        with patch.object(torch, "rand", return_value=torch.tensor(0.0)):
            manager.perturb_real_kv_unused_cache(None)

        self.assertTrue(torch.equal(source.tensor, snapshot))


def _make_group(
    *,
    kind: PoolKind,
    has_real_kv: bool,
    source: RealKvSource | None = None,
    swa_index_lut: torch.Tensor | None = None,
) -> CanaryBufferGroup:
    source = source or RealKvSource(
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
        swa_index_lut=swa_index_lut,
    )


if __name__ == "__main__":
    unittest.main()
