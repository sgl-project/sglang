from __future__ import annotations

import os
import unittest
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import torch

from sglang.jit_kernel.kv_canary.verify import RealKvSource
from sglang.srt.kv_canary.buffer_group import PoolKind
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
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kv_canary.fixtures import (
    DEFAULT_DEVICE,
    make_buffer_group,
    make_forward_batch,
    make_radix_cache,
    make_req_to_token_pool,
)
from sglang.test.test_utils import CustomTestCase

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

register_cuda_ci(est_time=9, stage="extra-a", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="extra-a-test-1-gpu-small-amd")


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

    def test_from_env_allows_missing_target_when_real_kv_perturb_is_disabled(
        self,
    ) -> None:
        """Verify normal canary startup does not require a perturb target group."""
        with patch.dict(
            os.environ,
            {
                "SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_PROB": "0",
                "SGLANG_KV_CANARY_PERTURB_REAL_KV_USED_PROB": "0",
                "SGLANG_KV_CANARY_PERTURB_REAL_KV_UNUSED_CACHE_PROB": "0",
                "SGLANG_KV_CANARY_PERTURB_REAL_KV_POST_FORWARD_PROB": "0",
            },
            clear=False,
        ):
            os.environ.pop("SGLANG_KV_CANARY_PERTURB_TARGET_GROUP", None)
            config = PerturbConfig.from_env()

        self.assertIsNone(config.target_group_kind)


class TestPickTargetGroup(CustomTestCase):
    def test_pick_target_group_filters_exact_kind(self) -> None:
        """Verify target group selection returns only the requested pool kind."""
        cases = [
            (TargetGroupKind.FULL, PoolKind.FULL),
            (TargetGroupKind.SWA, PoolKind.SWA),
        ]

        for target_kind, expected_kind in cases:
            with self.subTest(target_kind=target_kind):
                full_group = make_buffer_group(kind=PoolKind.FULL, has_real_kv=True)
                swa_group = make_buffer_group(kind=PoolKind.SWA, has_real_kv=True)

                group = pick_target_group(
                    buffer_groups=(full_group, swa_group),
                    target_kind=target_kind,
                )

                self.assertIsNotNone(group)
                self.assertEqual(group.kind, expected_kind)

    def test_pick_target_group_rejects_unsupported_kind(self) -> None:
        """Verify target group selection rejects unsupported enum values."""
        full_group = make_buffer_group(kind=PoolKind.FULL, has_real_kv=True)

        with self.assertRaisesRegex(ValueError, "Unsupported target_group_kind"):
            pick_target_group(
                buffer_groups=(full_group,),
                target_kind=cast(TargetGroupKind, 2),
            )

    def test_pick_target_group_ignores_groups_without_real_kv_sources(self) -> None:
        """Verify target group selection skips groups without real KV sources."""
        full_group = make_buffer_group(kind=PoolKind.FULL, has_real_kv=False)
        swa_group = make_buffer_group(kind=PoolKind.SWA, has_real_kv=True)

        group = pick_target_group(
            buffer_groups=(full_group, swa_group),
            target_kind=TargetGroupKind.FULL,
        )

        self.assertIsNone(group)


class TestPerturbManager(CustomTestCase):
    def test_perturb_manager_perturb_post_forward_dispatches_real_kv_post_forward(
        self,
    ) -> None:
        """Verify perturb_post_forward() routes only to the post_forward dispatch."""
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
            req_to_token_pool=make_req_to_token_pool(device, max_reqs=4, max_seq_len=8),
            buffer_groups=(),
            outer_step_counter_getter=lambda: 10,
        )
        forward_batch = make_forward_batch(device, bs=1, seq_lens_list=(1,))
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
            manager.perturb_post_forward(maybe_inaccurate_forward_batch=forward_batch)

        self.assertEqual(calls, ["real_kv_post_forward"])


class TestRealKvPostForwardPerturb(CustomTestCase):
    def test_real_kv_post_forward_flips_a_byte_in_out_cache_loc_slot(self) -> None:
        """Verify post-forward perturbation flips one real-KV byte and leaves canary buffers untouched."""
        device = DEFAULT_DEVICE
        group = make_buffer_group(kind=PoolKind.FULL, has_real_kv=True)
        source = group.real_kv_sources_k[0]
        config = PerturbConfig(
            req_to_token_prob=0.0,
            real_kv_used_prob=0.0,
            real_kv_unused_cache_prob=0.0,
            real_kv_post_forward_prob=1.0,
            target_group_kind=TargetGroupKind.FULL,
            warmup_steps=0,
        )
        warmup_gate = WarmupGate(config=config, outer_step_counter_getter=lambda: 10)

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
                maybe_inaccurate_forward_batch=forward_batch,
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
        pool = make_req_to_token_pool(device, max_reqs=4, max_seq_len=8)
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
            outer_step_counter_getter=lambda: 10,
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
        pool = make_req_to_token_pool(device, max_reqs=4, max_seq_len=8)
        pool.req_to_token[1, :2] = torch.tensor(
            [0, 7], dtype=torch.int32, device=device
        )
        forward_batch = make_forward_batch(device, bs=1, seq_lens_list=(2,))
        forward_batch.out_cache_loc = torch.tensor(
            [7, 0, 0], dtype=torch.int32, device=device
        )
        forward_batch.num_token_non_padded_cpu = 1

        targets = collect_active_slots(
            maybe_inaccurate_forward_batch=forward_batch,
            req_to_token_pool=pool,
        )

        self.assertEqual([target.value for target in targets], [0])


class TestRealKvUsedPerturb(CustomTestCase):
    def test_real_kv_used_flips_first_real_kv_byte_for_active_full_slot(
        self,
    ) -> None:
        """Verify real_kv_used flips only the first real KV byte for an active FULL slot."""
        device = DEFAULT_DEVICE
        pool = make_req_to_token_pool(device, max_reqs=4, max_seq_len=8)
        pool.req_to_token.fill_(-1)
        pool.req_to_token[1, 0] = 2
        group = make_buffer_group(kind=PoolKind.FULL, has_real_kv=True)
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
            outer_step_counter_getter=lambda: 10,
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
            tensor=torch.arange(64, dtype=torch.uint8).view(2, 32),
            page_size=2,
            num_bytes_per_token=16,
            read_bytes=16,
        )
        group = make_buffer_group(
            kind=PoolKind.SWA,
            has_real_kv=True,
            real_kv_source=source,
            swa_index_lut=torch.tensor([0, 3], dtype=torch.int32),
        )

        snapshot = source.tensor.clone()
        result = flip_first_byte_in_source(group=group, source=source, slot_idx=1)

        self.assertEqual(result, (1, 16, int(snapshot[1, 16].item())))
        expected = snapshot.clone()
        expected[1, 16] = int(snapshot[1, 16].item()) ^ 0xFF
        self.assertTrue(torch.equal(source.tensor, expected))

    def test_warmup_gate_prevents_perturbation_when_probabilities_are_one(self) -> None:
        """Verify warmup prevents all perturbations even when every probability is one."""
        device = DEFAULT_DEVICE
        pool = make_req_to_token_pool(device, max_reqs=4, max_seq_len=8)
        pool.req_to_token.fill_(-1)
        pool.req_to_token[1, 0] = 2
        group = make_buffer_group(kind=PoolKind.FULL, has_real_kv=True)
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
            outer_step_counter_getter=lambda: 10,
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
            "_pick_sweep_slot_for_group",
            return_value=3,
        ):
            manager.perturb(maybe_inaccurate_forward_batch=forward_batch)

        self.assertTrue(torch.equal(pool.req_to_token, pool_snapshot))
        self.assertTrue(torch.equal(source.tensor, source_snapshot))


class TestRealKvUnusedCachePerturb(CustomTestCase):
    def test_real_kv_unused_cache_flips_first_real_kv_byte_for_orphan_slot(
        self,
    ) -> None:
        """Verify real_kv_unused_cache flips only the first real KV byte for an orphan slot."""
        device = DEFAULT_DEVICE
        pool = make_req_to_token_pool(device, max_reqs=4, max_seq_len=8)
        group = make_buffer_group(kind=PoolKind.FULL, has_real_kv=True)
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
            outer_step_counter_getter=lambda: 10,
            sweep_interval=1,
        )
        manager.attach_radix_cache(make_radix_cache([[], [3]], device=device))

        snapshot = source.tensor.clone()
        with patch.object(torch, "rand", return_value=torch.tensor(0.0)), patch.object(
            torch,
            "randint",
            return_value=torch.tensor(0),
        ):
            manager.perturb_real_kv_unused_cache(None)

        expected = snapshot.clone()
        expected[3, 0] = int(snapshot[3, 0].item()) ^ 0xFF
        self.assertTrue(torch.equal(source.tensor, expected))

    def test_pick_sweep_slot_for_group_skips_locked_radix_nodes(self) -> None:
        """Verify unused-cache perturbation chooses only unlocked radix-cache slots."""
        device = DEFAULT_DEVICE
        group = make_buffer_group(kind=PoolKind.FULL, has_real_kv=True)
        cache = make_radix_cache([[], [1, 2], [3]], device=device)
        locked_node = next(iter(cache.root_node.children.values()))
        locked_node.lock_ref = 1

        with patch.object(torch, "randint", return_value=torch.tensor(0)):
            slot = real_kv_unused_cache_module._pick_sweep_slot_for_group(
                radix_cache=cache,
                group=group,
                swa_window_size=0,
            )

        self.assertEqual(slot, 3)

    def test_pick_sweep_slot_for_group_translates_swa_slots(self) -> None:
        """Verify unused-cache SWA perturbation translates full slots to physical SWA slots."""
        device = DEFAULT_DEVICE
        lut = torch.tensor([-1, 2], dtype=torch.int64, device=device)
        group = make_buffer_group(
            kind=PoolKind.SWA, has_real_kv=True, swa_index_lut=lut
        )
        cache = make_radix_cache([[], [1]], device=device)

        with patch.object(torch, "randint", return_value=torch.tensor(0)):
            slot = real_kv_unused_cache_module._pick_sweep_slot_for_group(
                radix_cache=cache,
                group=group,
                swa_window_size=4,
            )

        self.assertEqual(slot, 2)

    def test_real_kv_unused_cache_skips_without_radix_cache_when_forward_batch_is_none(
        self,
    ) -> None:
        """Verify unused-cache perturbation accepts no forward batch but skips without radix_cache."""
        device = DEFAULT_DEVICE
        group = make_buffer_group(kind=PoolKind.FULL, has_real_kv=True)
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
            req_to_token_pool=make_req_to_token_pool(device, max_reqs=4, max_seq_len=8),
            buffer_groups=(group,),
            outer_step_counter_getter=lambda: 10,
            sweep_interval=1,
        )

        snapshot = source.tensor.clone()
        with patch.object(torch, "rand", return_value=torch.tensor(0.0)):
            manager.perturb_real_kv_unused_cache(None)

        self.assertTrue(torch.equal(source.tensor, snapshot))


class TestPerturbUtils(CustomTestCase):
    def test_flip_first_byte_in_physical_swa_slot_does_not_translate_twice(
        self,
    ) -> None:
        """Verify a physical SWA slot selected from sweep is not LUT-translated again."""
        group = make_buffer_group(kind=PoolKind.SWA, has_real_kv=True)
        source = group.real_kv_sources_k[0]
        source.tensor[2, 0] = 0x12
        source.tensor[3, 0] = 0x34

        result = flip_first_byte_in_source(
            group=group,
            source=source,
            slot_idx=2,
            slot_is_physical=True,
        )

        self.assertEqual(result, (2, 0, 0x12))
        self.assertEqual(int(source.tensor[2, 0].item()), 0xED)
        self.assertEqual(int(source.tensor[3, 0].item()), 0x34)


if __name__ == "__main__":
    unittest.main()
