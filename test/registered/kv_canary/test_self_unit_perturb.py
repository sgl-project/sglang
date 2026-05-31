from __future__ import annotations

import os
import unittest
from typing import cast
from unittest.mock import patch

import torch

from sglang.jit_kernel.kv_canary.verify import RealKvSource
from sglang.srt.kv_canary.buffer_group import PoolKind
from sglang.srt.kv_canary.perturb.config import (
    PerturbConfig,
    TargetGroupKind,
    _parse_target_group_kind,
)
from sglang.srt.kv_canary.perturb.utils import (
    flip_first_byte_in_source,
    pick_target_group,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.fixtures import (
    make_buffer_group,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="extra-a", runner_config="1-gpu-small")


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

    def test_from_env_allows_missing_target(
        self,
    ) -> None:
        """Verify normal canary startup does not require a perturb target group."""
        with patch.dict(os.environ, {}, clear=False):
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


class TestPerturbWarmupAndUtils(CustomTestCase):
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
