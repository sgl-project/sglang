from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from sglang.srt.kv_canary.perturb.config import (
    PerturbConfig,
    TargetGroupKind,
    _parse_target_group_kind,
)
from sglang.test.ci.ci_register import register_cuda_ci
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


if __name__ == "__main__":
    unittest.main()
