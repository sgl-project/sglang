"""Startup validation for UNO configuration."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.speculative_hook import _handle_uno
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestUnoTreeConfig(CustomTestCase):
    def test_unsupported_runtime_modes_are_rejected_at_startup(self):
        cases = {
            "deterministic inference": (
                {"enable_deterministic_inference": True},
                "enable-deterministic-inference",
            ),
            "strict thinking": (
                {"enable_strict_thinking": True},
                "enable-strict-thinking",
            ),
        }

        for name, (overrides, expected) in cases.items():
            with self.subTest(name=name):
                values = {
                    "device": "cuda",
                    "speculative_draft_model_path": None,
                    "uno_lora_path": "/tmp/uno-lora",
                    "enable_deterministic_inference": False,
                    "enable_strict_thinking": False,
                }
                values.update(overrides)
                server_args = SimpleNamespace(**values)
                with (
                    patch(
                        "sglang.srt.arg_groups.speculative_hook.resolving_view",
                        side_effect=lambda args: args,
                    ),
                    self.assertRaisesRegex(ValueError, expected),
                ):
                    _handle_uno(server_args)

    def test_parent_list_overflow_is_rejected_at_startup(self):
        """An invalid tree must not survive startup and crash on first decode."""

        server_args = SimpleNamespace(
            device="cuda",
            enable_deterministic_inference=False,
            enable_strict_thinking=False,
            speculative_draft_model_path=None,
            uno_lora_path="/tmp/uno-lora",
            speculative_num_draft_tokens=8,
            speculative_num_steps=3,
            speculative_eagle_topk=2,
        )

        with (
            patch(
                "sglang.srt.arg_groups.speculative_hook.resolving_view",
                side_effect=lambda args: args,
            ),
            patch("sglang.srt.arg_groups.speculative_hook.declare_resolution"),
            self.assertRaisesRegex(ValueError, "parent-list ABI"),
        ):
            _handle_uno(server_args)


if __name__ == "__main__":
    unittest.main()
