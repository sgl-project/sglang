"""Startup validation for UNO configuration."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.speculative_hook import _handle_uno
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _uno_args(**overrides):
    """UNO linear 8/1/8 on XPU with intel_xpu attention, before overrides."""
    args = dict(
        model_path="dummy",
        device="xpu",
        speculative_algorithm="UNO",
        uno_lora_path="/tmp/uno-lora",
        speculative_num_draft_tokens=8,
        speculative_num_steps=1,
        speculative_eagle_topk=1,
        attention_backend="intel_xpu",
    )
    args.update(overrides)
    return ServerArgs(**args)


_TREE = dict(speculative_num_steps=7, speculative_eagle_topk=16)


class TestUnoAttentionBackendConfig(CustomTestCase):
    def test_xpu_backends_start(self):
        """Linear and tree UNO start on both XPU attention backends."""
        for tree in (False, True):
            for attention_backend in ("intel_xpu", "triton"):
                with self.subTest(tree=tree, attention_backend=attention_backend):
                    _handle_uno(
                        _uno_args(
                            attention_backend=attention_backend,
                            **(_TREE if tree else {}),
                        )
                    )

    def test_unsupported_backends_are_rejected(self):
        """A backend outside the device's list, or a prefill/decode split,
        fails at startup instead of mis-running UNO's draft or verify."""
        for case, overrides in (
            ("fa3 on xpu", dict(attention_backend="fa3")),
            (
                "split prefill/decode",
                dict(
                    prefill_attention_backend="triton",
                    decode_attention_backend="intel_xpu",
                ),
            ),
            ("intel_xpu on cuda", dict(device="cuda")),
        ):
            with (
                self.subTest(case=case),
                self.assertRaisesRegex(ValueError, "same attention backend"),
            ):
                _handle_uno(_uno_args(**overrides))


class TestUnoTreeConfig(CustomTestCase):
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
