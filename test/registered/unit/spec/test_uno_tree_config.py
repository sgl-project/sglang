"""Startup validation for UNO configuration."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.speculative_hook import _handle_uno
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _xpu_linear_args(decode_graph_backend):
    return SimpleNamespace(
        device="xpu",
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(backend=decode_graph_backend)
        ),
        enable_deterministic_inference=False,
        enable_strict_thinking=False,
        speculative_draft_model_path=None,
        uno_lora_path="/tmp/uno-lora",
        speculative_num_draft_tokens=8,
        speculative_num_steps=1,
        speculative_eagle_topk=1,
        tp_size=1,
        pp_size=1,
        attn_cp_size=1,
        enable_lora=False,
        lora_paths=None,
        speculative_use_rejection_sampling=False,
        enable_mixed_chunk=False,
        attention_backend="intel_xpu",
        prefill_attention_backend=None,
        decode_attention_backend=None,
    )


class TestUnoXpuConfig(CustomTestCase):
    def _handle(self, server_args):
        with (
            patch(
                "sglang.srt.arg_groups.speculative_hook.resolving_view",
                side_effect=lambda args: args,
            ),
            patch(
                "sglang.srt.arg_groups.speculative_hook.resolved_view",
                side_effect=lambda args: args,
            ),
            patch(
                "sglang.srt.arg_groups.speculative_hook.attn_dp_enabled_of",
                return_value=False,
            ),
            patch("sglang.srt.arg_groups.speculative_hook.declare_resolution"),
        ):
            _handle_uno(server_args)

    def test_eager_intel_xpu_is_accepted(self):
        """XPU with its native attention backend and no decode graph starts."""
        self._handle(_xpu_linear_args(Backend.DISABLED))

    def test_decode_graph_is_rejected(self):
        """The XPU graph runner would replay UNO's draft without its LoRA."""
        with self.assertRaisesRegex(ValueError, "decode graphs"):
            self._handle(_xpu_linear_args(Backend.FULL))


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
