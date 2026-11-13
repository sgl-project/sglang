"""Startup validation for UNO configuration."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.speculative_hook import _handle_uno
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _xpu_args(*, decode_graph_backend, attention_backend="intel_xpu", tree=False):
    """Startup args for UNO on XPU: linear 8/1/8, or tree 8/16/8."""
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
        speculative_num_steps=7 if tree else 1,
        speculative_eagle_topk=16 if tree else 1,
        enable_two_batch_overlap=False,
        speculative_accept_threshold_single=1.0,
        speculative_accept_threshold_acc=1.0,
        tp_size=1,
        pp_size=1,
        attn_cp_size=1,
        enable_lora=False,
        lora_paths=None,
        speculative_use_rejection_sampling=False,
        enable_mixed_chunk=False,
        attention_backend=attention_backend,
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

    def test_supported_xpu_configs_start(self):
        """Linear and tree UNO start on either XPU attention backend, with or
        without XPU decode graphs."""
        for tree in (False, True):
            for decode_graph_backend in (Backend.DISABLED, Backend.FULL):
                for attention_backend in ("intel_xpu", "triton"):
                    with self.subTest(
                        tree=tree,
                        decode_graph_backend=decode_graph_backend,
                        attention_backend=attention_backend,
                    ):
                        self._handle(
                            _xpu_args(
                                decode_graph_backend=decode_graph_backend,
                                attention_backend=attention_backend,
                                tree=tree,
                            )
                        )


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
