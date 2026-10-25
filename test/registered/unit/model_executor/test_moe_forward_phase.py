"""MoE phase follows the MLP batch through graph and nested TBO contexts."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.batch_overlap.operations import (  # noqa: E402
    ExecutionOperation,
    _StageExecutor,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.model_executor.forward_context import (  # noqa: E402
    ForwardContext,
    forward_context,
    get_forward_context,
    has_forward_context,
    moe_forward_phase_from_batch,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _batch(mode, *, global_phase=None):
    return SimpleNamespace(
        forward_mode=mode,
        global_forward_mode=None,
        global_moe_forward_phase=global_phase,
        # Capture deliberately clears this communication flag.
        is_extend_in_batch=False,
        moe_num_token_non_padded=lambda: None,
        global_dp_buffer_len=4,
        tbo_padded_len=2,
        global_num_tokens_cpu=[4],
        dp_padding_mode=SimpleNamespace(is_max_len=lambda: True),
    )


class TestMoeForwardPhase(CustomTestCase):
    def test_forward_modes(self):
        prefill_modes = {
            ForwardMode.EXTEND,
            ForwardMode.MIXED,
            ForwardMode.SPLIT_PREFILL,
            ForwardMode.DLLM_EXTEND,
        }
        for mode in ForwardMode:
            with self.subTest(mode=mode):
                with forward_context(
                    ForwardContext(attn_backend=None), forward_batch=_batch(mode)
                ):
                    self.assertEqual(
                        get_forward_context().moe_forward_phase,
                        "prefill" if mode in prefill_modes else "decode",
                    )

    def test_mlp_global_phase_wins_over_local_decode_and_idle(self):
        for mode in (ForwardMode.DECODE, ForwardMode.IDLE, ForwardMode.TARGET_VERIFY):
            with self.subTest(mode=mode):
                self.assertEqual(
                    moe_forward_phase_from_batch(_batch(mode, global_phase="prefill")),
                    "prefill",
                )
        self.assertEqual(
            moe_forward_phase_from_batch(
                _batch(ForwardMode.DRAFT_EXTEND_V2, global_phase="prefill")
            ),
            "decode",
        )

    def test_nested_attention_override_preserves_phase_and_restores_on_error(self):
        with forward_context(
            ForwardContext(attn_backend=None), forward_batch=_batch(ForwardMode.EXTEND)
        ):
            parent = get_forward_context()
            with forward_context(ForwardContext(attn_backend=object())):
                self.assertEqual(get_forward_context().moe_forward_phase, "prefill")
            with self.assertRaisesRegex(RuntimeError, "child failed"):
                with forward_context(
                    ForwardContext(attn_backend=None),
                    forward_batch=_batch(ForwardMode.DECODE),
                ):
                    self.assertEqual(get_forward_context().moe_forward_phase, "decode")
                    raise RuntimeError("child failed")
            self.assertIs(get_forward_context(), parent)
        self.assertFalse(has_forward_context())

    def test_coordinated_speculative_stages_choose_the_same_phase_on_all_ranks(self):
        for mode in (
            ForwardMode.EXTEND,
            ForwardMode.DRAFT_EXTEND_V2,
            ForwardMode.TARGET_VERIFY,
            ForwardMode.IDLE,
        ):
            for phase in ("prefill", "decode"):
                with self.subTest(mode=mode, phase=phase):
                    batch = _batch(mode, global_phase=phase)
                    batch.dp_spec_prefill_coordination_applied = True
                    self.assertEqual(moe_forward_phase_from_batch(batch), phase)
        self.assertEqual(
            moe_forward_phase_from_batch(
                _batch(ForwardMode.IDLE, global_phase="decode")
            ),
            "decode",
        )

    def test_tbo_stage_binds_its_own_batch_phase(self):
        seen = []

        def record_phase(state, forward_batch):
            seen.append(get_forward_context().moe_forward_phase)
            return {"forward_batch": forward_batch}

        stages = [[ExecutionOperation("record", record_phase)]] * 2
        with (
            patch("sglang.srt.batch_overlap.operations.set_dp_buffer_len"),
            forward_context(
                ForwardContext(attn_backend=None),
                forward_batch=_batch(ForwardMode.EXTEND),
            ),
        ):
            a = _StageExecutor(
                "a", stages, {"forward_batch": _batch(ForwardMode.DECODE)}
            )
            b = _StageExecutor(
                "b", stages, {"forward_batch": _batch(ForwardMode.EXTEND)}
            )
            for executor in (a, b, a, b):
                executor.next()
            self.assertEqual(get_forward_context().moe_forward_phase, "prefill")
        self.assertEqual(seen, ["decode", "prefill", "decode", "prefill"])


if __name__ == "__main__":
    unittest.main()
