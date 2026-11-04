"""Independent shared TP may span equal-padded PCP rows, never DP replicas."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import parallel_hook
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")


class TestKimiSharedExpertPCP(CustomTestCase):
    def validate(
        self,
        size,
        *,
        tp=8,
        cp=8,
        dp=1,
        enabled=True,
        strategy="interleave",
        backend="megamoe",
        arch="KimiK31ForConditionalGeneration",
    ):
        cfg = SimpleNamespace(
            shared_experts_tp_size=size,
            tp_size=tp,
            attn_cp_size=cp,
            attn_dp_size=dp,
            enable_prefill_cp=enabled,
            cp_strategy=strategy,
            model_path="dummy",
            moe_a2a_backend=backend,
        )
        model = SimpleNamespace(hf_config=SimpleNamespace(architectures=[arch]))
        with (
            patch.object(parallel_hook, "resolving_view", return_value=cfg),
            patch.object(parallel_hook, "resolved_view", return_value=cfg),
            patch.object(parallel_hook, "model_config_of", return_value=model),
        ):
            parallel_hook.handle_shared_experts_tp(cfg)

    def test_pcp8_accepts_shared_tp8(self):
        self.validate(8)

    def test_mixed_tp_cp_and_independent_dp_replicas(self):
        for size in (1, 2, 4, 8):
            with self.subTest(size=size):
                self.validate(size, tp=16, cp=4, dp=2)

    def test_cannot_cross_attention_dp_replica(self):
        with self.assertRaisesRegex(ValueError, "divisor"):
            self.validate(16, tp=16, cp=8, dp=2)

    def test_cross_cp_requires_equal_padded_interleave(self):
        for overrides in ({"enabled": False}, {"strategy": "zigzag"}):
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, "interleave"),
            ):
                self.validate(8, **overrides)

    def test_invalid_widths_rejected(self):
        for size in (0, -1, 3, 16):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "divisor"):
                self.validate(size)

    def test_legacy_attn_tp_and_default_unchanged(self):
        self.validate(None, enabled=False)
        for size in (1, 2, 4):
            self.validate(size, cp=1, dp=2, enabled=False)
        # A subgroup inside attention TP does not communicate across CP.
        self.validate(2, cp=4, enabled=False, strategy="zigzag")

    def test_model_and_backend_guards_remain(self):
        with self.assertRaisesRegex(ValueError, "Kimi-K3"):
            self.validate(8, arch="LlamaForCausalLM")
        with self.assertRaisesRegex(ValueError, "all-to-all"):
            self.validate(8, backend="none")


if __name__ == "__main__":
    unittest.main()
