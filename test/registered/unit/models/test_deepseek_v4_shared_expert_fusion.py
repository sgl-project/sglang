import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.models.deepseek_v4 as deepseek_v4
from sglang.srt.layers.moe.utils import (
    install_shared_experts_fusion_decision,
    is_shared_experts_fusion_disabled,
)
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.runtime_context import get_context, get_exec, get_flags
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestDeepseekV4SharedExpertFusionPolicy(unittest.TestCase):
    """V4 fuses its shared expert only when explicitly asked to.

    The gate is a question the loader asks the model class before any layer
    exists (``shared_experts_fusion_disable_reason``); the answer is installed
    on the ACTIVE moe flag, and the config bag keeps the user's intent.
    """

    def _publish(self, enforce):
        override = get_context().override_server_args(
            enforce_shared_experts_fusion=enforce
        )
        override.install()
        self.addCleanup(override.restore)
        get_flags().moe.disable_shared_experts_fusion = None
        self.addCleanup(
            lambda: setattr(get_flags().moe, "disable_shared_experts_fusion", None)
        )

    def _install(self, n_shared_experts=1):
        install_shared_experts_fusion_decision(
            DeepseekV4ForCausalLM,
            SimpleNamespace(n_shared_experts=n_shared_experts),
            None,
        )

    def _valid_config(self):
        return SimpleNamespace(
            n_shared_experts=1,
            n_routed_experts=384,
            num_experts_per_tok=6,
            hidden_size=7168,
            moe_intermediate_size=3072,
            hidden_act="silu",
        )

    def _valid_quant_config(self):
        return SimpleNamespace(
            is_fp4_experts=True,
            is_checkpoint_fp8_serialized=True,
            weight_block_size=[128, 128],
            ignored_layers=[],
        )

    def _fhmoe_environment(self):
        stack = ExitStack()
        stack.enter_context(patch.object(deepseek_v4, "_is_hip", True))
        stack.enter_context(patch.object(deepseek_v4, "_use_aiter", True))
        stack.enter_context(
            patch.object(deepseek_v4, "is_gfx95_supported", return_value=True)
        )
        stack.enter_context(
            patch.object(
                deepseek_v4,
                "aiter_fused_moe_supports_heterogeneous_shared_expert",
                return_value=True,
            )
        )
        stack.enter_context(
            patch.object(
                deepseek_v4,
                "get_moe_runner_backend",
                return_value=SimpleNamespace(
                    is_auto=lambda: True, is_aiter=lambda: False
                ),
            )
        )
        stack.enter_context(
            patch.object(
                deepseek_v4,
                "get_moe_a2a_backend",
                return_value=SimpleNamespace(is_none=lambda: True),
            )
        )
        stack.enter_context(
            patch.object(
                deepseek_v4,
                "get_parallel",
                return_value=SimpleNamespace(tp_size=8, moe_ep_size=1),
            )
        )
        stack.enter_context(
            patch.object(
                deepseek_v4,
                "get_server_args",
                return_value=SimpleNamespace(cpu_offload_gb=0),
            )
        )
        stack.enter_context(
            patch.object(
                deepseek_v4.envs.SGLANG_USE_AITER_MOE_GU_ITLV,
                "get",
                return_value=True,
            )
        )
        return stack

    def test_disables_shared_fusion_without_enforce(self):
        self._publish(enforce=False)
        self.assertEqual(
            DeepseekV4ForCausalLM.shared_experts_fusion_disable_reason(
                SimpleNamespace(n_shared_experts=1), None
            ),
            "Config does not support fused shared expert(s).",
        )
        self._install()
        # The decision lands on the ACTIVE flag; the config intent is untouched.
        self.assertTrue(is_shared_experts_fusion_disabled())
        self.assertFalse(get_exec().moe.disable_shared_experts_fusion)

    def test_enables_shared_fusion_when_enforced(self):
        self._publish(enforce=True)
        with self._fhmoe_environment():
            self.assertIsNone(
                DeepseekV4ForCausalLM.shared_experts_fusion_disable_reason(
                    self._valid_config(), self._valid_quant_config()
                )
            )

    def test_falls_back_when_aiter_fhmoe_abi_is_missing(self):
        self._publish(enforce=True)
        with self._fhmoe_environment(), patch.object(
            deepseek_v4,
            "aiter_fused_moe_supports_heterogeneous_shared_expert",
            return_value=False,
        ):
            reason = DeepseekV4ForCausalLM.shared_experts_fusion_disable_reason(
                self._valid_config(), self._valid_quant_config()
            )
        self.assertIn("does not expose", reason)

    def test_enforcing_with_more_than_one_shared_expert_is_rejected(self):
        self._publish(enforce=True)
        with self.assertRaisesRegex(ValueError, "exactly one shared"):
            DeepseekV4ForCausalLM.shared_experts_fusion_disable_reason(
                SimpleNamespace(n_shared_experts=2), None
            )


if __name__ == "__main__":
    unittest.main()
