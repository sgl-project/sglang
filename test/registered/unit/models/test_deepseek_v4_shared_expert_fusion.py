import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import sglang.srt.models.deepseek_v4 as deepseek_v4
from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.layers.moe.utils import (
    install_shared_experts_fusion_decision,
    is_shared_experts_fusion_disabled,
)
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
from sglang.srt.runtime_context import get_context, get_exec, get_flags
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestDeepseekV4SharedExpertFusionPolicy(CustomTestCase):
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

    def _install(
        self,
        model_class=DeepseekV4ForCausalLM,
        n_shared_experts=1,
        hf_config=None,
        quant_config=None,
    ):
        install_shared_experts_fusion_decision(
            model_class,
            hf_config or SimpleNamespace(n_shared_experts=n_shared_experts),
            quant_config,
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
                deepseek_v4.envs.SGLANG_USE_AITER_MOE_GU_ITLV,
                "get",
                return_value=True,
            )
        )
        return stack

    def _make_dspark_config(self):
        return DeepSeekV4Config(
            architectures=["DeepseekV4ForCausalLMDSpark"],
            quantization_config={},
            rope_scaling={},
            compress_ratios=[],
            n_shared_experts=1,
            dspark_markov_rank=1,
            num_nextn_predict_layers=1,
        )

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

    def test_dspark_entry_class_uses_the_v4_gate(self):
        """A DSV4 DSpark draft must inherit the target's default fusion policy."""
        self._publish(enforce=False)

        self._install(DeepseekV4ForCausalLMDSpark)

        self.assertTrue(is_shared_experts_fusion_disabled())

    def test_dspark_records_explicitly_forced_fusion(self):
        """A forced DSpark build must retain its fused shared-expert count."""
        self._publish(enforce=True)
        with self._fhmoe_environment():
            self._install(
                DeepseekV4ForCausalLMDSpark,
                hf_config=self._valid_config(),
                quant_config=self._valid_quant_config(),
            )

        class Stage(nn.Module):
            def __init__(self, **_kwargs):
                super().__init__()

        class MarkovHead(nn.Module):
            def __init__(self, **_kwargs):
                super().__init__()

        with (
            patch(
                "sglang.srt.models.deepseek_v4_dspark.DSparkV4Stage",
                Stage,
            ),
            patch(
                "sglang.srt.models.deepseek_v4_dspark.DSparkV4MarkovHead",
                MarkovHead,
            ),
            patch(
                "sglang.srt.models.deepseek_v4_dspark.build_dspark_v4_confidence_head",
                return_value=None,
            ),
        ):
            model = DeepseekV4ForCausalLMDSpark(self._make_dspark_config())

        self.assertEqual(model.num_fused_shared_experts, 1)

    def test_dspark_loads_forced_shared_expert_into_fused_slot(self):
        """Forced DSpark shared tensors must load instead of being skipped."""
        config = self._make_dspark_config()
        loaded = []

        class Param:
            def weight_loader(
                self,
                _param,
                loaded_weight,
                candidate,
                *,
                shard_id,
                expert_id,
            ):
                loaded.append((loaded_weight, candidate, shard_id, expert_id))

        class DraftModel:
            num_fused_shared_experts = 1
            confidence_head = None

            def __init__(self):
                self.config = config

            def named_parameters(self):
                return [("stages.0.mlp.experts.w13_weight", Param())]

            def _remap_dspark_weight_name(self, name):
                return DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name(self, name)

            def _assert_confidence_head_loaded(self, **_kwargs):
                return None

        weight = torch.ones(1)
        DeepseekV4ForCausalLMDSpark.load_weights(
            DraftModel(),
            [("mtp.0.ffn.shared_experts.w1.weight", weight)],
        )

        self.assertEqual(len(loaded), 1)
        self.assertIs(loaded[0][0], weight)
        self.assertEqual(loaded[0][1], "stages.0.mlp.experts.w13_weight")
        self.assertEqual(loaded[0][2:], ("w1", config.n_routed_experts))


if __name__ == "__main__":
    unittest.main()
