import unittest
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.models.deepseek_v4 import _resolve_num_fused_shared_experts
from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
from sglang.srt.runtime_context import get_context, get_exec
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestDeepseekV4SharedExpertFusionPolicy(CustomTestCase):
    """The disable decision is a load-time resolution: it lands on the
    published config bag via declare_load_time_override (bag-only; the
    ServerArgs instance stays pristine)."""

    def _make_config(self, n_shared_experts=1):
        return DeepSeekV4Config(
            architectures=["DeepseekV4ForCausalLM"],
            quantization_config={},
            rope_scaling={},
            compress_ratios=[],
            n_shared_experts=n_shared_experts,
        )

    def _publish(self, enforce):
        override = get_context().override_server_args(
            enforce_shared_experts_fusion=enforce
        )
        override.install()
        self.addCleanup(override.restore)

    def test_disables_shared_fusion_without_enforce(self):
        """The default DSV4 policy disables unsupported shared-expert fusion."""
        self._publish(enforce=False)

        num_fused_shared_experts = _resolve_num_fused_shared_experts(
            self._make_config()
        )

        self.assertEqual(num_fused_shared_experts, 0)
        # post-init declaration lands on the published config bag
        self.assertTrue(get_exec().moe.disable_shared_experts_fusion)

    def test_dspark_applies_policy_before_building_stages(self):
        """DSpark must resolve and publish the fusion policy before stages build."""
        self._publish(enforce=False)
        policy_seen_by_stages = []

        class Stage(nn.Module):
            def __init__(self, **_kwargs):
                super().__init__()
                policy_seen_by_stages.append(
                    get_exec().moe.disable_shared_experts_fusion
                )

        class MarkovHead(nn.Module):
            def __init__(self, **_kwargs):
                super().__init__()

        config = self._make_config()
        config.dspark_markov_rank = 1
        config.num_nextn_predict_layers = 1

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
            model = DeepseekV4ForCausalLMDSpark(config)

        self.assertEqual(model.num_fused_shared_experts, 0)
        self.assertEqual(policy_seen_by_stages, [True])

    def test_enables_shared_fusion_when_enforced(self):
        """An explicitly enforced single shared expert remains part of the MoE."""
        self._publish(enforce=True)

        num_fused_shared_experts = _resolve_num_fused_shared_experts(
            self._make_config()
        )

        self.assertEqual(num_fused_shared_experts, 1)
        self.assertFalse(get_exec().moe.disable_shared_experts_fusion)

    def test_dspark_loads_enforced_shared_expert_into_fused_slot(self):
        """DSpark shared-expert checkpoint tensors must load into the fused expert slot."""
        config = self._make_config()
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
