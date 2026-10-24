import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.layers.moe import MoeA2ABackend
from sglang.srt.layers.moe.utils import (
    is_shared_experts_fusion_disabled,
    resolve_shared_experts_config,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.models import deepseek_v4
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
from sglang.srt.runtime_context import (
    get_context,
    get_exec,
    get_flags,
    get_model,
    get_parallel,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDeepseekV4SharedExpertFusionPolicy(CustomTestCase):
    """V4 shared-expert fusion requires explicit startup intent, except by default
    with the gfx950 aiter MoE, MXFP4 routed experts and one shared expert."""

    def setUp(self):
        super().setUp()
        cm = get_parallel().override(moe_ep_size=1)
        cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)

    def _publish(self, enforce, disable=False):
        override = get_context().override_server_args(
            enforce_shared_experts_fusion=enforce,
            disable_shared_experts_fusion=disable,
        )
        override.install()
        self.addCleanup(override.restore)

    def _install(
        self, model_class=DeepseekV4ForCausalLM, n_shared_experts=1, quant_config=None
    ):
        config = resolve_shared_experts_config(
            model_class,
            SimpleNamespace(n_shared_experts=n_shared_experts),
            quant_config,
        )

        scope = get_model().override(shared_experts=config)
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)

    def _gfx950_aiter(self):
        for name in ("_use_aiter", "_is_gfx95_supported"):
            p = patch.object(deepseek_v4, name, True)
            p.start()
            self.addCleanup(p.stop)

    @staticmethod
    def _mxfp4(**overrides):
        config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            weight_block_size=[128, 128],
            is_fp4_experts=True,
        )
        for name, value in overrides.items():
            setattr(config, name, value)
        return config

    def _make_dspark_config(self):
        return DeepSeekV4Config(
            architectures=["DeepseekV4ForCausalLMDSpark"],
            quantization_config={},
            rope_scaling={},
            compress_ratios=[],
            kv_source_layer_ids=[],
            index_source_layer_ids=[],
            engram_layer_ids=[],
            engram_num_embeddings=[],
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
        # The resolved model decision leaves the startup intent untouched.
        self.assertTrue(is_shared_experts_fusion_disabled())
        self.assertFalse(get_exec().moe.disable_shared_experts_fusion)

    def test_v41_vision_admits_cuda_megamoe_with_dp_attention(self):
        """A vision-capable checkpoint must initialize for text or image serving
        with DP attention and CUDA MegaMoE. CP, PP and other A2A paths still lack
        vision routing support and must fail before constructing the tower.
        """
        self._publish(enforce=False)
        config = DeepSeekV4Config(
            architectures=["DeepseekV4ForCausalLM"],
            model_type="deepseek_v41",
            vision_n_layers=1,
            hidden_size=4,
            tie_word_embeddings=True,
            quantization_config={},
            rope_scaling={},
            compress_ratios=[],
            kv_source_layer_ids=[],
            index_source_layer_ids=[],
            engram_layer_ids=[],
            engram_num_embeddings=[],
        )
        backbone = nn.Module()
        backbone.embed_tokens = nn.Identity()
        backbone.start_layer = 0
        backbone.end_layer = 0
        cases = (
            (MoeA2ABackend.MEGAMOE, True, 1, 1, True),
            (MoeA2ABackend.NONE, True, 1, 1, True),
            (MoeA2ABackend.MEGAMOE, False, 1, 1, False),
            (MoeA2ABackend.DEEPEP, True, 1, 1, False),
            (MoeA2ABackend.MEGAMOE, True, 2, 1, False),
            (MoeA2ABackend.MEGAMOE, True, 1, 2, False),
        )
        for backend, cuda, cp, pp, admitted in cases:
            with self.subTest(backend=backend, cuda=cuda, cp=cp, pp=pp):
                with (
                    get_parallel().override(
                        tp_size=4,
                        attn_dp_size=2,
                        attn_tp_size=2 // cp,
                        attn_cp_size=cp,
                        moe_ep_size=4,
                        pp_size=pp,
                        pp_group=SimpleNamespace(world_size=pp, is_last_rank=True),
                    ),
                    get_flags().moe.override(a2a_backend=backend),
                    get_model().override(
                        shared_experts=SharedExpertsConfig(fusion_disabled=True)
                    ),
                    patch.object(deepseek_v4, "_is_cuda", cuda),
                    patch.object(deepseek_v4, "ViT", return_value=nn.Identity()),
                    patch.object(deepseek_v4, "Aligner", return_value=nn.Identity()),
                    patch.object(deepseek_v4, "DeepseekV4Model", return_value=backbone),
                    patch.object(
                        deepseek_v4, "LogitsProcessor", return_value=nn.Identity()
                    ),
                    patch.object(deepseek_v4, "get_attn_tp_context"),
                ):
                    if admitted:
                        model = DeepseekV4ForCausalLM(config)
                        self.assertIsInstance(model.vision, nn.Module)
                        self.assertEqual(model.image_start.shape, (4,))
                    else:
                        with self.assertRaisesRegex(ValueError, "V4.1 vision"):
                            DeepseekV4ForCausalLM(config)

    def test_enables_shared_fusion_when_enforced(self):
        self._publish(enforce=True)
        self.assertIsNone(
            DeepseekV4ForCausalLM.shared_experts_fusion_disable_reason(
                SimpleNamespace(n_shared_experts=1), None
            )
        )
        self._install()
        self.assertFalse(is_shared_experts_fusion_disabled())

    def test_enforcing_with_more_than_one_shared_expert_is_rejected(self):
        self._publish(enforce=True)
        with self.assertRaisesRegex(ValueError, "exactly one shared"):
            DeepseekV4ForCausalLM.shared_experts_fusion_disable_reason(
                SimpleNamespace(n_shared_experts=2), None
            )

    def test_mixed_precision_quant_vetoes_even_when_enforced(self):
        """A precision mismatch causes crash when shared expert fusion is enabled,
        so --enforce-shared-experts-fusion must not override it. Guards the gap
        where the enforce early-return skipped the quant check entirely."""
        self._publish(enforce=True)
        mixed = SimpleNamespace(
            get_name=lambda: "quark", can_fuse_shared_expert=lambda: False
        )
        self.assertIn(
            "higher precision",
            DeepseekV4ForCausalLM.shared_experts_fusion_disable_reason(
                SimpleNamespace(n_shared_experts=1), mixed
            ),
        )
        matched = SimpleNamespace(
            get_name=lambda: "quark", can_fuse_shared_expert=lambda: True
        )
        self.assertIsNone(
            DeepseekV4ForCausalLM.shared_experts_fusion_disable_reason(
                SimpleNamespace(n_shared_experts=1), matched
            )
        )

    def test_fuses_by_default_with_gfx950_aiter_and_mxfp4_experts(self):
        self._publish(enforce=False)
        self._gfx950_aiter()
        self.assertIsNone(
            DeepseekV4ForCausalLM.shared_experts_fusion_disable_reason(
                SimpleNamespace(n_shared_experts=1), self._mxfp4()
            )
        )
        self._install(quant_config=self._mxfp4())
        self.assertFalse(is_shared_experts_fusion_disabled())

    def test_default_needs_gfx950_aiter_mxfp4_and_one_shared_expert(self):
        """Outside the default rule the gate keeps the old answer, and a second
        shared expert falls back instead of raising."""
        self._publish(enforce=False)
        cases = {
            "not gfx950": (True, False, self._mxfp4(), 1),
            "no aiter": (False, True, self._mxfp4(), 1),
            "fp8 routed experts": (True, True, self._mxfp4(is_fp4_experts=False), 1),
            "dequant fp4 to fp8": (True, True, self._mxfp4(dequant_fp4_to_fp8=True), 1),
            "no quant config": (True, True, None, 1),
            "two shared experts": (True, True, self._mxfp4(), 2),
        }
        for label, (use_aiter, gfx95, quant, n_shared) in cases.items():
            with (
                self.subTest(label),
                patch.object(deepseek_v4, "_use_aiter", use_aiter),
                patch.object(deepseek_v4, "_is_gfx95_supported", gfx95),
            ):
                self.assertEqual(
                    DeepseekV4ForCausalLM.shared_experts_fusion_disable_reason(
                        SimpleNamespace(n_shared_experts=n_shared), quant
                    ),
                    "Config does not support fused shared expert(s).",
                )

    def test_disable_flag_wins_over_the_default(self):
        """--disable-shared-experts-fusion is applied before the gate is asked."""
        self._publish(enforce=False, disable=True)
        self._gfx950_aiter()
        self.assertIsNone(
            DeepseekV4ForCausalLM.shared_experts_fusion_disable_reason(
                SimpleNamespace(n_shared_experts=1), self._mxfp4()
            )
        )
        self._install(quant_config=self._mxfp4())
        self.assertTrue(is_shared_experts_fusion_disabled())

    def test_default_keeps_the_ep_and_precision_vetoes(self):
        self._publish(enforce=False)
        self._gfx950_aiter()
        with (
            get_parallel().override(tp_size=4, attn_tp_size=4, moe_ep_size=4),
            get_flags().moe.override(a2a_backend=MoeA2ABackend.NONE),
        ):
            self.assertIn(
                "Expert parallelism",
                DeepseekV4ForCausalLM.shared_experts_fusion_disable_reason(
                    SimpleNamespace(n_shared_experts=1), self._mxfp4()
                ),
            )
        self.assertIn(
            "higher precision",
            DeepseekV4ForCausalLM.shared_experts_fusion_disable_reason(
                SimpleNamespace(n_shared_experts=1),
                self._mxfp4(can_fuse_shared_expert=lambda: False),
            ),
        )

    def test_default_stays_off_with_an_all_to_all_backend(self):
        self._publish(enforce=False)
        self._gfx950_aiter()
        for backend in (MoeA2ABackend.MORI, MoeA2ABackend.MEGAMOE):
            with (
                self.subTest(backend=backend.value),
                get_parallel().override(tp_size=4, attn_tp_size=4, moe_ep_size=4),
                get_flags().moe.override(a2a_backend=backend),
            ):
                self.assertEqual(
                    DeepseekV4ForCausalLM.shared_experts_fusion_disable_reason(
                        SimpleNamespace(n_shared_experts=1), self._mxfp4()
                    ),
                    "Config does not support fused shared expert(s).",
                )

    def test_dspark_draft_fuses_by_default_too(self):
        self._publish(enforce=False)
        self._gfx950_aiter()
        self._install(DeepseekV4ForCausalLMDSpark, quant_config=self._mxfp4())
        self.assertFalse(is_shared_experts_fusion_disabled())

    def test_dspark_entry_class_uses_the_v4_gate(self):
        """A DSV4 DSpark draft must inherit the target's default fusion policy."""
        self._publish(enforce=False)

        self._install(DeepseekV4ForCausalLMDSpark)

        self.assertTrue(is_shared_experts_fusion_disabled())

    def test_dspark_records_explicitly_forced_fusion(self):
        """A forced DSpark build must retain its fused shared-expert count."""
        self._publish(enforce=True)
        self._install(DeepseekV4ForCausalLMDSpark)

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

            def post_load_weights(self):
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
