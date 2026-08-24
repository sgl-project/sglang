"""Every MoE family's fusion gate, asked the way the loader asks it.

`install_shared_experts_fusion_decision` calls
`<model class>.shared_experts_fusion_disable_reason(hf_config, quant_config)`
before the model is built, so the gate must answer from the config and
quantization it is handed — no instance, no layers. These cases pin each
family's branch table, which matters because most of these checkpoints cannot
be run on a single dev box: a wrong answer here is a silently wrong weight
remap (the loader remaps `mlp.shared_experts` into a fused slot the layers
never allocated), not a crash.

Conditions that depend on the device or the parallel topology are exercised
through `get_parallel().override(...)`; the ones that are pure config /
quantization are exercised directly.
"""

import unittest
import unittest.mock
from types import SimpleNamespace

from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _quant(name: str):
    return SimpleNamespace(get_name=lambda: name)


class _FusionGateCase(CustomTestCase):
    def _seed(self, **fields):
        override = get_context().override_server_args(**fields)
        override.install()
        self.addCleanup(override.restore)

    def _reason(self, model_class, hf_config, quant_config=None, moe_ep_size=1):
        # The gates consult the live EP size; without a group installed the
        # canonical getter asserts, so every case states a topology.
        with get_parallel().override(moe_ep_size=moe_ep_size):
            return model_class.shared_experts_fusion_disable_reason(
                hf_config, quant_config
            )


class TestDeepseekV2Gate(_FusionGateCase):
    def _config(self, **kw):
        base = dict(
            architectures=["DeepseekV3ForCausalLM"],
            n_routed_experts=256,
            n_shared_experts=1,
        )
        base.update(kw)
        return SimpleNamespace(**base)

    def test_a_foreign_architecture_cannot_fuse(self):
        from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM

        self._seed()
        self.assertIn(
            "does not support",
            self._reason(
                DeepseekV2ForCausalLM,
                self._config(architectures=["SomeOtherForCausalLM"]),
            ),
        )

    def test_an_unvalidated_expert_count_cannot_fuse(self):
        from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM

        self._seed()
        self.assertIn(
            "does not support",
            self._reason(DeepseekV2ForCausalLM, self._config(n_routed_experts=128)),
        )

    def test_the_384_expert_layout_needs_a_quark_checkpoint(self):
        from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM

        self._seed()
        config = self._config(n_routed_experts=384)
        self.assertIn(
            "does not support",
            self._reason(DeepseekV2ForCausalLM, config, _quant("compressed-tensors")),
        )
        # With Quark the layout is pre-fused, so this branch stops objecting.
        self.assertNotIn(
            "does not support",
            self._reason(DeepseekV2ForCausalLM, config, _quant("quark")) or "",
        )

    def test_the_nextn_draft_declares_its_own_architecture(self):
        from sglang.srt.models.deepseek_nextn import DeepseekV3ForCausalLMNextN
        from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM

        self.assertEqual(
            DeepseekV3ForCausalLMNextN.fused_shared_experts_architecture,
            "DeepseekV3ForCausalLMNextN",
        )
        self._seed()
        draft_config = self._config(architectures=["DeepseekV3ForCausalLMNextN"])
        # The draft's own class accepts it; the target's class does not.
        self.assertNotIn(
            "does not support",
            self._reason(DeepseekV3ForCausalLMNextN, draft_config) or "",
        )
        self.assertIn(
            "does not support", self._reason(DeepseekV2ForCausalLM, draft_config)
        )

    def test_expert_parallelism_blocks_fusion_off_rocm(self):
        from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM

        self._seed()
        self.assertTrue(
            self._reason(DeepseekV2ForCausalLM, self._config(), moe_ep_size=2)
        )

    def test_mixed_precision_quant_vetoes_even_when_enforced(self):
        """A precision mismatch causes crash when shared expert fusion is enabled,
        so --enforce-shared-experts-fusion must not override it. Guards the gap
        where the enforce early-return skipped the quant check entirely."""
        from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM

        self._seed(enforce_shared_experts_fusion=True)
        mixed = SimpleNamespace(
            get_name=lambda: "quark", can_fuse_shared_expert=lambda: False
        )
        self.assertIn(
            "higher precision",
            self._reason(DeepseekV2ForCausalLM, self._config(), mixed),
        )
        matched = SimpleNamespace(
            get_name=lambda: "quark", can_fuse_shared_expert=lambda: True
        )
        self.assertIsNone(self._reason(DeepseekV2ForCausalLM, self._config(), matched))


class TestGlmMoeLiteGate(_FusionGateCase):
    def _config(self, **kw):
        base = dict(architectures=["Glm4MoeLiteForCausalLM"], n_shared_experts=1)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_more_than_one_shared_expert_cannot_fuse(self):
        from sglang.srt.models.glm4_moe_lite import Glm4MoeLiteForCausalLM

        self._seed()
        self.assertTrue(
            self._reason(Glm4MoeLiteForCausalLM, self._config(n_shared_experts=2))
        )

    def test_expert_parallelism_blocks_fusion(self):
        from sglang.srt.models.glm4_moe_lite import Glm4MoeLiteForCausalLM

        self._seed()
        config = self._config()
        reason = self._reason(Glm4MoeLiteForCausalLM, config, moe_ep_size=2)
        self.assertTrue(reason)
        # This family checks the device capability before expert parallelism, so
        # only ask *which* branch refused on a device that would otherwise fuse
        # (a CPU runner never gets past the capability check).
        if self._reason(Glm4MoeLiteForCausalLM, config) is None:
            self.assertIn("expert parallelism", reason)

    def test_the_nextn_draft_declares_its_own_architecture(self):
        from sglang.srt.models.glm4_moe_lite_nextn import Glm4MoeLiteForCausalLMNextN

        self.assertEqual(
            Glm4MoeLiteForCausalLMNextN.fused_shared_experts_architecture,
            "Glm4MoeLiteForCausalLMNextN",
        )


class TestGlmMoeGate(_FusionGateCase):
    def test_a_w4afp8_checkpoint_cannot_fuse(self):
        from sglang.srt.models.glm4_moe import Glm4MoeForCausalLM

        self._seed()
        reason = self._reason(
            Glm4MoeForCausalLM, SimpleNamespace(n_shared_experts=1), _quant("w4afp8")
        )
        self.assertTrue(reason)

    def test_the_dsa_variant_declares_its_own_architecture(self):
        from sglang.srt.models.glm4_moe import GlmMoeDsaForCausalLM

        self.assertEqual(
            GlmMoeDsaForCausalLM.fused_shared_experts_architecture,
            "GlmMoeDsaForCausalLM",
        )


class TestMiniMaxGates(_FusionGateCase):
    def test_a_config_without_shared_experts_cannot_fuse(self):
        from sglang.srt.models.minimax_m3 import MiniMaxM3SparseForCausalLM

        self._seed()
        self.assertIn(
            "No shared experts",
            self._reason(
                MiniMaxM3SparseForCausalLM, SimpleNamespace(n_shared_experts=0)
            ),
        )

    def test_a_modelopt_mixed_checkpoint_cannot_fuse(self):
        from sglang.srt.models.minimax_m3 import MiniMaxM3SparseForCausalLM

        self._seed()
        reason = self._reason(
            MiniMaxM3SparseForCausalLM,
            SimpleNamespace(n_shared_experts=1),
            _quant("modelopt_mixed"),
        )
        self.assertIn("quantization formats", reason)

    def test_the_vl_variant_reads_the_text_config(self):
        from sglang.srt.models.minimax_m3_vl import (
            MiniMaxM3SparseForConditionalGeneration,
        )

        self._seed()
        wrapper = SimpleNamespace(text_config=SimpleNamespace(n_shared_experts=0))
        self.assertIn(
            "No shared experts",
            self._reason(MiniMaxM3SparseForConditionalGeneration, wrapper),
        )


class TestQwen3_5Gate(_FusionGateCase):
    def test_every_entry_class_answers(self):
        import sglang.srt.models.qwen3_5 as qwen3_5

        for cls in (
            qwen3_5.Qwen3_5ForCausalLM,
            qwen3_5.Qwen3_5MoeForCausalLM,
            qwen3_5.Qwen3_5ForConditionalGeneration,
            qwen3_5.Qwen3_5MoeForConditionalGeneration,
        ):
            self.assertTrue(
                hasattr(cls, "shared_experts_fusion_disable_reason"),
                f"{cls.__name__} would silently skip the ROCm auto-disable",
            )

    def test_the_auto_disable_is_rocm_only(self):
        import sglang.srt.models.qwen3_5 as qwen3_5

        self._seed()
        # On a non-ROCm build the gate never objects, whatever the checkpoint is.
        wrapper = SimpleNamespace(
            text_config=SimpleNamespace(model_type="qwen3_5_moe_text")
        )
        if not qwen3_5._is_hip:
            self.assertIsNone(
                self._reason(qwen3_5.Qwen3_5MoeForConditionalGeneration, wrapper)
            )


class TestWrapperEntryClassGates(_FusionGateCase):
    """A wrapper model answers with the config it hands its nested family.

    The loader asks the class it instantiates, which for these models is the
    wrapper — not the DeepSeek/Qwen3.5 body inside it. Each wrapper therefore
    delegates to its family's gate with the config (and quantization) the
    nested construction uses; these cases pin *what gets handed over*, because
    handing over the top-level config instead would answer for the wrong
    checkpoint (or raise on a config that has no expert counts at all).
    """

    def _recording_gate(self, family_cls):
        seen = {}

        def recorder(hf_config, quant_config):
            seen["config"] = hf_config
            seen["quant"] = quant_config
            return None

        return seen, unittest.mock.patch.object(
            family_cls,
            "shared_experts_fusion_disable_reason",
            staticmethod(recorder),
        )

    def test_kimi_vl_never_fuses_and_says_why(self):
        from sglang.srt.models.kimi_vl import KimiVLForConditionalGeneration

        self._seed()
        config = SimpleNamespace(
            encoder_only=False,
            text_config=SimpleNamespace(
                architectures=["Whatever"], n_routed_experts=256, n_shared_experts=1
            ),
        )
        # The construction rewrites the architecture to DeepseekV2ForCausalLM,
        # which is not the architecture the fused path validated.
        self.assertIn(
            "does not support",
            self._reason(KimiVLForConditionalGeneration, config),
        )
        self.assertIsNone(
            self._reason(
                KimiVLForConditionalGeneration,
                SimpleNamespace(encoder_only=True, text_config=None),
            ),
            "an encoder-only Kimi-VL builds no language model",
        )

    def test_kimi_k25_hands_over_its_text_config(self):
        from sglang.srt.models.deepseek_v2 import DeepseekV3ForCausalLM
        from sglang.srt.models.kimi_k25 import KimiK25ForConditionalGeneration

        self._seed()
        text_config = SimpleNamespace(
            architectures=["DeepseekV3ForCausalLM"],
            n_routed_experts=384,
            n_shared_experts=1,
        )
        config = SimpleNamespace(encoder_only=False, text_config=text_config)
        # The standard compressed-tensors Kimi-K2.5 checkpoint stores its shared
        # expert loose, so this must refuse to fuse.
        self.assertIn(
            "does not support",
            self._reason(
                KimiK25ForConditionalGeneration,
                config,
                _quant("compressed-tensors"),
            ),
        )
        seen, patcher = self._recording_gate(DeepseekV3ForCausalLM)
        with patcher:
            self._reason(KimiK25ForConditionalGeneration, config, _quant("quark"))
        self.assertIs(seen["config"], text_config)
        self.assertIsNone(
            self._reason(
                KimiK25ForConditionalGeneration,
                SimpleNamespace(encoder_only=True, text_config=None),
            )
        )

    def test_pixtral_only_asks_for_its_mla_backbone(self):
        from sglang.srt.models.mistral_large_3 import MistralLarge3ForCausalLM
        from sglang.srt.models.pixtral import PixtralForConditionalGeneration

        self._seed()
        mla_text = SimpleNamespace(
            model_type="deepseek_v3",
            architectures=["DeepseekV3ForCausalLM"],
            n_routed_experts=256,
            n_shared_experts=1,
        )
        seen, patcher = self._recording_gate(MistralLarge3ForCausalLM)
        with patcher:
            self._reason(
                PixtralForConditionalGeneration,
                SimpleNamespace(text_config=mla_text),
            )
        self.assertIs(seen["config"], mla_text)
        # A GQA text config builds the dense Mistral backbone instead.
        self.assertIsNone(
            self._reason(
                PixtralForConditionalGeneration,
                SimpleNamespace(text_config=SimpleNamespace(model_type="mistral")),
            )
        )

    def test_dots_vlm_hands_over_the_language_config(self):
        from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM
        from sglang.srt.models.dots_vlm import DotsVLMForCausalLM

        language_config = SimpleNamespace(
            architectures=["DeepseekV3ForCausalLM"],
            n_routed_experts=256,
            n_shared_experts=1,
        )
        config = SimpleNamespace(encoder_only=False, language_config=language_config)
        seen, patcher = self._recording_gate(DeepseekV2ForCausalLM)
        with patcher:
            self._reason(DotsVLMForCausalLM, config, _quant("fp8"))
        self.assertIs(seen["config"], language_config)
        self.assertEqual(seen["quant"].get_name(), "fp8")

    def test_deepseek_vl2_mirrors_its_unquantized_language_model(self):
        from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM
        from sglang.srt.models.deepseek_vl2 import DeepseekVL2ForCausalLM

        language_config = SimpleNamespace(
            use_mla=True,
            architectures=["DeepseekV3ForCausalLM"],
            n_routed_experts=256,
            n_shared_experts=1,
        )
        seen, patcher = self._recording_gate(DeepseekV2ForCausalLM)
        with patcher:
            self._reason(
                DeepseekVL2ForCausalLM,
                SimpleNamespace(language_config=language_config),
                _quant("fp8"),
            )
        self.assertIs(seen["config"], language_config)
        self.assertIsNone(
            seen["quant"], "the language model is constructed without quantization"
        )
        # deepseek-vl2-tiny forbids MLA and builds the dense model instead.
        self.assertIsNone(
            self._reason(
                DeepseekVL2ForCausalLM,
                SimpleNamespace(language_config=SimpleNamespace(use_mla=False)),
            )
        )

    def test_deepseek_ocr_only_asks_for_its_moe_branches(self):
        from sglang.srt.models.deepseek_ocr import DeepseekOCRForCausalLM
        from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM

        text_config = SimpleNamespace(
            topk_method="noaux_tc",
            use_mla=True,
            architectures=["DeepseekV3ForCausalLM"],
            n_routed_experts=256,
            n_shared_experts=1,
        )
        moe_config = SimpleNamespace(
            vision_config=SimpleNamespace(model_name="deepencoder"),
            projector_config=SimpleNamespace(input_dim=1280),
            text_config=text_config,
        )
        seen, patcher = self._recording_gate(DeepseekV2ForCausalLM)
        with patcher:
            self._reason(DeepseekOCRForCausalLM, moe_config, _quant("fp8"))
        self.assertIs(seen["config"], text_config)

        # OCR2 (and any non-MLA, non-noaux_tc config) builds the dense model.
        ocr2 = SimpleNamespace(
            vision_config=SimpleNamespace(model_name="DeepEncoderV2"),
            projector_config=SimpleNamespace(input_dim=896),
            text_config=text_config,
        )
        self.assertIsNone(self._reason(DeepseekOCRForCausalLM, ocr2))
        dense = SimpleNamespace(
            vision_config=SimpleNamespace(model_name="deepencoder"),
            projector_config=SimpleNamespace(input_dim=1280),
            text_config=SimpleNamespace(topk_method="greedy", use_mla=False),
        )
        self.assertIsNone(self._reason(DeepseekOCRForCausalLM, dense))

    def test_minicpmv_entries_delegate_to_the_qwen3_5_gate(self):
        from sglang.srt.models.minicpmv import (
            MiniCPMV,
            MiniCPMV4_6ForConditionalGeneration,
        )
        from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM

        text_config = SimpleNamespace(model_type="qwen3_5_moe_text")
        for cls in (MiniCPMV, MiniCPMV4_6ForConditionalGeneration):
            seen, patcher = self._recording_gate(Qwen3_5ForCausalLM)
            with patcher:
                self._reason(cls, SimpleNamespace(text_config=text_config))
            self.assertIs(seen["config"], text_config, cls.__name__)

    def test_the_text_only_qwen3_5_entries_delegate_to_their_body(self):
        import sglang.srt.models.qwen3_5 as qwen3_5
        import sglang.srt.models.qwen3_5_text as qwen3_5_text

        # A text-only Qwen3.5 checkpoint resolves to these classes, which shadow
        # the multimodal ones by name — attaching the gate to the multimodal
        # classes alone leaves the registry's text-only entries gate-less.
        self.assertIs(
            qwen3_5_text.Qwen3_5MoeForCausalLM.body_cls,
            qwen3_5.Qwen3_5MoeForCausalLM,
        )
        text_config = SimpleNamespace(model_type="qwen3_5_moe_text")
        seen, patcher = self._recording_gate(qwen3_5.Qwen3_5MoeForCausalLM)
        with patcher:
            self._reason(
                qwen3_5_text.Qwen3_5MoeForCausalLM, text_config, _quant("quark")
            )
        self.assertIs(seen["config"], text_config)
        self.assertEqual(seen["quant"].get_name(), "quark")

    def test_the_qwen3_5_mtp_entry_normalizes_its_quantization(self):
        from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM
        from sglang.srt.models.qwen3_5_mtp import (
            Qwen3_5ForCausalLMMTP,
            _mtp_quant_config,
        )

        # The normalization the constructor applies, shared with the gate.
        self.assertIsNone(_mtp_quant_config(_quant("modelopt_mixed")))
        serialized = SimpleNamespace(
            get_name=lambda: "modelopt_fp4", is_checkpoint_nvfp4_serialized=True
        )
        self.assertIsNone(_mtp_quant_config(serialized))
        # A non-serialized modelopt_fp4 checkpoint still converts on load, so
        # the MTP module keeps the quantization.
        online = SimpleNamespace(
            get_name=lambda: "modelopt_fp4", is_checkpoint_nvfp4_serialized=False
        )
        self.assertIs(_mtp_quant_config(online), online)
        quark_mtp = SimpleNamespace(
            get_name=lambda: "quark", exclude_layers=["mtp.mlp.experts"]
        )
        self.assertIsNone(_mtp_quant_config(quark_mtp))
        kept = _quant("fp8")
        self.assertIs(_mtp_quant_config(kept), kept)

        text_config = SimpleNamespace(model_type="qwen3_5_moe_text")
        seen, patcher = self._recording_gate(Qwen3_5ForCausalLM)
        with patcher:
            self._reason(
                Qwen3_5ForCausalLMMTP,
                SimpleNamespace(text_config=text_config),
                serialized,
            )
        self.assertIs(seen["config"], text_config)
        self.assertIsNone(
            seen["quant"], "the MTP module ships unquantized in that checkpoint"
        )


class TestFamiliesWithoutAGate(_FusionGateCase):
    def test_qwen2_moe_style_families_follow_the_intent(self):
        """A family with no gate must not grow one by accident: the installer
        falls back to the user's intent for it."""
        from sglang.srt.models.qwen2_moe import Qwen2MoeForCausalLM

        self.assertFalse(
            hasattr(Qwen2MoeForCausalLM, "shared_experts_fusion_disable_reason")
        )


if __name__ == "__main__":
    unittest.main()
