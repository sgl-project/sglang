"""Unit tests for the GLM-5.2 (GlmMoeDsa) fp32 MoE correction-bias fix.

GlmMoeDsa's MoE ``e_score_correction_bias`` values are ~34. bf16 has ULP 0.25 at
that magnitude, so downcasting collapses the ~174 distinct biases to ~3 levels,
which scrambles top-k expert routing (noaux_tc picks experts by sigmoid-score +
bias, and the ~34 bias dominates the [0, 1] sigmoid term). The fix keeps the bias
in fp32 for GlmMoeDsa at both the parameter-construction site (MoEGate) and the
aiter routing boundary (layers/moe/topk.py). These tests are pure dtype / CPU
logic -- no server, no weight loading, no GPU required.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.deepseek_v2 import MoEGate, _is_glm_moe_dsa
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GLM_MAIN_ARCH = "GlmMoeDsaForCausalLM"
GLM_NEXTN_ARCH = "GlmMoeDsaForCausalLMNextN"  # draft head, rewritten in model_config.py
NON_GLM_ARCH = "DeepseekV3ForCausalLM"

# Biases spanning a ~0.5-wide window around 34: distinct in fp32 (fp32 ULP ~4e-6
# there), but within ~2-3 bf16 bins (bf16 ULP 0.25 at magnitude 34).
NUM_EXPERTS = 174
BIAS_BASE = 34.0


def _glm_bias_values() -> torch.Tensor:
    step = 0.5 / NUM_EXPERTS
    return torch.tensor(
        [BIAS_BASE + i * step for i in range(NUM_EXPERTS)], dtype=torch.float32
    )


class TestMoEGateCorrectionBiasDtype(CustomTestCase):
    """Behavioral guard on the production dtype block in MoEGate.__init__.

    Fails on pre-fix code (GlmMoeDsa was downcast to bf16 like every other aiter
    fp8 model); passes once the arch-gate skips the downcast for GlmMoeDsa.
    """

    def _build_gate(self, arch: str) -> MoEGate:
        config = SimpleNamespace(
            n_routed_experts=NUM_EXPERTS,
            hidden_size=16,
            topk_method="noaux_tc",
            architectures=[arch],
        )
        quant_config = SimpleNamespace(get_name=lambda: "fp8")
        # Force the aiter fp8 path (the branch that downcasts to bf16 for non-GLM);
        # _is_cpu=False avoids the AMX PackWeightMethod branch so the test is
        # deterministic across CPU and GPU CI runners.
        import sglang.srt.models.deepseek_v2 as dv2

        with patch.object(dv2, "_use_aiter", True), patch.object(dv2, "_is_cpu", False):
            return MoEGate(config=config, quant_config=quant_config)

    def test_glm_main_keeps_fp32(self):
        gate = self._build_gate(GLM_MAIN_ARCH)
        self.assertEqual(gate.e_score_correction_bias.dtype, torch.float32)

    def test_glm_nextn_keeps_fp32(self):
        # Guards the NextN draft head: the "GlmMoeDsa" substring must cover it too.
        gate = self._build_gate(GLM_NEXTN_ARCH)
        self.assertEqual(gate.e_score_correction_bias.dtype, torch.float32)

    def test_non_glm_still_downcasts_bf16(self):
        # Blast-radius guard: the fix must not widen dtype for other aiter models.
        # Fails if the gate is accidentally made too broad (e.g. always-skip).
        gate = self._build_gate(NON_GLM_ARCH)
        self.assertEqual(gate.e_score_correction_bias.dtype, torch.bfloat16)


class TestIsGlmMoeDsaHelper(CustomTestCase):
    """The arch-gate predicate used at both fix sites."""

    def test_matches_main_and_nextn(self):
        self.assertTrue(_is_glm_moe_dsa(SimpleNamespace(architectures=[GLM_MAIN_ARCH])))
        self.assertTrue(
            _is_glm_moe_dsa(SimpleNamespace(architectures=[GLM_NEXTN_ARCH]))
        )

    def test_matches_when_present_among_others(self):
        self.assertTrue(
            _is_glm_moe_dsa(SimpleNamespace(architectures=["Foo", GLM_MAIN_ARCH]))
        )

    def test_rejects_non_glm(self):
        self.assertFalse(_is_glm_moe_dsa(SimpleNamespace(architectures=[NON_GLM_ARCH])))

    def test_returns_false_for_none_or_empty_architectures(self):
        # A config with architectures=None or [] must return False, never
        # mis-gating a non-GLM model (matches the direct-access idiom in
        # configs/model_config.py, which reads config.architectures unguarded).
        self.assertFalse(_is_glm_moe_dsa(SimpleNamespace(architectures=None)))
        self.assertFalse(_is_glm_moe_dsa(SimpleNamespace(architectures=[])))


class TestCorrectionBiasBf16Collapse(CustomTestCase):
    """Pins the numeric mechanism the fp32 fix protects against."""

    def test_bf16_collapses_distinct_biases(self):
        biases = _glm_bias_values()
        fp32_distinct = torch.unique(biases).numel()
        bf16_distinct = torch.unique(biases.to(torch.bfloat16)).numel()
        # fp32 keeps every distinct bias; bf16 collapses the 174 values to a
        # handful (the documented ~3), i.e. an order-of-magnitude information loss.
        self.assertEqual(fp32_distinct, NUM_EXPERTS)
        self.assertLessEqual(bf16_distinct, 4)
        self.assertLess(bf16_distinct * 20, fp32_distinct)

    def test_bf16_bias_scrambles_topk_routing(self):
        # noaux_tc selects top-k experts by (sigmoid(logits) + correction_bias).
        # With the bias collapsed to ~3 levels, selection within a level is decided
        # by the tiny sigmoid term instead of the intended bias order -> the chosen
        # expert set diverges from the fp32 (correct) selection for most tokens.
        torch.manual_seed(0)
        topk = 8
        num_tokens = 64
        biases_fp32 = _glm_bias_values()[torch.randperm(NUM_EXPERTS)]
        biases_bf16 = biases_fp32.to(torch.bfloat16).to(torch.float32)

        scores = torch.randn(num_tokens, NUM_EXPERTS).sigmoid()
        top_fp32 = (scores + biases_fp32).topk(topk, dim=-1).indices
        top_bf16 = (scores + biases_bf16).topk(topk, dim=-1).indices

        differ = sum(
            set(top_fp32[t].tolist()) != set(top_bf16[t].tolist())
            for t in range(num_tokens)
        )
        self.assertGreater(differ / num_tokens, 0.5)


if __name__ == "__main__":
    unittest.main()
