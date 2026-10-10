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

from sglang.srt.configs.model_config import is_glm_moe_dsa
from sglang.srt.models.deepseek_v2 import MoEGate
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

    def test_reads_the_first_architecture_like_its_neighbours(self):
        # is_deepseek_dsa and is_kimi_k3 next to it both decide on
        # architectures[0]; a HF config carries the model's own arch there.
        self.assertFalse(
            is_glm_moe_dsa(SimpleNamespace(architectures=["Foo", GLM_MAIN_ARCH]))
        )

    def test_returns_false_for_none_or_empty_architectures(self):
        # A config with architectures=None or [] must return False, never
        # mis-gating a non-GLM model. _hf_arch() returns None for both.
        self.assertFalse(is_glm_moe_dsa(SimpleNamespace(architectures=None)))
        self.assertFalse(is_glm_moe_dsa(SimpleNamespace(architectures=[])))


if __name__ == "__main__":
    unittest.main()
