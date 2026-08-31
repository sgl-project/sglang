"""The CuTe DSL AR fusion core is shared; only the RMSNorm flavour differs."""

import unittest

from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sglang.srt.layers.moe.cutedsl_ar_fusion import (
    CuteDSLFusionLayerCommunicator,
    CuteDSLFusionService,
    MoeFinalizeHandoff,
)
from sglang.srt.layers.moe.deepseek_flashinfer_fusion import (
    DeepseekFlashInferLayerCommunicator,
)
from sglang.srt.layers.moe.qwen35_flashinfer_fusion import (
    Qwen35FlashInferFusionService,
    Qwen35FlashInferLayerCommunicator,
    Qwen35MoeFinalizeHandoff,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-c-test-cpu")

HIDDEN = 8


class TestCuteDslFusionNormFlavour(CustomTestCase):
    def setUp(self):
        self.plain = RMSNorm(HIDDEN, eps=1e-6)
        self.gemma = GemmaRMSNorm(HIDDEN, eps=1e-6)

    def test_each_family_reads_its_own_norm(self):
        self.assertIs(
            Qwen35FlashInferLayerCommunicator._norm_gamma(self.gemma),
            self.gemma.gemma_weight,
        )
        self.assertIs(
            DeepseekFlashInferLayerCommunicator._norm_gamma(self.plain),
            self.plain.weight,
        )

    def test_norm_gamma_is_none_on_the_wrong_flavour(self):
        # The eligibility predicates rely on None to decline rather than raise.
        self.assertIsNone(Qwen35FlashInferLayerCommunicator._norm_gamma(self.plain))
        self.assertIsNone(DeepseekFlashInferLayerCommunicator._norm_gamma(self.gemma))

    def test_the_core_leaves_the_hook_abstract(self):
        with self.assertRaises(NotImplementedError):
            CuteDSLFusionLayerCommunicator._norm_gamma(self.plain)

    def test_finalize_is_unreachable_without_a_deferring_runner(self):
        # experts_can_defer_finalize is recorded at install time, so a layer
        # whose MoE runner cannot defer never advertises the finalize pattern.
        self.assertTrue(CuteDSLFusionLayerCommunicator.experts_can_defer_finalize)

    def test_both_families_subclass_the_shared_core(self):
        for communicator in (
            Qwen35FlashInferLayerCommunicator,
            DeepseekFlashInferLayerCommunicator,
        ):
            self.assertTrue(issubclass(communicator, CuteDSLFusionLayerCommunicator))

    def test_qwen_names_still_resolve(self):
        self.assertIs(Qwen35MoeFinalizeHandoff, MoeFinalizeHandoff)
        self.assertIs(Qwen35FlashInferFusionService, CuteDSLFusionService)


if __name__ == "__main__":
    unittest.main()
