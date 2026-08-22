import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.deepseek_v2 import MoEGate
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestMoEGateCorrectionBiasDtype(CustomTestCase):
    """The router's e_score_correction_bias must always be fp32.

    Checkpoints (e.g. GLM-5.2's mlp.gate.e_score_correction_bias) store this
    tensor in fp32. On the ROCm/aiter path it used to be allocated as bf16 for
    fp8/compressed_tensors/quark quant configs, silently rounding the bias on
    load and again on every forward -- see issue #34857. Every other branch
    (CUDA, no-quant) already used fp32; this pins the aiter branch to match.
    """

    def _make_config(self):
        return SimpleNamespace(
            n_routed_experts=8, hidden_size=16, topk_method="noaux_tc"
        )

    def test_aiter_fp8_quant_config_stays_fp32(self):
        quant_config = SimpleNamespace(get_name=lambda: "fp8")
        gate = MoEGate(config=self._make_config(), quant_config=quant_config)
        self.assertEqual(gate.e_score_correction_bias.dtype, torch.float32)

    def test_no_quant_config_is_fp32(self):
        gate = MoEGate(config=self._make_config(), quant_config=None)
        self.assertEqual(gate.e_score_correction_bias.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
