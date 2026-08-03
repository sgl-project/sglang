"""Hermetic unit test for the DSpark draft checkpoint-name remapping.

The ``.scale`` rename must be anchored at the suffix: packed checkpoints
(gptq / awq / auto_round) store scales as ``.scales``, and the previous
substring replace turned that into ``.weight_scale_invs`` — matching no
parameter, dropped with only a warning, leaving the draft with
uninitialised weights and a speculative accept rate silently pinned to
zero. Pure Python, no GPU, no weights.
"""

import unittest

from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

_remap = DeepseekV4ForCausalLMDSpark._remap_mtp_rest


class TestDSparkWeightNameRemap(CustomTestCase):
    def test_packed_scales_suffix_is_preserved(self):
        self.assertEqual(_remap("attn.wo_b.scales"), "self_attn.wo_b.scales")
        self.assertEqual(_remap("ffn.w1.scales"), "mlp.gate_proj.scales")

    def test_fp8_scale_suffix_still_renamed(self):
        self.assertEqual(_remap("attn.wo_b.scale"), "self_attn.wo_b.weight_scale_inv")

    def test_other_packed_tensors_untouched(self):
        self.assertEqual(_remap("ffn.w2.qweight"), "mlp.down_proj.qweight")
        self.assertEqual(_remap("ffn.w3.qzeros"), "mlp.up_proj.qzeros")

    def test_existing_renames_intact(self):
        self.assertEqual(_remap("attn_norm.weight"), "input_layernorm.weight")
        self.assertEqual(_remap("ffn.gate.bias"), "mlp.gate.e_score_correction_bias")


if __name__ == "__main__":
    unittest.main()
