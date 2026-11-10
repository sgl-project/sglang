"""DSpark head collapses with last FFN pre-mix, not untrained hc_head."""

from __future__ import annotations

import inspect
import unittest

import torch

from sglang.srt.models.deepseek_v4_dspark import (
    DSparkAttention,
    DeepseekV4ForCausalLMDSpark,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDsv41DsparkHcPre(CustomTestCase):
    def test_collapse_uses_last_ffn_pre_not_copy0(self):
        model = DeepseekV4ForCausalLMDSpark.__new__(DeepseekV4ForCausalLMDSpark)
        x = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        pre = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32
        )
        model._last_ffn_pre = pre
        got = DeepseekV4ForCausalLMDSpark.collapse_hc_head(model, x)
        want = (pre.unsqueeze(-1) * x).sum(dim=1)
        self.assertTrue(torch.allclose(got, want))
        self.assertFalse(torch.allclose(got, x[:, 0, :]))

    def test_collapse_identity_pre_mix_is_copy0(self):
        model = DeepseekV4ForCausalLMDSpark.__new__(DeepseekV4ForCausalLMDSpark)
        model._last_ffn_pre = None
        x = torch.randn(3, 4, 5)
        got = DeepseekV4ForCausalLMDSpark.collapse_hc_head(model, x)
        self.assertTrue(torch.equal(got, x[:, 0, :]))

    def test_collapse_2d_is_already_done(self):
        model = DeepseekV4ForCausalLMDSpark.__new__(DeepseekV4ForCausalLMDSpark)
        model._last_ffn_pre = torch.ones(2, 4)
        x = torch.randn(2, 8)
        self.assertIs(DeepseekV4ForCausalLMDSpark.collapse_hc_head(model, x), x)

    def test_dspark_attn_matches_mqa_forward_kwargs(self):
        params = inspect.signature(DSparkAttention.forward).parameters
        self.assertEqual(list(params)[:4], ["self", "x", "positions", "forward_batch"])
        self.assertIn("x_quant", params)

    def test_compute_q_uses_v41_qb_not_head_rmsnorm(self):
        src = inspect.getsource(DSparkAttention._compute_q)
        self.assertIn("return q_out", src)
        self.assertIn("fused_rope_inplace", src)
        self.assertIn("self.wq_b", src)
        self.assertNotIn("npu_rms_norm", src)

    def test_sm70_shared_block_flag_is_on_radix_attn(self):
        src = inspect.getsource(DSparkAttention.__init__)
        self.assertIn("self.attn.sm70_swa_ring_from_inject = True", src)


if __name__ == "__main__":
    unittest.main()
