import unittest

import torch

from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.srt.layers.attention.fla.kda import fused_kda_gate, fused_recurrent_kda
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="stage-b-test-large-1-gpu")


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestKDAFusedSigmoidGatingRecurrent(unittest.TestCase):
    def setUp(self):
        self.token_num = 4
        self.query_start_loc = torch.tensor([0, 1, 2, 3, 4], device="cuda")
        self.cache_indices = torch.tensor([0, 2, 5, 8], device="cuda")
        self.local_num_heads = 8
        self.head_dim = 128
        self.cache_len = 64

        self.A_log = torch.randn(
            1, 1, self.local_num_heads, 1, dtype=torch.float32, device="cuda"
        )
        self.a = torch.randn(
            1,
            self.token_num,
            self.local_num_heads * self.head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        self.dt_bias = torch.randn(
            self.local_num_heads * self.head_dim, dtype=torch.bfloat16, device="cuda"
        )
        self.softplus_beta = 1.0
        self.softplus_threshold = 20.0
        self.q = torch.randn(
            1,
            self.token_num,
            self.local_num_heads,
            self.head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        self.k = torch.randn(
            1,
            self.token_num,
            self.local_num_heads,
            self.head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        self.v = torch.randn(
            1,
            self.token_num,
            self.local_num_heads,
            self.head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        self.beta = torch.randn(
            1, self.token_num, self.local_num_heads, dtype=torch.bfloat16, device="cuda"
        )

        self.ssm_states = torch.zeros(
            self.cache_len,
            self.local_num_heads,
            self.head_dim,
            self.head_dim,
            dtype=torch.float32,
            device="cuda",
        )

    def run_fused(self):
        ssm_states = self.ssm_states.clone()
        core_attn_out = fused_sigmoid_gating_delta_rule_update(
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            q=self.q,
            k=self.k,
            v=self.v,
            a=self.a,
            b=self.beta,
            initial_state_source=ssm_states,
            initial_state_indices=self.cache_indices,
            cu_seqlens=self.query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=self.softplus_beta,
            softplus_threshold=self.softplus_threshold,
            is_kda=True,
        )
        return core_attn_out, ssm_states[self.cache_indices]

    def run_kda(self):
        b = self.beta.float().sigmoid()
        g = fused_kda_gate(self.a, self.A_log, self.head_dim, g_bias=self.dt_bias)
        initial_state = self.ssm_states[self.cache_indices].clone()
        core_attn_out, last_state = fused_recurrent_kda(
            q=self.q,
            k=self.k,
            v=self.v,
            g=g,
            beta=b,
            initial_state=initial_state,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=self.query_start_loc,
        )
        return core_attn_out, last_state

    def test_kda_fused_sigmoid_gating_recurrent(self):
        core_attn_out, last_state = self.run_fused()
        core_attn_out_ref, last_state_ref = self.run_kda()
        abs_diff_out = (core_attn_out - core_attn_out_ref).abs().max()
        abs_diff_state = (last_state - last_state_ref).abs().max()
        print(f"{abs_diff_out=}, {abs_diff_state=}")
        self.assertTrue(torch.allclose(core_attn_out, core_attn_out_ref))
        self.assertTrue(torch.allclose(last_state, last_state_ref))


if __name__ == "__main__":
    unittest.main()
