import unittest

import torch

from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.srt.layers.attention.fla.index import prepare_chunk_indices
from sglang.srt.layers.attention.fla.kda import (
    fused_recurrent_kda,
    kda_gate_chunk_cumsum,
)
from sglang.srt.utils.common import get_device
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=12, suite="stage-b-test-1-gpu-large")


@unittest.skipIf(
    not (torch.cuda.is_available() or torch.xpu.is_available()),
    "Test requires CUDA or XPU",
)
class TestKDAFusedSigmoidGatingRecurrent(unittest.TestCase):
    def setUp(self):
        self.device = get_device()
        self.token_num = 4
        self.query_start_loc = torch.tensor([0, 1, 2, 3, 4], device=self.device)
        self.cache_indices = torch.tensor([0, 2, 5, 8], device=self.device)
        self.local_num_heads = 8
        self.head_dim = 128
        self.cache_len = 64

        self.A_log = torch.randn(
            1, 1, self.local_num_heads, 1, dtype=torch.float32, device=self.device
        )
        self.a = torch.randn(
            1,
            self.token_num,
            self.local_num_heads * self.head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.dt_bias = torch.randn(
            self.local_num_heads * self.head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.softplus_beta = 1.0
        self.softplus_threshold = 20.0
        self.q = torch.randn(
            1,
            self.token_num,
            self.local_num_heads,
            self.head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.k = torch.randn(
            1,
            self.token_num,
            self.local_num_heads,
            self.head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.v = torch.randn(
            1,
            self.token_num,
            self.local_num_heads,
            self.head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.beta = torch.randn(
            1,
            self.token_num,
            self.local_num_heads,
            dtype=torch.bfloat16,
            device=self.device,
        )

        self.ssm_states = torch.zeros(
            self.cache_len,
            self.local_num_heads,
            self.head_dim,
            self.head_dim,
            dtype=torch.float32,
            device=self.device,
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
        # Reference gate activation using torch ops:
        #   g = -exp(A_log) * softplus(raw_g + dt_bias)
        H, K = self.local_num_heads, self.head_dim
        raw_g = self.a.float()  # [1, T, H*K]
        if self.dt_bias is not None:
            raw_g = raw_g + self.dt_bias.float()
        g = -torch.exp(
            self.A_log.float().view(1, 1, H, 1)
        ) * torch.nn.functional.softplus(raw_g.view(1, -1, H, K))
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
        self.assertTrue(
            torch.allclose(core_attn_out, core_attn_out_ref, rtol=1e-3, atol=1e-4)
        )
        self.assertTrue(torch.allclose(last_state, last_state_ref))


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestKDAGateChunkCumsum(unittest.TestCase):
    """Test kda_gate_chunk_cumsum against torch reference (gate activation + cumsum)."""

    CHUNK_SIZE = 64

    def _ref_gate_cumsum(self, raw_g, A_log, dt_bias, cu_seqlens, chunk_size):
        """Reference: torch gate activation then chunk_local_cumsum."""
        B, T, H, K = raw_g.shape
        g = raw_g.float()
        if dt_bias is not None:
            g = g + dt_bias.float().view(1, 1, H, K)
        g = -torch.exp(A_log.float().view(1, 1, H, 1)) * torch.nn.functional.softplus(g)
        chunk_indices = (
            prepare_chunk_indices(cu_seqlens, chunk_size)
            if cu_seqlens is not None
            else None
        )
        return chunk_local_cumsum(
            g, chunk_size=chunk_size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
        )

    def _run_case(self, B, T_per_seq, H, K, use_bias, use_varlen):
        T = B * T_per_seq
        torch.manual_seed(42)
        raw_g = torch.randn(1, T, H, K, dtype=torch.bfloat16, device="cuda")
        A_log = torch.randn(H, dtype=torch.float32, device="cuda") * 0.5
        dt_bias = (
            torch.randn(H * K, dtype=torch.float32, device="cuda") * 0.1
            if use_bias
            else None
        )
        cu_seqlens = (
            torch.arange(
                0, (B + 1) * T_per_seq, T_per_seq, dtype=torch.long, device="cuda"
            )
            if use_varlen
            else None
        )

        out_fused = kda_gate_chunk_cumsum(
            raw_g,
            A_log=A_log,
            chunk_size=self.CHUNK_SIZE,
            dt_bias=dt_bias,
            cu_seqlens=cu_seqlens,
        )
        out_ref = self._ref_gate_cumsum(
            raw_g, A_log, dt_bias, cu_seqlens, self.CHUNK_SIZE
        )

        max_diff = (out_fused - out_ref).abs().max().item()
        rel_diff = max_diff / (out_ref.abs().mean().item() + 1e-8)
        return max_diff, rel_diff

    def test_varlen_with_bias(self):
        max_diff, rel_diff = self._run_case(
            B=4, T_per_seq=256, H=16, K=128, use_bias=True, use_varlen=True
        )
        self.assertLess(
            max_diff, 1e-3, f"max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}"
        )

    def test_varlen_no_bias(self):
        max_diff, rel_diff = self._run_case(
            B=4, T_per_seq=256, H=16, K=128, use_bias=False, use_varlen=True
        )
        self.assertLess(
            max_diff, 1e-3, f"max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}"
        )

    def test_fixed_len_with_bias(self):
        max_diff, rel_diff = self._run_case(
            B=4, T_per_seq=256, H=16, K=128, use_bias=True, use_varlen=False
        )
        self.assertLess(
            max_diff, 1e-3, f"max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}"
        )

    def test_single_seq_long(self):
        max_diff, rel_diff = self._run_case(
            B=1, T_per_seq=2048, H=16, K=128, use_bias=True, use_varlen=True
        )
        self.assertLess(
            max_diff, 1e-3, f"max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}"
        )

    def test_small_head_dim(self):
        max_diff, rel_diff = self._run_case(
            B=4, T_per_seq=128, H=8, K=64, use_bias=True, use_varlen=True
        )
        self.assertLess(
            max_diff, 1e-3, f"max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}"
        )


if __name__ == "__main__":
    unittest.main()
