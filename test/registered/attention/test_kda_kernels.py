import unittest

import torch

from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
from sglang.srt.layers.attention.fla.fused_recurrent import (
    fused_recurrent_kda_packed_decode,
)
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

register_cuda_ci(est_time=16, stage="base-b", runner_config="1-gpu-large")


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


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestKDAPackedDecode(unittest.TestCase):
    """Verify ``fused_recurrent_kda_packed_decode`` matches the existing decode
    path (split + unflatten + ``fused_sigmoid_gating_delta_rule_update``)."""

    @staticmethod
    def _make_inputs(B, H, HV, K, V, pool_size, dtype, device, seed=42):
        torch.manual_seed(seed)
        qkv_dim = 2 * H * K + HV * V
        mixed_qkv = (
            torch.randn(B, qkv_dim, dtype=dtype, device=device) * 0.1
        ).contiguous()
        a = (
            torch.randn(B, HV * K, dtype=dtype, device=device) * 0.5 - 1.0
        ).contiguous()
        b = (torch.randn(B, HV, dtype=dtype, device=device) * 0.5).contiguous()
        A_log = torch.randn(HV, dtype=torch.float32, device=device) * 0.2
        dt_bias = torch.randn(HV * K, dtype=torch.float32, device=device) * 0.1
        ssm_states = (
            torch.randn(pool_size, HV, V, K, dtype=dtype, device=device) * 0.01
        ).contiguous()
        cache_indices = torch.arange(B, device=device, dtype=torch.int32)
        return mixed_qkv, a, b, A_log, dt_bias, ssm_states, cache_indices

    @staticmethod
    def _run_baseline(
        mixed_qkv, a, b, A_log, dt_bias, ssm_states, cache_indices, H, HV, K, V
    ):
        B = mixed_qkv.shape[0]
        q_flat, k_flat, v_flat = torch.split(mixed_qkv, [H * K, H * K, HV * V], dim=-1)
        q = q_flat.view(1, B, H, K)
        k = k_flat.view(1, B, H, K)
        v = v_flat.view(1, B, HV, V)
        # The real backend passes query_start_loc = [0, 1, ..., B] so that
        # each of the B tokens becomes its own length-1 sequence with an
        # independent state; without this the kernel would share state.
        cu_seqlens = torch.arange(B + 1, device=mixed_qkv.device, dtype=torch.int32)
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=cu_seqlens,
            scale=K**-0.5,
            use_qk_l2norm_in_kernel=True,
            is_kda=True,
        )

    @staticmethod
    def _run_packed(
        mixed_qkv, a, b, A_log, dt_bias, ssm_states, cache_indices, HV, K, V
    ):
        B = mixed_qkv.shape[0]
        out = mixed_qkv.new_empty(B, 1, HV, V)
        fused_recurrent_kda_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=K**-0.5,
            initial_state=ssm_states,
            out=out,
            ssm_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
        )
        return out.transpose(0, 1)

    def _check(self, B, H, HV, K, V):
        device = get_device()
        dtype = torch.bfloat16
        pool_size = B + 4
        mixed_qkv, a, b, A_log, dt_bias, ssm_states, cache_indices = self._make_inputs(
            B, H, HV, K, V, pool_size, dtype, device
        )
        s_packed = ssm_states.clone()
        s_baseline = ssm_states.clone()

        o_packed = self._run_packed(
            mixed_qkv, a, b, A_log, dt_bias, s_packed, cache_indices, HV, K, V
        )
        o_baseline = self._run_baseline(
            mixed_qkv, a, b, A_log, dt_bias, s_baseline, cache_indices, H, HV, K, V
        )

        torch.testing.assert_close(
            o_packed.float(), o_baseline.float(), atol=2e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            s_packed[cache_indices].float(),
            s_baseline[cache_indices].float(),
            atol=2e-2,
            rtol=1e-2,
        )

    def test_b1(self):
        self._check(B=1, H=16, HV=16, K=128, V=128)

    def test_b4(self):
        self._check(B=4, H=16, HV=16, K=128, V=128)

    def test_b32(self):
        self._check(B=32, H=16, HV=16, K=128, V=128)

    def test_b128(self):
        self._check(B=128, H=16, HV=16, K=128, V=128)

    def test_asymmetric_heads(self):
        # Common KDA config with HV > H (grouped query).
        self._check(B=8, H=8, HV=16, K=128, V=128)

    def test_pad_slot(self):
        """Entries with state_idx == -1 must produce zero output and skip state writeback."""
        device = get_device()
        dtype = torch.bfloat16
        B, H, HV, K, V = 8, 16, 16, 128, 128
        pool_size = B + 4
        mixed_qkv, a, b, A_log, dt_bias, ssm_states, cache_indices = self._make_inputs(
            B, H, HV, K, V, pool_size, dtype, device
        )
        # Mark every other request as padded.
        cache_indices = cache_indices.clone()
        cache_indices[::2] = -1

        s_packed = ssm_states.clone()
        s_baseline = ssm_states.clone()
        o_packed = self._run_packed(
            mixed_qkv, a, b, A_log, dt_bias, s_packed, cache_indices, HV, K, V
        )
        o_baseline = self._run_baseline(
            mixed_qkv, a, b, A_log, dt_bias, s_baseline, cache_indices, H, HV, K, V
        )
        torch.testing.assert_close(
            o_packed.float(), o_baseline.float(), atol=2e-2, rtol=1e-2
        )

    def test_production_shapes_through_dispatcher(self):
        """Go through ``TritonKDAKernel.packed_decode`` with the exact tensor
        shapes the KimiDeltaAttention model produces at decode time, so the
        a/b/A_log/dt_bias reshape-normalization is unit-tested (not only E2E).

        Production decode shapes (see kimi_linear.py forward + __init__):
          - a (forget_gate): [B, HV*K]   (2D, not unflattened in decode)
          - b (beta):        [1, B, HV]  (unsqueeze(0), pre-sigmoid)
          - A_log:           [1, 1, HV, 1]
          - dt_bias:         [HV*K]
        """
        from sglang.srt.layers.attention.linear.kernels.kda_triton import (
            TritonKDAKernel,
        )

        device = get_device()
        dtype = torch.bfloat16
        B, H, HV, K, V = 4, 16, 16, 128, 128
        pool_size = B + 4
        mixed_qkv, a, b, A_log, dt_bias, ssm_states, cache_indices = self._make_inputs(
            B, H, HV, K, V, pool_size, dtype, device
        )

        # Reshape the flat reference tensors into the production layouts.
        b_prod = b.unsqueeze(0)  # [B, HV] -> [1, B, HV]
        A_log_prod = A_log.view(1, 1, HV, 1)  # [HV] -> [1, 1, HV, 1]

        kernel = TritonKDAKernel()
        self.assertTrue(kernel.supports_packed_decode)

        s_packed = ssm_states.clone()
        out = kernel.packed_decode(
            mixed_qkv,
            a,
            b_prod,
            A_log=A_log_prod,
            dt_bias=dt_bias,
            scale=K**-0.5,
            ssm_states=s_packed,
            cache_indices=cache_indices,
            num_v_heads=HV,
            head_v_dim=V,
        )

        s_baseline = ssm_states.clone()
        o_baseline = self._run_baseline(
            mixed_qkv, a, b, A_log, dt_bias, s_baseline, cache_indices, H, HV, K, V
        )

        # Dispatcher returns [1, B, HV, V], same layout as the baseline.
        torch.testing.assert_close(
            out.float(), o_baseline.float(), atol=2e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            s_packed[cache_indices].float(),
            s_baseline[cache_indices].float(),
            atol=2e-2,
            rtol=1e-2,
        )


if __name__ == "__main__":
    unittest.main()
