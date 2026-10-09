"""
Tests that fused_gdn_gating and fused_sigmoid_gating_delta_rule_update
produce correct results when a/b inputs are non-contiguous,
as happens with Qwen3.5-27B (v_per_group=3) via mixed_ba.split(), and that
the Triton decode chain reads row-strided qkvz/ba projection views exactly.
"""

import unittest

import torch

from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.kernels.ops.attention.fla.layernorm_gated import rms_norm_gated
from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkvzba_split_reshape_cat_contiguous,
    qwen3_5_gdn_prefill_projection_views,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=7, stage="stage-b", runner_config="1-gpu-large-amd")


def _make_noncontiguous_ab(batch, num_heads, dtype=torch.bfloat16, device="cuda"):
    """
    Simulate Qwen3.5 fallback: mixed_ba.split([nv_tp, nv_tp], dim=-1).
    Returns (b, a) as split views with stride(0) = 2 * num_heads.
    Also returns contiguous copies for reference comparison.
    """
    mixed_ba = torch.randn(batch, 2 * num_heads, dtype=dtype, device=device)
    b, a = mixed_ba.split([num_heads, num_heads], dim=-1)

    # For batch=1, PyTorch may still report contiguous even when split keeps
    # a widened leading stride. Validate stride semantics unconditionally.
    if batch > 1:
        assert not a.is_contiguous(), "a should be non-contiguous from split"
        assert not b.is_contiguous(), "b should be non-contiguous from split"
    assert a.stride(0) == 2 * num_heads
    assert b.stride(0) == 2 * num_heads
    return b, a, b.contiguous(), a.contiguous()


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestFusedGdnGatingNonContiguous(unittest.TestCase):
    """Test fused_gdn_gating with non-contiguous a/b."""

    def _run_test(self, batch, num_heads):
        A_log = torch.randn(num_heads, dtype=torch.float32, device="cuda")
        dt_bias = torch.randn(num_heads, dtype=torch.bfloat16, device="cuda")

        b, a, b_contig, a_contig = _make_noncontiguous_ab(batch, num_heads)

        g_ref, beta_ref = fused_gdn_gating(A_log, a_contig, b_contig, dt_bias)
        g_test, beta_test = fused_gdn_gating(A_log, a, b, dt_bias)

        self.assertTrue(
            torch.allclose(g_test, g_ref, rtol=0, atol=0),
            f"g mismatch: max diff = {(g_test - g_ref).abs().max().item()}",
        )
        self.assertTrue(
            torch.allclose(beta_test, beta_ref, rtol=0, atol=0),
            f"beta mismatch: max diff = {(beta_test - beta_ref).abs().max().item()}",
        )

    def test_small(self):
        self._run_test(batch=4, num_heads=8)

    def test_qwen35_27b_tp1(self):
        """Qwen3.5-27B TP=1: nv_tp=48."""
        self._run_test(batch=16, num_heads=48)

    def test_qwen35_27b_tp2(self):
        """Qwen3.5-27B TP=2: nv_tp=24."""
        self._run_test(batch=32, num_heads=24)

    def test_single_batch(self):
        self._run_test(batch=1, num_heads=48)


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestFusedSigmoidGatingDeltaRuleUpdateNonContiguous(unittest.TestCase):
    """Test fused_sigmoid_gating_delta_rule_update with non-contiguous a/b."""

    def _run_test(self, batch, T, num_v_heads, head_k_dim, head_v_dim):
        num_k_heads = num_v_heads  # simplification for GDN
        HV = num_v_heads
        K = head_k_dim
        V = head_v_dim

        A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
        dt_bias = torch.randn(HV, dtype=torch.bfloat16, device="cuda")

        q = torch.randn(batch, T, num_k_heads, K, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(batch, T, num_k_heads, K, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(batch, T, HV, V, dtype=torch.bfloat16, device="cuda")

        # Simulate non-contiguous a/b from split
        mixed_ba = torch.randn(batch * T, 2 * HV, dtype=torch.bfloat16, device="cuda")
        b_nc, a_nc = mixed_ba.split([HV, HV], dim=-1)
        b_c, a_c = b_nc.contiguous(), a_nc.contiguous()

        # Build cu_seqlens for varlen (one token per sequence)
        cu_seqlens = torch.arange(0, batch * T + 1, T, dtype=torch.int32, device="cuda")

        cache_len = batch + 4
        ssm_states = torch.zeros(
            cache_len, HV, K, V, dtype=torch.float32, device="cuda"
        )
        state_indices = torch.arange(batch, dtype=torch.int32, device="cuda")

        # Reference: contiguous a/b
        ssm_ref = ssm_states.clone()
        out_ref = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a_c,
            b=b_c,
            initial_state_source=ssm_ref,
            initial_state_indices=state_indices,
            cu_seqlens=cu_seqlens,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=False,
        )

        # Test: non-contiguous a/b
        ssm_test = ssm_states.clone()
        out_test = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a_nc,
            b=b_nc,
            initial_state_source=ssm_test,
            initial_state_indices=state_indices,
            cu_seqlens=cu_seqlens,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=False,
        )

        max_out_diff = (out_test - out_ref).abs().max().item()
        max_state_diff = (ssm_test - ssm_ref).abs().max().item()

        self.assertTrue(
            torch.allclose(out_test, out_ref, rtol=0, atol=0),
            f"output mismatch: max diff = {max_out_diff}",
        )
        self.assertTrue(
            torch.allclose(ssm_test, ssm_ref, rtol=0, atol=0),
            f"state mismatch: max diff = {max_state_diff}",
        )

    def test_decode_single_token(self):
        """Standard decode: T=1, batch>1."""
        self._run_test(batch=4, T=1, num_v_heads=8, head_k_dim=64, head_v_dim=32)

    def test_qwen35_decode(self):
        """Qwen3.5-27B like config: HV=48."""
        self._run_test(batch=8, T=1, num_v_heads=48, head_k_dim=128, head_v_dim=128)

    def test_multi_token(self):
        """target_verify style: T>1."""
        self._run_test(batch=4, T=4, num_v_heads=8, head_k_dim=64, head_v_dim=32)


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestFusedSigmoidGatingKDAStride(unittest.TestCase):
    """Regression test: KDA path handles non-contiguous a/b after stride_a refactor."""

    def test_kda_noncontiguous_matches_contiguous(self):
        """KDA path should produce identical outputs/states for contiguous vs non-contiguous a/b."""
        token_num = 4
        num_heads = 8
        head_dim = 128
        HV = num_heads
        K = head_dim

        A_log = torch.randn(1, 1, HV, 1, dtype=torch.float32, device="cuda")
        dt_bias = torch.randn(HV * K, dtype=torch.bfloat16, device="cuda")

        mixed_a = torch.randn(
            token_num, 2 * HV * K, dtype=torch.bfloat16, device="cuda"
        )
        a_nc, _ = mixed_a.split([HV * K, HV * K], dim=-1)
        a_c = a_nc.contiguous()
        self.assertFalse(a_nc.is_contiguous())

        mixed_b = torch.randn(1, token_num, 2 * HV, dtype=torch.bfloat16, device="cuda")
        b_nc, _ = mixed_b.split([HV, HV], dim=-1)
        b_c = b_nc.contiguous()
        self.assertFalse(b_nc.is_contiguous())

        q = torch.randn(1, token_num, HV, K, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, token_num, HV, K, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(1, token_num, HV, K, dtype=torch.bfloat16, device="cuda")

        cu_seqlens = torch.tensor([0, 1, 2, 3, 4], device="cuda", dtype=torch.int32)
        cache_len = 64
        ssm_states = torch.zeros(
            cache_len, HV, K, K, dtype=torch.float32, device="cuda"
        )
        cache_indices = torch.tensor([0, 2, 5, 8], device="cuda", dtype=torch.int32)

        # Reference: contiguous a/b
        ssm_ref = ssm_states.clone()
        out_ref = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a_c,
            b=b_c,
            initial_state_source=ssm_ref,
            initial_state_indices=cache_indices,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=True,
        )

        # Test: non-contiguous a/b from split
        ssm_test = ssm_states.clone()
        out_test = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a_nc,
            b=b_nc,
            initial_state_source=ssm_test,
            initial_state_indices=cache_indices,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=True,
        )

        self.assertTrue(
            torch.allclose(out_test, out_ref, rtol=0, atol=0),
            f"KDA output mismatch: max diff = {(out_test - out_ref).abs().max().item()}",
        )
        self.assertTrue(
            torch.allclose(ssm_test, ssm_ref, rtol=0, atol=0),
            f"KDA state mismatch: max diff = {(ssm_test - ssm_ref).abs().max().item()}",
        )


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestGdnDecodeOnProjectionViews(unittest.TestCase):
    """Batched Triton decode on row-strided projection views must be bit-identical
    to decode on the tensors materialized by the qkvzba split kernel."""

    HEAD_DIM = 128
    CONV_WIDTH = 4

    @staticmethod
    def _decode(mixed_qkv, norm_z, b, a, *, conv_state, ssm_state, params):
        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_state,
            params["conv_weight"],
            None,
            "silu",
            conv_state_indices=params["indices"],
        )
        batch, num_v_heads = a.shape
        head_dim = norm_z.shape[-1]
        out = mixed_qkv.new_empty(batch, 1, num_v_heads, head_dim)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=params["A_log"],
            dt_bias=params["dt_bias"],
            scale=head_dim**-0.5,
            initial_state=ssm_state,
            out=out,
            ssm_state_indices=params["indices"],
            use_qk_l2norm_in_kernel=True,
        )
        gated = rms_norm_gated(
            x=out.reshape(-1, head_dim),
            weight=params["norm_weight"],
            bias=None,
            z=norm_z,
            eps=1e-6,
            is_rms_norm=True,
        )
        return mixed_qkv, gated

    def _run(self, *, batch, num_qk_heads, num_v_heads):
        torch.manual_seed(0)
        hd = self.HEAD_DIM
        qkv_dim = (2 * num_qk_heads + num_v_heads) * hd
        num_slots = batch + 3
        qkvz = torch.randn(
            batch, qkv_dim + num_v_heads * hd, dtype=torch.bfloat16, device="cuda"
        )
        ba = torch.randn(batch, 2 * num_v_heads, dtype=torch.bfloat16, device="cuda")
        conv_state = torch.randn(
            num_slots,
            qkv_dim,
            self.CONV_WIDTH - 1,
            dtype=torch.bfloat16,
            device="cuda",
        )
        ssm_state = 0.1 * torch.randn(
            num_slots, num_v_heads, hd, hd, dtype=torch.float32, device="cuda"
        )
        params = {
            "conv_weight": torch.randn(
                qkv_dim, self.CONV_WIDTH, dtype=torch.bfloat16, device="cuda"
            ),
            "indices": torch.randperm(num_slots, device="cuda")[:batch].to(torch.int32),
            "A_log": torch.randn(num_v_heads, dtype=torch.float32, device="cuda"),
            "dt_bias": torch.randn(num_v_heads, dtype=torch.bfloat16, device="cuda"),
            "norm_weight": torch.randn(hd, dtype=torch.bfloat16, device="cuda"),
        }
        split_args = (qkvz, ba, num_qk_heads, num_v_heads, hd, hd)

        ref_qkv, ref_z, ref_b, ref_a = fused_qkvzba_split_reshape_cat_contiguous(
            *split_args
        )
        qkv, z, b, a = qwen3_5_gdn_prefill_projection_views(*split_args)
        # Zero-copy: every view aliases the projection outputs.
        self.assertEqual(qkv.data_ptr(), qkvz.data_ptr())
        self.assertEqual(b.data_ptr(), ba.data_ptr())
        self.assertFalse(any(t.is_contiguous() for t in (qkv, z, b, a)))

        conv_ref, ssm_ref = conv_state.clone(), ssm_state.clone()
        conv_test, ssm_test = conv_state.clone(), ssm_state.clone()
        # The model flattens a materialized z but hands the norm the 3D view.
        ref = self._decode(
            ref_qkv,
            ref_z.reshape(-1, hd),
            ref_b,
            ref_a,
            conv_state=conv_ref,
            ssm_state=ssm_ref,
            params=params,
        )
        test = self._decode(
            qkv, z, b, a, conv_state=conv_test, ssm_state=ssm_test, params=params
        )

        for name, got, want in (
            ("conv1d output", test[0], ref[0]),
            ("gated output", test[1], ref[1]),
            ("conv state", conv_test, conv_ref),
            ("ssm state", ssm_test, ssm_ref),
        ):
            self.assertTrue(torch.equal(got, want), f"{name} differs")

    def test_qwen38_27b_tp2_ratio3(self):
        self._run(batch=7, num_qk_heads=8, num_v_heads=24)

    def test_ratio2(self):
        self._run(batch=5, num_qk_heads=2, num_v_heads=4)


if __name__ == "__main__":
    unittest.main()
