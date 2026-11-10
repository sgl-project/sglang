"""
Tests that fused_gdn_gating and fused_sigmoid_gating_delta_rule_update
produce correct results when a/b inputs are non-contiguous,
as happens with Qwen3.5-27B (v_per_group=3) via mixed_ba.split().
"""

import unittest

import torch

from sglang.kernels.ops.attention.linear.fused_gdn_gating import fused_gdn_gating
from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=7, stage="base-b", runner_config="1-gpu-small")
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

    def _run_fp16_verify_block_matches_repeated_decode(
        self,
        *,
        key_heads,
        value_heads,
        steps,
    ):
        """Low-precision recurrent state must cross the same boundary per token."""
        torch.manual_seed(7)
        batch, key_dim, value_dim = 1, 128, 128
        state_shape = (batch, value_heads, key_dim, value_dim)

        A_log = torch.randn(value_heads, dtype=torch.float32, device="cuda") * 0.1
        dt_bias = torch.randn(value_heads, dtype=torch.float16, device="cuda") * 0.1
        q = torch.randn(
            batch,
            steps,
            key_heads,
            key_dim,
            dtype=torch.float16,
            device="cuda",
        )
        k = torch.randn_like(q)
        v = torch.randn(
            batch,
            steps,
            value_heads,
            value_dim,
            dtype=torch.float16,
            device="cuda",
        )
        a = torch.randn(batch * steps, value_heads, dtype=torch.float16, device="cuda")
        b = torch.randn_like(a)
        initial_state = (
            torch.randn(*state_shape, dtype=torch.float16, device="cuda") * 0.01
        )
        state_indices = torch.tensor([0], dtype=torch.int32, device="cuda")

        intermediate_states = torch.empty(
            batch,
            steps,
            value_heads,
            key_dim,
            value_dim,
            dtype=torch.float16,
            device="cuda",
        )
        block_output = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=initial_state.clone(),
            initial_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            disable_state_update=True,
            intermediate_states_buffer=intermediate_states,
            intermediate_state_indices=state_indices,
        )

        decode_state = initial_state.clone()
        decode_outputs = []
        for step in range(steps):
            decode_outputs.append(
                fused_sigmoid_gating_delta_rule_update(
                    A_log=A_log,
                    dt_bias=dt_bias,
                    q=q[:, step : step + 1],
                    k=k[:, step : step + 1],
                    v=v[:, step : step + 1],
                    a=a[step : step + 1],
                    b=b[step : step + 1],
                    initial_state_source=decode_state,
                    initial_state_indices=state_indices,
                    use_qk_l2norm_in_kernel=True,
                    softplus_beta=1.0,
                    softplus_threshold=20.0,
                )
            )
        decode_output = torch.cat(decode_outputs, dim=1)

        self.assertTrue(
            torch.equal(block_output, decode_output),
            (
                "output mismatch: "
                f"max diff={(block_output - decode_output).abs().max().item()}, "
                f"count={torch.count_nonzero(block_output != decode_output).item()}"
            ),
        )
        self.assertTrue(
            torch.equal(intermediate_states[0, -1], decode_state[0]),
            (
                "final state mismatch: "
                f"max diff={(intermediate_states[0, -1] - decode_state[0]).abs().max().item()}, "
                f"count={torch.count_nonzero(intermediate_states[0, -1] != decode_state[0]).item()}"
            ),
        )

    def test_qwen35_122b_tp4_fp16_verify_block_matches_repeated_decode(self):
        # Qwen3.5-122B: global H=16/HV=64, therefore TP4 H=4/HV=16.
        self._run_fp16_verify_block_matches_repeated_decode(
            key_heads=4,
            value_heads=16,
            steps=16,
        )

    def test_qwen36_27b_tp4_fp16_verify_block_matches_repeated_decode(self):
        # Qwen3.6-27B: global H=16/HV=48, therefore TP4 H=4/HV=12.
        # The 3:1 mapping also exercises the recurrent kernel shape used by the
        # model's non-fused QKVZBA split path.
        self._run_fp16_verify_block_matches_repeated_decode(
            key_heads=4,
            value_heads=12,
            steps=16,
        )

    def test_qwen36_35b_a3b_tp4_fp16_verify_block8_matches_repeated_decode(self):
        # Qwen3.6-35B-A3B: global H=16/HV=32, therefore TP4 H=4/HV=8.
        # The model card currently demonstrates an eight-token override.
        self._run_fp16_verify_block_matches_repeated_decode(
            key_heads=4,
            value_heads=8,
            steps=8,
        )

    def test_qwen36_35b_a3b_tp4_fp16_verify_block16_matches_repeated_decode(self):
        # Also cover the checkpoint's configured block size and the common
        # explicit --speculative-dflash-block-size 16 deployment.
        self._run_fp16_verify_block_matches_repeated_decode(
            key_heads=4,
            value_heads=8,
            steps=16,
        )


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


if __name__ == "__main__":
    unittest.main()
