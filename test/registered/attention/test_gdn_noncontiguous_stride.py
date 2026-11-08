"""
Tests that fused_gdn_gating and fused_sigmoid_gating_delta_rule_update
produce correct results when a/b inputs are non-contiguous,
as happens with Qwen3.5-27B (v_per_group=3) via mixed_ba.split().
"""

import itertools
import unittest

import torch

from sglang.kernels.ops.attention import triton_gdn_fused_proj as fused_gdn_projection
from sglang.kernels.ops.attention.fla import layernorm_gated
from sglang.kernels.ops.attention.linear.fused_gdn_gating import fused_gdn_gating
from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    warmup_gdn_projection_norm,
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


def reference_norm(
    x,
    weight,
    z=None,
    *,
    bias=None,
    group_size=None,
    before=True,
    activation="swish",
    is_rms_norm=True,
    eps=1e-6,
):
    values = x.float()
    gate = None
    if z is not None:
        gate = z.reshape_as(x).float()
        gate = (
            gate * torch.sigmoid(gate) if activation == "swish" else torch.sigmoid(gate)
        )
        if not before:
            values = values * gate
    grouped = values.reshape(values.shape[0], -1, group_size or values.shape[1])
    if not is_rms_norm:
        grouped = grouped - grouped.mean(-1, keepdim=True)
    normalized = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + eps)
    output = normalized.reshape_as(values) * weight.float()
    if bias is not None:
        output = output + bias.float()
    if gate is not None and before:
        output = output * gate
    return output.to(x.dtype)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestGDNProjectionViews(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda", torch.cuda.current_device())
        cls.generator = torch.Generator(device=cls.device).manual_seed(397)

    def random(self, *shape, dtype):
        return torch.randn(
            *shape, device=self.device, dtype=dtype, generator=self.generator
        )

    def check_oracle(self, actual, expected):
        tolerance = (
            dict(rtol=1e-2, atol=1.5625e-2)
            if actual.dtype == torch.bfloat16
            else dict(rtol=2e-3, atol=2e-3)
        )
        torch.testing.assert_close(actual, expected, **tolerance)

    def test_00_warmup_covers_actual_projection_layout(self):
        kernel = layernorm_gated._layer_norm_fwd_1pass_kernel
        device_index = self.device.index
        with torch.inference_mode():
            for dtype, weight_dtype in (
                (torch.bfloat16, torch.bfloat16),
                (torch.bfloat16, torch.float32),
                (torch.float16, torch.float16),
                (torch.float16, torch.float32),
            ):
                self.assertTrue(
                    warmup_gdn_projection_norm(
                        device=self.device,
                        dtype=dtype,
                        weight_dtype=weight_dtype,
                        num_heads_qk=16,
                        num_heads_v=64,
                        head_qk=128,
                        head_v=128,
                        eps=1e-6,
                        group_size=None,
                        norm_before_gate=True,
                        activation="swish",
                    )
                )
                variants = set(kernel.device_caches[device_index][0])
                self.assertTrue(variants)
                for tokens in (2, 3, 5, 17, 65, 4096):
                    packed = torch.zeros(
                        (tokens, 20480), device=self.device, dtype=dtype
                    )
                    z = packed[:, 12288:].view(tokens, 64, 128)
                    x = torch.zeros((tokens * 64, 128), device=self.device, dtype=dtype)
                    weight = torch.ones(128, device=self.device, dtype=weight_dtype)
                    output = layernorm_gated.rms_norm_gated(
                        x=x,
                        weight=weight,
                        bias=None,
                        z=z,
                        is_rms_norm=True,
                    )
                    torch.testing.assert_close(output, x, rtol=0, atol=0)
                torch.cuda.synchronize(self.device)
                self.assertEqual(variants, set(kernel.device_caches[device_index][0]))
                for compiled in kernel.device_caches[device_index][0].values():
                    self.assertIsNotNone(compiled.module)
                    self.assertIsNotNone(compiled.function)
                    self.assertTrue(callable(compiled._run))

    def test_interleaved_projection_matches_torch_split(self):
        with torch.inference_mode():
            for dtype, tokens, geometry in itertools.product(
                (torch.bfloat16, torch.float16),
                (1, 17),
                ((2, 4, 64, 128), (4, 16, 128, 64)),
            ):
                kh, vh, kd, vd = geometry
                v_per_group = vh // kh
                packed = self.random(tokens, 2 * kh * kd + 2 * vh * vd, dtype=dtype)
                gates = self.random(tokens, 2 * vh, dtype=dtype)
                q, k, v, z = packed.view(tokens, kh, -1).split(
                    (kd, kd, v_per_group * vd, v_per_group * vd), dim=-1
                )
                b, a = gates.view(tokens, kh, 2 * v_per_group).chunk(2, dim=-1)
                expected = (
                    torch.cat([x.reshape(tokens, -1) for x in (q, k, v)], dim=-1),
                    z.reshape(tokens, vh, vd),
                    b.reshape(tokens, vh),
                    a.reshape(tokens, vh),
                )
                actual = fused_gdn_projection.fused_qkvzba_split_reshape_cat(
                    packed, gates, *geometry
                )
                for value, reference in zip(actual, expected):
                    torch.testing.assert_close(value, reference, rtol=0, atol=0)

    def test_projection_views_match_copy_helper(self):
        count = 0
        with torch.inference_mode():
            for dtype, tokens, geometry, padding in itertools.product(
                (torch.bfloat16, torch.float16),
                (0, 1, 17),
                (
                    (2, 2, 64, 128),
                    (2, 4, 128, 64),
                    (8, 24, 128, 128),
                    (16, 64, 128, 128),
                ),
                (0, 16),
            ):
                kh, vh, kd, vd = geometry
                width = 2 * kh * kd + 2 * vh * vd
                packed = self.random(tokens, width + padding, dtype=dtype)[:, :width]
                gates = self.random(
                    tokens, 2 * vh + padding, dtype=torch.float32 if padding else dtype
                )[:, : 2 * vh]
                saved_packed, saved_gates = packed.clone(), gates.clone()
                actual = fused_gdn_projection.qwen3_5_gdn_prefill_projection_views(
                    packed, gates, *geometry
                )
                if tokens:
                    expected = (
                        fused_gdn_projection.fused_qkvzba_split_reshape_cat_contiguous(
                            packed.contiguous(), gates.contiguous(), *geometry
                        )
                    )
                    for output, reference in zip(actual, expected):
                        torch.testing.assert_close(output, reference, rtol=0, atol=0)
                    for output in actual[:2]:
                        self.assertEqual(
                            output.untyped_storage().data_ptr(),
                            packed.untyped_storage().data_ptr(),
                        )
                    for output in actual[2:]:
                        self.assertEqual(
                            output.untyped_storage().data_ptr(),
                            gates.untyped_storage().data_ptr(),
                        )
                else:
                    self.assertEqual(tuple(actual[1].shape), (0, vh, vd))
                torch.testing.assert_close(packed, saved_packed, rtol=0, atol=0)
                torch.testing.assert_close(gates, saved_gates, rtol=0, atol=0)
                count += 1
        self.assertEqual(count, 48)

    def test_3d_z_matches_2d_and_torch(self):
        count = 0
        with torch.inference_mode():
            for dtype, shape, before, activation, group in itertools.product(
                (torch.bfloat16, torch.float16),
                ((1, 4), (17, 4), (97, 3)),
                (False, True),
                ("swish", "sigmoid"),
                (None, 64),
            ):
                tokens, heads = shape
                with self.subTest(
                    dtype=dtype,
                    shape=shape,
                    before=before,
                    activation=activation,
                    group=group,
                ):
                    packed = self.random(tokens, 256 + heads * 128, dtype=dtype)
                    z = packed[:, 256:].view(tokens, heads, 128)
                    x = self.random(tokens * heads, 128, dtype=dtype)
                    weight = self.random(128, dtype=dtype if before else torch.float32)
                    arguments = dict(
                        x=x,
                        weight=weight,
                        bias=None,
                        group_size=group,
                        norm_before_gate=before,
                        is_rms_norm=True,
                        activation=activation,
                    )
                    expected = layernorm_gated.rms_norm_gated(
                        z=z.reshape_as(x), **arguments
                    )
                    actual = layernorm_gated.rms_norm_gated(z=z, **arguments)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.check_oracle(
                        actual,
                        reference_norm(
                            x,
                            weight,
                            z,
                            group_size=group,
                            before=before,
                            activation=activation,
                        ),
                    )
                    count += 1
        self.assertEqual(count, 48)

    def test_model_sized_norm(self):
        with torch.inference_mode():
            for dtype in (torch.bfloat16, torch.float16):
                packed = self.random(4096, 20480, dtype=dtype)
                z = packed[:, 12288:].view(4096, 64, 128)
                x = self.random(4096 * 64, 128, dtype=dtype)
                weight = self.random(128, dtype=dtype)
                expected = layernorm_gated.rms_norm_gated(
                    x=x, weight=weight, bias=None, z=z.reshape_as(x), is_rms_norm=True
                )
                actual = layernorm_gated.rms_norm_gated(
                    x=x, weight=weight, bias=None, z=z, is_rms_norm=True
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_shared_2d_layernorm_bias_and_strides(self):
        count = 0
        with torch.inference_mode():
            for dtype, has_gate, strided in itertools.product(
                (torch.bfloat16, torch.float16), (False, True), (False, True)
            ):
                x = self.random(35, 256 if strided else 128, dtype=dtype)[:, :128]
                z = self.random(35, 256, dtype=dtype)[:, :128] if has_gate else None
                weight, bias = (self.random(128, dtype=torch.float32) for _ in range(2))
                group = 64 if strided else None
                saved_x, saved_z = x.clone(), z.clone() if z is not None else None
                actual = layernorm_gated.rms_norm_gated(
                    x=x, weight=weight, bias=bias, z=z, group_size=group
                )
                expected = reference_norm(
                    x, weight, z, bias=bias, group_size=group, is_rms_norm=False
                )
                self.check_oracle(actual, expected)
                torch.testing.assert_close(x, saved_x, rtol=0, atol=0)
                if z is not None:
                    torch.testing.assert_close(z, saved_z, rtol=0, atol=0)
                count += 1
        self.assertEqual(count, 8)

    def test_backward_remains_unsupported(self):
        for strided in (False, True):
            packed = self.random(2, 20480, dtype=torch.bfloat16)
            z = packed[:, 12288:].view(2, 64, 128)
            x = self.random(128, 128, dtype=torch.bfloat16).requires_grad_()
            weight = self.random(128, dtype=torch.bfloat16).requires_grad_()
            output = layernorm_gated.layernorm_fn(
                x, weight, None, z=z if strided else z.reshape_as(x), is_rms_norm=True
            )
            with self.assertRaisesRegex(NotImplementedError, "backward|vjp"):
                output.float().sum().backward()


if __name__ == "__main__":
    unittest.main()
