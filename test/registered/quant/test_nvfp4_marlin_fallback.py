"""Tests for NVFP4 Marlin fallback on non-Blackwell GPUs (SM75+)."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=480, suite="stage-b-test-1-gpu-large")

_FP4_MARLIN_GROUP_SIZE = 16

_FP4_E2M1_LUT_VALUES = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _check_requirements():
    from sglang.srt.utils import is_cuda

    if not is_cuda():
        return False
    from sglang.srt.layers.quantization.marlin_utils_fp4 import is_fp4_marlin_supported

    if not is_fp4_marlin_supported():
        return False
    return True


def _dequant_fp4_weights(
    raw_weight: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Dequantize uint8-packed FP4 E2M1 weights to float32 via lookup table."""
    lut = torch.tensor(_FP4_E2M1_LUT_VALUES, dtype=torch.float32, device=device)
    lo = (raw_weight.int() & 0x0F).long()
    hi = ((raw_weight.int() >> 4) & 0x0F).long()
    return torch.stack([lut[lo], lut[hi]], dim=-1).reshape(
        raw_weight.shape[0], raw_weight.shape[1] * 2
    )


class _FakeLayer(torch.nn.Module):
    """Minimal stand-in for a quantized layer in unit tests."""

    pass


# ---------------------------------------------------------------------------
# Linear (non-MoE) tests
# ---------------------------------------------------------------------------
class TestNvfp4MarlinLinear(CustomTestCase):
    """Test the FP4 Marlin linear layer fallback (non-MoE)."""

    def setUp(self):
        if not _check_requirements():
            self.skipTest("Requirements not met (CUDA unavailable or SM < 75)")
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

    # -- helpers -------------------------------------------------------------

    def _make_fake_fp4_layer(self, N, K):
        layer = _FakeLayer()
        layer.params_dtype = self.dtype
        layer.input_size_per_partition = K
        layer.output_size_per_partition = N

        layer.weight = torch.nn.Parameter(
            torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=self.device),
            requires_grad=False,
        )
        layer.weight_scale = torch.nn.Parameter(
            torch.ones(
                N,
                K // _FP4_MARLIN_GROUP_SIZE,
                dtype=torch.float8_e4m3fn,
                device=self.device,
            ),
            requires_grad=False,
        )
        layer.weight_scale_2_marlin = torch.nn.Parameter(
            torch.tensor(1.0, dtype=torch.float32, device=self.device),
            requires_grad=False,
        )
        return layer

    def _run_fp4_marlin_vs_reference(self, M, N, K):
        """Prepare a layer, run the Marlin kernel, return (kernel_out, ref_out)."""
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            apply_fp4_marlin_linear,
            prepare_fp4_layer_for_marlin,
        )

        raw_weight = torch.randint(
            0, 256, (N, K // 2), dtype=torch.uint8, device=self.device
        )
        dq_weight = _dequant_fp4_weights(raw_weight, self.device)

        raw_scale = torch.full(
            (N, K // _FP4_MARLIN_GROUP_SIZE),
            1.0,
            dtype=torch.float8_e4m3fn,
            device=self.device,
        )
        global_scale_val = torch.tensor(1.0, dtype=torch.float32, device=self.device)

        x = torch.randn(M, K, dtype=self.dtype, device=self.device)
        ref_output = (x.float() @ dq_weight.T).to(self.dtype)

        layer = self._make_fake_fp4_layer(N, K)
        layer.weight = torch.nn.Parameter(raw_weight, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(raw_scale, requires_grad=False)
        layer.weight_scale_2_marlin = torch.nn.Parameter(
            global_scale_val.to(self.dtype), requires_grad=False
        )

        prepare_fp4_layer_for_marlin(
            layer,
            weight_attr="weight",
            weight_scale_attr="weight_scale",
            weight_global_scale_attr="weight_scale_2_marlin",
        )

        marlin_output = apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_scale_2_marlin,
            workspace=layer.marlin_workspace,
            size_n=N,
            size_k=K,
        )
        return marlin_output, ref_output

    # -- tests ---------------------------------------------------------------

    def test_prepare_and_apply_fp4_marlin_linear(self):
        """Smoke test: shape and dtype are correct after prepare + apply."""
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            apply_fp4_marlin_linear,
            prepare_fp4_layer_for_marlin,
        )

        N, K, M = 256, 128, 16
        layer = self._make_fake_fp4_layer(N, K)
        prepare_fp4_layer_for_marlin(
            layer,
            weight_attr="weight",
            weight_scale_attr="weight_scale",
            weight_global_scale_attr="weight_scale_2_marlin",
        )

        self.assertTrue(hasattr(layer, "marlin_workspace"))

        x = torch.randn(M, K, dtype=self.dtype, device=self.device)
        output = apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_scale_2_marlin,
            workspace=layer.marlin_workspace,
            size_n=N,
            size_k=K,
        )
        self.assertEqual(output.shape, (M, N))
        self.assertEqual(output.dtype, self.dtype)

    def test_fp4_marlin_numerical_correctness(self):
        """Kernel output vs BF16 dequant reference (cosine sim, MAE, assert_close)."""
        N, K, M = 256, 256, 32
        marlin_output, ref_output = self._run_fp4_marlin_vs_reference(M, N, K)

        self.assertEqual(marlin_output.shape, ref_output.shape)
        self.assertEqual(marlin_output.dtype, ref_output.dtype)

        cos_sim = torch.nn.functional.cosine_similarity(
            marlin_output.float().flatten(), ref_output.float().flatten(), dim=0
        )
        self.assertGreater(
            cos_sim.item(),
            0.99,
            f"Cosine similarity {cos_sim.item():.6f} too low",
        )

        rel_mae = torch.mean(
            torch.abs(marlin_output.float() - ref_output.float())
        ) / torch.mean(torch.abs(ref_output.float()))
        self.assertLess(
            rel_mae.item(),
            0.04,
            f"Relative MAE {rel_mae.item():.6f} >= 0.04",
        )

        torch.testing.assert_close(marlin_output, ref_output, atol=1e-1, rtol=1e-1)

    def test_fp4_marlin_multiple_shapes(self):
        """Numerical correctness across various (M, N, K) dimensions."""
        shapes = [
            (1, 256, 256),
            (16, 512, 128),
            (64, 128, 512),
            (32, 256, 256),
        ]
        for M, N, K in shapes:
            with self.subTest(M=M, N=N, K=K):
                marlin_out, ref_out = self._run_fp4_marlin_vs_reference(M, N, K)
                rel_mae = torch.mean(
                    torch.abs(marlin_out.float() - ref_out.float())
                ) / torch.mean(torch.abs(ref_out.float()))
                self.assertLess(
                    rel_mae.item(),
                    0.04,
                    f"Shape ({M},{N},{K}): relative MAE {rel_mae.item():.6f} >= 0.04",
                )

    def test_fp4_marlin_linear_with_bias(self):
        """Verify output_with_bias == output_no_bias + bias."""
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            apply_fp4_marlin_linear,
            prepare_fp4_layer_for_marlin,
        )

        N, K, M = 256, 128, 16
        layer = self._make_fake_fp4_layer(N, K)
        prepare_fp4_layer_for_marlin(
            layer,
            weight_attr="weight",
            weight_scale_attr="weight_scale",
            weight_global_scale_attr="weight_scale_2_marlin",
        )

        x = torch.randn(M, K, dtype=self.dtype, device=self.device)
        bias = torch.randn(N, dtype=self.dtype, device=self.device)

        common = dict(
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_scale_2_marlin,
            workspace=layer.marlin_workspace,
            size_n=N,
            size_k=K,
        )
        output_no_bias = apply_fp4_marlin_linear(input=x, **common)
        output_with_bias = apply_fp4_marlin_linear(input=x, bias=bias, **common)

        torch.testing.assert_close(
            output_with_bias, output_no_bias + bias, atol=1e-5, rtol=1e-5
        )

    def test_fp4_marlin_registered_op_numerical(self):
        """torch.ops.sglang.apply_fp4_marlin_linear matches the direct Python call."""
        import sglang.srt.layers.quantization.marlin_utils_fp4  # noqa: F401
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            apply_fp4_marlin_linear,
            prepare_fp4_layer_for_marlin,
        )

        N, K, M = 256, 128, 16
        layer = self._make_fake_fp4_layer(N, K)
        prepare_fp4_layer_for_marlin(
            layer,
            weight_attr="weight",
            weight_scale_attr="weight_scale",
            weight_global_scale_attr="weight_scale_2_marlin",
        )

        x = torch.randn(M, K, dtype=self.dtype, device=self.device)

        common = dict(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_scale_2_marlin,
            workspace=layer.marlin_workspace,
            size_n=N,
            size_k=K,
        )

        direct_out = apply_fp4_marlin_linear(**common)
        op_out = torch.ops.sglang.apply_fp4_marlin_linear(**common)

        self.assertEqual(op_out.shape, direct_out.shape)
        self.assertEqual(op_out.dtype, direct_out.dtype)
        torch.testing.assert_close(op_out, direct_out, atol=0, rtol=0)

    def test_fp4_marlin_3d_input(self):
        """Verify correct reshape for 3-D input (batch, seq_len, K)."""
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            apply_fp4_marlin_linear,
            prepare_fp4_layer_for_marlin,
        )

        N, K = 256, 128
        batch, seq_len = 2, 8
        layer = self._make_fake_fp4_layer(N, K)
        prepare_fp4_layer_for_marlin(
            layer,
            weight_attr="weight",
            weight_scale_attr="weight_scale",
            weight_global_scale_attr="weight_scale_2_marlin",
        )

        x_3d = torch.randn(batch, seq_len, K, dtype=self.dtype, device=self.device)
        x_2d = x_3d.reshape(-1, K)

        common = dict(
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_scale_2_marlin,
            workspace=layer.marlin_workspace,
            size_n=N,
            size_k=K,
        )

        out_3d = apply_fp4_marlin_linear(input=x_3d, **common)
        out_2d = apply_fp4_marlin_linear(input=x_2d, **common)

        self.assertEqual(out_3d.shape, (batch, seq_len, N))
        self.assertEqual(out_3d.dtype, self.dtype)
        torch.testing.assert_close(out_3d.reshape(-1, N), out_2d, atol=0, rtol=0)

    def test_fake_apply_fp4_marlin_linear(self):
        """Fake impl for PCG tracing must return the correct shape and dtype."""
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            fake_apply_fp4_marlin_linear,
        )

        N, K = 256, 128

        for input_shape in [(16, K), (2, 8, K)]:
            with self.subTest(input_shape=input_shape):
                x = torch.randn(*input_shape, dtype=self.dtype, device=self.device)
                out = fake_apply_fp4_marlin_linear(
                    input=x,
                    weight=torch.empty(0, device=self.device),
                    weight_scale=torch.empty(0, device=self.device),
                    weight_global_scale=torch.empty(0, device=self.device),
                    workspace=torch.empty(0, device=self.device),
                    size_n=N,
                    size_k=K,
                )
                expected_shape = input_shape[:-1] + (N,)
                self.assertEqual(out.shape, expected_shape)
                self.assertEqual(out.dtype, self.dtype)

    def test_prepare_rejects_bad_weight_shape(self):
        """prepare_fp4_layer_for_marlin must raise on mismatched weight shape."""
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            prepare_fp4_layer_for_marlin,
        )

        N, K = 256, 128
        layer = _FakeLayer()
        layer.params_dtype = self.dtype
        layer.input_size_per_partition = K
        layer.output_size_per_partition = N

        layer.weight = torch.nn.Parameter(
            torch.randint(
                0, 256, (N + 1, K // 2), dtype=torch.uint8, device=self.device
            ),
            requires_grad=False,
        )
        layer.weight_scale = torch.nn.Parameter(
            torch.ones(
                N,
                K // _FP4_MARLIN_GROUP_SIZE,
                dtype=torch.float8_e4m3fn,
                device=self.device,
            ),
            requires_grad=False,
        )
        layer.weight_scale_2_marlin = torch.nn.Parameter(
            torch.tensor(1.0, dtype=torch.float32, device=self.device),
            requires_grad=False,
        )

        with self.assertRaises(AssertionError):
            prepare_fp4_layer_for_marlin(
                layer,
                weight_attr="weight",
                weight_scale_attr="weight_scale",
                weight_global_scale_attr="weight_scale_2_marlin",
            )

    def test_prepare_fp4_layer_permutes_bias(self):
        """prepare_fp4_layer_for_marlin must permute layer.bias when present."""
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            prepare_fp4_layer_for_marlin,
        )

        N, K = 256, 128
        layer = self._make_fake_fp4_layer(N, K)
        original_bias = torch.randn(N, dtype=self.dtype, device=self.device)
        layer.bias = torch.nn.Parameter(original_bias.clone(), requires_grad=False)

        prepare_fp4_layer_for_marlin(
            layer,
            weight_attr="weight",
            weight_scale_attr="weight_scale",
            weight_global_scale_attr="weight_scale_2_marlin",
        )

        self.assertEqual(layer.bias.shape, (N,))
        self.assertEqual(layer.bias.dtype, self.dtype)
        self.assertFalse(
            torch.equal(layer.bias.data, original_bias),
            "Bias should be permuted by prepare_fp4_layer_for_marlin",
        )

    def test_fp4_marlin_custom_op_registration(self):
        """apply_fp4_marlin_linear must be registered as torch.ops.sglang for PCG."""
        import sglang.srt.layers.quantization.marlin_utils_fp4  # noqa: F401

        self.assertTrue(
            hasattr(torch.ops.sglang, "apply_fp4_marlin_linear"),
            "apply_fp4_marlin_linear not registered as a custom op",
        )

    def test_nvfp4_marlin_scale_values_correctness(self):
        """Verify scale conversion produces analytically correct values."""
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            nvfp4_marlin_process_global_scale,
            nvfp4_marlin_process_scales,
        )

        # -- global scale: BF16 --
        # fp4_exp=2, target_exp=8  =>  bias = 2^7 - 2^1 = 126
        # result = 1.0 * 2^(126-7) = 2^119
        gs_bf16 = torch.tensor(1.0, dtype=torch.bfloat16, device=self.device)
        result_bf16 = nvfp4_marlin_process_global_scale(gs_bf16)
        expected_bf16 = torch.tensor(2.0**119, dtype=torch.bfloat16, device=self.device)
        self.assertEqual(
            result_bf16.item(),
            expected_bf16.item(),
            f"BF16 global_scale(1.0): expected 2^119, got {result_bf16.item()}",
        )
        self.assertEqual(result_bf16.dtype, torch.bfloat16)

        # -- global scale: FP16 --
        # fp4_exp=2, target_exp=5  =>  bias = 2^4 - 2^1 = 14
        # result = 1.0 * 2^(14-7) = 128
        gs_fp16 = torch.tensor(1.0, dtype=torch.float16, device=self.device)
        result_fp16 = nvfp4_marlin_process_global_scale(gs_fp16)
        self.assertEqual(
            result_fp16.item(),
            128.0,
            f"FP16 global_scale(1.0): expected 128.0, got {result_fp16.item()}",
        )
        self.assertEqual(result_fp16.dtype, torch.float16)

        # -- global scale: linearity --
        gs_2 = torch.tensor(2.0, dtype=torch.bfloat16, device=self.device)
        result_2 = nvfp4_marlin_process_global_scale(gs_2)
        self.assertAlmostEqual(
            result_2.item(),
            2.0 * result_bf16.item(),
            places=0,
            msg="Global scale processing should be linear",
        )

        # -- per-group scales: structural properties --
        N, K_div_group = 64, 16
        raw_scale = torch.ones(
            N, K_div_group, dtype=torch.float8_e4m3fn, device=self.device
        ).to(self.dtype)
        processed = nvfp4_marlin_process_scales(raw_scale)

        self.assertEqual(processed.dtype, torch.float8_e4m3fn)
        self.assertEqual(processed.shape, (N, K_div_group))
        self.assertFalse(torch.isnan(processed.to(self.dtype)).any())

        # Deterministic
        self.assertTrue(torch.equal(processed, nvfp4_marlin_process_scales(raw_scale)))

        # Large scales (448 = FP8 E4M3 max) must not produce NaN
        large_scale = torch.full(
            (N, K_div_group), 448.0, dtype=self.dtype, device=self.device
        )
        proc_large = nvfp4_marlin_process_scales(large_scale)
        self.assertFalse(torch.isnan(proc_large.to(self.dtype)).any())
        self.assertEqual(proc_large.shape, (N, K_div_group))


# ---------------------------------------------------------------------------
# MoE tests
# ---------------------------------------------------------------------------
class TestNvfp4MarlinMoe(CustomTestCase):
    """Test the FP4 Marlin MoE fallback."""

    def setUp(self):
        if not _check_requirements():
            self.skipTest("Requirements not met (CUDA unavailable or SM < 75)")
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        try:
            from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack

            self._gptq_marlin_repack = gptq_marlin_repack
        except ImportError:
            self.skipTest("gptq_marlin_repack JIT compilation not available")
        self._perm = torch.empty(0, dtype=torch.int, device=self.device)

    # -- helpers -------------------------------------------------------------

    def _repack_fp4_weight(self, raw_fp4, size_k, size_n):
        """Repack raw uint8 FP4 weights into Marlin tile layout."""
        qw = raw_fp4.view(torch.int32).T.contiguous()
        return self._gptq_marlin_repack(qw, self._perm, size_k, size_n, num_bits=4)

    def _make_marlin_scale(self, size_k, size_n):
        from sglang.srt.layers.quantization.marlin_utils import marlin_permute_scales
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            nvfp4_marlin_process_scales,
        )

        raw = torch.ones(
            size_k // _FP4_MARLIN_GROUP_SIZE,
            size_n,
            dtype=self.dtype,
            device=self.device,
        )
        permuted = marlin_permute_scales(raw, size_k, size_n, _FP4_MARLIN_GROUP_SIZE)
        return nvfp4_marlin_process_scales(permuted)

    def _make_processed_global_scale(self):
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            nvfp4_marlin_process_global_scale,
        )

        return nvfp4_marlin_process_global_scale(
            torch.tensor(1.0, dtype=self.dtype, device=self.device)
        )

    # -- tests ---------------------------------------------------------------

    def test_fused_marlin_moe_fp4(self):
        """Smoke test: shape, dtype, no NaN for multi-expert MoE."""
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )

        E, K, N, topk, M = 4, 128, 64, 2, 8

        def _rand_weight(size_k, size_n):
            raw = torch.randint(
                0, 256, (size_n, size_k // 2), dtype=torch.uint8, device=self.device
            )
            return self._repack_fp4_weight(raw, size_k, size_n)

        w1 = torch.stack([_rand_weight(K, 2 * N) for _ in range(E)])
        w2 = torch.stack([_rand_weight(N, K) for _ in range(E)])
        w1_scale = torch.stack([self._make_marlin_scale(K, 2 * N) for _ in range(E)])
        w2_scale = torch.stack([self._make_marlin_scale(N, K) for _ in range(E)])

        gs = self._make_processed_global_scale()
        w1_gs = gs.expand(E)
        w2_gs = gs.expand(E)

        hidden = torch.randn(M, K, dtype=self.dtype, device=self.device)
        gating = torch.randn(M, E, dtype=self.dtype, device=self.device)
        topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)

        output = fused_marlin_moe(
            hidden_states=hidden,
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            gating_output=gating,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            num_bits=4,
            w1_global_scale=w1_gs,
            w2_global_scale=w2_gs,
        )

        self.assertEqual(output.shape, (M, K))
        self.assertEqual(output.dtype, self.dtype)
        self.assertFalse(torch.isnan(output).any(), "Output contains NaN!")

    def test_fused_marlin_moe_fp4_numerical(self):
        """E=1, topk=1 MoE output vs dequant reference (SiLU-gated)."""
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )

        E, K, N, topk, M = 1, 128, 64, 1, 8

        raw_w1 = torch.randint(
            0, 256, (2 * N, K // 2), dtype=torch.uint8, device=self.device
        )
        raw_w2 = torch.randint(
            0, 256, (K, N // 2), dtype=torch.uint8, device=self.device
        )
        dq_w1 = _dequant_fp4_weights(raw_w1, self.device)
        dq_w2 = _dequant_fp4_weights(raw_w2, self.device)

        w1 = self._repack_fp4_weight(raw_w1, K, 2 * N).unsqueeze(0)
        w2 = self._repack_fp4_weight(raw_w2, N, K).unsqueeze(0)
        w1_scale = self._make_marlin_scale(K, 2 * N).unsqueeze(0)
        w2_scale = self._make_marlin_scale(N, K).unsqueeze(0)

        gs = self._make_processed_global_scale()
        w1_gs = gs.unsqueeze(0)
        w2_gs = gs.unsqueeze(0)

        x = torch.randn(M, K, dtype=self.dtype, device=self.device) * 0.1
        gating = torch.ones(M, E, dtype=self.dtype, device=self.device)
        topk_weights = torch.ones(M, topk, dtype=self.dtype, device=self.device)
        topk_ids = torch.zeros(M, topk, dtype=torch.int64, device=self.device)

        output = fused_marlin_moe(
            hidden_states=x,
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            gating_output=gating,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            num_bits=4,
            w1_global_scale=w1_gs,
            w2_global_scale=w2_gs,
        )

        gate_up = x.float() @ dq_w1.T
        gate, up = gate_up[:, :N], gate_up[:, N:]
        ref_output = ((torch.nn.functional.silu(gate) * up) @ dq_w2.T).to(self.dtype)

        self.assertEqual(output.shape, ref_output.shape)
        self.assertFalse(torch.isinf(output).any(), "MoE output contains Inf")
        self.assertFalse(torch.isnan(output).any(), "MoE output contains NaN")

        finite = torch.isfinite(ref_output) & torch.isfinite(output)
        if finite.any():
            cos_sim = torch.nn.functional.cosine_similarity(
                output[finite].float().flatten(),
                ref_output[finite].float().flatten(),
                dim=0,
            )
            self.assertGreater(
                cos_sim.item(),
                0.90,
                f"MoE cosine similarity {cos_sim.item():.4f} too low",
            )

    def test_prepare_moe_fp4_layer_for_marlin(self):
        """Weight repacking produces correct shapes for all expert tensors."""
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            prepare_moe_fp4_layer_for_marlin,
        )

        E, K, N = 4, 128, 64

        class _FakeMoeRunnerConfig:
            is_gated = True

        layer = _FakeLayer()
        layer.num_local_experts = E
        layer.intermediate_size_per_partition = N
        layer.params_dtype = self.dtype
        layer.moe_runner_config = _FakeMoeRunnerConfig()

        layer.w13_weight = torch.nn.Parameter(
            torch.randint(
                0, 256, (E, 2 * N, K // 2), dtype=torch.uint8, device=self.device
            ),
            requires_grad=False,
        )
        layer.w2_weight = torch.nn.Parameter(
            torch.randint(
                0, 256, (E, K, N // 2), dtype=torch.uint8, device=self.device
            ),
            requires_grad=False,
        )
        layer.w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                E,
                2 * N,
                K // _FP4_MARLIN_GROUP_SIZE,
                dtype=torch.float8_e4m3fn,
                device=self.device,
            ),
            requires_grad=False,
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                E,
                K,
                N // _FP4_MARLIN_GROUP_SIZE,
                dtype=torch.float8_e4m3fn,
                device=self.device,
            ),
            requires_grad=False,
        )
        layer.w13_weight_scale_2 = torch.nn.Parameter(
            torch.ones(E, 2, dtype=torch.float32, device=self.device),
            requires_grad=False,
        )
        layer.w2_weight_scale_2 = torch.nn.Parameter(
            torch.ones(E, dtype=torch.float32, device=self.device),
            requires_grad=False,
        )

        prepare_moe_fp4_layer_for_marlin(layer)

        self.assertEqual(layer.w13_weight.shape[0], E)
        self.assertEqual(layer.w2_weight.shape[0], E)
        self.assertEqual(layer.w13_weight_scale_2.shape, (E,))
        self.assertEqual(layer.w2_weight_scale_2.shape, (E,))


# ---------------------------------------------------------------------------
# Support / capability tests
# ---------------------------------------------------------------------------
class TestFp4MarlinSupport(CustomTestCase):
    """Test the capability detection functions."""

    def test_is_fp4_marlin_supported(self):
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            is_fp4_marlin_supported,
        )

        result = is_fp4_marlin_supported()
        if torch.cuda.is_available() and torch.version.hip is None:
            cap = torch.cuda.get_device_capability()
            sm = cap[0] * 10 + cap[1]
            expected = sm >= 75
            self.assertEqual(result, expected)
        elif torch.version.hip is not None:
            self.assertFalse(result, "FP4 Marlin should not be supported on ROCm/HIP")

    def test_min_capability_changed(self):
        """get_min_capability() must return 75 (not 100)."""
        from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config

        cap = ModelOptFp4Config.get_min_capability()
        self.assertEqual(cap, 75, f"Expected 75, got {cap}")

    def test_should_use_fp4_marlin_fallback(self):
        """should_use_fp4_marlin_fallback returns True on non-Blackwell SM>=75."""
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            should_use_fp4_marlin_fallback,
        )

        result = should_use_fp4_marlin_fallback()
        self.assertIsInstance(result, bool)

        if torch.cuda.is_available() and torch.version.hip is None:
            cap = torch.cuda.get_device_capability()
            sm = cap[0] * 10 + cap[1]
            is_blackwell = sm >= 100
            if is_blackwell:
                self.assertFalse(
                    result,
                    "Blackwell GPUs should NOT use Marlin fallback (native FP4)",
                )
            elif sm >= 75:
                self.assertTrue(
                    result,
                    f"SM{sm} should use Marlin fallback, but got False",
                )
            else:
                self.assertFalse(
                    result,
                    f"SM{sm} should not support FP4 Marlin at all",
                )


if __name__ == "__main__":
    unittest.main(verbosity=3)
