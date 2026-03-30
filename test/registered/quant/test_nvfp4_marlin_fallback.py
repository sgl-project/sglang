"""Tests for NVFP4 Marlin fallback on non-Blackwell GPUs (SM75+)."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=300, suite="stage-b-test-large-1-gpu")


def _check_requirements():
    from sglang.srt.utils import is_cuda

    if not is_cuda():
        return False
    from sglang.srt.layers.quantization.marlin_utils_fp4 import is_fp4_marlin_supported

    if not is_fp4_marlin_supported():
        return False
    return True


class TestNvfp4MarlinLinear(CustomTestCase):
    """Test the FP4 Marlin linear layer fallback (non-MoE)."""

    def setUp(self):
        if not _check_requirements():
            self.skipTest("Requirements not met (CUDA unavailable or SM < 75)")
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

    def _make_fake_fp4_layer(self, N, K):
        """Build a fake layer with NVFP4 weight attributes."""

        class FakeLayer(torch.nn.Module):
            pass

        layer = FakeLayer()
        layer.params_dtype = self.dtype
        layer.input_size_per_partition = K
        layer.output_size_per_partition = N

        layer.weight = torch.nn.Parameter(
            torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=self.device),
            requires_grad=False,
        )

        layer.weight_scale = torch.nn.Parameter(
            torch.ones(N, K // 16, dtype=torch.float8_e4m3fn, device=self.device),
            requires_grad=False,
        )

        layer.weight_scale_2_marlin = torch.nn.Parameter(
            torch.tensor(1.0, dtype=torch.float32, device=self.device),
            requires_grad=False,
        )

        return layer

    def test_prepare_and_apply_fp4_marlin_linear(self):
        """Test prepare_fp4_layer_for_marlin + apply_fp4_marlin_linear."""
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            apply_fp4_marlin_linear,
            prepare_fp4_layer_for_marlin,
        )

        N, K = 256, 128
        layer = self._make_fake_fp4_layer(N, K)

        prepare_fp4_layer_for_marlin(
            layer,
            weight_attr="weight",
            weight_scale_attr="weight_scale",
            weight_global_scale_attr="weight_scale_2_marlin",
        )

        self.assertTrue(hasattr(layer, "marlin_workspace"))

        M = 16
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
        """Compare FP4 Marlin output against a BF16 dequantized reference."""
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            apply_fp4_marlin_linear,
            prepare_fp4_layer_for_marlin,
        )

        N, K = 256, 256
        M = 32

        # FP4 E2M1 lookup table (nibble value → float)
        FP4_E2M1_LUT = torch.tensor(
            [
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
            ],
            dtype=torch.float32,
            device=self.device,
        )

        raw_weight = torch.randint(
            0, 256, (N, K // 2), dtype=torch.uint8, device=self.device
        )

        # Dequantize FP4 to float for reference
        lo = (raw_weight.int() & 0x0F).long()
        hi = ((raw_weight.int() >> 4) & 0x0F).long()
        dq_weight = torch.stack([FP4_E2M1_LUT[lo], FP4_E2M1_LUT[hi]], dim=-1).reshape(
            N, K
        )

        # Per-group scale = 1.0 and global_scale = 1.0 for clean comparison
        raw_scale = torch.full(
            (N, K // 16), 1.0, dtype=torch.float8_e4m3fn, device=self.device
        )
        global_scale_val = torch.tensor(1.0, dtype=torch.float32, device=self.device)

        # Reference: fp4_dequant * per_group_scale(=1) * global_scale(=1)
        x = torch.randn(M, K, dtype=self.dtype, device=self.device)
        ref_output = (x.float() @ dq_weight.T).to(self.dtype)

        # Build a fake layer and run through the Marlin pipeline
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

        self.assertEqual(marlin_output.shape, ref_output.shape)
        self.assertEqual(marlin_output.dtype, ref_output.dtype)

        cos_sim = torch.nn.functional.cosine_similarity(
            marlin_output.float().flatten(), ref_output.float().flatten(), dim=0
        )
        self.assertGreater(
            cos_sim.item(),
            0.99,
            f"Cosine similarity {cos_sim.item():.6f} too low; "
            "Marlin output diverges from BF16 dequant reference.",
        )

    def test_fp4_marlin_custom_op_registration(self):
        """Verify apply_fp4_marlin_linear is registered as torch.ops.sglang for PCG."""
        import sglang.srt.layers.quantization.marlin_utils_fp4  # noqa: F401

        self.assertTrue(
            hasattr(torch.ops.sglang, "apply_fp4_marlin_linear"),
            "apply_fp4_marlin_linear not registered as a custom op",
        )

    def test_nvfp4_marlin_process_scales(self):
        """Test that scale conversion functions produce non-NaN outputs."""
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            nvfp4_marlin_process_global_scale,
            nvfp4_marlin_process_scales,
        )

        N, K_div_group = 64, 16

        raw_scale = torch.ones(
            N, K_div_group, dtype=torch.float8_e4m3fn, device=self.device
        )
        raw_scale = raw_scale.to(self.dtype)
        processed = nvfp4_marlin_process_scales(raw_scale)

        self.assertFalse(torch.isnan(processed.to(self.dtype)).any())
        self.assertEqual(processed.dtype, torch.float8_e4m3fn)

        # Large scales (448 = FP8 E4M3 max): int16 wrapping preserves bit patterns
        large_scale = torch.full(
            (N, K_div_group), 448.0, dtype=self.dtype, device=self.device
        )
        proc_large = nvfp4_marlin_process_scales(large_scale)
        self.assertFalse(torch.isnan(proc_large.to(self.dtype)).any())

        global_scale = torch.tensor(1.0, dtype=self.dtype, device=self.device)
        processed_global = nvfp4_marlin_process_global_scale(global_scale)
        self.assertFalse(torch.isnan(processed_global).any())
        self.assertEqual(processed_global.dtype, self.dtype)


class TestNvfp4MarlinMoe(CustomTestCase):
    """Test the FP4 Marlin MoE fallback."""

    def setUp(self):
        if not _check_requirements():
            self.skipTest("Requirements not met (CUDA unavailable or SM < 75)")
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

    def test_fused_marlin_moe_fp4(self):
        """Test fused_marlin_moe with FP4 global scales."""
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )
        from sglang.srt.layers.quantization.marlin_utils import marlin_permute_scales
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            nvfp4_marlin_process_scales,
        )

        try:
            from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack
        except ImportError:
            self.skipTest("gptq_marlin_repack not available")

        E, K, N, topk, M = 4, 128, 64, 2, 8
        FP4_MARLIN_GROUP_SIZE = 16
        perm = torch.empty(0, dtype=torch.int, device=self.device)

        def _make_marlin_weight(size_k, size_n):
            raw_fp4 = torch.randint(0, 256, (size_n, size_k // 2), dtype=torch.uint8)
            qweight = raw_fp4.view(torch.int32).T.contiguous().to(self.device)
            return gptq_marlin_repack(qweight, perm, size_k, size_n, num_bits=4)

        def _make_marlin_scale(size_k, size_n):
            raw = torch.ones(
                size_k // FP4_MARLIN_GROUP_SIZE,
                size_n,
                dtype=self.dtype,
                device=self.device,
            )
            permuted = marlin_permute_scales(raw, size_k, size_n, FP4_MARLIN_GROUP_SIZE)
            return nvfp4_marlin_process_scales(permuted)

        def _make_global_scale():
            return torch.tensor(1.0, dtype=self.dtype, device=self.device)

        w1 = torch.stack([_make_marlin_weight(K, 2 * N) for _ in range(E)])
        w2 = torch.stack([_make_marlin_weight(N, K) for _ in range(E)])
        w1_scale = torch.stack([_make_marlin_scale(K, 2 * N) for _ in range(E)])
        w2_scale = torch.stack([_make_marlin_scale(N, K) for _ in range(E)])
        w1_global_scale = torch.stack([_make_global_scale() for _ in range(E)])
        w2_global_scale = torch.stack([_make_global_scale() for _ in range(E)])

        hidden_states = torch.randn(M, K, dtype=self.dtype, device=self.device)
        gating_output = torch.randn(M, E, dtype=self.dtype, device=self.device)
        topk_weights, topk_ids = torch.topk(
            torch.softmax(gating_output, dim=-1), topk, dim=-1
        )

        output = fused_marlin_moe(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            gating_output=gating_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            num_bits=4,
            w1_global_scale=w1_global_scale,
            w2_global_scale=w2_global_scale,
        )

        self.assertEqual(output.shape, (M, K))
        self.assertEqual(output.dtype, self.dtype)
        self.assertFalse(torch.isnan(output).any(), "Output contains NaN!")

    def test_prepare_moe_fp4_layer_for_marlin(self):
        """Test that prepare_moe_fp4_layer_for_marlin correctly repacks weights."""
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            prepare_moe_fp4_layer_for_marlin,
        )

        try:
            from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack  # noqa
        except ImportError:
            self.skipTest("gptq_marlin_repack not available")

        E, K, N = 4, 128, 64

        class FakeMoeRunnerConfig:
            is_gated = True

        class FakeLayer(torch.nn.Module):
            pass

        layer = FakeLayer()
        layer.num_local_experts = E
        layer.intermediate_size_per_partition = N
        layer.params_dtype = self.dtype
        layer.moe_runner_config = FakeMoeRunnerConfig()

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
                E, 2 * N, K // 16, dtype=torch.float8_e4m3fn, device=self.device
            ),
            requires_grad=False,
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            torch.ones(E, K, N // 16, dtype=torch.float8_e4m3fn, device=self.device),
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
        """Verify get_min_capability() returns 75 (not 100)."""
        from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config

        cap = ModelOptFp4Config.get_min_capability()
        self.assertEqual(cap, 75, f"Expected 75, got {cap}")


if __name__ == "__main__":
    unittest.main(verbosity=3)
