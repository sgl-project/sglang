"""Unit tests for FP4 KV cache quantization strategy pattern - no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.test.test_utils import CustomTestCase


def skip_if_no_blackwell_nvfp4(func):
    """Skip test if Blackwell NVFP4 is not available."""
    from sglang.srt.utils import is_blackwell

    return unittest.skipUnless(
        is_blackwell(), "Blackwell (SM100/SM120) with CUDA >= 12.8 is required"
    )(func)


class TestKVCacheQuantRegistry(CustomTestCase):
    """Test the registry and factory function."""

    def test_registry_contains_nvfp4_and_blockfp4(self):
        from sglang.srt.layers.quantization.kv_cache_quant_method import (
            KV_CACHE_QUANT_REGISTRY,
        )

        self.assertIn("nvfp4", KV_CACHE_QUANT_REGISTRY)
        self.assertIn("fp4_e2m1_block16", KV_CACHE_QUANT_REGISTRY)

    def test_factory_nvfp4(self):
        from sglang.srt.layers.quantization.kv_cache_quant_method import (
            NVFP4Method,
            get_kv_cache_quant_method,
        )

        method = get_kv_cache_quant_method(
            "nvfp4", num_layers=4, device="cpu", sm_version=120
        )
        self.assertIsInstance(method, NVFP4Method)
        self.assertEqual(method.name, "nvfp4")

    def test_factory_blockfp4(self):
        from sglang.srt.layers.quantization.kv_cache_quant_method import (
            BlockFP4Method,
            get_kv_cache_quant_method,
        )

        method = get_kv_cache_quant_method("fp4_e2m1_block16")
        self.assertIsInstance(method, BlockFP4Method)
        self.assertEqual(method.name, "fp4_e2m1_block16")

    def test_resolve_explicit_recipes(self):
        from sglang.srt.layers.quantization.kv_cache_quant_method import (
            resolve_kv_cache_quant,
        )

        self.assertEqual(resolve_kv_cache_quant("nvfp4"), "nvfp4")
        self.assertEqual(resolve_kv_cache_quant("fp4_e2m1_block16"), "fp4_e2m1_block16")
        self.assertIsNone(resolve_kv_cache_quant("fp8_e4m3"))

    def test_resolve_legacy_fp4_alias_raises(self):
        from sglang.srt.layers.quantization.kv_cache_quant_method import (
            resolve_kv_cache_quant,
        )

        with self.assertRaises(ValueError):
            resolve_kv_cache_quant("fp4_e2m1")

    def test_resolve_mxfp4_name_raises(self):
        from sglang.srt.layers.quantization.kv_cache_quant_method import (
            resolve_kv_cache_quant,
        )

        with self.assertRaises(ValueError):
            resolve_kv_cache_quant("mxfp4")

    def test_factory_unknown_raises(self):
        from sglang.srt.layers.quantization.kv_cache_quant_method import (
            get_kv_cache_quant_method,
        )

        with self.assertRaises(ValueError):
            get_kv_cache_quant_method("unknown_method")


class TestNVFP4Method(CustomTestCase):
    """Test NVFP4Method buffer creation and properties."""

    def test_properties(self):
        from sglang.srt.layers.quantization.kv_cache_quant_method import (
            NVFP4Method,
        )

        m = NVFP4Method(num_layers=4, device="cpu", sm_version=120)
        self.assertEqual(m.name, "nvfp4")
        self.assertEqual(m.scale_block_size, 16)
        self.assertTrue(m.needs_dequant_workspace())
        self.assertTrue(m.needs_global_scale())

    def test_create_buffers_shapes(self):
        from sglang.srt.layers.quantization.kv_cache_quant_method import (
            NVFP4Method,
        )

        m = NVFP4Method(num_layers=4, device="cpu", sm_version=120)
        size, heads, dim, layers = 64, 8, 128, 4
        bufs = m.create_buffers(size, heads, dim, layers, "cpu")

        self.assertEqual(len(bufs["k_buffer"]), layers)
        self.assertEqual(len(bufs["v_buffer"]), layers)
        self.assertEqual(len(bufs["k_scale_buffer"]), layers)
        self.assertEqual(len(bufs["v_scale_buffer"]), layers)

        # FP4 packed: (size, heads, dim//2)
        self.assertEqual(bufs["k_buffer"][0].shape, (size, heads, dim // 2))
        # Block scales: (size, heads, dim//16)
        self.assertEqual(bufs["k_scale_buffer"][0].shape, (size, heads, dim // 16))
        # Dequant workspace: (size, heads, dim), FP8
        self.assertEqual(bufs["dq_k_buffer"].shape, (size, heads, dim))
        self.assertEqual(bufs["dq_k_buffer"].dtype, torch.float8_e4m3fn)
        self.assertEqual(bufs["store_dtype"], torch.uint8)

    def test_compute_cell_size(self):
        from sglang.srt.layers.quantization.kv_cache_quant_method import (
            NVFP4Method,
        )

        m = NVFP4Method(num_layers=4, device="cpu")
        cell = m.compute_cell_size(head_num=8, head_dim=128, num_layers=4, kv_size=1)
        # FP4: 8*64*4*2 = 4096, scales: 8*8*4*2 = 512, dq: 8*128*2 = 2048
        self.assertEqual(cell, 4096 + 512 + 2048)

    def test_scales_init(self):
        from sglang.srt.layers.quantization.kv_cache_quant_method import (
            NVFP4Method,
        )

        m = NVFP4Method(num_layers=4, device="cpu")
        # Default scales should be 1.0
        self.assertTrue(torch.all(m.k_scales_gpu == 1.0))
        self.assertTrue(torch.all(m.v_scales_gpu == 1.0))
        self.assertEqual(len(m.k_scales_gpu), 4)

    @skip_if_no_blackwell_nvfp4
    def test_quantize_dequantize_roundtrip(self):
        """Test NVFP4 quantize->dequantize roundtrip on CUDA."""
        from sglang.srt.layers.quantization.kv_cache_quant_method import (
            NVFP4Method,
        )

        major, minor = torch.cuda.get_device_capability()
        m = NVFP4Method(num_layers=1, device="cuda", sm_version=major * 10 + minor)

        size, heads, dim = 32, 8, 128
        bufs = m.create_buffers(size, heads, dim, 1, "cuda")

        # Create random input
        k = torch.randn(4, heads, dim, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(4, heads, dim, dtype=torch.bfloat16, device="cuda")
        loc = torch.arange(4, device="cuda")

        # Quantize
        m.quantize_and_store(
            bufs["k_buffer"][0],
            bufs["v_buffer"][0],
            bufs["k_scale_buffer"][0],
            bufs["v_scale_buffer"][0],
            loc,
            k,
            v,
            k_scale=m.k_scales_gpu[0:1],
            v_scale=m.v_scales_gpu[0:1],
        )

        # Dequantize
        k_fp4 = bufs["k_buffer"][0][loc]
        k_scales = bufs["k_scale_buffer"][0][loc]
        v_fp4 = bufs["v_buffer"][0][loc]
        v_scales = bufs["v_scale_buffer"][0][loc]
        k_out, v_out = m.dequantize_prev_kv(k_fp4, k_scales, v_fp4, v_scales, 0)

        # Check shapes
        self.assertEqual(k_out.shape, (4, heads, dim))
        self.assertEqual(k_out.dtype, torch.float8_e4m3fn)

        # Check roundtrip error is bounded (FP4 is very lossy, ~20% relative error)
        k_ref = k.float()
        k_rec = k_out.float()
        rel_error = (k_ref - k_rec).abs().mean() / k_ref.abs().mean()
        self.assertLess(
            rel_error, 0.5, f"NVFP4 roundtrip error too high: {rel_error:.3f}"
        )


class TestBlockFP4Method(CustomTestCase):
    """Test BlockFP4Method buffer creation and roundtrip."""

    def test_properties(self):
        from sglang.srt.layers.quantization.kv_cache_quant_method import (
            BlockFP4Method,
        )

        m = BlockFP4Method()
        self.assertEqual(m.name, "fp4_e2m1_block16")
        self.assertTrue(m.needs_dequant_workspace())
        self.assertFalse(m.needs_global_scale())

    def test_create_buffers_shapes(self):
        from sglang.srt.layers.quantization.kv_cache_quant_method import (
            BlockFP4Method,
        )

        m = BlockFP4Method()
        size, heads, dim, layers = 64, 8, 128, 4
        bufs = m.create_buffers(size, heads, dim, layers, "cpu")

        self.assertEqual(len(bufs["k_buffer"]), layers)
        self.assertEqual(bufs["k_buffer"][0].shape, (size, heads, dim // 2))
        # Block-16 FP4 flattens head dims for scales
        self.assertEqual(bufs["k_scale_buffer"][0].shape, (size, (heads * dim) // 16))

    def test_quantize_dequantize_roundtrip_cpu(self):
        """Test block-16 FP4 quantize->dequantize roundtrip on CPU."""
        from sglang.srt.layers.quantization.kv_cache_quant_method import (
            BlockFP4Method,
        )

        m = BlockFP4Method()
        size, heads, dim = 32, 8, 128
        bufs = m.create_buffers(size, heads, dim, 1, "cpu")

        k = torch.randn(4, heads, dim, dtype=torch.bfloat16)
        v = torch.randn(4, heads, dim, dtype=torch.bfloat16)
        loc = torch.arange(4)

        # Quantize
        m.quantize_and_store(
            bufs["k_buffer"][0],
            bufs["v_buffer"][0],
            bufs["k_scale_buffer"][0],
            bufs["v_scale_buffer"][0],
            loc,
            k,
            v,
        )

        # Dequantize
        k_fp4 = bufs["k_buffer"][0][loc]
        k_scales = bufs["k_scale_buffer"][0][loc]
        v_fp4 = bufs["v_buffer"][0][loc]
        v_scales = bufs["v_scale_buffer"][0][loc]
        k_out, v_out = m.dequantize_prev_kv(k_fp4, k_scales, v_fp4, v_scales, 0)

        self.assertEqual(k_out.shape, (4, heads, dim))
        self.assertEqual(k_out.dtype, torch.float8_e4m3fn)


class TestKVFP4QuantizeUtil(CustomTestCase):
    """Test the existing block-16 FP4 KVFP4QuantizeUtil roundtrip."""

    def test_roundtrip_cpu(self):
        from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

        x = torch.randn(4, 8, 128, dtype=torch.bfloat16)
        packed, scales = KVFP4QuantizeUtil.batched_quantize(x)
        reconstructed = KVFP4QuantizeUtil.batched_dequantize(packed, scales)

        self.assertEqual(reconstructed.shape, x.shape)
        rel_error = (
            x.float() - reconstructed.float()
        ).abs().mean() / x.float().abs().mean()
        self.assertLess(rel_error, 0.5)


if __name__ == "__main__":
    unittest.main()
