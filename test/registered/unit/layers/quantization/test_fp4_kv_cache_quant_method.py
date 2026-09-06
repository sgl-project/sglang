"""Unit tests for KV cache quantization strategies - no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.runtime_context import override_platform
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
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            KV_CACHE_QUANT_REGISTRY,
        )

        self.assertIn("nvfp4", KV_CACHE_QUANT_REGISTRY)
        self.assertIn("fp4_mx_block16", KV_CACHE_QUANT_REGISTRY)
        self.assertIn("cpu_fp8_e4m3", KV_CACHE_QUANT_REGISTRY)

    def test_factory_nvfp4(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            NVFP4KVCacheMethod,
            get_kv_cache_quant_method,
        )

        method = get_kv_cache_quant_method("nvfp4", num_layers=4, device="cpu")
        self.assertIsInstance(method, NVFP4KVCacheMethod)
        self.assertEqual(method.name, "nvfp4")

    def test_factory_blockfp4(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            FP4MXBlock16KVCacheMethod,
            get_kv_cache_quant_method,
        )

        method = get_kv_cache_quant_method("fp4_mx_block16")
        self.assertIsInstance(method, FP4MXBlock16KVCacheMethod)
        self.assertEqual(method.name, "fp4_mx_block16")

    def test_resolve_explicit_recipes(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            resolve_kv_cache_quant,
        )

        self.assertEqual(resolve_kv_cache_quant("nvfp4"), "nvfp4")
        self.assertEqual(resolve_kv_cache_quant("fp4_mx_block16"), "fp4_mx_block16")
        self.assertIsNone(resolve_kv_cache_quant("fp8_e4m3"))

    def test_resolve_legacy_fp4_alias_raises(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            resolve_kv_cache_quant,
        )

        with self.assertRaisesRegex(ValueError, "fp4_mx_block16"):
            resolve_kv_cache_quant("fp4_e2m1")

    def test_model_runner_rejects_legacy_fp4_alias(self):
        from sglang.srt.model_executor.model_runner import ModelRunner
        from sglang.srt.runtime_context import get_context

        runner = object.__new__(ModelRunner)
        runner.server_args = SimpleNamespace(speculative_draft_kv_cache_dtype=None)
        runner.draft_attention_backend = None
        # The runner reads the requested dtype off the model bag, so the double
        # publishes it rather than carrying it on a stand-in config.
        override = get_context().override_server_args(kv_cache_dtype="fp4_e2m1")
        override.install()
        self.addCleanup(override.restore)
        with self.assertRaisesRegex(ValueError, "fp4_mx_block16"):
            runner.configure_kv_cache_dtype()

    def test_resolve_mxfp4_name_raises(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            resolve_kv_cache_quant,
        )

        with self.assertRaises(ValueError):
            resolve_kv_cache_quant("mxfp4")

    def test_factory_unknown_raises(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            get_kv_cache_quant_method,
        )

        with self.assertRaises(ValueError):
            get_kv_cache_quant_method("unknown_method")


class TestCPUFP8KVCacheMethod(CustomTestCase):
    def test_static_scale_quantize_and_store(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            CPUFP8KVCacheMethod,
        )

        method = CPUFP8KVCacheMethod()
        buffers = method.create_buffers(4, 2, 8, 1, "cpu")
        loc = torch.tensor([1, 3])
        cache_k = torch.randn(2, 2, 8, dtype=torch.bfloat16)
        cache_v = torch.randn(2, 2, 8, dtype=torch.bfloat16)

        method.quantize_and_store(
            buffers["k_buffer"][0],
            buffers["v_buffer"][0],
            buffers["k_scale_buffer"],
            buffers["v_scale_buffer"],
            loc,
            cache_k,
            cache_v,
            k_scale=0.5,
            v_scale=0.25,
        )

        torch.testing.assert_close(
            buffers["k_buffer"][0][loc].float(),
            (cache_k / 0.5).to(torch.float8_e4m3fn).float(),
        )
        torch.testing.assert_close(
            buffers["v_buffer"][0][loc].float(),
            (cache_v / 0.25).to(torch.float8_e4m3fn).float(),
        )
        self.assertIsNone(buffers["k_scale_buffer"])
        self.assertIsNone(buffers["v_scale_buffer"])
        self.assertEqual(method.compute_cell_size(2, 8, 1, 4), 128)

    def test_defaults_to_unit_scales(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            CPUFP8KVCacheMethod,
        )

        method = CPUFP8KVCacheMethod()
        buffers = method.create_buffers(1, 1, 8, 1, "cpu")
        cache = torch.ones(1, 1, 8, dtype=torch.bfloat16)
        method.quantize_and_store(
            buffers["k_buffer"][0],
            buffers["v_buffer"][0],
            buffers["k_scale_buffer"],
            buffers["v_scale_buffer"],
            torch.tensor([0]),
            cache,
            cache,
        )

        expected = cache.to(torch.float8_e4m3fn)
        torch.testing.assert_close(buffers["k_buffer"][0][0], expected[0])
        torch.testing.assert_close(buffers["v_buffer"][0][0], expected[0])


class TestNVFP4KVCacheMethod(CustomTestCase):
    """Test NVFP4KVCacheMethod buffer creation and properties."""

    def test_properties(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            NVFP4KVCacheMethod,
        )

        m = NVFP4KVCacheMethod(num_layers=4, device="cpu", native_scale_layout=True)
        self.assertEqual(m.name, "nvfp4")
        self.assertEqual(m.SCALE_BLOCK_SIZE, 16)
        self.assertTrue(m.needs_dequant_workspace())
        self.assertTrue(m.needs_native_fp4_scales())
        self.assertTrue(m.needs_global_scale())

        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            KVCacheAttentionAccessKind,
        )

        self.assertEqual(
            m.resolve_attention_access("prefill", "trtllm_mha").kind,
            KVCacheAttentionAccessKind.NATIVE_FP4,
        )
        self.assertEqual(
            m.resolve_attention_access("prefill", "flashinfer").kind,
            KVCacheAttentionAccessKind.DEQUANT_WORKSPACE,
        )

    def test_create_buffers_shapes(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            NVFP4KVCacheMethod,
        )

        m = NVFP4KVCacheMethod(num_layers=4, device="cpu", native_scale_layout=True)
        size, heads, dim, layers = 64, 8, 128, 4
        bufs = m.create_buffers(size, heads, dim, layers, "cpu")

        self.assertEqual(len(bufs["k_buffer"]), layers)
        self.assertEqual(len(bufs["v_buffer"]), layers)
        self.assertEqual(len(bufs["k_scale_buffer"]), layers)
        self.assertEqual(len(bufs["v_scale_buffer"]), layers)
        self.assertEqual(len(bufs["native_k_scale_buffer"]), layers)
        self.assertEqual(len(bufs["native_v_scale_buffer"]), layers)

        # FP4 packed: (size, heads, dim//2)
        self.assertEqual(bufs["k_buffer"][0].shape, (size, heads, dim // 2))
        # Block scales: (size, heads, dim//16)
        self.assertEqual(bufs["k_scale_buffer"][0].shape, (size, heads, dim // 16))
        self.assertEqual(
            bufs["native_k_scale_buffer"][0].shape,
            (size // 16, heads, 16, dim // 16),
        )
        # Dequant workspace: (size, heads, dim), FP8
        self.assertEqual(bufs["dq_k_buffer"].shape, (size, heads, dim))
        self.assertEqual(bufs["dq_k_buffer"].dtype, torch.float8_e4m3fn)
        self.assertEqual(bufs["store_dtype"], torch.uint8)

    def test_compute_cell_size(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            NVFP4KVCacheMethod,
        )

        m = NVFP4KVCacheMethod(num_layers=4, device="cpu", native_scale_layout=True)
        cell = m.compute_cell_size(head_num=8, head_dim=128, num_layers=4, kv_size=1)
        # FP4: 4096, linear scales: 512, native scales: 512, shared DQ: 2048.
        self.assertEqual(cell, 4096 + 512 + 512 + 2048)

    def test_active_prefill_recipe_controls_auxiliary_memory(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            NVFP4KVCacheMethod,
        )

        size, heads, dim, layers = 64, 8, 128, 4

        native = NVFP4KVCacheMethod(
            num_layers=layers,
            device="cpu",
            page_size=16,
            native_scale_layout=True,
        )
        native.configure_attention_backends("trtllm_mha", "trtllm_mha")
        native_bufs = native.create_buffers(size, heads, dim, layers, "cpu")
        self.assertIsNone(native_bufs["k_scale_buffer"])
        self.assertIsNone(native_bufs["v_scale_buffer"])
        self.assertIsNone(native_bufs["dq_k_buffer"])
        self.assertIsNotNone(native_bufs["native_k_scale_buffer"])
        self.assertEqual(native.compute_cell_size(heads, dim, layers, 1), 4096 + 512)

        mixed = NVFP4KVCacheMethod(
            num_layers=layers,
            device="cpu",
            page_size=16,
            native_scale_layout=True,
        )
        mixed.configure_attention_backends("flashinfer", "trtllm_mha")
        mixed_bufs = mixed.create_buffers(size, heads, dim, layers, "cpu")
        self.assertIsNotNone(mixed_bufs["k_scale_buffer"])
        self.assertIsNotNone(mixed_bufs["dq_k_buffer"])
        self.assertIsNotNone(mixed_bufs["native_k_scale_buffer"])
        self.assertEqual(
            mixed.compute_cell_size(heads, dim, layers, 1),
            4096 + 512 + 512 + 2048,
        )

    def test_server_args_backend_selection_uses_resolution_projection(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            KVCacheAttentionAccessKind,
            NVFP4KVCacheMethod,
        )

        # Model/backend hooks declare overrides without mutating the raw
        # ServerArgs fields. Pool sizing and allocation must observe the same
        # resolved pair, and must not depend on a private ServerArgs method.
        server_args = SimpleNamespace(
            attention_backend="triton",
            prefill_attention_backend=None,
            decode_attention_backend=None,
            _resolved_overrides=[
                (
                    "test_model_override",
                    {
                        "prefill_attention_backend": "flashinfer",
                        "decode_attention_backend": "trtllm_mha",
                    },
                )
            ],
        )
        method = NVFP4KVCacheMethod(
            num_layers=1,
            device="cpu",
            page_size=16,
            native_scale_layout=True,
        )

        method.configure_attention_backends_from_server_args(server_args)

        accesses = method.active_attention_accesses()
        self.assertEqual(
            [access.kind for access in accesses],
            [
                KVCacheAttentionAccessKind.DEQUANT_WORKSPACE,
                KVCacheAttentionAccessKind.NATIVE_FP4,
            ],
        )

    def test_xqa_recipe_retains_linear_scales(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            NVFP4KVCacheMethod,
        )

        size, heads, dim, layers = 64, 8, 128, 4
        xqa = NVFP4KVCacheMethod(
            num_layers=layers,
            device="cpu",
            page_size=16,
            native_scale_layout=False,
        )
        xqa.configure_attention_backends("flashinfer", "trtllm_mha")
        bufs = xqa.create_buffers(size, heads, dim, layers, "cpu")

        self.assertIsNotNone(bufs["k_scale_buffer"])
        self.assertIsNotNone(bufs["dq_k_buffer"])
        self.assertIsNone(bufs["native_k_scale_buffer"])
        self.assertFalse(xqa.needs_native_fp4_scales())
        self.assertEqual(
            xqa.compute_cell_size(heads, dim, layers, 1),
            4096 + 512 + 2048,
        )

    def test_native_v_scale_swizzle_reference(self):
        from sglang.srt.layers.quantization.nvfp4_kv_cache import (
            nvfp4_v_scale_swizzle_indices,
        )

        token = torch.arange(16)[:, None]
        scale = torch.arange(8)[None, :]
        swizzled_token, swizzled_scale = nvfp4_v_scale_swizzle_indices(
            token, scale, scale_dim=8
        )

        # Every logical (token, scale) pair maps bijectively inside each
        # four-token group and agrees with FlashInfer/TRT-LLM's published map.
        flat = (swizzled_token * 8 + swizzled_scale).flatten()
        self.assertEqual(torch.unique(flat).numel(), 16 * 8)
        self.assertEqual(
            (swizzled_token[3, 7].item(), swizzled_scale[3, 7].item()), (3, 7)
        )
        self.assertEqual(
            (swizzled_token[1, 4].item(), swizzled_scale[1, 4].item()), (2, 1)
        )

    def test_scales_init(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            NVFP4KVCacheMethod,
        )

        m = NVFP4KVCacheMethod(num_layers=4, device="cpu")
        # Default scales should be 1.0
        self.assertTrue(torch.all(m.k_scales_gpu == 1.0))
        self.assertTrue(torch.all(m.v_scales_gpu == 1.0))
        self.assertEqual(len(m.k_scales_gpu), 4)

    def test_sm100_scale_loading_preserves_uncalibrated_fallback(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            NVFP4KVCacheMethod,
        )

        attention = SimpleNamespace(
            layer_id=0,
            k_scale=torch.tensor(1.0),
            v_scale=torch.tensor(1.0),
        )
        model = SimpleNamespace(
            layers=[SimpleNamespace(self_attn=SimpleNamespace(attn=attention))]
        )
        method = NVFP4KVCacheMethod(num_layers=1, device="cpu")

        with override_platform(is_sm100=True):
            method.load_scales_from_model(model)

        self.assertEqual(method.get_bmm_scales(0), (1.0, 1.0))

    def test_sm100_scale_loading_converts_calibrated_checkpoint_scales(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            NVFP4KVCacheMethod,
        )

        attention = SimpleNamespace(
            layer_id=0,
            k_scale=torch.tensor(0.002),
            v_scale=torch.tensor(0.003),
        )
        model = SimpleNamespace(
            layers=[SimpleNamespace(self_attn=SimpleNamespace(attn=attention))]
        )
        method = NVFP4KVCacheMethod(num_layers=1, device="cpu")

        with override_platform(is_sm100=True):
            method.load_scales_from_model(model)

        k_scale, v_scale = method.get_bmm_scales(0)
        self.assertAlmostEqual(k_scale, 0.012)
        self.assertAlmostEqual(v_scale, 0.018)

    @skip_if_no_blackwell_nvfp4
    def test_quantize_dequantize_roundtrip(self):
        """Test NVFP4 quantize->dequantize roundtrip on CUDA."""
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            NVFP4KVCacheMethod,
        )

        major, minor = torch.cuda.get_device_capability()
        m = NVFP4KVCacheMethod(num_layers=1, device="cuda")

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


class TestFP4MXBlock16KVCacheMethod(CustomTestCase):
    """Test FP4MXBlock16KVCacheMethod buffer creation and roundtrip."""

    def test_properties(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            FP4MXBlock16KVCacheMethod,
            KVCacheAttentionAccessKind,
        )

        m = FP4MXBlock16KVCacheMethod()
        self.assertEqual(m.name, "fp4_mx_block16")
        self.assertFalse(m.needs_dequant_workspace())
        self.assertTrue(m.needs_plain_kv_dequant_read())
        self.assertFalse(m.needs_global_scale())
        self.assertEqual(m.plain_attention_kv_dtype(), torch.bfloat16)
        self.assertEqual(
            m.resolve_attention_access("prefill", "triton").kind,
            KVCacheAttentionAccessKind.PLAIN,
        )
        self.assertEqual(
            m.resolve_attention_access("decode", "trtllm_mha").kind,
            KVCacheAttentionAccessKind.PLAIN,
        )
        self.assertIsNone(m.resolve_attention_access("prefill", "flashinfer"))
        self.assertIsNone(m.resolve_attention_access("decode", "flashinfer"))

    def test_create_buffers_shapes(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            FP4MXBlock16KVCacheMethod,
        )

        m = FP4MXBlock16KVCacheMethod()
        size, heads, dim, layers = 64, 8, 128, 4
        bufs = m.create_buffers(size, heads, dim, layers, "cpu")

        self.assertEqual(len(bufs["k_buffer"]), layers)
        self.assertEqual(bufs["k_buffer"][0].shape, (size, heads, dim // 2))
        # Block-16 FP4 flattens head dims for scales
        self.assertEqual(bufs["k_scale_buffer"][0].shape, (size, (heads * dim) // 16))
        self.assertIsNone(bufs["dq_k_buffer"])
        self.assertIsNone(bufs["dq_v_buffer"])

    def test_quantize_dequantize_roundtrip_cpu(self):
        """Test block-16 FP4 quantize->dequantize roundtrip on CPU."""
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            FP4MXBlock16KVCacheMethod,
        )

        m = FP4MXBlock16KVCacheMethod()
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
        self.assertEqual(v_out.shape, (4, heads, dim))
        self.assertEqual(k_out.dtype, torch.bfloat16)
        self.assertEqual(v_out.dtype, torch.bfloat16)


class TestFP4MXBlock16KVQuantizeUtil(CustomTestCase):
    """Test the existing block-16 FP4 FP4MXBlock16KVQuantizeUtil roundtrip."""

    def test_roundtrip_cpu(self):
        from sglang.srt.layers.quantization.kvfp4_tensor import (
            FP4MXBlock16KVQuantizeUtil,
        )

        x = torch.randn(4, 8, 128, dtype=torch.bfloat16)
        packed, scales = FP4MXBlock16KVQuantizeUtil.batched_quantize(x)
        reconstructed = FP4MXBlock16KVQuantizeUtil.batched_dequantize(packed, scales)

        self.assertEqual(reconstructed.shape, x.shape)
        rel_error = (
            x.float() - reconstructed.float()
        ).abs().mean() / x.float().abs().mean()
        self.assertLess(rel_error, 0.5)


if __name__ == "__main__":
    unittest.main()
