"""ROCm W4A16 format/dispatch regressions; no GPU or vLLM installation needed."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
from compressed_tensors.quantization import ActivationOrdering

from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_wNa16 as wna16,
)


def make_layer(
    k=256, n=128, group_size=128, dtype=torch.float16, full_k=None, full_n=None
):
    scheme = wna16.CompressedTensorsWNA16(
        strategy="channel" if group_size == -1 else "group",
        num_bits=4,
        group_size=group_size,
    )
    layer = torch.nn.Module()
    scheme.create_weights(
        layer,
        output_size=full_n or n,
        input_size=full_k or k,
        output_partition_sizes=[n],
        input_size_per_partition=k,
        params_dtype=dtype,
        weight_loader=lambda *args, **kwargs: None,
    )
    # Deterministic signed values exercise every nibble, including -8 and +7.
    signed = (
        (torch.arange(n * k).reshape(n, k) * 7 + torch.arange(n).unsqueeze(1)) % 16 - 8
    ).to(torch.int8)
    groups = 1 if group_size == -1 else k // group_size
    scales = (torch.arange(n * groups).reshape(n, groups) % 17 + 1).to(dtype) / 128
    layer.weight_packed.data.copy_(pack_to_int32(signed, 4))
    layer.weight_scale.data.copy_(scales)
    group_size = k if group_size == -1 else group_size
    reference = signed.float() * scales.float().repeat_interleave(group_size, dim=1)
    return scheme, layer, reference


class TestRocmWNA16(unittest.TestCase):
    def setUp(self):
        self.platform = patch.object(wna16, "_is_hip", True)
        self.platform.start()
        self.addCleanup(self.platform.stop)
        self.ops = types.ModuleType("vllm._custom_ops")
        self.ops.gptq_shuffle = Mock()
        self.ops.gptq_gemm = Mock(side_effect=self.reference_gemm)
        modules = patch.dict(
            sys.modules,
            {
                "vllm": types.ModuleType("vllm"),
                "vllm._custom_ops": self.ops,
            },
        )
        modules.start()
        self.addCleanup(modules.stop)

    @staticmethod
    def reference_gemm(x, qw, qz, scales, g_idx, exllama, v2, bits):
        # Independent GPTQ v1 decoder. Real packed CT data must survive the
        # layout and zero-point conversion before a kernel sees it.
        assert exllama and not v2 and bits == 4
        assert x.dtype == scales.dtype == torch.float16
        assert x.is_contiguous() and qw.is_contiguous() and scales.is_contiguous()
        shifts = torch.arange(8, dtype=torch.int32) * 4
        q = ((qw[:, None, :] >> shifts[None, :, None]) & 15).reshape(-1, qw.shape[1])
        zeros = ((qz[:, :, None] >> shifts) & 15).reshape(qz.shape[0], -1) + 1
        group_size = q.shape[0] // scales.shape[0]
        weight = (
            q.float() - zeros.repeat_interleave(group_size, 0)
        ) * scales.repeat_interleave(group_size, 0)
        return (x.float() @ weight).half()

    def test_conversion_and_output(self):
        for k, n, group in [
            (256, 128, 32),
            (256, 128, 64),
            (256, 128, 128),
            (256, 128, -1),
            (1024, 128, 128),
        ]:
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(k=k, n=n, group=group, dtype=dtype):
                    scheme, layer, weight = make_layer(k, n, group, dtype)
                    scheme.process_weights_after_loading(layer)
                    x = (torch.arange(2 * 3 * k).reshape(2, 3, k) % 19).to(dtype) / 32
                    bias = torch.linspace(-0.1, 0.1, n).to(dtype)
                    got = scheme.apply_weights(layer, x, bias)
                    expected = (x.half().float() @ weight.t()).half().to(dtype) + bias
                    torch.testing.assert_close(got, expected, rtol=0, atol=0)
                    self.assertEqual(layer.weight_packed.shape, (k // 8, n))
                    self.assertIs(type(layer.weight_packed), torch.nn.Parameter)
                    self.assertFalse(layer.weight_packed.requires_grad)
                    self.assertFalse(hasattr(layer, "rocm_qweight"))
                    self.assertEqual(got.dtype, dtype)
                    self.assertEqual(got.shape, (2, 3, n))

    def test_partitioned_weights(self):
        for full_k, full_n, group in [(512, 128, 128), (256, 256, 128), (512, 128, -1)]:
            with self.subTest(full_k=full_k, full_n=full_n, group=group):
                scheme, layer, weight = make_layer(
                    group_size=group, full_k=full_k, full_n=full_n
                )
                scheme.process_weights_after_loading(layer)
                x = torch.ones(2, 256, dtype=torch.float16)
                torch.testing.assert_close(
                    scheme.apply_weights(layer, x, None),
                    (x.float() @ weight.t()).half(),
                )

    def test_noncontiguous_and_empty_input(self):
        scheme, layer, weight = make_layer()
        scheme.process_weights_after_loading(layer)
        x = torch.ones(256, 3, dtype=torch.float16).t()
        torch.testing.assert_close(
            scheme.apply_weights(layer, x, None), (x.float() @ weight.t()).half()
        )
        self.ops.gptq_gemm.reset_mock()
        got = scheme.apply_weights(
            layer, torch.empty(2, 0, 256, dtype=torch.float16), None
        )
        self.assertEqual(got.shape, (2, 0, 128))
        self.ops.gptq_gemm.assert_not_called()

    def test_reject_unsupported_formats(self):
        for kwargs in [
            {"num_bits": 8},
            {"symmetric": False},
            {"actorder": ActivationOrdering.GROUP},
            {"group_size": 16},
        ]:
            args = dict(strategy="group", num_bits=4, group_size=128)
            args.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(NotImplementedError):
                wna16.CompressedTensorsWNA16(**args)
        self.ops.gptq_shuffle.assert_not_called()

    def test_reject_invalid_shapes_and_dtypes(self):
        for kwargs in [{"k": 192}, {"n": 136}, {"dtype": torch.float32}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(NotImplementedError):
                make_layer(**kwargs)
        scheme, layer, _ = make_layer()
        layer.weight_scale.data = torch.ones(128, 3, dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "scale shape"):
            scheme.process_weights_after_loading(layer)
        self.ops.gptq_shuffle.assert_not_called()

    def test_reject_overflowing_scales(self):
        scheme, layer, _ = make_layer(dtype=torch.bfloat16)
        layer.weight_scale.data.fill_(1e10)
        with self.assertRaisesRegex(ValueError, "finite in float16"):
            scheme.process_weights_after_loading(layer)
        self.ops.gptq_shuffle.assert_not_called()

    def test_missing_optional_dependency(self):
        scheme, layer, _ = make_layer()
        with patch.dict(sys.modules, {"vllm._custom_ops": None}):
            with self.assertRaisesRegex(ImportError, "ROCm vLLM build"):
                scheme.process_weights_after_loading(layer)

    def test_other_platform_does_not_use_rocm(self):
        with patch.object(wna16, "_is_hip", False):
            scheme = wna16.CompressedTensorsWNA16("group", 8, 128)
            with patch.object(scheme, "_process_weights_rocm") as rocm:
                # The original Marlin path still owns this call.
                with self.assertRaises(AttributeError):
                    scheme.process_weights_after_loading(torch.nn.Module())
                rocm.assert_not_called()


@unittest.skipUnless(torch.version.hip and torch.cuda.is_available(), "requires ROCm")
class TestRocmWNA16GPU(unittest.TestCase):
    """Also runnable on a ROCm host with vLLM GPTQ ops for numerical validation."""

    @classmethod
    def setUpClass(cls):
        try:
            from vllm._custom_ops import gptq_gemm, gptq_shuffle  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("requires ROCm vLLM GPTQ ops")

    def test_kernel_against_dequantized_reference(self):
        for k, n, group in [
            (256, 128, 32),
            (256, 128, 64),
            (256, 128, 128),
            (256, 128, -1),
            (1024, 128, 128),
            (2048, 256, 128),
        ]:
            for dtype in (torch.float16, torch.bfloat16):
                for batch in (1, 8, 64):
                    with self.subTest(k=k, n=n, group=group, dtype=dtype, batch=batch):
                        scheme, layer, weight = make_layer(k, n, group, dtype)
                        layer = layer.to("cuda")
                        scheme.process_weights_after_loading(layer)
                        torch.manual_seed(42)
                        x = (torch.randn(batch, k, device="cuda") * 0.125).to(dtype)
                        bias = torch.linspace(-0.1, 0.1, n, device="cuda").to(dtype)
                        got = scheme.apply_weights(layer, x, bias)
                        ref = (x.half().float() @ weight.to("cuda").t()).half().to(
                            dtype
                        ) + bias
                        # ExLlama accumulates in FP16; compare against independent
                        # FP32 dequantization/matmul, not another GPTQ execution.
                        torch.testing.assert_close(got, ref, atol=0.04, rtol=0.02)
                        self.assertTrue(torch.isfinite(got).all())

    def test_cuda_graph_replay(self):
        scheme, layer, _ = make_layer()
        layer = layer.to("cuda")
        scheme.process_weights_after_loading(layer)
        x = torch.ones(1, 256, device="cuda", dtype=torch.float16)
        for _ in range(3):
            expected = scheme.apply_weights(layer, x, None)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = scheme.apply_weights(layer, x, None)
        x.mul_(0.5)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(got, expected * 0.5, rtol=0.001, atol=0.001)


if __name__ == "__main__":
    unittest.main()
