"""ROCm W4A16 format/dispatch regressions; no GPU or vLLM installation needed."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import sys
import types
import unittest
from unittest.mock import create_autospec, patch

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


def _old_api_gptq_shuffle(q_weight, g_idx, bit):
    """Autospec template: gptq_shuffle before vllm-project/vllm#54809."""


def _old_api_gptq_gemm(
    a, b_q_weight, b_gptq_qzeros, b_gptq_scales, g_idx, use_exllama, use_v2_format, bit
):
    """Autospec template: gptq_gemm before vllm-project/vllm#54809."""


def _new_api_gptq_shuffle(q_weight, bit):
    """Autospec template: gptq_shuffle after vllm-project/vllm#54809 dropped
    g_idx (activation ordering was removed from the ROCm GPTQ kernels)."""


def _new_api_gptq_gemm(
    a, b_q_weight, b_gptq_qzeros, b_gptq_scales, use_exllama, use_v2_format, bit
):
    """Autospec template: gptq_gemm after vllm-project/vllm#54809 dropped
    g_idx (activation ordering was removed from the ROCm GPTQ kernels)."""


class _RocmWNA16TestMixin:
    """Shared coverage for both vLLM ROCm GPTQ op signatures.

    vllm-project/vllm#54809 ("[Quant][Kernel] Remove GPTQ Group/Dynamic
    Activation Ordering", 2026-09-08) dropped the g_idx argument from
    gptq_shuffle/gptq_gemm. The two concrete subclasses below run this whole
    suite against mocks shaped like the old (<= vLLM 0.16, e.g.
    rocm/sgl-dev:rocm7-vllm-20250904) and new signatures respectively, using
    ``create_autospec`` so ``inspect.signature`` -- which
    ``_bind_rocm_gptq_ops`` relies on -- sees the real parameter count.

    This is a plain mixin, not a ``unittest.TestCase`` itself: both unittest
    and pytest discover *every* TestCase subclass in a module regardless of
    name, so if this carried the test methods directly it would also run
    (with its default ``NEW_API = False``) as its own, third, duplicate test
    case. Only the two concrete subclasses below are TestCases.
    """

    NEW_API = False

    def setUp(self):
        self.platform = patch.object(wna16, "_is_hip", True)
        self.platform.start()
        self.addCleanup(self.platform.stop)
        self.ops = types.ModuleType("vllm._custom_ops")
        if self.NEW_API:
            self.ops.gptq_shuffle = create_autospec(_new_api_gptq_shuffle)
            self.ops.gptq_gemm = create_autospec(
                _new_api_gptq_gemm, side_effect=self._reference_gemm_new_api
            )
        else:
            self.ops.gptq_shuffle = create_autospec(_old_api_gptq_shuffle)
            self.ops.gptq_gemm = create_autospec(
                _old_api_gptq_gemm, side_effect=self.reference_gemm
            )
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

    @classmethod
    def _reference_gemm_new_api(cls, x, qw, qz, scales, exllama, v2, bits):
        # The bound wrapper never forwards a non-trivial g_idx to the new
        # API (see _bind_rocm_gptq_ops), so plug in the identity value here.
        empty_g_idx = torch.empty(0, dtype=torch.int32)
        return cls.reference_gemm(x, qw, qz, scales, empty_g_idx, exllama, v2, bits)

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


class TestRocmWNA16(_RocmWNA16TestMixin, unittest.TestCase):
    """Runs against the pre-#54809 (old) gptq_gemm/gptq_shuffle signatures."""


class TestRocmWNA16NewAPI(_RocmWNA16TestMixin, unittest.TestCase):
    """Runs the same suite against the post-#54809 (new) signatures, which
    dropped g_idx."""

    NEW_API = True


class TestSignatureKind(unittest.TestCase):
    """Direct coverage of the old/new signature classifier.

    Only vLLM's real Python wrappers are supported. In particular, an
    opaque ``(*args, **kwargs)`` callable -- what an un-wrapped
    ``torch.ops._C`` op looks like from Python -- must never be silently
    classified as either API, even though it happens to have as few
    apparent parameters as the new signature.
    """

    def test_old_and_new_param_counts_are_classified(self):
        self.assertEqual(
            wna16._signature_kind(_old_api_gptq_shuffle, "gptq_shuffle", 3, 2),
            "old",
        )
        self.assertEqual(
            wna16._signature_kind(_new_api_gptq_shuffle, "gptq_shuffle", 3, 2),
            "new",
        )
        self.assertEqual(
            wna16._signature_kind(_old_api_gptq_gemm, "gptq_gemm", 8, 7), "old"
        )
        self.assertEqual(
            wna16._signature_kind(_new_api_gptq_gemm, "gptq_gemm", 8, 7), "new"
        )

    def test_opaque_varargs_signature_is_rejected(self):
        def opaque(*args, **kwargs):
            pass

        with self.assertRaisesRegex(NotImplementedError, "opaque"):
            wna16._signature_kind(opaque, "gptq_shuffle", 3, 2)

    def test_unrecognized_param_count_is_rejected(self):
        def three_params(a, b, c):
            pass

        with self.assertRaisesRegex(NotImplementedError, "does not recognize"):
            wna16._signature_kind(three_params, "gptq_gemm", 8, 7)


class TestBindRocmGptqOps(unittest.TestCase):
    """Direct coverage of the signature detection/adaptation itself."""

    def test_old_signature_is_passed_through_unchanged(self):
        shuffle = create_autospec(_old_api_gptq_shuffle)
        gemm = create_autospec(_old_api_gptq_gemm)
        bound_shuffle, bound_gemm = wna16._bind_rocm_gptq_ops(shuffle, gemm)
        self.assertIs(bound_shuffle, shuffle)
        self.assertIs(bound_gemm, gemm)

    def test_new_signature_drops_trivial_g_idx(self):
        shuffle = create_autospec(_new_api_gptq_shuffle)
        gemm = create_autospec(_new_api_gptq_gemm, return_value="gemm-out")
        bound_shuffle, bound_gemm = wna16._bind_rocm_gptq_ops(shuffle, gemm)

        q_weight = torch.zeros(1, dtype=torch.int32)
        empty_g_idx = torch.empty(0, dtype=torch.int32)
        bound_shuffle(q_weight, empty_g_idx, 4)
        shuffle.assert_called_once_with(q_weight, 4)

        args = (torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1))
        result = bound_gemm(*args, empty_g_idx, True, False, 4)
        self.assertEqual(result, "gemm-out")
        gemm.assert_called_once_with(*args, True, False, 4)

    def test_new_signature_rejects_nontrivial_g_idx(self):
        shuffle = create_autospec(_new_api_gptq_shuffle)
        gemm = create_autospec(_new_api_gptq_gemm)
        bound_shuffle, bound_gemm = wna16._bind_rocm_gptq_ops(shuffle, gemm)

        q_weight = torch.zeros(1, dtype=torch.int32)
        act_order_g_idx = torch.tensor([1, 0], dtype=torch.int32)
        with self.assertRaisesRegex(NotImplementedError, "activation ordering"):
            bound_shuffle(q_weight, act_order_g_idx, 4)
        args = (torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1))
        with self.assertRaisesRegex(NotImplementedError, "activation ordering"):
            bound_gemm(*args, act_order_g_idx, True, False, 4)
        shuffle.assert_not_called()
        gemm.assert_not_called()

    def test_opaque_ops_are_rejected_not_treated_as_new(self):
        # An un-wrapped torch.ops._C op is (*args, **kwargs) from Python and
        # happens to be callable with either arity; it must be rejected, not
        # silently guessed at as the new (g_idx-less) API.
        def opaque_shuffle(*args, **kwargs):
            pass

        def opaque_gemm(*args, **kwargs):
            pass

        with self.assertRaisesRegex(NotImplementedError, "opaque"):
            wna16._bind_rocm_gptq_ops(opaque_shuffle, opaque_gemm)

    def test_mismatched_signature_pair_is_rejected(self):
        shuffle = create_autospec(_old_api_gptq_shuffle)  # old (3 params)
        gemm = create_autospec(_new_api_gptq_gemm)  # new (7 params)
        with self.assertRaisesRegex(NotImplementedError, "mismatched"):
            wna16._bind_rocm_gptq_ops(shuffle, gemm)


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
