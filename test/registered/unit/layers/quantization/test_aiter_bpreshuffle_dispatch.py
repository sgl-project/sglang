"""Keep FP8 weight layouts and GEMM selection consistent across HIP devices."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy

from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w8a8_fp8 as ct_fp8,
)
from sglang.srt.utils import common
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAiterBpreshuffleDispatch(unittest.TestCase):
    def test_rdna4_device_detection(self):
        for arch, expected in (
            ("gfx1200", True),
            ("gfx1201:sramecc-:xnack-", True),
            ("gfx942", False),
            ("gfx950", False),
            ("gfx1250", False),
        ):
            with (
                self.subTest(arch=arch),
                patch.object(torch.version, "hip", "7.2"),
                patch.object(
                    torch.cuda,
                    "get_device_properties",
                    return_value=SimpleNamespace(gcnArchName=arch),
                ) as props,
            ):
                common.is_gfx120x_supported.cache_clear()
                try:
                    self.assertEqual(common.is_gfx120x_supported(), expected)
                    self.assertEqual(common.is_gfx120x_supported(), expected)
                    props.assert_called_once_with(0)
                finally:
                    common.is_gfx120x_supported.cache_clear()

        with (
            patch.object(torch.version, "hip", None),
            patch.object(torch.cuda, "get_device_properties") as props,
        ):
            self.assertFalse(common.is_gfx120x_supported())
            props.assert_not_called()
        common.is_gfx120x_supported.cache_clear()

    def test_device_and_output_width_gates(self):
        for use_aiter in (False, True):
            for rdna4 in (False, True):
                for n in (32, 64, 128, 8192):
                    with (
                        self.subTest(aiter=use_aiter, rdna4=rdna4, n=n),
                        patch.multiple(
                            fp8_utils,
                            _use_aiter=use_aiter,
                            _is_gfx120x_supported=rdna4,
                        ),
                    ):
                        self.assertEqual(
                            fp8_utils.use_aiter_bpreshuffle_gemm(n),
                            use_aiter and not rdna4 and n % 64 == 0,
                        )

    def test_compressed_tensors_load_and_apply_agree(self):
        # N != K also catches gating runtime dispatch by the wrong weight axis
        # after load has transposed the unshuffled weight.
        for rdna4, n in ((True, 64), (True, 32), (False, 32), (False, 64)):
            with (
                self.subTest(rdna4=rdna4, n=n),
                patch.multiple(fp8_utils, _use_aiter=True, _is_gfx120x_supported=rdna4),
                patch.object(ct_fp8, "is_fp8_fnuz", return_value=False),
                patch.object(
                    ct_fp8,
                    "shuffle_weight",
                    side_effect=lambda w, _: w.clone(),
                    create=True,
                ) as shuffle,
            ):
                scheme = ct_fp8.CompressedTensorsW8A8Fp8(
                    QuantizationArgs(
                        num_bits=8, type="float", strategy=QuantizationStrategy.CHANNEL
                    ),
                    is_static_input_scheme=False,
                )
                weight = torch.arange(n * 128, dtype=torch.float32).reshape(n, 128)
                weight = weight.remainder(16).to(torch.float8_e4m3fn)
                layer = torch.nn.Module()
                layer.weight = torch.nn.Parameter(weight, requires_grad=False)
                layer.weight_scale = torch.nn.Parameter(
                    torch.ones(n, 1), requires_grad=False
                )
                scheme.process_weights_after_loading(layer)
                shuffled = not rdna4 and n % 64 == 0
                if shuffled:
                    shuffle.assert_called_once()
                    self.assertEqual(layer.weight.shape, (n, 128))
                else:
                    shuffle.assert_not_called()
                    torch.testing.assert_close(
                        layer.weight.view(torch.uint8), weight.t().view(torch.uint8)
                    )

                x = torch.ones(2, 128, dtype=torch.bfloat16)
                sentinel = object()
                with (
                    patch.object(
                        ct_fp8, "apply_fp8_linear", return_value=sentinel
                    ) as plain,
                    patch.object(
                        ct_fp8, "apply_fp8_ptpc_linear", return_value=sentinel
                    ) as preshuffled,
                ):
                    self.assertIs(scheme.apply_weights(layer, x), sentinel)
                    self.assertEqual(plain.call_count, int(not shuffled))
                    self.assertEqual(preshuffled.call_count, int(shuffled))

    def test_rowwise_fallback_preserves_quantization_and_bias(self):
        # Exercise the real dispatch function; emulate only the GPU operations.
        # A per-channel scale makes an accidental per-tensor path fail.
        qx = torch.ones(4, 128, dtype=torch.float8_e4m3fn)
        sx = torch.arange(1, 5, dtype=torch.float32).view(-1, 1)
        weight = torch.ones(64, 128, dtype=torch.float8_e4m3fn).t()
        sw = torch.arange(1, 65, dtype=torch.float32).view(-1, 1)
        bias = torch.arange(64, dtype=torch.bfloat16)
        expected = ((qx.float() @ weight.float()) * sx * sw.t() + bias).bfloat16()

        def scaled_mm(a, b, *, scale_a, scale_b, out_dtype, bias):
            self.assertEqual(tuple(scale_b.shape), (1, 64))
            return ((a.float() @ b.float()) * scale_a * scale_b + bias).to(out_dtype)

        with (
            patch.multiple(fp8_utils, _use_aiter=True, _is_gfx120x_supported=True),
            patch.object(fp8_utils, "scaled_fp8_quant", return_value=(qx, sx)) as quant,
            patch.object(
                fp8_utils, "gemm_a8w8_bpreshuffle", Mock(), create=True
            ) as preshuffled,
            patch.object(torch, "_scaled_mm", side_effect=scaled_mm),
        ):
            output = fp8_utils.apply_fp8_linear(
                torch.ones(2, 2, 128, dtype=torch.bfloat16),
                weight,
                sw,
                bias=bias,
                cutlass_fp8_supported=False,
                use_per_token_if_dynamic=True,
                compressed_tensor_quant=True,
                pad_output=False,
            )
            torch.testing.assert_close(output, expected.reshape(2, 2, 64))
            quant.assert_called_once()
            self.assertTrue(fp8_utils._use_aiter)
            preshuffled.assert_not_called()
            # Online FP8 loading stores the same channel scales as (1, N).
            output = fp8_utils.apply_fp8_linear(
                torch.ones(2, 2, 128, dtype=torch.bfloat16),
                weight,
                sw.t(),
                bias=bias,
                cutlass_fp8_supported=False,
                use_per_token_if_dynamic=True,
                compressed_tensor_quant=True,
                pad_output=False,
            )
            torch.testing.assert_close(output, expected.reshape(2, 2, 64))
            preshuffled.assert_not_called()


if __name__ == "__main__":
    unittest.main()
