import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=3, suite="extra-a-test-npu")

import sglang.srt.hardware_backend.npu.quantization.dtype_utils as dtype_utils

# Initialize the quantization registry as in model-config loading.
import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.hardware_backend.npu.compile_backend import (
    _register_npu_mxfp8_fusion_patterns,
)
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    npu_w8a8_mxfp8_linear,
)
from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
from sglang.srt.layers.layer_boundary.residual.add_norm import _norm_quant_kernel
from sglang.srt.utils import is_npu


def _is_arch35_hardware() -> bool:
    if not is_npu():
        return False
    try:
        return is_npu_arch35()
    except ImportError:
        return False


class TestNPUMXFP8Fusion(CustomTestCase):
    def test_prequantized_mxfp8_tuple_skips_dynamic_quant(self):
        qx = torch.empty((6, 128), dtype=torch.float8_e4m3fn)
        xs = torch.empty((6, 2, 2), dtype=torch.uint8)
        weight = torch.empty((128, 64), dtype=torch.float8_e4m3fn)
        weight_scale = torch.empty((2, 64, 2), dtype=torch.uint8)
        matmul_out = torch.randn((6, 64), dtype=torch.bfloat16)

        npu_ops = MagicMock()
        npu_ops.npu_quant_matmul.return_value = matmul_out
        with patch.object(torch.ops, "npu", npu_ops, create=True):
            out = npu_w8a8_mxfp8_linear(
                qx,
                weight,
                [64, 128],
                weight_scale,
                input_scale=xs,
            )

        npu_ops.npu_dynamic_mx_quant.assert_not_called()
        npu_ops.npu_quant_matmul.assert_called_once()
        self.assertIs(npu_ops.npu_quant_matmul.call_args.args[0], qx)
        self.assertIs(npu_ops.npu_quant_matmul.call_args.kwargs["pertoken_scale"], xs)
        self.assertEqual(out.shape, (6, 64))

    def test_dtype_utils_handles_missing_torch_npu_on_cpu(self):
        old_cache = dtype_utils._E8M0_DTYPE
        try:
            dtype_utils._E8M0_DTYPE = None
            with patch("sglang.srt.utils.is_npu", return_value=False):
                resolved = dtype_utils._get_float8_e8m0fnu_dtype()
            self.assertEqual(resolved, getattr(torch, "float8_e8m0fnu", None))
        finally:
            dtype_utils._E8M0_DTYPE = old_cache

    def test_add_norm_npu_mxfp8_dispatch(self):
        fn = _norm_quant_kernel("npu_mxfp8")
        self.assertIsNotNone(fn)

        norm = SimpleNamespace(
            weight=torch.ones(64, dtype=torch.bfloat16), variance_epsilon=1e-6
        )
        x = torch.randn((4, 64), dtype=torch.bfloat16)
        r = torch.randn((4, 64), dtype=torch.bfloat16)
        qx = torch.empty((4, 64), dtype=torch.float8_e4m3fn)
        xs = torch.empty((4, 1, 2), dtype=torch.uint8)

        npu_ops = MagicMock()
        npu_ops.npu_rms_norm_dynamic_mx_quant.return_value = (qx, xs, None)
        npu_ops.npu_add_rms_norm_dynamic_mx_quant.return_value = (qx, r, xs, None)
        with patch.object(torch.ops, "npu", npu_ops, create=True):
            out_no_res, res_out = fn(norm, x, None, None)
            self.assertIs(out_no_res[0], qx)
            self.assertIs(out_no_res[1], xs)
            self.assertIs(res_out, x)
            npu_ops.npu_rms_norm_dynamic_mx_quant.assert_called_once()

            out_res, res_out2 = fn(norm, x, r, None)
            self.assertIs(out_res[0], qx)
            self.assertIs(out_res[1], xs)
            self.assertIs(res_out2, r)
            npu_ops.npu_add_rms_norm_dynamic_mx_quant.assert_called_once()

    def test_compile_backend_skips_mxfp8_registration_on_non_arch35(self):
        _register_npu_mxfp8_fusion_patterns.cache_clear()
        try:
            with patch(
                "sglang.srt.hardware_backend.npu.utils.is_npu_arch35",
                return_value=False,
            ):
                # Must no-op cleanly without importing npugraph_ex or touching FakeTensorMode
                _register_npu_mxfp8_fusion_patterns()
        finally:
            _register_npu_mxfp8_fusion_patterns.cache_clear()

    @unittest.skipUnless(
        _is_arch35_hardware(), "Requires Ascend 950 (arch35) NPU hardware"
    )
    def test_npu_swiglu_and_add_rmsnorm_mx_quant_match_unfused(self):
        torch.manual_seed(0)
        x = torch.randn((32, 256), device="npu", dtype=torch.bfloat16)
        r = torch.randn((32, 256), device="npu", dtype=torch.bfloat16)
        w = torch.ones((256,), device="npu", dtype=torch.bfloat16)
        gate_up = torch.randn((32, 512), device="npu", dtype=torch.bfloat16)

        qx_f, xs_f = torch.ops.npu.npu_swiglu_mx_quant(
            gate_up, activate_dim=-1, activate_left=True, dst_type=torch.float8_e4m3fn
        )
        y_ref = torch.ops.npu.npu_swiglu(gate_up)
        qx_r, xs_r = torch.ops.npu.npu_dynamic_mx_quant(
            y_ref, dst_type=torch.float8_e4m3fn
        )
        self.assertTrue(torch.equal(qx_f.view(torch.uint8), qx_r.view(torch.uint8)))
        self.assertTrue(torch.equal(xs_f.view(torch.uint8), xs_r.view(torch.uint8)))

        qx_n, res_n, xs_n, _ = torch.ops.npu.npu_add_rms_norm_dynamic_mx_quant(
            r, x, w, epsilon=1e-6, dst_type=torch.float8_e4m3fn
        )
        y_n, _, res_ref = torch.ops.npu.npu_add_rms_norm(r, x, w, 1e-6)
        qx_nr, xs_nr = torch.ops.npu.npu_dynamic_mx_quant(
            y_n, dst_type=torch.float8_e4m3fn
        )
        torch.testing.assert_close(res_n, res_ref, atol=1e-3, rtol=1e-3)
        self.assertTrue(torch.equal(qx_n.view(torch.uint8), qx_nr.view(torch.uint8)))
        self.assertTrue(torch.equal(xs_n.view(torch.uint8), xs_nr.view(torch.uint8)))


if __name__ == "__main__":
    unittest.main()
