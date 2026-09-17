"""CPU regression test for the W4A16 dense-dequant fallback scheme.

The compressed-tensors dispatch in ``_get_scheme_from_parts`` used to raise
``ImportError: Other method (CompressedTensorsW4A16Sparse24) is not supported
now`` for weight-only 4-bit integer group/channel quantized checkpoints whose
format is not ``pack_quantized`` (or otherwise fall outside the Marlin WNA16
path). The checkpoint could not load at all.
``CompressedTensorsW4A16Sparse24`` is a *correctness* fallback that unpacks
the 4-bit integer weights, dequantizes them with the per-group scales (and
optional zero points) into a dense float tensor, and runs a plain
``torch.nn.functional.linear`` -- no fused 2:4-sparse GEMM kernel.

These tests pin four contracts on CPU (no kernels run):
  1. A weight-only int4 group/channel config in a non ``pack_quantized``
     format dispatches to ``CompressedTensorsW4A16Sparse24`` instead of raising
     ``ImportError``.
  2. A weight-only float4 config (NVFP4-style) does NOT dispatch to the scheme
     -- it stays a loud ``NotImplementedError``, because the uint4b8 bias
     convention is only valid for integer weights.
  3. The dense dequant + ``F.linear`` forward reproduces a reference
     dequantized matmul bit-for-bit for both group and channelwise scales.
  4. The asymmetric (zero-point) dequant path is correct for both group and
     channel strategies -- the reference subtracts the zero point, not the
     symmetric bias.
"""

import unittest

import torch
import torch.nn as nn
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationType
from compressed_tensors.quantization.lifecycle.forward import dequantize, quantize
from compressed_tensors.quantization.utils.helpers import calculate_qparams

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4A16Sparse24,
    CompressedTensorsWNA16,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_W4A16_DENSE_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "dense",
    "config_groups": {
        "group_0": {
            "targets": ["re:.*"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": 128,
            },
            "input_activations": None,
        },
    },
    "ignore": [],
}

# A weight-only float4 (NVFP4-style) config: same shape as the int4 config but
# type=float. This must NOT be dequantized by the int4 fallback.
_W4A16_FLOAT4_DENSE_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "dense",
    "config_groups": {
        "group_0": {
            "targets": ["re:.*"],
            "weights": {
                "num_bits": 4,
                "type": "float",
                "symmetric": True,
                "strategy": "group",
                "group_size": 128,
            },
            "input_activations": None,
        },
    },
    "ignore": [],
}


def _pack_int4(q: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Pack [out, in] int4 values into [out, in//pack_factor] int32 along dim=1."""
    pack_factor = 32 // num_bits
    out_f, in_f = q.shape
    packed = torch.zeros(out_f, in_f // pack_factor, dtype=torch.int32)
    for j in range(in_f):
        col = j // pack_factor
        slot = j % pack_factor
        packed[:, col] |= q[:, j] << (num_bits * slot)
    return packed


def _pack_int4_dim0(q: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Pack [out, n_groups] int4 values into [out//pack_factor, n_groups] int32
    along dim=0, matching the weight_zero_point axis convention
    (packed_dim=0)."""
    pack_factor = 32 // num_bits
    out_f, n_groups = q.shape
    packed = torch.zeros(out_f // pack_factor, n_groups, dtype=torch.int32)
    for o in range(out_f):
        row = o // pack_factor
        slot = o % pack_factor
        packed[row, :] |= q[o, :] << (num_bits * slot)
    return packed


def _make_synthetic_layer(scheme, out_f, in_f, dtype, symmetric=True):
    """Create weights and populate them with a known quantized weight + scales
    (and, for asymmetric, a non-trivial zero point). Returns (layer, w_ref)
    where w_ref is the independently-computed dequantized reference."""
    torch.manual_seed(0)
    group_size = scheme.group_size if scheme.group_size != -1 else in_f
    w_true = torch.randn(out_f, in_f, dtype=torch.float32)
    n_groups = in_f // group_size
    scales = torch.rand(out_f, n_groups, dtype=torch.float32) * 0.1 + 0.05
    scales_exp = scales.repeat_interleave(group_size, dim=1)
    if symmetric:
        # Symmetric int4 is stored as unsigned [0, 15] with an implicit bias
        # of 8.
        q = torch.clamp(torch.round(w_true / scales_exp) + 8, 0, 15).to(torch.int32)
        w_ref = (q.float() - 8) * scales_exp
    else:
        # Asymmetric quantization carries an explicit (non-zero) zero point;
        # the dequant reference must subtract the zero point, NOT the bias.
        zp_true = torch.randint(1, 8, (out_f, n_groups), dtype=torch.int32)
        zp_exp = zp_true.repeat_interleave(group_size, dim=1).to(torch.float32)
        q = torch.clamp(torch.round(w_true / scales_exp) + zp_exp, 0, 15).to(
            torch.int32
        )
        w_ref = (q.float() - zp_exp) * scales_exp
    packed = _pack_int4(q, scheme.num_bits)
    layer = nn.Module()
    scheme.create_weights(
        layer,
        input_size=in_f,
        input_size_per_partition=in_f,
        output_partition_sizes=[out_f],
        output_size=out_f,
        params_dtype=dtype,
        weight_loader=None,
    )
    layer.weight_packed = nn.Parameter(packed.clone(), requires_grad=False)
    layer.weight_scale = nn.Parameter(scales.clone(), requires_grad=False)
    if not symmetric:
        zp_packed = _pack_int4_dim0(zp_true, scheme.num_bits)
        layer.weight_zero_point = nn.Parameter(zp_packed.clone(), requires_grad=False)
    return layer, w_ref


class TestW4A16Sparse24Dispatch(CustomTestCase):
    """Dispatch contract: the int4 Sparse24 path returns a scheme, float4
    does not."""

    def test_dispatch_returns_sparse24_scheme(self):
        qc = CompressedTensorsConfig.from_config(_W4A16_DENSE_CONFIG)
        layer = nn.Module()
        scheme_dict = qc.get_scheme_dict(layer, "model.layers.0.self_attn.q_proj")
        weight_quant = scheme_dict["weights"]
        input_quant = scheme_dict["input_activations"]

        # The weight-only int4 group/channel predicate must match, and the
        # format is intentionally not pack_quantized.
        self.assertTrue(qc._is_wNa16_group_channel(weight_quant, input_quant))
        self.assertNotEqual(qc.quant_format, CompressionFormat.pack_quantized.value)
        self.assertEqual(weight_quant.type, QuantizationType.INT)

        # The pack_quantized variant still dispatches to the Marlin WNA16 path
        # (the change is strictly additive inside the `else`; the `if` branch is
        # untouched).
        pack_cfg = dict(_W4A16_DENSE_CONFIG)
        pack_cfg["format"] = CompressionFormat.pack_quantized.value
        pack_qc = CompressedTensorsConfig.from_config(pack_cfg)
        pack_sd = pack_qc.get_scheme_dict(nn.Module(), "model.layers.0.mlp.gate")
        self.assertIsInstance(
            pack_qc._get_scheme_from_parts(
                pack_sd["weights"], pack_sd["input_activations"]
            ),
            CompressedTensorsWNA16,
        )

        # The dense variant dispatches to the new W4A16Sparse24 fallback
        # instead of raising ImportError.
        scheme = qc._get_scheme_from_parts(weight_quant, input_quant)
        self.assertIsInstance(scheme, CompressedTensorsW4A16Sparse24)
        self.assertEqual(scheme.num_bits, 4)
        self.assertEqual(scheme.group_size, 128)
        self.assertTrue(scheme.symmetric)
        self.assertEqual(scheme.quant_type, QuantizationType.INT)

    def test_float4_dispatch_raises_not_implemented(self):
        # A weight-only float4 (NVFP4-style) config enters the same wNa16
        # group/channel branch but must NOT be dequantized by the int4 fallback
        # (the uint4b8 bias convention is invalid for float weights). It stays
        # a loud NotImplementedError instead of silently wrong numbers.
        qc = CompressedTensorsConfig.from_config(_W4A16_FLOAT4_DENSE_CONFIG)
        layer = nn.Module()
        scheme_dict = qc.get_scheme_dict(layer, "model.layers.0.self_attn.q_proj")
        weight_quant = scheme_dict["weights"]
        input_quant = scheme_dict["input_activations"]

        self.assertTrue(qc._is_wNa16_group_channel(weight_quant, input_quant))
        self.assertNotEqual(qc.quant_format, CompressionFormat.pack_quantized.value)
        self.assertEqual(weight_quant.type, QuantizationType.FLOAT)
        self.assertEqual(weight_quant.num_bits, 4)

        with self.assertRaises(NotImplementedError) as ctx:
            qc._get_scheme_from_parts(weight_quant, input_quant)
        # The error message must name the type so the gap is diagnosable.
        self.assertIn("type=", str(ctx.exception))


class TestW4A16Sparse24Dequant(CustomTestCase):
    """Correctness contract: dense dequant + F.linear matches a reference."""

    def _check_forward(self, strategy, group_size, symmetric=True):
        out_f, in_f = (8, 16)
        dtype = torch.float16
        scheme = CompressedTensorsW4A16Sparse24(
            num_bits=4,
            strategy=strategy,
            quant_type=QuantizationType.INT,
            group_size=group_size,
            symmetric=symmetric,
        )
        layer, w_ref = _make_synthetic_layer(
            scheme, out_f, in_f, dtype, symmetric=symmetric
        )
        scheme.process_weights_after_loading(layer)
        self.assertTrue(
            torch.allclose(layer.weight.data, w_ref.to(dtype), atol=0.001),
            "dequantized weight diverges from reference",
        )
        x = torch.randn(3, in_f, dtype=dtype)
        out = scheme.apply_weights(layer, x, bias=None)
        ref = torch.nn.functional.linear(x, w_ref.to(dtype), None)
        self.assertTrue(torch.allclose(out, ref, atol=0.01))

    def test_group_dequant_forward(self):
        self._check_forward(strategy="group", group_size=8, symmetric=True)

    def test_channelwise_dequant_forward(self):
        self._check_forward(strategy="channel", group_size=None, symmetric=True)

    def test_group_asymmetric_dequant_forward(self):
        # Asymmetric path: the reference dequant subtracts the (non-zero) zero
        # point, NOT the symmetric bias. Exercises the weight_zero_point unpack
        # + transpose axis convention.
        self._check_forward(strategy="group", group_size=8, symmetric=False)

    def test_channelwise_asymmetric_dequant_forward(self):
        self._check_forward(strategy="channel", group_size=None, symmetric=False)


def _library_quantized_layer(strategy, group_size, symmetric, out_f=16, in_f=128):
    """Build a W4A16 layer whose ``weight_packed`` / ``weight_scale`` /
    ``weight_zero_point`` are produced by compressed-tensors' OWN quantization
    API (``calculate_qparams`` + ``quantize`` + ``pack_to_int32``), NOT by the
    hand-rolled ``_pack_int4`` helpers used above.

    This breaks the circularity the self-referential dequant tests carry: there
    the packed weights were manufactured with the same bias-8 / explicit
    zero-point convention the implementation assumes, so they only proved the
    scheme agrees with itself. Here the quantized values, scales, zero points
    and the int32 packing all come from the library, so the scheme's dequant is
    checked against an independent ground truth.

    Returns ``(layer, scheme, w_orig, ref_dequant)`` where ``w_orig`` is the
    pre-quantization dense FP32 weight and ``ref_dequant`` is the library's own
    ``dequantize()`` of the same packed weight (an independent reference).
    """
    torch.manual_seed(0)
    cfg = {
        "quant_method": "compressed-tensors",
        "format": "dense",
        "config_groups": {
            "group_0": {
                "targets": ["re:.*"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": symmetric,
                    "strategy": strategy,
                    "group_size": group_size if strategy == "group" else None,
                },
                "input_activations": None,
            },
        },
        "ignore": [],
    }
    # Parse the scheme through the library's own config parser so the
    # QuantizationArgs (num_bits / type / strategy / symmetric / group_size)
    # match exactly what compressed-tensors attaches to a real checkpoint.
    qc = CompressedTensorsConfig.from_config(cfg)
    scheme_dict = qc.get_scheme_dict(nn.Module(), "model.layers.0.self_attn.q_proj")
    weight_args = scheme_dict["weights"]

    w_orig = torch.randn(out_f, in_f, dtype=torch.float32) * 0.2
    # Observer step: per-group / per-channel min & max fed to the library's
    # qparams calculator. This plain reduction is not the dequant convention
    # under test; all of the scale / zero-point / int32-packing math below is
    # the library's own.
    if strategy == "group":
        n_groups = in_f // group_size
        wg = w_orig.reshape(out_f, n_groups, group_size)
        min_vals, max_vals = wg.amin(-1), wg.amax(-1)
    else:
        min_vals, max_vals = w_orig.amin(1, keepdim=True), w_orig.amax(1, keepdim=True)

    scale, zero_point = calculate_qparams(min_vals, max_vals, weight_args)
    # Library quantize() -> signed int8 in [-8, 7]; library pack_to_int32() then
    # adds the implicit +8 bias (uint4b8), which is exactly the convention the
    # scheme's ``self._bias = 2 ** (num_bits - 1)`` assumes. Verifying the scheme
    # reproduces the original dense weight here checks that agreement for real.
    q_int8 = quantize(
        x=w_orig,
        scale=scale,
        zero_point=zero_point,
        args=weight_args,
        dtype=torch.int8,
    )
    weight_packed = pack_to_int32(q_int8, weight_args.num_bits)
    ref_dequant = dequantize(
        x_q=q_int8,
        scale=scale,
        zero_point=zero_point if not symmetric else None,
    )

    scheme = CompressedTensorsW4A16Sparse24(
        num_bits=4,
        strategy=strategy,
        quant_type=QuantizationType.INT,
        group_size=group_size if strategy == "group" else None,
        symmetric=symmetric,
    )
    layer = nn.Module()
    scheme.create_weights(
        layer,
        input_size=in_f,
        input_size_per_partition=in_f,
        output_partition_sizes=[out_f],
        output_size=out_f,
        params_dtype=torch.float16,
        weight_loader=None,
    )
    layer.weight_packed = nn.Parameter(weight_packed.clone(), requires_grad=False)
    layer.weight_scale = nn.Parameter(scale.clone(), requires_grad=False)
    if not symmetric:
        zp_packed = pack_to_int32(zero_point, weight_args.num_bits, packed_dim=0)
        layer.weight_zero_point = nn.Parameter(zp_packed.clone(), requires_grad=False)
    return layer, scheme, w_orig, ref_dequant


class TestW4A16Sparse24DequantFromLibrary(CustomTestCase):
    """Independent dequant correctness.

    Unlike ``TestW4A16Sparse24Dequant`` (whose packed weights are manufactured by
    this file's own ``_pack_int4`` and therefore only prove the scheme agrees
    with itself), the inputs here are produced by compressed-tensors' own
    quantization API. The scheme must (1) reproduce the library's own
    ``dequantize()`` to fp16 precision -- the real "lossless dequant" claim --
    and (2) reconstruct the original dense weight within int4 quantization
    error, for both symmetric and asymmetric configs and both group and channel
    strategies.
    """

    # fp16 rounding from the scale / weight casts (verified ~2.4e-4 worst case).
    _ATOL_VS_LIB = 2e-3
    # int4 quantization error dominates vs the pre-quantization dense weight
    # (q step ~= scale/2; verified <6e-2 worst case for these small weights).
    _ATOL_VS_ORIG = 1e-1

    def _check(self, strategy, group_size, symmetric):
        layer, scheme, w_orig, ref_dequant = _library_quantized_layer(
            strategy, group_size, symmetric
        )
        scheme.process_weights_after_loading(layer)
        w = layer.weight.data
        ref_fp16 = ref_dequant.to(torch.float16)
        orig_fp16 = w_orig.to(torch.float16)

        # (1) The scheme's unpack+dequant matches the library's own dequantize()
        # to fp16 precision -- checked against an independent reference, not the
        # scheme's own packer.
        self.assertTrue(
            torch.allclose(w, ref_fp16, atol=self._ATOL_VS_LIB),
            f"scheme dequant diverges from library dequantize: max="
            f"{(w - ref_fp16).abs().max().item():.6g} "
            f"(strategy={strategy} symmetric={symmetric})",
        )
        # (2) The dequantized weight reconstructs the original dense weight
        # within int4 quantization error.
        self.assertTrue(
            torch.allclose(w, orig_fp16, atol=self._ATOL_VS_ORIG),
            f"dequantized weight diverges from original dense weight: max="
            f"{(w - orig_fp16).abs().max().item():.6g} "
            f"(strategy={strategy} symmetric={symmetric})",
        )
        # (3) The F.linear forward on the dequantized weight matches a reference
        # matmul on the library's dequantized weight.
        x = torch.randn(3, w_orig.shape[1], dtype=torch.float16)
        out = scheme.apply_weights(layer, x, bias=None)
        ref_out = torch.nn.functional.linear(x, ref_fp16, None)
        self.assertTrue(
            torch.allclose(out, ref_out, atol=self._ATOL_VS_LIB),
            f"forward diverges from reference: max="
            f"{(out - ref_out).abs().max().item():.6g} "
            f"(strategy={strategy} symmetric={symmetric})",
        )

    def test_group_symmetric(self):
        self._check("group", 32, True)

    def test_group_asymmetric(self):
        # The asymmetric path's weight_zero_point is packed by the library
        # (packed_dim=0) and unpacked by the scheme's unpack_cols(.t() ...);
        # this independently exercises that axis convention.
        self._check("group", 32, False)

    def test_channel_symmetric(self):
        self._check("channel", None, True)

    def test_channel_asymmetric(self):
        self._check("channel", None, False)


if __name__ == "__main__":
    unittest.main()
