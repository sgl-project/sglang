import logging
from typing import TYPE_CHECKING, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.hardware_backend.npu.utils import NPUACLFormat, npu_format_cast
from sglang.srt.layers.quantization.base_config import LinearMethodBase

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

logger = logging.getLogger(__name__)

MXFP8_BLOCK_SIZE = 32
# W4A8_MXFP block (group) size — fixed at 32 by the msmodelslim export format.
MXFP4_BLOCK_SIZE = 32


# NPU ops are reached via torch.ops.npu.* (registered when torch_npu is imported
# by the runtime), so this module needs no top-level `import torch_npu` and stays
# importable on CUDA/CPU/AMD/XPU CI.
def _get_float8_e8m0fnu_dtype():
    # Resolve lazily rather than as a module-level constant: this module is
    # imported early (during quant-scheme registration), so reading the dtype at
    # call time keeps it correct regardless of import order / platform.
    return getattr(torch, "float8_e8m0fnu", None)


def _get_float4_e2m1fn_x2_dtype():
    # Same lazy pattern as _get_float8_e8m0fnu_dtype: the packed-FP4 dtype is a
    # torch core dtype on recent builds, so it needs no torch_npu. Resolved at
    # call time on the NPU-only W4A8 paths.
    return getattr(torch, "float4_e2m1fn_x2", None)


def _mxfp4_quantize_weight(weight_fp: torch.Tensor):
    """BF16/FP16 weight ``[out, in]`` -> msmodelslim ``W4A8_MXFP`` layout, pure torch.

    Returns ``(weight_packed, weight_scale)``:

      * ``weight_packed``: uint8 ``[out, in // 2]`` -- packed FP4
        (``float4_e2m1fn_x2``), two e2m1 nibbles per byte, low nibble first.
      * ``weight_scale``:  uint8 ``[out, in // 32]`` -- UE8M0 per-block shared
        exponent, ``+127`` biased.

    The output is byte-identical in *format* to what msmodelslim emits
    (``pack_fp4_to_uint8`` + ``on_w4a8_mx_dynamic_per_block``), so the tensors feed
    straight into the offline W4A8 kernel path (FRACTAL_NZ cast + transpose +
    ``npu_quant_matmul(x2_dtype=fp4)``) that is already verified on the A5. Online
    vs offline then differ only in the weight *source* (RTN here vs msmodelslim
    calibration), not the layout.

    Deliberately avoids ``npu_dynamic_mx_quant(dst=float4_e2m1fn_x2)``: on current
    CANN builds its arch35 tiling rejects the shape-halving packed-FP4 output
    ("output y must be same shape as input x"). The offline path never calls that
    op (weights come pre-quantised in the checkpoint) -- which is exactly why the
    offline script keeps running while this online one broke.

    MXFP4 constants mirror msmodelslim (``ir/qal/qbase.py``,
    ``core/quantizer/mse_round.py``): ebits=2 -> ``emax = 2**(ebits-1) = 2``, the
    e2m1 grid is ``[0, .5, 1, 1.5, 2, 3, 4, 6]`` (max_norm 6.0), and the per-block
    shared exponent is ``floor(log2(amax)) - emax``.
    """
    block = MXFP4_BLOCK_SIZE
    out_features, in_features = weight_fp.shape
    w_blocks = weight_fp.to(torch.float32).reshape(out_features, -1, block)

    # Per-block shared exponent (UE8M0): floor(log2(amax)) - emax. All-zero blocks
    # get exponent 0 (scale 1); clamp keeps it inside the int8 (+127) storage range.
    amax = w_blocks.abs().amax(dim=-1, keepdim=True)
    nonzero = amax > 0
    exp = torch.floor(torch.log2(torch.where(nonzero, amax, torch.ones_like(amax)))) - 2
    exp = torch.where(nonzero, exp, torch.zeros_like(exp)).clamp_(-127, 127)

    # Normalise into the e2m1 range, then round-to-nearest onto the non-uniform
    # e2m1 grid via midpoint thresholds. Counting how many midpoints each value
    # exceeds is equivalent to argmin|x - grid| but uses only elementwise compares
    # (NPU-friendly; no torch.bucketize and no [out, in, 8] broadcast). Values above
    # 6.0 saturate to the top grid index (7), matching the reference packer.
    w_norm = (w_blocks / torch.exp2(exp)).reshape(out_features, in_features)
    w_abs = w_norm.abs()
    mag = torch.zeros_like(w_abs, dtype=torch.int32)
    for boundary in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        mag += (w_abs > boundary).to(torch.int32)
    # 4-bit code = sign bit (bit 3) | magnitude index (bits 0-2).
    codes = mag + (w_norm < 0).to(torch.int32) * 8

    # Pack consecutive pairs low-nibble-first: byte = code[2j] | (code[2j+1] << 4).
    codes = codes.reshape(out_features, in_features // 2, 2)
    weight_packed = (codes[..., 0] | (codes[..., 1] << 4)).to(torch.uint8)
    weight_scale = (exp.reshape(out_features, in_features // block) + 127).to(torch.uint8)
    return weight_packed, weight_scale


class _NPULinearMethodBase(LinearMethodBase):

    def __init__(
        self,
        quant_config: Optional["QuantizationConfig"] = None,
    ):
        self.quant_config = quant_config


class NPUW8A8Int8LinearMethod(_NPULinearMethodBase):

    def process_weights_after_loading(self, layer: torch.nn.Module):
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight.data = npu_format_cast(layer.weight.data)

        layer.weight_scale.data = layer.weight_scale.data.flatten()
        # Compressed-tensors format doesn't have this field
        if hasattr(layer, "weight_offset"):
            layer.weight_offset.data = layer.weight_offset.data.flatten()

        expanding_factor = layer.weight.data.shape[0]
        layer.aclnn_input_scale = torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor).to(device="npu"),
            requires_grad=False,
        )
        layer.aclnn_input_scale_reciprocal = 1 / torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor).to(device="npu"),
            requires_grad=False,
        )
        layer.aclnn_input_offset = torch.nn.Parameter(
            layer.input_offset.data.repeat(expanding_factor).to(device="npu"),
            requires_grad=False,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from sglang.srt.layers.linear import RowParallelLinear

        original_dtype = x.dtype
        if original_dtype != torch.int8:
            x = torch.ops.npu.npu_quantize(
                x,
                layer.aclnn_input_scale_reciprocal,
                layer.aclnn_input_offset,
                torch.qint8,
                -1,
                False,
            )
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in Attention TP>1 case)
        if isinstance(layer, RowParallelLinear) and layer.tp_rank > 0:
            quant_bias = None
        else:
            quant_bias = layer.quant_bias
        return torch.ops.npu.npu_quant_matmul(
            x,
            layer.weight,
            layer.deq_scale,
            bias=quant_bias,
            output_dtype=original_dtype,
        )


class NPUW8A8Int8DynamicLinearMethod(_NPULinearMethodBase):

    def process_weights_after_loading(self, layer: torch.nn.Module):
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight.data = npu_format_cast(layer.weight.data)

        layer.weight_scale.data = layer.weight_scale.data.flatten()
        # Compressed-tensors format doesn't have this field
        if hasattr(layer, "weight_offset"):
            layer.weight_offset.data = layer.weight_offset.data.flatten()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if isinstance(x, tuple):
            """dynamic_scale is calculated in malprolog kernel"""
            original_dtype = torch.bfloat16
            quant_out, dynamic_scale = x
        else:
            original_dtype = x.dtype
            quant_out, dynamic_scale = torch.ops.npu.npu_dynamic_quant(x)
        return torch.ops.npu.npu_quant_matmul(
            quant_out,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=dynamic_scale.flatten(),
            bias=bias,
            output_dtype=original_dtype,
        )


class NPUMXFP8LinearMethod(_NPULinearMethodBase):
    """Ascend NPU MXFP8 linear method for LLM (SRT) models.

    Shared kernel for both the online config path (``--quantization mxfp8``) and
    the offline ModelSlimMXFP8Scheme (which delegates to this as ``self.kernel``).
    process_weights_after_loading branches on weight dtype: FP16/BF16 weights are
    quantised to MXFP8 at load time (online); pre-quantised float8_e4m3fn weights
    are only re-laid-out (offline). Inference: dynamic MXFP8 activation quant +
    MXFP8 matmul (block_size=32).
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.parameter import ModelWeightParameter

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # Load weights in original dtype; quantise later in process_weights_after_loading
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        if weight.dtype == torch.float8_e4m3fn:
            # Offline (ModelSlim) path: weight is already MXFP8-quantised and
            # layer.weight_scale holds the uint8 block scales [out, in/32]. Only
            # re-layout to [in, out] / [in//64, out, 2] strided views below.
            n_dim, k_dim = layer.weight_scale.data.shape
            scale = layer.weight_scale.data.reshape(n_dim, k_dim // 2, 2)
            layer.weight = Parameter(weight.transpose(0, 1), requires_grad=False)
            layer.weight_scale_inv = Parameter(
                scale.transpose(0, 1), requires_grad=False
            )
            # weight_scale is now folded into weight_scale_inv (which keeps the
            # underlying storage alive via its view); drop the stale parameter so
            # it doesn't linger in named_parameters() / state_dict().
            del layer.weight_scale
        else:
            # Online path: quantise FP16/BF16 weights to MXFP8 at load time.
            if weight.dtype not in (torch.float16, torch.bfloat16):
                logger.warning(
                    "NPUMXFP8LinearMethod: weight dtype %s is not float16/bfloat16; "
                    "casting to bfloat16 before MXFP8 quantisation.",
                    weight.dtype,
                )
                weight = weight.to(torch.bfloat16)
            # Move weight to NPU if needed (cpu offload may move it back to CPU).
            if not weight.is_npu:
                weight = weight.to(f"npu:{torch.npu.current_device()}")
            # Online MXFP8 quantisation of weights (block_size=32).
            # qw: [out, in] float8_e4m3fn, w_scale: [out, in//64, 2] uint8.
            qw, w_scale = torch.ops.npu.npu_dynamic_mx_quant(
                weight, dst_type=torch.float8_e4m3fn
            )
            layer.weight = Parameter(qw.transpose(0, 1), requires_grad=False)
            layer.weight_scale_inv = Parameter(
                w_scale.transpose(0, 1), requires_grad=False
            )

        # Both paths produce weight [in, out] and weight_scale_inv [in//64, out,
        # 2] as strided transpose views — DO NOT call .contiguous(). The matmul
        # reduction loop scans the in-dim per output column; the [out, in]
        # row-major source gives stride-1 access for that scan via the transpose
        # view (matches msmodelslim's offline layout and vllm-ascend's
        # AscendW8A8MXFP8DynamicLinearMethod). Calling .contiguous() physically
        # reorders to [in, out] row-major, making the inner-loop stride = out and
        # tanking HBM bandwidth.

        # Cache FP32 bias once to avoid a per-forward dtype conversion + alloc.
        if (
            getattr(layer, "bias", None) is not None
            and layer.bias.dtype != torch.float32
        ):
            layer.bias_fp32 = Parameter(
                layer.bias.data.to(torch.float32), requires_grad=False
            )
        else:
            layer.bias_fp32 = None

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        original_dtype = x.dtype
        if original_dtype not in (torch.float16, torch.bfloat16):
            x = x.to(torch.bfloat16)
            original_dtype = torch.bfloat16

        # Flatten to 2D [tokens, hidden] for npu_dynamic_mx_quant
        input_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        # Dynamic MXFP8 activation quantisation
        qx, input_scale = torch.ops.npu.npu_dynamic_mx_quant(
            x_2d, dst_type=torch.float8_e4m3fn
        )

        # MXFP8 matmul (weight & scale already transposed at load time)
        # Use the cached FP32 bias from process_weights_after_loading; fall back
        # to per-call conversion if the cache was bypassed (e.g. dynamic bias).
        if bias is None:
            quant_bias = None
        elif (
            bias is getattr(layer, "bias", None)
            and getattr(layer, "bias_fp32", None) is not None
        ):
            quant_bias = layer.bias_fp32
        else:
            quant_bias = bias.to(torch.float32)

        e8m0_dtype = _get_float8_e8m0fnu_dtype()
        output = torch.ops.npu.npu_quant_matmul(
            qx,
            layer.weight,
            layer.weight_scale_inv,
            scale_dtype=e8m0_dtype,
            pertoken_scale=input_scale,
            pertoken_scale_dtype=e8m0_dtype,
            bias=quant_bias,
            output_dtype=original_dtype,
            group_sizes=[1, 1, MXFP8_BLOCK_SIZE],
        )

        # Restore original shape (replace last dim with output features)
        output_shape = list(input_shape[:-1]) + [output.shape[-1]]
        return output.reshape(output_shape)


class NPU_W4A4DynamicLinearMethod(_NPULinearMethodBase):

    def process_weights_after_loading(self, layer):
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight_scale.data = layer.weight_scale.data.flatten()
        layer.weight_scale_fp32 = layer.weight_scale.data.to(torch.float32)
        layer.weight_offset.data = layer.weight_offset.data.flatten()
        layer.weight.data = torch.ops.npu.npu_convert_weight_to_int4pack(
            layer.weight.data.to(torch.int32)
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = 0,
    ) -> torch.Tensor:
        original_dtype = x.dtype
        quant_out, dynamic_scale = torch.ops.npu.npu_dynamic_quant(
            x, dst_type=torch.quint4x2
        )
        return torch.ops.npu.npu_quant_matmul(
            quant_out,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=dynamic_scale.flatten(),
            bias=bias,
            output_dtype=original_dtype,
        )


class NPUMXFP4W4A8LinearMethod(_NPULinearMethodBase):
    """Ascend NPU W4A8 online quantization: MXFP4 weights + MXFP8 activations.

    This is a *true* W4(weight) A8(activation) path: it mirrors the offline
    ``W4A8_MXFP`` kernel (``NPUMXFP4W4A8OfflineLinearMethod``) exactly — the only
    difference is that the FP4 weights are produced online from BF16/FP16
    (round-to-nearest, no calibration) instead of being loaded from a msmodelslim
    checkpoint. An earlier version of this method ran a *dual-level* scheme that
    also compressed the activation to FP4 (W4A4 compute via
    ``npu_dual_level_quant_matmul``); that was a large accuracy regression — 4-bit
    activations — so it was replaced with the single-level FP8-activation path
    below, aligned with the offline W4A8 implementation.

    Weight quantization (process_weights_after_loading):
        BF16/FP16 weight → npu_dynamic_mx_quant(dst=float4_e2m1fn_x2) → packed FP4
        + UE8M0 block scale → npu_format_cast to FRACTAL_NZ → transpose [in//2, out]

    Inference (apply):
        BF16/FP16 activation → npu_dynamic_mx_quant(dst=float8_e4m3fn)  (A8, FP8)
        → npu_quant_matmul(x2_dtype=float4_e2m1fn_x2, group_sizes=[0, 0, block])

    Hardware: Ascend 950 (A5) + a recent torch_npu with the FP4 npu_quant_matmul
    (same requirement as the offline W4A8 path — see that class's docstring).
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Register an unquantized (``params_dtype``) weight placeholder.

        Online quantization needs its own ``create_weights`` because the
        checkpoint still holds full-precision BF16/FP16 weights: the loader
        fills this buffer, then ``process_weights_after_loading`` quantizes it to
        MXFP4 in place. This differs from the offline/int8 methods, whose weights
        are created by the scheme's own ``create_weights`` to match the
        already-quantized (FP8 / uint8-packed) layout the checkpoint provides.
        """
        from sglang.srt.layers.parameter import ModelWeightParameter

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # Load weights in original dtype; quantise to MXFP4 in
        # process_weights_after_loading.
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Online single-level MXFP4 weight quant, then lay the weight out exactly
        # like the offline W4A8 path so the same npu_quant_matmul(x2_dtype=fp4)
        # kernel accepts it. All NPU ops go through torch.ops.npu.* (no torch_npu).
        fp4_dtype = _get_float4_e2m1fn_x2_dtype()

        weight_fp = layer.weight.data
        if weight_fp.dtype not in (torch.float16, torch.bfloat16):
            weight_fp = weight_fp.to(torch.bfloat16)
        # Move to NPU if needed (cpu offload may have put it on CPU).
        if not weight_fp.is_npu:
            weight_fp = weight_fp.to(f"npu:{torch.npu.current_device()}")

        # BF16/FP16 -> packed FP4 [out, in//2] + UE8M0 block scale [out, in//32],
        # byte-identical to the msmodelslim W4A8_MXFP checkpoint. Pure torch (see
        # _mxfp4_quantize_weight): deliberately NOT npu_dynamic_mx_quant(dst=fp4),
        # whose arch35 tiling rejects the shape-halving packed-FP4 output on current
        # CANN builds ("output y must be same shape as input x"). Offline never
        # calls that op, which is why offline keeps running and online broke.
        qw, w_scale = _mxfp4_quantize_weight(weight_fp)

        # weight: packed FP4 -> FRACTAL_NZ (float8_e4m3fn view) -> transpose
        # [in//2, out]; weight_scale [out, in//32] -> [in//64, out, 2]. Identical to
        # NPUMXFP4W4A8OfflineLinearMethod.process_weights_after_loading below.
        qw_nz = npu_format_cast(
            qw,
            NPUACLFormat.ACL_FORMAT_FRACTAL_NZ,
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=fp4_dtype,
        )
        layer.weight = Parameter(qw_nz.transpose(-1, -2), requires_grad=False)

        n, k = w_scale.shape
        layer.weight_scale = Parameter(
            w_scale.reshape(n, k // 2, 2).transpose(-3, -2), requires_grad=False
        )

        # Cache FP32 bias once to avoid a per-forward dtype conversion + alloc.
        if (
            getattr(layer, "bias", None) is not None
            and layer.bias.dtype != torch.float32
        ):
            layer.bias_fp32 = Parameter(
                layer.bias.data.to(torch.float32), requires_grad=False
            )
        else:
            layer.bias_fp32 = None

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        e8m0_dtype = _get_float8_e8m0fnu_dtype()
        fp4_dtype = _get_float4_e2m1fn_x2_dtype()

        original_dtype = x.dtype
        if original_dtype not in (torch.float16, torch.bfloat16):
            x = x.to(torch.bfloat16)
            original_dtype = torch.bfloat16

        # Flatten to 2D [tokens, hidden] for npu_dynamic_mx_quant.
        input_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        # Dynamic MXFP8 activation quantisation (A8 — FP8, not FP4).
        quantized_x, dynamic_scale = torch.ops.npu.npu_dynamic_mx_quant(
            x_2d, dst_type=torch.float8_e4m3fn
        )

        # Use the cached FP32 bias from process_weights_after_loading; fall back
        # to per-call conversion if the cache was bypassed (e.g. dynamic bias).
        if bias is None:
            quant_bias = None
        elif (
            bias is getattr(layer, "bias", None)
            and getattr(layer, "bias_fp32", None) is not None
        ):
            quant_bias = layer.bias_fp32
        else:
            quant_bias = bias.to(torch.float32)

        # True W4(weight)A8(activation) matmul, identical to the offline path.
        output = torch.ops.npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            scale_dtype=e8m0_dtype,
            pertoken_scale=dynamic_scale,
            pertoken_scale_dtype=e8m0_dtype,
            bias=quant_bias,
            output_dtype=original_dtype,
            x2_dtype=fp4_dtype,
            group_sizes=[0, 0, MXFP4_BLOCK_SIZE],
        )

        # Restore original shape (replace last dim with output features).
        output_shape = list(input_shape[:-1]) + [output.shape[-1]]
        return output.reshape(output_shape)


class NPUMXFP4W4A8OfflineLinearMethod(_NPULinearMethodBase):
    """Ascend NPU offline W4A8 (ModelSlim ``W4A8_MXFP``): packed-FP4 weights + MXFP8 activations.

    Kernel for the offline ModelSlimMXFP4W4A8Scheme (delegated as ``self.kernel``).
    The msmodelslim ``W4A8_MXFP`` checkpoint stores weights as *packed FP4*
    (``pack_fp4_to_uint8`` → ``uint8`` shape ``[out, in//2]``) plus UE8M0 block
    scales (``uint8`` shape ``[out, in//group_size]``):

      process_weights_after_loading:
        weight (uint8 packed FP4 [out, in//2]) → npu_format_cast(29,
            customize_dtype=float8_e4m3fn, input_dtype=float4_e2m1fn_x2) → FRACTAL_NZ
            → transpose [in//2, out]
        weight_scale [out, in/32] → reshape [out, in/64, 2] → transpose → [in/64, out, 2]

      apply:
        BF16/FP16 activation → npu_dynamic_mx_quant(dst=float8_e4m3fn)  (A8, MXFP8)
        → npu_quant_matmul(x2_dtype=float4_e2m1fn_x2, group_sizes=[0, 0, block])

    Mirrors vllm-ascend ``AscendW4A8MXFPDynamicLinearMethod`` exactly (Ascend 950/A5).
    The weight is cast to FRACTAL_NZ then transposed; ``npu_dynamic_mx_quant`` already
    returns a 3D ``[tokens, in//64, 2]`` block scale so the matmul needs no extra
    scale-layout normalization.

    ⚠️ REQUIRES a recent torch_npu build for the FP4 ``npu_quant_matmul``. On the
    A5 this device forces ``allow_internal_format=False`` (the NZ cast still produces
    a ``FRACTAL_NZ_C0_16`` tensor, which is fine). Older torch_npu (e.g.
    ``2.10.0.dev20260320``) had a broken FP4 matmul that rejected the NZ weight in
    *prefill* with ``x2 should be in ... nz format, but it is 2``;
    ``2.10.0.post1.dev20260624`` (and later) runs the vllm-aligned NZ path
    correctly. If you hit ``it is 2``, update torch_npu — do NOT "fix" it by
    switching the weight to ND.

    ⚠️ A ``atb::OperationSetup`` *segfault during decode* (not prefill) is a
    DIFFERENT, unrelated issue: it is the eager-decode ``ascend`` attention
    backend, NOT this matmul (verified by stage-sync bisection — qkv's matmul
    syncs clean, the fault surfaces at the entry-sync of the next layer, i.e. the
    decode attention between qkv and o_proj). Run with the NPU decode graph (do
    NOT pass ``--disable-cuda-graph``); graph mode is the NPU default and what
    vllm uses. This attention issue is model-agnostic and out of scope for W4A8.

    This is a true W4(weight) A8(activation) single-level matmul. The *online*
    ``NPUMXFP4W4A8LinearMethod`` now uses this exact apply path — the only
    difference is that it quantizes BF16/FP16 weights to FP4 at load time instead
    of loading them from a msmodelslim checkpoint. ``group_size`` is fixed at 32
    by the ``W4A8_MXFP`` export format.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Mirror vllm-ascend AscendW4A8MXFPDynamicLinearMethod: cast the packed-FP4
        # weight to FRACTAL_NZ then transpose. All NPU ops go through
        # torch.ops.npu.* (no torch_npu). Requires a recent torch_npu build (see
        # class docstring): older builds reject the NZ weight ("x2 ... it is 2").
        fp4_dtype = _get_float4_e2m1fn_x2_dtype()

        # weight: packed-FP4 uint8 [out, in//2] -> FRACTAL_NZ (float8_e4m3fn view)
        # -> transpose to [in//2, out].
        layer.weight.data = npu_format_cast(
            layer.weight.data,
            NPUACLFormat.ACL_FORMAT_FRACTAL_NZ,
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=fp4_dtype,
        )
        layer.weight.data = layer.weight.data.transpose(-1, -2)
        # weight_scale: [out, in/32] uint8 -> [in/64, out, 2].
        n, k = layer.weight_scale.data.shape
        layer.weight_scale.data = layer.weight_scale.data.reshape(
            n, k // 2, 2
        ).transpose(-3, -2)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        e8m0_dtype = _get_float8_e8m0fnu_dtype()
        fp4_dtype = _get_float4_e2m1fn_x2_dtype()

        original_dtype = x.dtype
        if original_dtype not in (torch.float16, torch.bfloat16):
            x = x.to(torch.bfloat16)
            original_dtype = torch.bfloat16

        # Flatten to 2D [tokens, hidden] for npu_dynamic_mx_quant.
        input_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        # Dynamic MXFP8 activation quantisation (A8).
        quantized_x, dynamic_scale = torch.ops.npu.npu_dynamic_mx_quant(
            x_2d, dst_type=torch.float8_e4m3fn
        )

        if bias is not None and bias.dtype != torch.float32:
            bias = bias.to(torch.float32)

        # W4(weight)A8(activation) matmul, mirroring vllm-ascend exactly.
        output = torch.ops.npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            scale_dtype=e8m0_dtype,
            pertoken_scale=dynamic_scale,
            pertoken_scale_dtype=e8m0_dtype,
            bias=bias,
            output_dtype=original_dtype,
            x2_dtype=fp4_dtype,
            group_sizes=[0, 0, MXFP4_BLOCK_SIZE],
        )

        # Restore original shape (replace last dim with output features).
        output_shape = list(input_shape[:-1]) + [output.shape[-1]]
        return output.reshape(output_shape)
