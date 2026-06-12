import logging
from typing import TYPE_CHECKING, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.hardware_backend.npu.utils import npu_format_cast
from sglang.srt.layers.quantization.base_config import LinearMethodBase
from sglang.srt.platforms import current_platform

_is_npu = current_platform.is_npu()

if _is_npu:
    import torch_npu

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

logger = logging.getLogger(__name__)

MXFP8_BLOCK_SIZE = 32
_FLOAT8_E8M0FNU_DTYPE = (
    getattr(torch_npu, "float8_e8m0fnu", getattr(torch, "float8_e8m0fnu", None))
    if _is_npu
    else getattr(torch, "float8_e8m0fnu", None)
)


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

    Online mode: loads FP16/BF16 weights → quantises to MXFP8 at load time.
    Inference: dynamic MXFP8 activation quant + MXFP8 matmul (block_size=32).
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
        weight_fp = layer.weight.data
        if weight_fp.dtype not in (torch.float16, torch.bfloat16):
            logger.warning(
                "NPUMXFP8LinearMethod: weight dtype %s is not float16/bfloat16; "
                "casting to bfloat16 before MXFP8 quantisation.",
                weight_fp.dtype,
            )
            weight_fp = weight_fp.to(torch.bfloat16)

        # Move weight to NPU if needed (cpu offload may have moved it back to CPU)
        if not weight_fp.is_npu:
            weight_fp = weight_fp.to(f"npu:{torch.npu.current_device()}")

        # Online MXFP8 quantisation of weights (block_size=32).
        # qw: [out, in] float8_e4m3fn, w_scale: [out, in//64, 2] uint8.
        qw, w_scale = torch_npu.npu_dynamic_mx_quant(
            weight_fp, dst_type=torch_npu.float8_e4m3fn
        )
        # Transpose to [in, out] / [in//64, out, 2] as a strided view — DO NOT
        # call .contiguous(). The matmul reduction loop scans the in-dim per
        # output column; the [out, in] row-major layout gives stride-1 access
        # for that scan via the transpose view (matches msmodelslim's offline
        # layout and vllm-ascend's AscendW8A8MXFP8DynamicLinearMethod). Calling
        # .contiguous() physically reorders to [in, out] row-major, which makes
        # the inner-loop stride = out and tanks HBM bandwidth.
        layer.weight = Parameter(qw.transpose(0, 1), requires_grad=False)
        layer.weight_scale_inv = Parameter(w_scale.transpose(0, 1), requires_grad=False)
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
        qx, input_scale = torch_npu.npu_dynamic_mx_quant(
            x_2d, dst_type=torch_npu.float8_e4m3fn
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

        output = torch_npu.npu_quant_matmul(
            qx,
            layer.weight,
            layer.weight_scale_inv,
            scale_dtype=_FLOAT8_E8M0FNU_DTYPE,
            pertoken_scale=input_scale,
            pertoken_scale_dtype=_FLOAT8_E8M0FNU_DTYPE,
            bias=quant_bias,
            output_dtype=original_dtype,
            group_sizes=[1, 1, MXFP8_BLOCK_SIZE],
        )

        # Restore original shape (replace last dim with output features)
        output_shape = list(input_shape[:-1]) + [output.shape[-1]]
        return output.reshape(output_shape)


class NPUMXFP4W4A8LinearMethod(_NPULinearMethodBase):
    """Ascend NPU W4A8 online quantization: MXFP4 weights + MXFP8 activations.

    Weight quantization flow (process_weights_after_loading):
        BF16/FP16 weight → npu_dynamic_dual_level_mx_quant → FP4 + l0_scale(FP32) + l1_scale(FP8_E8M0)
        → npu_format_cast to FRACTAL_NZ (required by npu_dual_level_quant_matmul)
        → w_dual_scale transposed to [in/512, out] (required by matmul API)

    Inference flow (apply):
        FP16/BF16 activation → npu_dynamic_dual_level_mx_quant → FP4 + act_l0_scale + act_l1_scale
        → npu_dual_level_quant_matmul(FP4_act, FP4_weight, scales...) → FP16/BF16 output

    Note: The "A8" refers to the MXFP8 intermediate scale format (FP8_E8M0 l1_scale).
    The actual matmul compute is W4A4 (both operands in FP4) since there is no
    W4A8 mixed-precision kernel in the current torch_npu public API.
    """

    _FLOAT4_E2M1FN_X2_DTYPE = (
        getattr(torch_npu, "float4_e2m1fn_x2", getattr(torch, "float4_e2m1fn_x2", None))
        if _is_npu
        else getattr(torch, "float4_e2m1fn_x2", None)
    )

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

        # Load weights in original dtype; quantise to MXFP4 in process_weights_after_loading
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
        import logging

        from sglang.srt.utils import get_npu_memory_capacity

        _logger = logging.getLogger(__name__)

        # Heuristic hardware check: npu_dynamic_dual_level_mx_quant requires Ascend 950.
        # Atlas A2/A3 have ≤64 GB per card; Ascend 950 has ≥96 GB per card.
        npu_mem_mb = get_npu_memory_capacity()
        if npu_mem_mb < 96 * 1024:
            _logger.warning(
                "MXFP4 W4A8 dual-level quantization may not be supported on this "
                "hardware (detected NPU memory %.1f GB < 96 GB). "
                "npu_dynamic_dual_level_mx_quant requires Ascend 950 (Atlas A3). "
                "Continuing — expect a RuntimeError if the kernel is unavailable.",
                npu_mem_mb / 1024,
            )

        weight_fp = layer.weight.data
        if weight_fp.dtype not in (torch.float16, torch.bfloat16):
            weight_fp = weight_fp.to(torch.bfloat16)

        # Move to NPU if needed (cpu offload may have put it on CPU)
        if not weight_fp.is_npu:
            weight_fp = weight_fp.to(f"npu:{torch.npu.current_device()}")

        # Online MXFP4 dual-level quantisation of weights
        # qw:          float4_e2m1fn_x2, shape [out, in]
        # w_dual_scale: float32,          shape [out, in/512, 1]  (L0)
        # w_scale:      float8_e8m0,      shape [out, (ceil(in/32)+1)//2, 2]  (L1)
        try:
            qw, w_dual_scale, w_scale = torch_npu.npu_dynamic_dual_level_mx_quant(
                weight_fp, smooth_scale=None
            )
        except (RuntimeError, AttributeError) as e:
            raise RuntimeError(
                "npu_dynamic_dual_level_mx_quant failed — this operation requires "
                "Ascend 950 (Atlas A3). Atlas 800I A2/A3 and earlier chips do NOT "
                "support DualLevelQuantBatchMatmul. "
                f"Original error: {e}"
            ) from e

        # npu_dual_level_quant_matmul requires x2 in FRACTAL_NZ format (format=29)
        # view as int8 first because npu_format_cast only accepts int-dtype tensors
        qw = torch_npu.npu_format_cast(qw.view(torch.int8), 29)

        # npu_dual_level_quant_matmul expects x2_level0_scale shape [in/512, out]:
        # squeeze the trailing dim-1 axis, then transpose as a strided view.
        # Mirrors the MXFP8 dense path: skipping .contiguous() avoids a physical
        # reorder + alloc and keeps stride-1 access for the in-dim scan.
        # TODO(NPU-validate): npu_dual_level_quant_matmul is a different kernel
        # than npu_quant_matmul; its strided-view tolerance is NOT yet verified
        # on Ascend 950/A3. Re-validate output (no garbled tokens) on hardware;
        # if it regresses, restore `.contiguous()` on this line only.
        w_dual_scale = w_dual_scale.squeeze(-1).transpose(0, 1)

        layer.weight = Parameter(qw, requires_grad=False)
        layer.weight_dual_scale = Parameter(w_dual_scale, requires_grad=False)
        layer.weight_scale = Parameter(w_scale, requires_grad=False)
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

        # Flatten to 2D [tokens, hidden] for dual-level quant API
        input_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        # Dynamic MXFP4 activation quantisation (W4 activations → A4 for matmul)
        qx, act_l0_scale, act_l1_scale = torch_npu.npu_dynamic_dual_level_mx_quant(
            x_2d, smooth_scale=None
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

        # MXFP4 matmul: W4A4 compute (weight already in NZ format + transposed scales)
        output = torch_npu.npu_dual_level_quant_matmul(
            qx,
            layer.weight,
            act_l0_scale,
            layer.weight_dual_scale,
            act_l1_scale,
            layer.weight_scale,
            bias=quant_bias,
            output_dtype=original_dtype,
        )

        # Restore original shape (replace last dim with output features)
        output_shape = list(input_shape[:-1]) + [output.shape[-1]]
        return output.reshape(output_shape)


class NPUW4A8DynamicLinearMethod(_NPULinearMethodBase):
    """Ascend NPU W4A8 offline quantization linear method.

    Offline mode: loads ModelSlim pre-quantized INT4 weights.
    For ``new_quant_version=True`` (version "1.0.0"): 2 int4 values are pre-packed
    into 1 int8 in the checkpoint (shape ``[N/2, K]``).
    For old version: plain int4 stored as int8 (shape ``[N, K]``).

    Uses ``torch_npu.npu_weight_quant_batchmatmul`` for inference — activations
    stay in high precision and INT4 weights are dequantized on-the-fly.
    """

    def __init__(
        self,
        group_size: int = 256,
        new_quant_version: bool = True,
    ):
        super().__init__()
        self.group_size = group_size
        self.new_quant_version = new_quant_version

    @staticmethod
    def _process_scale_second(
        weight: torch.Tensor,
        scale: torch.Tensor,
        per_group_scale: torch.Tensor,
        is_new_quant: bool = False,
    ):
        """Merge per-channel (L1) and per-group (L2) scales into antiquant_scale.

        Args:
            weight: weight after transpose, shape ``[K, N/2]`` (new) or ``[K, N]`` (old)
            scale: per-channel L1 scale, shape ``[N]``
            per_group_scale: per-group L2 scale after transpose, shape ``[K//group_size, N]``
            is_new_quant: whether weight dim is compressed (N/2)

        Returns:
            (antiquant_scale, bias): ``antiquant_scale`` shape ``[K//group_size, N]``;
            ``bias`` is non-None only for old version (asymmetric compensation term).
        """
        k, n_compressed = weight.shape
        group_num, n_scale = per_group_scale.shape

        # Logical N dimension
        n = n_compressed * 2 if is_new_quant else n_compressed

        bias = None
        if not is_new_quant:
            weight_high = weight.to(torch.float32).reshape(
                group_num, -1, n
            ) * per_group_scale.reshape(group_num, 1, n)
            weight_high = weight_high.reshape(k, n)
            bias = 8 * (weight_high.to(torch.float32) * scale).sum(dim=0)

        antiquant_scale = (scale * per_group_scale).reshape(group_num, n)
        return antiquant_scale, bias

    def process_weights_after_loading(self, layer: torch.nn.Module):
        from sglang.srt.hardware_backend.npu.utils import npu_format_cast

        # Transpose [N, K] → [K, N] (or [N/2, K] → [K, N/2] for packed)
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        # Cast to FRACTAL_NZ format for NPU matmul efficiency
        layer.weight.data = npu_format_cast(layer.weight.data)

        # Flatten per-channel scales to 1-D float32
        layer.weight_scale.data = layer.weight_scale.data.flatten().to(torch.float32)
        layer.weight_offset.data = layer.weight_offset.data.flatten()

        # Merge L1/L2 scales: weight_scale_second loaded as [N, K//group_size],
        # transpose to [K//group_size, N] for process_scale_second
        layer.weight_scale_second.data, scale_bias = self._process_scale_second(
            layer.weight.data,
            layer.weight_scale.data,
            layer.weight_scale_second.data.transpose(0, 1).contiguous(),
            is_new_quant=self.new_quant_version,
        )

        if self.new_quant_version:
            # Handle optional scale_bias parameter
            if hasattr(layer, "scale_bias"):
                if layer.scale_bias.data.shape[1] == 1:
                    layer.scale_bias.data = layer.scale_bias.data.flatten()
                else:
                    layer.scale_bias.data = layer.scale_bias.data.contiguous()
            # Pack 4 int8 (2×int4) into int32 for NPU kernel
            assert (
                layer.weight.data.shape[-1] % 4 == 0
            ), f"Last dim of weight must be divisible by 4, got {layer.weight.data.shape}"
            layer.weight.data = layer.weight.data.view(torch.int32).contiguous()
        else:
            # Old version: use NPU int4-pack conversion
            if scale_bias is not None:
                param = torch.nn.Parameter(scale_bias, requires_grad=False)
                layer.register_parameter("weight_scale_bias", param)
            layer.weight.data = torch_npu.npu_convert_weight_to_int4pack(
                layer.weight.data.to(torch.int32)
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Weight-dequant path: INT4 weights dequantized on-the-fly, activations in high precision
        return torch_npu.npu_weight_quant_batchmatmul(
            x,
            layer.weight,
            antiquant_scale=layer.weight_scale_second.to(x.dtype),
            antiquant_group_size=self.group_size,
        )


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
