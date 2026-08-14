import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import torch
from torch.nn.parameter import Parameter

from sglang.srt.hardware_backend.npu.utils import NPUACLFormat, npu_format_cast
from sglang.srt.layers.quantization.base_config import LinearMethodBase

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

MXFP8_BLOCK_SIZE = 32
# OCP UE8M0 reserves 0xFF for NaN, so it is an unambiguous not-loaded
# sentinel for ModelSlim block scales.
MXFP_E8M0_NOT_LOADED = 0xFF
# W4A8_MXFP block (group) size — fixed at 32 by the msmodelslim export format.
MXFP4_BLOCK_SIZE = 32
BLOCK_FP8_SIZE = (128, 128)


# NPU ops are reached via torch.ops.npu.* (registered when torch_npu is imported
# by the runtime), so this module needs no top-level `import torch_npu` and stays
# importable on CUDA/CPU/AMD/XPU CI.
def _get_float8_e8m0fnu_dtype():
    # Resolve lazily rather than as a module-level constant: this module is
    # imported early (during quant-scheme registration), so reading the dtype at
    # call time keeps it correct regardless of import order / platform.
    return getattr(torch, "float8_e8m0fnu", None)


def _get_float4_e2m1fn_x2_dtype():
    # The packed-FP4 dtype MUST come from torch_npu (an int enum, e.g. 296), not
    # from torch. The NPU ops that consume it -- npu_dynamic_mx_quant(dst_type=),
    # npu_quant_matmul(x2_dtype=), npu_format_cast(input_dtype=) -- REJECT the
    # torch dtype object torch.float4_e2m1fn_x2 in op-plugin on recent torch_npu
    # builds (it raises, or with None gives "output y must be same shape as input
    # x"), even though torch.float4_e2m1fn_x2 exists. This is fp4-specific: fp8 /
    # float8_e8m0fnu is accepted from torch either way. Verified on A5 /
    # torch_npu 2.10.0.post2.dev20260704 (see llm/probe_fp4_w4a8_chain.py: dst=296
    # passes the full quant->format_cast->matmul chain, dst=torch dtype fails).
    #
    # Lazy import so this NPU-only path keeps the module importable on
    # CUDA/CPU/AMD/XPU CI (no top-level torch_npu; see AGENTS.md known pitfalls).
    from sglang.srt.utils import is_npu

    if is_npu():
        import torch_npu

        npu_dtype = getattr(torch_npu, "float4_e2m1fn_x2", None)
        if npu_dtype is not None:
            return npu_dtype
    return getattr(torch, "float4_e2m1fn_x2", None)


def _get_npu_ops():
    """Resolve the torch_npu op namespace lazily for CPU-only imports/tests."""
    return torch.ops.npu


def _npu_device_index(tensor: torch.Tensor) -> int:
    device_index = tensor.device.index
    if device_index is not None:
        return device_index

    npu_module = vars(torch).get("npu")
    if npu_module is None or not npu_module.is_available():
        raise RuntimeError("An available Ascend NPU is required for block-FP8.")
    return int(npu_module.current_device())


@lru_cache(maxsize=None)
def _npu_device_is_a5(device_index: int) -> bool:
    from sglang.srt.utils import is_npu_atlas_a5

    return is_npu_atlas_a5(device_index)


def _npu_is_a5_for_tensor(tensor: torch.Tensor) -> bool:
    """Probe only after an NPU tensor exists, then cache that stable device."""
    return _npu_device_is_a5(_npu_device_index(tensor))


def _require_same_npu_device(*tensors: Optional[torch.Tensor]) -> None:
    devices = {tensor.device for tensor in tensors if tensor is not None}
    if not devices or any(device.type != "npu" for device in devices):
        raise RuntimeError(
            "NPU block-FP8 operands must all be resident on an Ascend NPU."
        )
    if len(devices) != 1:
        raise RuntimeError(
            f"NPU block-FP8 operands must share one device, got {sorted(map(str, devices))}."
        )


def relayout_npu_block_fp8_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: Sequence[int],
    *,
    before_a5: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert checkpoint ``[..., N, K]`` block-FP8 tensors to NPU ``[..., K, N]``.

    Atlas A2/A3 soft-FP8 kernels consume the FP8 payload as raw bytes, while A5
    consumes the native ``float8_e4m3fn`` dtype. Scales remain FP32 and follow
    the same last-two-dimension transpose as the weight block grid.
    """
    if tuple(block_size) != BLOCK_FP8_SIZE:
        raise ValueError(
            "Ascend block-FP8 only supports weight_block_size=[128, 128], "
            f"got {list(block_size)}."
        )
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError(
            "Serialized Ascend block-FP8 weights must use "
            f"torch.float8_e4m3fn, got {weight.dtype}."
        )
    if weight_scale.dtype != torch.float32:
        raise TypeError(
            f"Ascend block-FP8 weight scales must be float32, got {weight_scale.dtype}."
        )
    if weight.ndim not in (2, 3) or weight_scale.ndim != weight.ndim:
        raise ValueError(
            "Ascend block-FP8 expects dense [N, K] or expert [E, N, K] "
            f"weights and matching-rank scales, got {weight.shape} and "
            f"{weight_scale.shape}."
        )

    n_dim, k_dim = weight.shape[-2:]
    block_n, block_k = BLOCK_FP8_SIZE
    if n_dim % block_n or k_dim % block_k:
        raise ValueError(
            "Ascend block-FP8 weight dimensions must both be divisible by 128, "
            f"got N={n_dim}, K={k_dim}."
        )
    expected_scale_shape = (*weight.shape[:-2], n_dim // block_n, k_dim // block_k)
    if tuple(weight_scale.shape) != expected_scale_shape:
        raise ValueError(
            "Ascend block-FP8 checkpoint scale shape mismatch: expected "
            f"{expected_scale_shape} for weight {tuple(weight.shape)}, got "
            f"{tuple(weight_scale.shape)}."
        )

    if before_a5:
        weight = weight.view(torch.uint8)
    return (
        weight.transpose(-1, -2).contiguous(),
        weight_scale.transpose(-1, -2).contiguous(),
    )


def _validate_npu_block_fp8_runtime(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor],
    *,
    before_a5: bool,
    expert: bool,
) -> Tuple[int, int, int]:
    expected_rank = 3 if expert else 2
    if weight.ndim != expected_rank or weight_scale.ndim != expected_rank:
        kind = "expert [E, K, N]" if expert else "dense [K, N]"
        raise ValueError(
            f"Ascend block-FP8 expects {kind} weights and matching-rank scales, "
            f"got {weight.shape} and {weight_scale.shape}."
        )
    if input.ndim < 2 or (expert and input.ndim != 2):
        raise ValueError(
            "Ascend block-FP8 input must be at least 2D for dense GEMM and "
            f"exactly 2D for grouped GEMM, got {input.shape}."
        )

    k_dim, n_dim = weight.shape[-2:]
    if input.shape[-1] != k_dim:
        raise ValueError(
            f"Ascend block-FP8 K mismatch: input K={input.shape[-1]}, weight K={k_dim}."
        )
    if k_dim % 128 or n_dim % 128:
        raise ValueError(
            "Ascend block-FP8 runtime dimensions must both be divisible by 128, "
            f"got K={k_dim}, N={n_dim}."
        )
    expected_scale_shape = (*weight.shape[:-2], k_dim // 128, n_dim // 128)
    if tuple(weight_scale.shape) != expected_scale_shape:
        raise ValueError(
            "Ascend block-FP8 runtime scale shape mismatch: expected "
            f"{expected_scale_shape}, got {tuple(weight_scale.shape)}."
        )
    if weight_scale.dtype != torch.float32:
        raise TypeError(
            f"Ascend block-FP8 weight scales must be float32, got {weight_scale.dtype}."
        )

    expected_weight_dtype = torch.uint8 if before_a5 else torch.float8_e4m3fn
    if weight.dtype != expected_weight_dtype:
        generation = "pre-A5" if before_a5 else "A5"
        raise TypeError(
            f"Ascend {generation} block-FP8 weight payload must use "
            f"{expected_weight_dtype}, got {weight.dtype}."
        )

    rows = input.numel() // input.shape[-1]
    if input_scale is None:
        supported_input_dtypes = (
            (torch.float16, torch.bfloat16)
            if expert and not before_a5
            else (torch.bfloat16,)
        )
        if input.dtype not in supported_input_dtypes:
            raise TypeError(
                "Unquantized Ascend block-FP8 activation dtype is unsupported: "
                f"expected one of {supported_input_dtypes}, got {input.dtype}."
            )
    else:
        if before_a5:
            raise ValueError("Pre-A5 soft-FP8 does not accept pre-quantized inputs.")
        if input.dtype != torch.float8_e4m3fn:
            raise TypeError(
                "Pre-quantized A5 block-FP8 activations must use "
                f"torch.float8_e4m3fn, got {input.dtype}."
            )
        if input_scale.dtype != torch.float32:
            raise TypeError(
                "A5 block-FP8 activation scales must be float32, "
                f"got {input_scale.dtype}."
            )
        expected_input_scale_numel = rows * (k_dim // 128)
        if input_scale.numel() != expected_input_scale_numel:
            raise ValueError(
                "A5 block-FP8 activation scale shape mismatch: expected "
                f"{(rows, k_dim // 128)} ({expected_input_scale_numel} values), "
                f"got {tuple(input_scale.shape)}."
            )
    return rows, k_dim, n_dim


def fp8_matmul_npu(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: Sequence[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run standard [128,128] block-FP8 dense GEMM on Ascend."""
    if tuple(block_size) != BLOCK_FP8_SIZE:
        raise ValueError(
            "fp8_matmul_npu only supports block_size=[128, 128], "
            f"got {list(block_size)}."
        )
    _require_same_npu_device(input, weight, weight_scale, input_scale, bias)
    before_a5 = not _npu_is_a5_for_tensor(input)
    rows, k_dim, n_dim = _validate_npu_block_fp8_runtime(
        input,
        weight,
        weight_scale,
        input_scale,
        before_a5=before_a5,
        expert=False,
    )

    if bias is not None:
        if bias.ndim != 1 or bias.numel() != n_dim or bias.dtype != torch.bfloat16:
            raise ValueError(
                "Ascend block-FP8 bias must be bfloat16 [N], "
                f"got shape={tuple(bias.shape)}, dtype={bias.dtype}."
            )

    input_shape = input.shape
    input_2d = input.reshape(rows, k_dim).contiguous()
    ops = _get_npu_ops()
    if before_a5:
        output_2d = ops.softfp8_w8a16_matmul(input_2d, weight, weight_scale, "bf16")
    else:
        if input_scale is None:
            input_2d, input_scale = ops.npu_dynamic_block_quant(
                input_2d,
                dst_type=torch.float8_e4m3fn,
                row_block_size=1,
                col_block_size=128,
            )
        else:
            input_scale = input_scale.reshape(rows, k_dim // 128).contiguous()
        output_2d = ops.npu_quant_matmul(
            input_2d,
            weight,
            scale=weight_scale,
            pertoken_scale=input_scale,
            output_dtype=torch.bfloat16,
            group_sizes=(1, 128, 128),
        )

    if bias is not None:
        output_2d = output_2d + bias
    return output_2d.reshape(*input_shape[:-1], n_dim)


def fp8_grouped_matmul_npu(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run one standard [128,128] block-FP8 expert grouped GEMM on Ascend."""
    if group_list_type not in (0, 1):
        raise ValueError(f"group_list_type must be 0 or 1, got {group_list_type}.")
    _require_same_npu_device(input, weight, weight_scale, group_list, input_scale, bias)
    before_a5 = not _npu_is_a5_for_tensor(input)
    rows, k_dim, n_dim = _validate_npu_block_fp8_runtime(
        input,
        weight,
        weight_scale,
        input_scale,
        before_a5=before_a5,
        expert=True,
    )
    supported_output_dtypes = (
        (torch.bfloat16,) if before_a5 else (torch.float16, torch.bfloat16)
    )
    if output_dtype not in supported_output_dtypes:
        raise TypeError(
            "Ascend block-FP8 grouped output dtype is unsupported: expected one "
            f"of {supported_output_dtypes}, got {output_dtype}."
        )
    num_experts = weight.shape[0]
    if group_list.ndim != 1 or group_list.numel() != num_experts:
        raise ValueError(
            "Ascend block-FP8 group_list must contain one entry per local expert, "
            f"expected {num_experts}, got {tuple(group_list.shape)}."
        )
    if bias is not None and (
        bias.shape != torch.Size((num_experts, n_dim)) or bias.dtype != output_dtype
    ):
        raise ValueError(
            "Ascend block-FP8 grouped bias must be [E, N] in the output dtype, "
            f"got shape={tuple(bias.shape)}, dtype={bias.dtype}."
        )

    ops = _get_npu_ops()
    group_list = group_list.to(torch.int64)
    if before_a5:
        if bias is not None:
            raise ValueError("Pre-A5 soft-FP8 grouped GEMM does not support bias.")
        if group_list_type == 1:
            group_list = group_list.cumsum(dim=0)
        return ops.softfp8_w8a16_grouped_matmul(
            input, weight, weight_scale, group_list, "bf16"
        )

    if input_scale is None:
        input, input_scale = ops.npu_dynamic_block_quant(
            input,
            dst_type=torch.float8_e4m3fn,
            row_block_size=1,
            col_block_size=128,
        )
    else:
        input_scale = input_scale.reshape(rows, k_dim // 128).contiguous()

    scale_args = {
        "scale": [weight_scale],
        "per_token_scale": [input_scale],
    }
    if bias is not None:
        scale_args["bias"] = [bias]
    return ops.npu_grouped_matmul(
        x=[input],
        weight=[weight],
        **scale_args,
        split_item=2,
        group_type=0,
        group_list=group_list,
        group_list_type=group_list_type,
        output_dtype=output_dtype,
    )[0]


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

    def __init__(
        self,
        quant_config=None,
        *,
        preserve_mlaprolog_source: bool = False,
    ) -> None:
        self.quant_config = quant_config
        self.preserve_mlaprolog_source = preserve_mlaprolog_source

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
            weight_scale_param = layer._parameters.get("weight_scale")
            if weight_scale_param is None:
                raise RuntimeError(
                    "weight_scale is required for ModelSlim MXFP8 linear; "
                    "unit-scale fallback is not allowed."
                )
            weight_scale = weight_scale_param.data
            if (weight_scale == MXFP_E8M0_NOT_LOADED).any():
                raise RuntimeError(
                    "ModelSlim MXFP8 weight_scale was not fully loaded "
                    "(found the UE8M0 NaN sentinel 0xFF)."
                )
            if self.preserve_mlaprolog_source:
                # MLAProlog consumes the checkpoint's [out, in] payload and
                # flat [out, in/32] scale, while the ordinary linear path below
                # consumes transposed views.  Keep plain Tensor views so both
                # paths share storage; registering cloned Parameters here would
                # duplicate several giant attention projections per layer.
                layer.mlaprolog_weight_source = weight
                layer.mlaprolog_weight_scale_source = weight_scale
            # Offline (ModelSlim) path: weight is already MXFP8-quantised and
            # layer.weight_scale holds the uint8 block scales [out, in/32]. Only
            # re-layout to [in, out] / [in//64, out, 2] strided views below.
            n_dim, k_dim = weight_scale.shape
            scale = weight_scale.reshape(n_dim, k_dim // 2, 2)
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
        if envs.SGLANG_NPU_W4A4_NEW_PACKING.get():
            layer.weight.data = layer.weight.data.view(torch.int32).contiguous()
        else:
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

        # BF16 -> packed FP4 (float4_e2m1fn_x2, [out, in//2]) + UE8M0 block scale.
        # npu_dynamic_mx_quant returns the scale as [out, in//64, 2] (3D); older
        # builds may return [out, in//32] (2D) — handle both before the transpose.
        qw, w_scale = torch.ops.npu.npu_dynamic_mx_quant(
            weight_fp, dst_type=fp4_dtype, round_mode="round"
        )

        # weight: packed FP4 -> FRACTAL_NZ (float8_e4m3fn view) -> transpose
        # [in//2, out]. Mirror the offline path (no .contiguous() on the NZ view);
        # view as uint8 first because npu_format_cast only accepts int-dtype tensors.
        qw_nz = npu_format_cast(
            qw.view(torch.uint8),
            NPUACLFormat.ACL_FORMAT_FRACTAL_NZ,
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=fp4_dtype,
        )
        layer.weight = Parameter(qw_nz.transpose(-1, -2), requires_grad=False)

        # weight_scale -> [in//64, out, 2] to match npu_quant_matmul.
        if w_scale.dim() == 2:
            n, k = w_scale.shape
            w_scale = w_scale.reshape(n, k // 2, 2)
        layer.weight_scale = Parameter(w_scale.transpose(-3, -2), requires_grad=False)

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


class NPUSingleLevelMXFP4LinearMethod(_NPULinearMethodBase):
    """Ascend NPU W4A4 online quantization: single-level MXFP4.

    True W4(weight) A4(activation): both weights and activations are quantised to
    single-level MXFP4 (``float4_e2m1fn_x2``), unlike the W4A8 path which keeps FP8
    activations. All NPU ops go through ``torch.ops.npu.*`` (no top-level
    ``torch_npu``) and the fp4 dtype comes from ``_get_float4_e2m1fn_x2_dtype()``.

    Weight quantization (process_weights_after_loading):
        BF16/FP16 weight → npu_dynamic_mx_quant(dst=float4_e2m1fn_x2)
        → (packed FP4 [out, in//2], UE8M0 block scale) → transpose [in//2, out]

    Inference (apply):
        BF16/FP16 activation → npu_dynamic_mx_quant(dst=float4_e2m1fn_x2)  (A4)
        → npu_quant_matmul(x1_dtype = x2_dtype = float4_e2m1fn_x2,
                           group_sizes=[1, 1, MXFP4_BLOCK_SIZE])

    Triggered by ``--quantization mxfp4`` on Ascend NPU. Hardware: Ascend 950 (A5)
    with a recent torch_npu exposing ``float4_e2m1fn_x2``.
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

        The checkpoint still holds full-precision BF16/FP16 weights: the loader
        fills this buffer, then ``process_weights_after_loading`` quantizes it to
        MXFP4 in place.
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
        # Online single-level MXFP4 weight quant. All NPU ops go through
        # torch.ops.npu.* (no torch_npu); the fp4 dtype comes from the shared
        # _get_float4_e2m1fn_x2_dtype() helper (the torch_npu int enum).
        fp4_dtype = _get_float4_e2m1fn_x2_dtype()

        weight_fp = layer.weight.data
        if weight_fp.dtype not in (torch.float16, torch.bfloat16):
            weight_fp = weight_fp.to(torch.bfloat16)
        # Move to NPU if needed (cpu offload may have put it on CPU).
        if not weight_fp.is_npu:
            weight_fp = weight_fp.to(f"npu:{torch.npu.current_device()}")

        # BF16 -> packed FP4 (float4_e2m1fn_x2, [out, in//2]) + UE8M0 block scale.
        qw, w_scale = torch.ops.npu.npu_dynamic_mx_quant(
            weight_fp, dst_type=fp4_dtype, round_mode="round"
        )
        # Pre-transpose the weight to [in//2, out] for npu_quant_matmul; use
        # .data= to preserve the non-contiguous transpose view (npu_quant_matmul
        # reads strides directly — .contiguous() would reorder data and break
        # block-scale alignment).
        layer.weight = Parameter(qw, requires_grad=False)
        layer.weight.data = layer.weight.data.transpose(0, 1)

        # weight_scale -> [in//64, out, 2] (3D), matching the offline W4A4 path,
        # the W4A8 path and vllm-ascend's W4A4_MXFP4 layout. npu_dynamic_mx_quant
        # already returns the scale as [out, in//64, 2] (3D) on current builds;
        # older builds may return [out, in//32] (2D) — reshape those first so the
        # transpose always yields the 3D layout npu_quant_matmul requires.
        if w_scale.dim() == 2:
            n, k = w_scale.shape
            w_scale = w_scale.reshape(n, k // 2, 2)
        layer.weight_scale = Parameter(w_scale.transpose(-3, -2), requires_grad=False)

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

        # Dynamic single-level MXFP4 activation quantisation (A4 — FP4).
        qx, input_scale = torch.ops.npu.npu_dynamic_mx_quant(
            x_2d, dst_type=fp4_dtype, round_mode="round"
        )

        # Single-level MXFP4 matmul (weight & scale already transposed at load
        # time): x1_dtype = x2_dtype = fp4, group_sizes=[1, 1, block].
        output = torch.ops.npu.npu_quant_matmul(
            qx,
            layer.weight,
            layer.weight_scale,
            scale_dtype=e8m0_dtype,
            pertoken_scale=input_scale,
            pertoken_scale_dtype=e8m0_dtype,
            bias=bias.to(torch.float32) if bias is not None else None,
            output_dtype=original_dtype,
            x1_dtype=fp4_dtype,
            x2_dtype=fp4_dtype,
            group_sizes=[1, 1, MXFP4_BLOCK_SIZE],
        )

        # Restore original shape (replace last dim with output features).
        output_shape = list(input_shape[:-1]) + [output.shape[-1]]
        return output.reshape(output_shape)


class NPUSingleLevelMXFP4OfflineLinearMethod(NPUSingleLevelMXFP4LinearMethod):
    """Ascend NPU offline W4A4 (ModelSlim ``W4A4_MXFP4``): packed FP4 weights.

    Kernel for the offline ``ModelSlimMXFP4Scheme`` (delegated as ``self.kernel``).
    The msmodelslim ``W4A4_MXFP4`` checkpoint stores weights as packed ``uint8``
    [out, in//2] (two FP4 values per byte) plus UE8M0 block scales (``uint8``
    [out, in//32]). The weight is transposed and the scale reshaped to 3D; it then
    shares the online :class:`NPUSingleLevelMXFP4LinearMethod` matmul (``apply``)
    exactly — only the weight source differs (msmodelslim checkpoint vs online RTN).
    Mirrors vllm-ascend's single-level W4A4 MXFP4 layout.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        if not weight.is_npu:
            weight = weight.to(f"npu:{torch.npu.current_device()}")
        # The checkpoint is already packed two-FP4-per-byte. Preserve the strided
        # transpose used by vllm-ascend and by the online path.
        layer.weight = Parameter(weight.transpose(0, 1), requires_grad=False)

        weight_scale = layer.weight_scale.data
        if not weight_scale.is_npu:
            weight_scale = weight_scale.to(f"npu:{torch.npu.current_device()}")
        # npu_quant_matmul with float4_e2m1fn_x2 requires x2Scale to be 3D:
        # [out, in/32] -> [out, in/64, 2] -> transpose to [in/64, out, 2].
        n_dim, k_dim = weight_scale.shape
        layer.weight_scale = Parameter(
            weight_scale.reshape(n_dim, k_dim // 2, 2).transpose(0, 1),
            requires_grad=False,
        )


class NPUDualLevelMXFP4LinearMethod(NPUSingleLevelMXFP4LinearMethod):
    """Ascend NPU W4A4 online quantization: dual-level MXFP4 (higher accuracy).

    This is the sole online ``--quantization mxfp4`` linear path. Instead of a single
    UE8M0 (power-of-2) block scale, dual-level MX quant produces a finer L0 (FP8 E4M3)
    block scale plus a coarser L1 scale, so per-block dynamic range is captured far
    more accurately — this fixed the online-RTN degradation that made single-level
    decoding loop (never emitting EOS) under greedy sampling. (The single-level
    :class:`NPUSingleLevelMXFP4LinearMethod` is retained only as the offline path's
    base — msmodelslim checkpoints ship single-level UE8M0 scales.)

    All NPU ops go through ``torch.ops.npu.*`` (no top-level ``torch_npu``). Only
    ``create_weights`` (the BF16/FP16 placeholder) is shared with the single-level
    base; weight post-processing and the matmul are fully dual-level.

    Weight quantization (process_weights_after_loading):
        BF16/FP16 weight → npu_dynamic_dual_level_mx_quant
        → (packed FP4 weight, L0 scale, L1 scale); weight cast to FRACTAL_NZ,
          L0 scale transposed to [in//l0_block, out].

    Inference (apply):
        BF16/FP16 activation → npu_dynamic_dual_level_mx_quant  (A4, dual-level)
        → npu_dual_level_quant_matmul(act, weight, act_l0, w_l0, act_l1, w_l1)

    Reference: Diffusion ``NPUMXFP4DiffusionLinearMethod`` / MindIE-SD
    ``W4A4MXFP4DualQuantLinear``. Hardware: Ascend 950 (A5) only — the
    ``DualLevelQuantBatchMatmul`` op is unavailable on A2/A3.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_fp = layer.weight.data
        if weight_fp.dtype not in (torch.float16, torch.bfloat16):
            weight_fp = weight_fp.to(torch.bfloat16)
        # Move to NPU if needed (cpu offload may have put it on CPU).
        if not weight_fp.is_npu:
            weight_fp = weight_fp.to(f"npu:{torch.npu.current_device()}")

        # Dual-level MXFP4 weight quant: packed FP4 weight + L0 (fine, FP8 E4M3)
        # and L1 (coarse) block scales.
        qw, w_l0_scale, w_l1_scale = torch.ops.npu.npu_dynamic_dual_level_mx_quant(
            weight_fp, smooth_scale=None
        )

        # npu_dual_level_quant_matmul requires the weight (x2) in FRACTAL_NZ.
        # View the packed FP4 as int8 first (npu_format_cast takes int dtypes).
        qw_nz = npu_format_cast(
            qw.view(torch.int8),
            NPUACLFormat.ACL_FORMAT_FRACTAL_NZ,
            customize_dtype=torch.int8,
        )

        # L0 scale -> [in//l0_block, out] (op returns [out, in//l0_block, 1]).
        w_l0_scale = w_l0_scale.squeeze(-1).transpose(0, 1).contiguous()

        layer.weight = Parameter(qw_nz, requires_grad=False)
        layer.weight_l0_scale = Parameter(w_l0_scale, requires_grad=False)
        layer.weight_l1_scale = Parameter(w_l1_scale, requires_grad=False)

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

        # Flatten to 2D [tokens, hidden] for the quant operators.
        input_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        # Dynamic dual-level MXFP4 activation quant (A4): packed FP4 + L0/L1 scales.
        qx, act_l0_scale, act_l1_scale = torch.ops.npu.npu_dynamic_dual_level_mx_quant(
            x_2d, smooth_scale=None
        )

        # Dual-level matmul. Arg order (act, weight, act_l0, w_l0, act_l1, w_l1);
        # the weight is NOT transposed here (unlike the single-level path).
        output = torch.ops.npu.npu_dual_level_quant_matmul(
            qx,
            layer.weight,
            act_l0_scale,
            layer.weight_l0_scale,
            act_l1_scale,
            layer.weight_l1_scale,
            bias=bias.to(torch.float32) if bias is not None else None,
            output_dtype=original_dtype,
        )

        # Restore original shape (replace last dim with output features).
        output_shape = list(input_shape[:-1]) + [output.shape[-1]]
        return output.reshape(output_shape)
