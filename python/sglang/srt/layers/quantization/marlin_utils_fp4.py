from __future__ import annotations

import torch

from sglang.srt.layers.quantization.marlin_utils import (
    marlin_make_workspace,
    marlin_permute_bias,
    marlin_permute_scales,
)
from sglang.srt.utils import is_cuda
from sglang.srt.utils.common import get_bool_env_var

_is_cuda = is_cuda()
_INVERT_MXFP4_MARLIN_SCALES = get_bool_env_var("SGLANG_MXFP4_MARLIN_INVERT_SCALE")
_SKIP_MXFP4_MARLIN_SCALE_TRANSPOSE = get_bool_env_var(
    "SGLANG_MXFP4_MARLIN_SKIP_SCALE_TRANSPOSE"
)
_SKIP_MXFP4_MARLIN_SCALE_SWIZZLE = get_bool_env_var(
    "SGLANG_MXFP4_MARLIN_SKIP_SCALE_SWIZZLE"
)
_SKIP_MXFP4_MARLIN_W13_SCALE_TRANSPOSE = get_bool_env_var(
    "SGLANG_MXFP4_MARLIN_W13_SKIP_SCALE_TRANSPOSE"
)
_SKIP_MXFP4_MARLIN_W13_SCALE_PERMUTE = get_bool_env_var(
    "SGLANG_MXFP4_MARLIN_W13_SKIP_SCALE_PERMUTE"
)
_SKIP_MXFP4_MARLIN_W13_SCALE_SWIZZLE = get_bool_env_var(
    "SGLANG_MXFP4_MARLIN_W13_SKIP_SCALE_SWIZZLE"
)
_SKIP_MXFP4_MARLIN_WEIGHT_REPACK_TRANSPOSE = get_bool_env_var(
    "SGLANG_MXFP4_MARLIN_SKIP_WEIGHT_REPACK_TRANSPOSE"
)

if _is_cuda:
    from sgl_kernel import gptq_marlin_repack


def mxfp4_marlin_process_scales(
    marlin_scales: torch.Tensor,
    input_dtype: torch.dtype | None = None,
    apply_swizzle: bool = True,
) -> torch.Tensor:
    if (
        apply_swizzle
        and not _SKIP_MXFP4_MARLIN_SCALE_SWIZZLE
        and (input_dtype is None or input_dtype.itemsize == 2)
    ):
        marlin_scales = marlin_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(
            marlin_scales.size(0), -1
        )
    marlin_scales = marlin_scales.to(torch.float8_e8m0fnu)
    if input_dtype == torch.float8_e4m3fn:
        marlin_scales = marlin_scales.view(torch.uint8)
        assert marlin_scales.max() <= 249
        # exponent_bias (fp4->fp8) = 2 ** 3 - 2 ** 1 = 6
        marlin_scales = marlin_scales + 6
        marlin_scales = marlin_scales.view(torch.float8_e8m0fnu)
    return marlin_scales


def _normalize_scale_tensor(
    scales: torch.Tensor, target_dtype: torch.dtype
) -> torch.Tensor:
    # The kernel consumes E8M0 exponents. Regardless of the placeholder dtype
    # the loader used, we want the *numerical* value 2**e in ``target_dtype``.
    # float32/bfloat16/float16 containers hold the numerical 2**e directly
    # (they were filled via a dtype-promoting copy from uint8/e8m0).
    # uint8/int8 containers hold the raw E8M0 byte and must be reinterpreted.
    if scales.dtype == torch.float8_e8m0fnu:
        return scales.to(target_dtype)
    if scales.dtype == torch.uint8:
        return scales.view(torch.float8_e8m0fnu).to(target_dtype)
    if scales.dtype == torch.int8:
        return scales.view(torch.uint8).view(torch.float8_e8m0fnu).to(target_dtype)
    if scales.dtype in (torch.float32, torch.bfloat16, torch.float16):
        return scales.to(target_dtype)
    raise TypeError(f"Unsupported MXFP4 scale dtype for Marlin: {scales.dtype}")


def prepare_moe_mxfp4_layer_for_marlin(layer: torch.nn.Module) -> None:
    group_size = 32
    w13 = layer.w13_weight.data
    w2 = layer.w2_weight.data
    w13_scale = layer.w13_weight_scale_inv.data
    w2_scale = layer.w2_weight_scale_inv.data
    w13_bias = getattr(layer, "w13_bias", None)
    w2_bias = getattr(layer, "w2_bias", None)

    num_experts = w13.shape[0]
    intermediate_size = w13.shape[1] // 2
    hidden_size = w13.shape[2] * 2
    param_dtype = getattr(
        layer,
        "orig_dtype",
        w13_bias.dtype if w13_bias is not None else torch.bfloat16,
    )

    device = w13.device
    layer.workspace = marlin_make_workspace(device, 4)
    perm = torch.empty(0, dtype=torch.int, device=device)

    def _repack_weight(weight: torch.Tensor, is_w13: bool) -> torch.Tensor:
        if is_w13:
            size_n, size_k = intermediate_size * 2, hidden_size
        else:
            size_n, size_k = hidden_size, intermediate_size
        assert weight.shape == (num_experts, size_n, size_k // 2)

        tensor_list = []
        for i in range(num_experts):
            qweight = weight[i].view(torch.int32)
            expected_packed_k = size_k // (32 // 4)
            if (
                not _SKIP_MXFP4_MARLIN_WEIGHT_REPACK_TRANSPOSE
                or qweight.size(0) != expected_packed_k
            ):
                qweight = qweight.T
            qweight = qweight.contiguous()
            marlin_qweight = gptq_marlin_repack(
                b_q_weight=qweight,
                perm=perm,
                size_k=size_k,
                size_n=size_n,
                num_bits=4,
            )
            tensor_list.append(marlin_qweight)
        return torch.stack(tensor_list)

    def _permute_scales(scales: torch.Tensor, is_w13: bool) -> torch.Tensor:
        scales = _normalize_scale_tensor(scales, param_dtype)
        if _INVERT_MXFP4_MARLIN_SCALES:
            scales = torch.reciprocal(scales)

        if is_w13:
            size_n, size_k = intermediate_size * 2, hidden_size
        else:
            size_n, size_k = hidden_size, intermediate_size

        tensor_list = []
        for i in range(num_experts):
            scale = scales[i]
            skip_transpose = _SKIP_MXFP4_MARLIN_SCALE_TRANSPOSE or (
                is_w13 and _SKIP_MXFP4_MARLIN_W13_SCALE_TRANSPOSE
            )
            if not skip_transpose:
                scale = scale.T
            scale = scale.contiguous()
            skip_permute = is_w13 and _SKIP_MXFP4_MARLIN_W13_SCALE_PERMUTE
            if skip_permute:
                marlin_scales = scale
            else:
                marlin_scales = marlin_permute_scales(
                    s=scale,
                    size_k=size_k,
                    size_n=size_n,
                    group_size=group_size,
                )
            apply_swizzle = not (is_w13 and _SKIP_MXFP4_MARLIN_W13_SCALE_SWIZZLE)
            tensor_list.append(
                mxfp4_marlin_process_scales(
                    marlin_scales,
                    input_dtype=param_dtype,
                    apply_swizzle=apply_swizzle,
                )
            )
        return torch.stack(tensor_list)

    def _permute_bias(bias: torch.Tensor | None) -> torch.Tensor | None:
        if bias is None:
            return None
        tensor_list = []
        for i in range(num_experts):
            tensor_list.append(marlin_permute_bias(bias[i].to(param_dtype)))
        return torch.stack(tensor_list)

    w13_marlin = _repack_weight(w13, True)
    w2_marlin = _repack_weight(w2, False)
    w13_scale_marlin = _permute_scales(w13_scale, True)
    w2_scale_marlin = _permute_scales(w2_scale, False)

    layer.w13_weight = torch.nn.Parameter(w13_marlin, requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(w2_marlin, requires_grad=False)
    layer.w13_weight_scale_inv = torch.nn.Parameter(w13_scale_marlin, requires_grad=False)
    layer.w2_weight_scale_inv = torch.nn.Parameter(w2_scale_marlin, requires_grad=False)

    if w13_bias is not None:
        layer.w13_bias = torch.nn.Parameter(_permute_bias(w13_bias), requires_grad=False)
    if w2_bias is not None:
        layer.w2_bias = torch.nn.Parameter(_permute_bias(w2_bias), requires_grad=False)
