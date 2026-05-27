# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py

"""FP4 Marlin helpers: NVFP4 fallback (non-Blackwell) and MXFP4 weight preparation."""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.layers.quantization.fp8_utils import is_blackwell_supported
from sglang.srt.layers.quantization.marlin_utils import (
    USE_FP32_REDUCE_DEFAULT,
    marlin_make_workspace,
    marlin_permute_bias,
    marlin_permute_scales,
    should_use_atomic_add_reduce,
)
from sglang.srt.layers.quantization.utils import get_scalar_types
from sglang.srt.utils import direct_register_custom_op, get_device_capability, is_cuda
from sglang.srt.utils.common import print_info_once, print_warning_once

_is_cuda = is_cuda()
if _is_cuda:
    from sglang.jit_kernel.gptq_marlin import gptq_marlin_gemm
    from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack

ScalarType, scalar_types = get_scalar_types()

# NVFP4 always uses group_size=16
FP4_MARLIN_GROUP_SIZE = 16


def is_fp4_marlin_supported() -> bool:
    """Check if the current GPU supports FP4 Marlin fallback (CUDA SM >= 75)."""
    if not _is_cuda:
        return False
    if torch.version.hip is not None:
        return False
    major, minor = get_device_capability()
    if major is None or minor is None:
        return False
    return (major * 10 + minor) >= 75


def should_use_fp4_marlin_fallback() -> bool:
    """True if non-Blackwell GPU AND Marlin kernel available (SM >= 75)."""
    return (not is_blackwell_supported()) and is_fp4_marlin_supported()


def nvfp4_marlin_process_scales(marlin_scales: torch.Tensor) -> torch.Tensor:
    """Convert NVFP4 scales from FP8-S1E4M3 to FP8-S0E5M3 format for Marlin.

    The int16 <<1 may wrap for large scales (e.g. 448*128=57344), but the BIT
    PATTERN is preserved correctly — the kernel reads raw bytes, not int16 values.
    """
    marlin_scales = marlin_scales.to(torch.half)

    if not (marlin_scales >= 0).all():
        print_warning_once(
            "NVFP4 Marlin assumes scales >= 0, but encountered negative scales. "
            "Accuracy may be degraded. The scales are converted from FP8-S1E4M3 "
            "to a special FP8-S0E5M3 format to speed up dequantization."
        )

    marlin_scales = marlin_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(
        marlin_scales.size(0), -1
    )

    marlin_scales = (marlin_scales * (2**7)).view(torch.int16) << 1
    marlin_scales = marlin_scales.view(torch.float8_e4m3fn)
    marlin_scales = marlin_scales[:, 1::2].contiguous()

    return marlin_scales


def nvfp4_marlin_process_global_scale(global_scale: torch.Tensor) -> torch.Tensor:
    """Pre-adjust global scale with FP4/FP16/BF16 exponent bias for Marlin kernel."""
    assert global_scale.dtype in [
        torch.half,
        torch.bfloat16,
    ], f"global_scale dtype must be half or bfloat16, got {global_scale.dtype}"
    fp4_exponent = 2
    if global_scale.dtype == torch.half:
        target_exponent = 5
    elif global_scale.dtype == torch.bfloat16:
        target_exponent = 8
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp4_exponent - 1)
    return global_scale * (2.0 ** (exponent_bias - 7))


def apply_fp4_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: Optional[torch.Tensor],
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: Optional[torch.Tensor] = None,
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    """Apply FP4-quantized linear via Marlin kernel (non-Blackwell fallback)."""
    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (size_n,)

    use_atomic_add = should_use_atomic_add_reduce(
        m=reshaped_x.size(0),
        n=size_n,
        k=size_k,
        device=input.device,
        dtype=input.dtype,
    )

    output = gptq_marlin_gemm(
        a=reshaped_x,
        c=None,
        b_q_weight=weight,
        b_scales=weight_scale,
        global_scale=weight_global_scale.reshape(-1),
        b_zeros=None,
        g_idx=None,
        perm=None,
        workspace=workspace,
        b_q_type=scalar_types.float4_e2m1f,
        size_m=reshaped_x.size(0),
        size_n=size_n,
        size_k=size_k,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
    )

    if bias is not None:
        output.add_(bias)

    return output.reshape(out_shape)


def fake_apply_fp4_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: Optional[torch.Tensor],
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: Optional[torch.Tensor] = None,
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    out_shape = input.shape[:-1] + (size_n,)
    return torch.empty(out_shape, dtype=input.dtype, device=input.device)


direct_register_custom_op(
    op_name="apply_fp4_marlin_linear",
    op_func=apply_fp4_marlin_linear,
    mutates_args=[],
    fake_impl=fake_apply_fp4_marlin_linear,
)


def prepare_fp4_layer_for_marlin(
    layer: torch.nn.Module,
    weight_attr: str = "weight",
    weight_scale_attr: str = "weight_scale",
    weight_global_scale_attr: str = "weight_global_scale",
) -> None:
    """Repack NVFP4 linear layer weights into Marlin format in-place."""
    print_info_once(
        "Loading NVFP4 checkpoint via Marlin fallback (non-native FP4 GPU). "
        "Note: on non-Blackwell GPUs, INT4 AWQ checkpoints may offer "
        "comparable or better accuracy for dense models."
    )

    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition
    param_dtype = layer.params_dtype

    weight = getattr(layer, weight_attr)
    assert weight.shape == (part_size_n, part_size_k // 2), (
        f"Expected {weight_attr} shape ({part_size_n}, {part_size_k // 2}), "
        f"got {weight.shape}"
    )

    device = weight.device

    # WORKSPACE
    layer.marlin_workspace = marlin_make_workspace(device)

    # WEIGHT: repack from NVFP4 native layout to Marlin tile layout
    perm = torch.empty(0, dtype=torch.int, device=device)
    qweight = weight.data.view(torch.int32).T.contiguous()
    del weight
    marlin_qweight = gptq_marlin_repack(
        b_q_weight=qweight,
        perm=perm,
        size_k=part_size_k,
        size_n=part_size_n,
        num_bits=4,
    )
    del qweight
    setattr(layer, weight_attr, torch.nn.Parameter(marlin_qweight, requires_grad=False))

    # WEIGHT SCALES: transpose, permute, convert to FP8-S0E5M3
    weight_scale = getattr(layer, weight_scale_attr)
    weight_scale = weight_scale.data.T.contiguous().to(param_dtype)
    weight_scale = marlin_permute_scales(
        s=weight_scale,
        size_k=part_size_k,
        size_n=part_size_n,
        group_size=FP4_MARLIN_GROUP_SIZE,
    )
    weight_scale = nvfp4_marlin_process_scales(weight_scale)
    setattr(
        layer, weight_scale_attr, torch.nn.Parameter(weight_scale, requires_grad=False)
    )

    # GLOBAL SCALE: Pre-adjust exponent bias for Marlin kernel.
    weight_global_scale = getattr(layer, weight_global_scale_attr)
    weight_global_scale = weight_global_scale.to(param_dtype)
    weight_global_scale = nvfp4_marlin_process_global_scale(weight_global_scale)
    setattr(
        layer,
        weight_global_scale_attr,
        torch.nn.Parameter(weight_global_scale, requires_grad=False),
    )

    # BIAS (if present): Permute for Marlin's fast access pattern
    if hasattr(layer, "bias") and layer.bias is not None:
        assert layer.bias.shape == (part_size_n,)
        bias = marlin_permute_bias(layer.bias)
        layer.bias = torch.nn.Parameter(bias, requires_grad=False)


def prepare_moe_fp4_layer_for_marlin(layer: torch.nn.Module) -> None:
    """Repack NVFP4 MoE weights into Marlin format in-place (per-expert)."""
    print_info_once(
        "Loading NVFP4 checkpoint via Marlin fallback for MoE layers "
        "(non-native FP4 GPU). Note: on non-Blackwell GPUs, INT4 AWQ "
        "checkpoints may offer comparable or better accuracy."
    )

    e = layer.num_local_experts
    k = layer.w13_weight.shape[2] * 2  # hidden_size (packed: K//2 per uint8)
    n = layer.intermediate_size_per_partition
    param_dtype = layer.params_dtype
    num_shards = 2 if layer.moe_runner_config.is_gated else 1

    device = layer.w13_weight.device
    perm = torch.empty(0, dtype=torch.int, device=device)

    # (size_n, size_k) for each projection
    sizes = {"w13": (n * num_shards, k), "w2": (k, n)}

    # --- WEIGHT REPACKING ---
    for name in ["w13_weight", "w2_weight"]:
        prefix = name.split("_")[0]  # "w13" or "w2"
        size_n, size_k = sizes[prefix]
        weight = getattr(layer, name)

        assert weight.shape == (e, size_n, size_k // 2), (
            f"Expected {name} shape ({e}, {size_n}, {size_k // 2}), "
            f"got {weight.shape}"
        )

        repacked = []
        for i in range(e):
            qweight = weight.data[i].view(torch.int32).T.contiguous()
            repacked.append(
                gptq_marlin_repack(
                    b_q_weight=qweight,
                    perm=perm,
                    size_k=size_k,
                    size_n=size_n,
                    num_bits=4,
                )
            )

        del weight
        setattr(
            layer, name, torch.nn.Parameter(torch.stack(repacked), requires_grad=False)
        )

    # --- WEIGHT SCALE PROCESSING ---
    for prefix in ["w13", "w2"]:
        size_n, size_k = sizes[prefix]
        scales = getattr(layer, prefix + "_weight_scale").to(param_dtype)
        global_scale = getattr(layer, prefix + "_weight_scale_2").to(param_dtype)

        processed = []
        for i in range(e):
            s = marlin_permute_scales(
                s=scales.data[i].T,
                size_k=size_k,
                size_n=size_n,
                group_size=FP4_MARLIN_GROUP_SIZE,
            )
            processed.append(nvfp4_marlin_process_scales(s))

        del scales
        setattr(
            layer,
            prefix + "_weight_scale",
            torch.nn.Parameter(torch.stack(processed), requires_grad=False),
        )

        if global_scale.dim() > 1:
            global_scale = global_scale.max(dim=-1).values
        global_scale = nvfp4_marlin_process_global_scale(global_scale)
        setattr(
            layer,
            prefix + "_weight_scale_2",
            torch.nn.Parameter(global_scale, requires_grad=False),
        )


def mxfp4_marlin_process_scales(
    marlin_scales: torch.Tensor,
    input_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if input_dtype is None or input_dtype.itemsize == 2:
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


def _get_optional_param(layer: torch.nn.Module, *names: str) -> torch.Tensor | None:
    for name in names:
        value = getattr(layer, name, None)
        if value is not None:
            return value
    return None


def prepare_moe_mxfp4_layer_for_marlin(layer: torch.nn.Module) -> None:
    group_size = 32
    w13 = layer.w13_weight.data
    w2 = layer.w2_weight.data
    w13_scale = _get_optional_param(layer, "w13_weight_scale", "w13_weight_scale_inv")
    w2_scale = _get_optional_param(layer, "w2_weight_scale", "w2_weight_scale_inv")
    w13_bias = _get_optional_param(layer, "w13_weight_bias", "w13_bias")
    w2_bias = _get_optional_param(layer, "w2_weight_bias", "w2_bias")

    if w13_scale is None or w2_scale is None:
        raise ValueError("MXFP4 Marlin requires w13/w2 weight scales.")

    w13_scale_data = w13_scale.data if hasattr(w13_scale, "data") else w13_scale
    w2_scale_data = w2_scale.data if hasattr(w2_scale, "data") else w2_scale
    w13_bias_data = w13_bias.data if hasattr(w13_bias, "data") else w13_bias
    w2_bias_data = w2_bias.data if hasattr(w2_bias, "data") else w2_bias

    num_experts = w13.shape[0]
    intermediate_size = w13.shape[1] // 2
    hidden_size = w13.shape[2] * 2
    param_dtype = getattr(
        layer,
        "orig_dtype",
        w13_bias_data.dtype if w13_bias_data is not None else torch.bfloat16,
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
            qweight = weight[i].view(torch.int32).T.contiguous()
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

        if is_w13:
            size_n, size_k = intermediate_size * 2, hidden_size
        else:
            size_n, size_k = hidden_size, intermediate_size

        tensor_list = []
        for i in range(num_experts):
            scale = scales[i].T.contiguous()
            marlin_scales = marlin_permute_scales(
                s=scale,
                size_k=size_k,
                size_n=size_n,
                group_size=group_size,
            )
            tensor_list.append(
                mxfp4_marlin_process_scales(
                    marlin_scales,
                    input_dtype=param_dtype,
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
    w13_scale_marlin = _permute_scales(w13_scale_data, True)
    w2_scale_marlin = _permute_scales(w2_scale_data, False)

    layer.w13_weight = torch.nn.Parameter(w13_marlin, requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(w2_marlin, requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(w13_scale_marlin, requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(w2_scale_marlin, requires_grad=False)

    if w13_bias_data is not None:
        layer.w13_weight_bias = torch.nn.Parameter(
            _permute_bias(w13_bias_data), requires_grad=False
        )
    if w2_bias_data is not None:
        layer.w2_weight_bias = torch.nn.Parameter(
            _permute_bias(w2_bias_data), requires_grad=False
        )
