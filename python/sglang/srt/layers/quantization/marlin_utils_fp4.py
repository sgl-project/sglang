from __future__ import annotations

import torch

from sglang.srt.layers.quantization.marlin_utils import (
    USE_FP32_REDUCE_DEFAULT,
    marlin_make_workspace,
    marlin_permute_bias,
    marlin_permute_scales,
    should_use_atomic_add_reduce,
)
from sglang.srt.layers.quantization.utils import get_scalar_types
from sglang.srt.utils import is_cuda
from sglang.srt.utils.custom_op import register_custom_op

_is_cuda = is_cuda()

if _is_cuda:
    from sglang.jit_kernel.gptq_marlin import gptq_marlin_gemm
    from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack

ScalarType, scalar_types = get_scalar_types()


def nvfp4_marlin_process_scales(marlin_scales: torch.Tensor) -> torch.Tensor:
    if not (marlin_scales >= 0).all():
        # NVFP4 ModelOpt scales are expected to be non-negative. Keep this as
        # a warning so unusual checkpoints can still load for diagnosis.
        import logging

        logging.getLogger(__name__).warning_once(
            "NVFP4 Marlin assumes non-negative scales, but negative scales "
            "were found. Accuracy may be degraded."
        )

    marlin_scales = marlin_scales.to(torch.half)
    marlin_scales = marlin_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(
        marlin_scales.size(0), -1
    )
    marlin_scales = (marlin_scales * (2**7)).view(torch.int16) << 1
    marlin_scales = marlin_scales.view(torch.float8_e4m3fn)
    return marlin_scales[:, 1::2].contiguous()


def nvfp4_marlin_process_global_scale(global_scale: torch.Tensor) -> torch.Tensor:
    assert global_scale.dtype in [torch.half, torch.bfloat16]
    global_scale_shape = global_scale.shape
    fp4_exponent = 2
    if global_scale.dtype == torch.half:
        target_exponent = 5
    elif global_scale.dtype == torch.bfloat16:
        target_exponent = 8
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp4_exponent - 1)
    global_scale = global_scale * (2.0 ** (exponent_bias - 7))
    if global_scale_shape == torch.Size([]):
        global_scale = global_scale.reshape(1)
    return global_scale


def fake_apply_fp4_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: torch.Tensor | None = None,
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    del weight, weight_scale, weight_global_scale, workspace, size_k, bias
    out_shape = input.shape[:-1] + (size_n,)
    return input.new_empty(out_shape)


@register_custom_op(fake_impl=fake_apply_fp4_marlin_linear)
def apply_fp4_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: torch.Tensor | None = None,
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("NVFP4 Marlin requires FP16 or BF16 activations.")

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
        global_scale=weight_global_scale,
        b_zeros=None,
        g_idx=None,
        perm=None,
        workspace=workspace,
        b_q_type=scalar_types.float4_e2m1f,
        size_m=reshaped_x.size(0),
        size_n=size_n,
        size_k=size_k,
        is_k_full=True,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
    )

    if bias is not None:
        output.add_(bias)

    return output.reshape(out_shape)


def prepare_nvfp4_layer_for_marlin(layer: torch.nn.Module) -> None:
    if getattr(layer, "quant_config", None) is not None:
        group_size = layer.quant_config.group_size
        if group_size != 16:
            raise ValueError(f"NVFP4 Marlin requires group_size=16, got {group_size}.")

    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition
    param_dtype = getattr(layer, "params_dtype", getattr(layer, "orig_dtype", None))
    if param_dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("NVFP4 Marlin requires FP16 or BF16 activation dtype.")

    assert layer.weight.shape == (part_size_n, part_size_k // 2)

    if part_size_n % 64 != 0:
        raise ValueError(
            f"NVFP4 Marlin requires output_size_per_partition to be a multiple of 64, "
            f"got {part_size_n}."
        )

    device = layer.weight.device
    layer.workspace = marlin_make_workspace(device)

    perm = torch.empty(0, dtype=torch.int, device=device)
    qweight = layer.weight.view(torch.int32).T.contiguous()
    marlin_qweight = gptq_marlin_repack(
        b_q_weight=qweight,
        perm=perm,
        size_k=part_size_k,
        size_n=part_size_n,
        num_bits=4,
    )
    layer.weight = torch.nn.Parameter(marlin_qweight, requires_grad=False)

    weight_scale = layer.weight_scale.T.contiguous().to(param_dtype)
    weight_scale = marlin_permute_scales(
        s=weight_scale,
        size_k=part_size_k,
        size_n=part_size_n,
        group_size=16,
    )
    weight_scale = nvfp4_marlin_process_scales(weight_scale)
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)

    weight_global_scale = layer.weight_global_scale.to(param_dtype)
    weight_global_scale = nvfp4_marlin_process_global_scale(weight_global_scale)
    layer.weight_global_scale = torch.nn.Parameter(
        weight_global_scale, requires_grad=False
    )

    if hasattr(layer, "bias") and layer.bias is not None:
        assert layer.bias.shape == (part_size_n,)
        bias = marlin_permute_bias(layer.bias)
        layer.bias = torch.nn.Parameter(bias, requires_grad=False)


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


def _discard_mxfp4_marlin_repack_inputs(layer: torch.nn.Module) -> None:
    for name in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
        value = getattr(layer, name, None)
        if isinstance(value, torch.nn.Parameter):
            delattr(layer, name)


def deinterleave_moe_mxfp4_w13_for_marlin(layer: torch.nn.Module) -> None:
    """Convert GPT-OSS interleaved w13 rows to Marlin's contiguous halves.

    GPT-OSS stores gate/up rows as [gate0, up0, gate1, up1, ...]. The Marlin
    fused activation consumes [all_gate_rows, all_up_rows].
    """

    w13 = layer.w13_weight.data
    w13_scale = _get_optional_param(layer, "w13_weight_scale", "w13_weight_scale_inv")
    w13_bias = _get_optional_param(layer, "w13_weight_bias", "w13_bias")

    if w13.shape[1] % 2 != 0:
        raise ValueError(f"Expected even w13 row dimension, got {w13.shape}.")

    e, n, k = w13.shape
    layer.w13_weight.data = (
        w13.view(e, n // 2, 2, k).permute(0, 2, 1, 3).contiguous().view(e, n, k)
    )

    if w13_scale is not None:
        scale = w13_scale.data
        if scale.shape[1] != n:
            raise ValueError(
                f"Expected w13 scale row dimension {n}, got {scale.shape}."
            )
        w13_scale.data = (
            scale.view(e, n // 2, 2, scale.shape[-1])
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(e, n, scale.shape[-1])
        )

    if w13_bias is not None:
        bias = w13_bias.data
        if bias.shape[1] != n:
            raise ValueError(f"Expected w13 bias row dimension {n}, got {bias.shape}.")
        w13_bias.data = bias.view(e, n // 2, 2).permute(0, 2, 1).contiguous().view(e, n)


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
    if hidden_size % 128 == 0:
        padded_intermediate_size = ((intermediate_size + 63) // 64) * 64
    else:
        if hidden_size % 64 != 0:
            raise ValueError(
                f"MXFP4 Marlin requires hidden_size to be divisible by 64, "
                f"got {hidden_size}."
            )
        padded_intermediate_size = ((intermediate_size + 127) // 128) * 128
    param_dtype = getattr(
        layer,
        "orig_dtype",
        w13_bias_data.dtype if w13_bias_data is not None else torch.bfloat16,
    )

    device = w13.device
    layer.workspace = marlin_make_workspace(device, 4)
    perm = torch.empty(0, dtype=torch.int, device=device)

    def _pad_w13(x: torch.Tensor) -> torch.Tensor:
        if padded_intermediate_size == intermediate_size:
            return x
        x = x.view(num_experts, 2, intermediate_size, x.shape[-1])
        x = torch.nn.functional.pad(
            x, (0, 0, 0, padded_intermediate_size - intermediate_size)
        )
        return x.reshape(num_experts, 2 * padded_intermediate_size, -1)

    def _pad_w2(x: torch.Tensor, packing: int) -> torch.Tensor:
        if padded_intermediate_size == intermediate_size:
            return x
        return torch.nn.functional.pad(
            x, (0, (padded_intermediate_size - intermediate_size) // packing)
        )

    w13 = _pad_w13(w13)
    w2 = _pad_w2(w2, packing=2)
    w13_scale_data = _pad_w13(_normalize_scale_tensor(w13_scale_data, param_dtype))
    w2_scale_data = _pad_w2(
        _normalize_scale_tensor(w2_scale_data, param_dtype),
        packing=group_size,
    )
    if w13_bias_data is not None:
        w13_bias_data = _pad_w13(w13_bias_data.unsqueeze(-1)).squeeze(-1)

    def _repack_weight(weight: torch.Tensor, is_w13: bool) -> torch.Tensor:
        if is_w13:
            size_n, size_k = padded_intermediate_size * 2, hidden_size
        else:
            size_n, size_k = hidden_size, padded_intermediate_size
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
        if is_w13:
            size_n, size_k = padded_intermediate_size * 2, hidden_size
        else:
            size_n, size_k = hidden_size, padded_intermediate_size

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

    _discard_mxfp4_marlin_repack_inputs(layer)


def prepare_moe_nvfp4_layer_for_marlin(layer: torch.nn.Module) -> None:
    if layer.quant_config.group_size != 16:
        raise ValueError(
            f"NVFP4 Marlin MoE requires group_size=16, got {layer.quant_config.group_size}."
        )

    w13 = layer.w13_weight.data
    w2 = layer.w2_weight.data
    w13_scale = layer.w13_weight_scale.data
    w2_scale = layer.w2_weight_scale.data
    w13_global_scale = layer.w13_weight_scale_2.data
    w2_global_scale = layer.w2_weight_scale_2.data
    w13_bias = getattr(layer, "w13_bias", None)
    w2_bias = getattr(layer, "w2_bias", None)

    num_experts = w13.shape[0]
    num_shards = 2 if layer.moe_runner_config.is_gated else 1
    intermediate_size = layer.intermediate_size_per_partition
    hidden_size = w13.shape[2] * 2
    param_dtype = layer.params_dtype
    if param_dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("NVFP4 Marlin MoE requires FP16 or BF16 activations.")

    device = w13.device
    layer.workspace = marlin_make_workspace(device, 4)
    perm = torch.empty(0, dtype=torch.int, device=device)

    if not layer.moe_runner_config.is_gated:
        padded_intermediate_size = ((intermediate_size + 127) // 128) * 128
        intermediate_size_pad = padded_intermediate_size - intermediate_size
        if intermediate_size_pad:
            w13 = torch.nn.functional.pad(w13, (0, 0, 0, intermediate_size_pad))
            w13_scale = torch.nn.functional.pad(
                w13_scale, (0, 0, 0, intermediate_size_pad)
            )
            w2 = torch.nn.functional.pad(w2, (0, intermediate_size_pad // 2, 0, 0))
            w2_scale = torch.nn.functional.pad(
                w2_scale, (0, intermediate_size_pad // 16)
            )
            if w13_bias is not None:
                w13_bias = torch.nn.functional.pad(w13_bias, (0, intermediate_size_pad))
            intermediate_size = padded_intermediate_size

    def _repack_weight(weight: torch.Tensor, is_w13: bool) -> torch.Tensor:
        if is_w13:
            size_n, size_k = intermediate_size * num_shards, hidden_size
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
        scales = scales.to(param_dtype)
        if is_w13:
            size_n, size_k = intermediate_size * num_shards, hidden_size
        else:
            size_n, size_k = hidden_size, intermediate_size

        tensor_list = []
        for i in range(num_experts):
            scale = scales[i].T.contiguous()
            marlin_scales = marlin_permute_scales(
                s=scale,
                size_k=size_k,
                size_n=size_n,
                group_size=16,
            )
            tensor_list.append(nvfp4_marlin_process_scales(marlin_scales))
        return torch.stack(tensor_list)

    def _process_global_scale(global_scale: torch.Tensor) -> torch.Tensor:
        return nvfp4_marlin_process_global_scale(global_scale.to(param_dtype))

    def _permute_bias(bias: torch.Tensor | None) -> torch.Tensor | None:
        if bias is None:
            return None
        tensor_list = []
        for i in range(num_experts):
            tensor_list.append(marlin_permute_bias(bias[i].to(param_dtype)))
        return torch.stack(tensor_list)

    layer.w13_weight = torch.nn.Parameter(
        _repack_weight(w13, True), requires_grad=False
    )
    layer.w2_weight = torch.nn.Parameter(_repack_weight(w2, False), requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(
        _permute_scales(w13_scale, True), requires_grad=False
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        _permute_scales(w2_scale, False), requires_grad=False
    )
    layer.w13_weight_scale_2 = torch.nn.Parameter(
        _process_global_scale(w13_global_scale), requires_grad=False
    )
    layer.w2_weight_scale_2 = torch.nn.Parameter(
        _process_global_scale(w2_global_scale), requires_grad=False
    )

    if w13_bias is not None:
        layer.w13_bias = torch.nn.Parameter(
            _permute_bias(w13_bias), requires_grad=False
        )
    if w2_bias is not None:
        layer.w2_bias = torch.nn.Parameter(_permute_bias(w2_bias), requires_grad=False)
