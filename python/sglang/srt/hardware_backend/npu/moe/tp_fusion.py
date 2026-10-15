"""Opt-in A5 TP fusion contracts for MXFP8 activations and MXFP4 weights."""

import torch


def require_npu_op(name: str, arguments=()):
    op = getattr(torch.ops.npu, name, None)
    if op is None:
        raise RuntimeError(f"A5 TP fusion requires torch_npu operator {name}")
    schema = op.default._schema
    available = {arg.name for arg in schema.arguments}
    missing = set(arguments) - available
    if missing:
        raise RuntimeError(f"{name} ABI lacks {sorted(missing)}; found {schema}")
    return op


def mxfp8_input(hidden_states: torch.Tensor):
    scale = getattr(hidden_states, "_npu_mxfp8_scale", None)
    if scale is not None:
        return hidden_states, scale
    if hidden_states.dtype == torch.float8_e4m3fn:
        raise RuntimeError("MXFP8 input is missing its E8M0 scales")
    return torch.ops.npu.npu_dynamic_mx_quant(
        hidden_states, dst_type=torch.float8_e4m3fn
    )


def record_mxfp8_operand(operand):
    # Quantization can run on a different stream from either consumer.
    stream = torch.get_device_module().current_stream()
    for tensor in operand:
        tensor.record_stream(stream)


def supports_mxfp8_linear(layer) -> bool:
    from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
        NPUMXFP4W4A8OfflineLinearMethod,
        NPUMXFP8LinearMethod,
    )

    return isinstance(
        getattr(getattr(layer, "scheme", None), "kernel", layer.quant_method),
        (NPUMXFP4W4A8OfflineLinearMethod, NPUMXFP8LinearMethod),
    )


def prequantize_tp_input(experts, hidden_states):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.token_dispatcher.ascend_tp import AscendTPDispatcher
    from sglang.srt.layers.moe.utils import DispatcherOutputDtype

    dispatcher = experts.dispatcher
    if getattr(hidden_states, "_npu_mxfp8_scale", None) is not None:
        return mxfp8_input(hidden_states)
    if (
        not envs.SGLANG_NPU_TP_MOE_PREQUANT_INPUT.get()
        or hidden_states.shape[0] == 0
        or hidden_states.dtype != torch.bfloat16
        or not isinstance(dispatcher, AscendTPDispatcher)
        or experts.moe_ep_size != 1
        or dispatcher.ascend_dispatcher_output_dtype != DispatcherOutputDtype.MXFP8
    ):
        return None
    return mxfp8_input(hidden_states)


def shared_gmm1_weight_views(weight, weight_scale):
    # CANN checks full batch strides even for the singleton expert dimension.
    return (
        weight.transpose(0, 1).unsqueeze(0).transpose(1, 2),
        weight_scale.transpose(0, 1).unsqueeze(0).transpose(1, 2),
    )


def shared_gmm1_mode():
    from sglang.srt.environ import envs

    mode = envs.SGLANG_NPU_TP_MOE_SHARED_GMM1_MODE.get()
    if not mode:
        return (
            "grouped_fused" if envs.SGLANG_NPU_TP_MOE_SHARED_GMM1.get() else "baseline"
        )
    if mode not in ("baseline", "grouped_fused"):
        raise ValueError(f"Unknown SGLANG_NPU_TP_MOE_SHARED_GMM1_MODE: {mode!r}")
    return mode


def shared_gateup_quant(mlp, x, pre_quant_input=None, *, mode="grouped_fused"):
    from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
        NPUMXFP8LinearMethod,
    )
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
        _get_float4_e2m1fn_x2_dtype,
        _require_e8m0_dtype,
    )

    gate = mlp.gate_up_proj
    if not supports_mxfp8_linear(gate) or not supports_mxfp8_linear(mlp.down_proj):
        raise ValueError("Shared GMM1 fusion requires MXFP8-compatible shared linears")
    if getattr(mlp, "swiglu_limit", None) is not None:
        raise ValueError("Shared GMM1 fusion does not support a clamped activation")
    if getattr(gate, "bias", None) is not None:
        raise ValueError("Shared GMM1 fusion requires a bias-free gate/up projection")
    qx, scale = mxfp8_input(x) if pre_quant_input is None else pre_quant_input
    record_mxfp8_operand((qx, scale))
    if mode != "grouped_fused":
        raise ValueError(f"Unsupported shared gate/up mode: {mode!r}")
    kernel = getattr(getattr(gate, "scheme", None), "kernel", gate.quant_method)
    weight_dtype = (
        None
        if isinstance(kernel, NPUMXFP8LinearMethod)
        else _get_float4_e2m1fn_x2_dtype()
    )
    weight_scale = gate.weight_scale_inv if weight_dtype is None else gate.weight_scale
    grouped_weight, grouped_scale = shared_gmm1_weight_views(gate.weight, weight_scale)
    group_key = (qx.shape[0], qx.device)
    cached = getattr(mlp, "_npu_shared_group_list", None)
    if cached is None or cached[0] != group_key:
        group_list = torch.full((1,), qx.shape[0], dtype=torch.int64, device=qx.device)
        mlp._npu_shared_group_list = (group_key, group_list)
    else:
        group_list = cached[1]
    group_list.record_stream(torch.get_device_module().current_stream())
    op = require_npu_op("npu_grouped_matmul_swiglu_quant_v2")
    quantized, output_scale = op(
        x=qx,
        weight=[grouped_weight],
        weight_scale=[grouped_scale],
        x_scale=scale,
        group_list=group_list,
        dequant_mode=2,
        quant_mode=2,
        dequant_dtype=torch.float32,
        quant_dtype=torch.float8_e4m3fn,
        weight_dtype=weight_dtype,
        weight_scale_dtype=_require_e8m0_dtype(),
        x_scale_dtype=_require_e8m0_dtype(),
    )
    return quantized, output_scale


def shared_gmm1(mlp, x, pre_quant_input=None, *, mode="grouped_fused"):
    operand = shared_gateup_quant(mlp, x, pre_quant_input, mode=mode)
    return mlp.down_proj(operand)[0]
