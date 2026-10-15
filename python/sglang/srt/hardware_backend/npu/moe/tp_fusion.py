"""Native TP MoE operand reuse and shared-expert eligibility."""

from types import SimpleNamespace
from typing import NamedTuple

import torch


class NpuMxfp8Input(NamedTuple):
    data: torch.Tensor
    scale: torch.Tensor


def supports_npu_op(name, arguments=()):
    op = getattr(torch.ops.npu, name, None)
    if op is None or not hasattr(op, "default"):
        return False
    available = {arg.name for arg in op.default._schema.arguments}
    return set(arguments).issubset(available)


def mxfp8_input(hidden_states: torch.Tensor) -> NpuMxfp8Input:
    if hidden_states.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("MXFP8 quantization requires FP16/BF16 input")
    return NpuMxfp8Input(
        *torch.ops.npu.npu_dynamic_mx_quant(hidden_states, dst_type=torch.float8_e4m3fn)
    )


def record_mxfp8_operand(operand):
    stream = torch.get_device_module(operand[0].device).current_stream()
    for tensor in operand:
        tensor.record_stream(stream)


def supports_mxfp8_linear(layer) -> bool:
    from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
        NPUMXFP4W4A8OfflineLinearMethod,
        NPUMXFP8LinearMethod,
    )

    kernel = getattr(getattr(layer, "scheme", None), "kernel", None)
    if kernel is None:
        kernel = getattr(layer, "quant_method", None)
    return isinstance(kernel, (NPUMXFP4W4A8OfflineLinearMethod, NPUMXFP8LinearMethod))


def prequantize_tp_input(experts, hidden_states, shared_experts=None):
    from sglang.srt.layers.moe.token_dispatcher.ascend_tp import AscendTPDispatcher
    from sglang.srt.layers.moe.utils import DispatcherOutputDtype

    dispatcher = experts.dispatcher
    if (
        hidden_states.shape[0] == 0
        or shared_experts is None
        or not supports_mxfp8_linear(shared_experts.gate_up_proj)
        or hidden_states.dtype != torch.bfloat16
        or hidden_states.shape[-1] % 64
        or not isinstance(dispatcher, AscendTPDispatcher)
        or experts.moe_ep_size != 1
        or dispatcher.ascend_dispatcher_output_dtype != DispatcherOutputDtype.MXFP8
        or getattr(getattr(experts, "w13_kernel", None), "dynamic_quant_kwargs", ())
        is None
        or not supports_npu_op("npu_moe_init_routing_v2", ("scale",))
    ):
        return None
    return mxfp8_input(hidden_states)


def shared_gmm1_weight_views(weight, weight_scale):
    # CANN checks full batch strides even for the singleton expert dimension.
    return (
        weight.transpose(0, 1).unsqueeze(0).transpose(1, 2),
        weight_scale.transpose(0, 1).unsqueeze(0).transpose(1, 2),
    )


def shared_gmm1_mode(mlp=None):
    if mlp is None:
        return "baseline"
    from sglang.srt.hardware_backend.npu.utils import is_npu_arch35

    if (
        not supports_mxfp8_linear(mlp.gate_up_proj)
        or not supports_mxfp8_linear(mlp.down_proj)
        or getattr(mlp, "swiglu_limit", None) is not None
        or getattr(mlp.gate_up_proj, "bias", None) is not None
        or not is_npu_arch35()
        or not supports_npu_op("npu_grouped_matmul_swiglu_quant_v2")
    ):
        return "baseline"
    return "grouped_fused"


def shared_gmm1(mlp, x, pre_quant_input=None):
    from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
        NPUMXFP8LinearMethod,
    )
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
        NPUMXFP8MoEMethod,
        NPUW4A8MXFP4MoEMethod,
    )

    gate = mlp.gate_up_proj
    linear_kernel = getattr(getattr(gate, "scheme", None), "kernel", None)
    if linear_kernel is None:
        linear_kernel = gate.quant_method
    is_mxfp8 = isinstance(linear_kernel, NPUMXFP8LinearMethod)
    kernel = getattr(mlp, "_npu_shared_gmm1_kernel", None)
    if kernel is None:
        kernel = NPUMXFP8MoEMethod("w13") if is_mxfp8 else NPUW4A8MXFP4MoEMethod()
        mlp._npu_shared_gmm1_kernel = kernel
    scale = gate.weight_scale_inv if is_mxfp8 else gate.weight_scale
    weight, scale = shared_gmm1_weight_views(gate.weight, scale)
    quant_info = SimpleNamespace(w13_weight=weight, w13_weight_scale=scale)
    operand = mxfp8_input(x) if pre_quant_input is None else pre_quant_input
    record_mxfp8_operand(operand)
    # A singleton expert uses the same compute component as routed GMM1.
    group_key = (x.shape[0], x.device)
    cached = getattr(mlp, "_npu_shared_group_list", None)
    if cached is None or cached[0] != group_key:
        group_list = torch.full((1,), x.shape[0], dtype=torch.int64, device=x.device)
        mlp._npu_shared_group_list = (group_key, group_list)
    else:
        group_list = cached[1]
    group_list.record_stream(torch.get_device_module(x.device).current_stream())
    out = kernel.apply_fused_gmm1_swiglu(
        quant_info, operand[0], group_list, operand[1], group_list_type=1
    )
    return mlp.down_proj(NpuMxfp8Input(*out))[0]


def tp_fused_shared_expert_reason(config, quant_config):
    """Validate matching routed/shared projection contracts before loading."""
    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.layers.moe.utils import is_sbo_enabled, is_tbo_enabled
    from sglang.srt.runtime_context import get_exec, get_parallel

    parallel = get_parallel()
    shared_count = getattr(config, "n_shared_experts", None)
    if shared_count is None:
        shared_count = int(getattr(config, "shared_expert_intermediate_size", 0) > 0)
    if (
        not get_moe_a2a_backend().is_none()
        or parallel.moe_ep_size != 1
        or parallel.moe_dp_size != 1
        or parallel.attn_dp_size != 1
        or parallel.attn_cp_size != 1
        or parallel.enable_prefill_cp
        or shared_count != 1
        or is_sbo_enabled()
        or is_tbo_enabled()
        or getattr(config, "swiglu_limit", None) is not None
        or get_exec().moe.enable_eplb
        or get_exec().moe.enable_waterfill
    ):
        return "NPU shared-expert fusion requires native TP/EP1 without overlap or expert remapping."
    if (
        getattr(config, "shared_expert_intermediate_size", config.moe_intermediate_size)
        != config.moe_intermediate_size
    ):
        return "Shared and routed expert intermediate sizes differ."
    if quant_config is None:
        return None
    if quant_config.get_name() != "modelslim":
        return "NPU shared-expert fusion requires matching ModelSlim or unquantized experts."
    description = quant_config.quant_description
    shared = [
        key
        for key in description
        if isinstance(key, str)
        and (".shared_experts." in key or ".shared_expert." in key)
        and key.endswith(".weight")
    ]
    if not shared:
        return "Missing shared-expert quantization descriptors."
    supported = {
        "FLOAT",
        "W8A8",
        "W8A8_DYNAMIC",
        "W8A8_MXFP8",
        "W4A8",
        "W4A4",
        "W4A8_MXFP",
        "W4A4_MXFP4",
    }
    expert_count = getattr(
        config, "n_routed_experts", getattr(config, "num_experts", 0)
    )
    if expert_count <= 0:
        return "Missing routed-expert count."
    for key in shared:
        marker = ".shared_experts." if ".shared_experts." in key else ".shared_expert."
        layer, projection = key.split(marker)
        scheme = description[key]
        if scheme not in supported or any(
            description.get(f"{layer}.experts.{i}.{projection}") != scheme
            for i in range(expert_count)
        ):
            return "Shared and routed expert quantization descriptors differ."
        if any(
            f"{layer}{marker}{proj}.weight" not in description
            for proj in ("gate_proj", "up_proj", "down_proj")
        ):
            return "Missing shared-expert projection descriptors."
    return None
