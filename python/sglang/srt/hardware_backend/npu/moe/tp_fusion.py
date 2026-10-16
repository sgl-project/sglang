"""Native TP MoE operand reuse and shared-expert eligibility."""

import ctypes
from functools import lru_cache
from types import SimpleNamespace

import torch

from sglang.srt.hardware_backend.npu.moe.contracts import (
    INT8,
    MXFP8,
    QuantizedOperand,
    linear_input_contract,
    moe_input_contract,
)


@lru_cache(maxsize=None)
def supports_npu_op(name, arguments=()):
    op = getattr(torch.ops.npu, name, None)
    if op is None or not hasattr(op, "default"):
        return False
    available = {arg.name for arg in op.default._schema.arguments}
    return set(arguments).issubset(available)


@lru_cache(maxsize=None)
def supports_cann_symbol(symbol):
    for name in (None, "libopapi.so", "libopapi_transformer.so"):
        try:
            library = ctypes.CDLL(name)
        except OSError:
            continue
        if hasattr(library, symbol) and hasattr(library, symbol + "GetWorkspaceSize"):
            return True
    return False


def supports_fused_mxfp_gmm(kernel):
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
        NPUW4A8MXFP4MoEMethod,
    )
    from sglang.srt.hardware_backend.npu.utils import is_npu_arch35

    symbol = (
        "aclnnGroupedMatmulSwigluQuantWeightNzV2"
        if isinstance(kernel, NPUW4A8MXFP4MoEMethod)
        else "aclnnGroupedMatmulSwigluQuantV2"
    )
    return (
        is_npu_arch35()
        and supports_npu_op("npu_grouped_matmul_swiglu_quant_v2")
        and supports_cann_symbol(symbol)
    )


def mxfp8_input(hidden_states: torch.Tensor) -> QuantizedOperand:
    if hidden_states.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("MXFP8 quantization requires FP16/BF16 input")
    data, scale = torch.ops.npu.npu_dynamic_mx_quant(
        hidden_states, dst_type=torch.float8_e4m3fn
    )
    return QuantizedOperand(
        data, scale, MXFP8, tuple(hidden_states.shape), hidden_states.dtype
    )


def record_quantized_operand(operand):
    stream = torch.get_device_module(operand[0].device).current_stream()
    for tensor in operand:
        tensor.record_stream(stream)


def supports_mxfp8_linear(layer) -> bool:
    return linear_input_contract(layer) == MXFP8


def supports_quantized_linear(layer) -> bool:
    return linear_input_contract(layer) is not None


def prequantize_tp_input(experts, hidden_states, shared_experts=None):
    from sglang.srt.layers.moe.token_dispatcher.ascend_tp import AscendTPDispatcher
    from sglang.srt.layers.moe.utils import DispatcherOutputDtype

    dispatcher = experts.dispatcher
    if (
        hidden_states.shape[0] == 0
        or shared_experts is None
        or hidden_states.dtype != torch.bfloat16
        or not isinstance(dispatcher, AscendTPDispatcher)
        or experts.moe_ep_size != 1
        or not supports_npu_op("npu_moe_init_routing_v2", ("scale",))
    ):
        return None
    contract = moe_input_contract(getattr(experts, "w13_kernel", None))
    if contract != linear_input_contract(shared_experts.gate_up_proj):
        return None
    if (
        contract == MXFP8
        and hidden_states.shape[-1] % 64 == 0
        and dispatcher.ascend_dispatcher_output_dtype == DispatcherOutputDtype.MXFP8
    ):
        return mxfp8_input(hidden_states)
    if (
        contract == INT8
        and dispatcher.ascend_dispatcher_output_dtype == DispatcherOutputDtype.INT8
        and supports_npu_op("npu_dynamic_quant")
    ):
        data, scale = torch.ops.npu.npu_dynamic_quant(hidden_states)
        return QuantizedOperand(data, scale, INT8, tuple(hidden_states.shape))
    return None


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
        linear_input_contract(mlp.down_proj) == INT8
        and getattr(mlp, "swiglu_limit", None) is None
        and supports_npu_op("npu_dequant_swiglu_quant")
        and supports_cann_symbol("aclnnDequantSwigluQuant")
    ):
        return "activation_quant"
    if (
        not supports_mxfp8_linear(mlp.gate_up_proj)
        or not supports_mxfp8_linear(mlp.down_proj)
        or getattr(mlp, "swiglu_limit", None) is not None
        or getattr(mlp.gate_up_proj, "bias", None) is not None
        or not is_npu_arch35()
        or not supports_npu_op("npu_grouped_matmul_swiglu_quant_v2")
        or not supports_cann_symbol("aclnnGroupedMatmulSwigluQuantV2")
    ):
        return "baseline"
    from sglang.srt.hardware_backend.npu.moe.contracts import linear_kernel
    from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
        NPUMXFP4W4A8OfflineLinearMethod,
    )

    if isinstance(linear_kernel(mlp.gate_up_proj), NPUMXFP4W4A8OfflineLinearMethod):
        if not supports_cann_symbol("aclnnGroupedMatmulSwigluQuantWeightNzV2"):
            return "baseline"
    return "grouped_fused"


def shared_gmm1(mlp, x, pre_quant_input=None):
    if shared_gmm1_mode(mlp) == "activation_quant":
        from sglang.srt.hardware_backend.npu.moe.activation import NPUSwigluQuant

        gate_up = mlp.gate_up_proj(x if pre_quant_input is None else pre_quant_input)[0]
        data, scale = NPUSwigluQuant()._apply_activation(gate_up)
        operand = QuantizedOperand(data, scale, INT8, tuple(data.shape), x.dtype)
        return mlp.down_proj(operand)[0]
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
    record_quantized_operand(operand)
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
    operand = QuantizedOperand(*out, MXFP8, (x.shape[0], out[0].shape[-1]), x.dtype)
    return mlp.down_proj(operand)[0]


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
    supported = quant_config.moe_scheme_types()
    expert_count = getattr(
        config, "n_routed_experts", getattr(config, "num_experts", 0)
    )
    if expert_count <= 0:
        return "Missing routed-expert count."
    layers = set()
    for key in shared:
        marker = ".shared_experts." if ".shared_experts." in key else ".shared_expert."
        layers.add((key.split(marker)[0], marker))
    for layer, marker in layers:
        schemes = []
        for projection in ("gate_proj", "up_proj", "down_proj"):
            scheme = quant_config.resolve_projection_scheme(
                f"{layer}{marker}{projection}"
            )
            if scheme is None:
                return "Missing shared-expert projection descriptors."
            if scheme not in supported:
                return f"No native MoE method for shared projection {projection}: {scheme}."
            if any(
                quant_config.resolve_projection_scheme(
                    f"{layer}.experts.{i}.{projection}"
                )
                != scheme
                for i in range(expert_count)
            ):
                return "Shared and routed expert quantization descriptors differ."
            schemes.append(scheme)
        if schemes[0] != schemes[1]:
            return "Gate and up projections require the same W13 scheme."

    return None


def record_tp_shared_load(layer, param, shard_id, expert_id):
    from sglang.srt.layers.moe.token_dispatcher.ascend_tp import AscendTPDispatcher

    if (
        isinstance(layer.dispatcher, AscendTPDispatcher)
        and layer.moe_ep_size == 1
        and layer.num_fused_shared_experts == 1
        and expert_id == layer.num_local_experts - 1
        and layer.quant_config is not None
        and layer.quant_config.get_name() == "modelslim"
    ):
        loaded = getattr(param, "_npu_shared_loaded_shards", set())
        loaded.add(shard_id)
        param._npu_shared_loaded_shards = loaded


def validate_tp_expert_weights(layer, *, prepared=True):
    from sglang.srt.hardware_backend.npu.moe.contracts import projection_contract
    from sglang.srt.layers.moe.token_dispatcher.ascend_tp import AscendTPDispatcher

    if (
        not isinstance(getattr(layer, "dispatcher", None), AscendTPDispatcher)
        or layer.moe_ep_size != 1
    ):
        return
    if prepared:
        layer._npu_projection_contracts = tuple(
            projection_contract(layer, prefix) for prefix in ("w13", "w2")
        )
    if not layer.num_fused_shared_experts:
        return
    for prefix in ("w13", "w2"):
        weight = getattr(layer, f"{prefix}_weight", None)
        if weight is None or weight.ndim != 3:
            raise ValueError(f"NPU appended {prefix} requires a grouped weight tensor")
        for name, tensor in layer.named_parameters(recurse=False):
            if not name.startswith(prefix + "_"):
                continue
            if tensor.ndim == 0 or tensor.shape[0] != layer.num_local_experts:
                raise ValueError(
                    f"NPU appended expert metadata {name} has invalid expert count"
                )
            if tensor.device != weight.device:
                raise ValueError(
                    f"NPU appended expert metadata {name} is on the wrong device"
                )
            if not prepared:
                needed = {"w1", "w3"} if prefix == "w13" else {"w2"}
                loaded = getattr(tensor, "_npu_shared_loaded_shards", set())
                if not needed.issubset(loaded) and not (
                    prefix == "w13" and "w13" in loaded
                ):
                    raise ValueError(
                        f"NPU shared expert {name} is missing checkpoint shards {needed - loaded}"
                    )
        if prepared:
            kernel = getattr(layer, f"{prefix}_kernel")
            contract = moe_input_contract(kernel)
            if contract is not None:
                scale = getattr(layer, f"{prefix}_weight_scale", None)
                if scale is None or scale.numel() == 0:
                    raise ValueError(
                        f"NPU appended {prefix} is missing quantization scales"
                    )
                if contract.block_size and scale.element_size() != 1:
                    raise ValueError(f"NPU appended {prefix} requires E8M0 byte scales")
                if contract.block_size and (scale.ndim != 4 or scale.shape[-1] != 2):
                    raise ValueError(
                        f"NPU appended {prefix} requires paired block scales"
                    )
            if contract is not None and contract.kind in ("int4", "int8"):
                from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
                    NPUW8A8Int8MoEMethod,
                )

                expected = (
                    torch.int8
                    if isinstance(kernel, NPUW8A8Int8MoEMethod)
                    else torch.int32
                )
                if weight.dtype != expected:
                    raise ValueError(
                        f"NPU appended {prefix} requires prepared {expected} weights"
                    )
