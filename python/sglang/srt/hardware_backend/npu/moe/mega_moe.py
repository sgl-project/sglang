"""Runtime adapter for the registered Ascend MegaMoE operators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from sglang.srt.runtime_context import get_buffer, get_resources

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher.ascend_ops import _MegaMoEState


_BUFFER_NAME = "ascend_ops_mega_moe"


def _get_symm_buffer(x, config, state, quant_mode, quant_dtype):
    from sgl_kernel_npu.operators.moe import get_symm_buffer_for_mega_moe

    buffers = get_buffer(_BUFFER_NAME, dict)
    topk = state.topk_ids.shape[1]
    key = (
        state.group,
        x.device,
        config.num_experts,
        state.capacity,
        topk,
        config.hidden_size,
        config.intermediate_size_per_partition,
        quant_mode,
        quant_dtype,
    )
    if key not in buffers:
        # The CANN provider looks up an initialized HCCL communicator.
        backend = state.group._get_backend(x.device)
        rank = torch.distributed.get_rank(state.group)
        if not backend.get_hccl_comm_name(rank, init_comm=True):
            raise RuntimeError("Failed to initialize the MegaMoE EP communicator")
        buffers[key] = get_symm_buffer_for_mega_moe(
            state.group,
            num_experts=config.num_experts,
            num_max_tokens_per_rank=state.capacity,
            num_topk=topk,
            hidden=config.hidden_size,
            intermediate_hidden=config.intermediate_size_per_partition,
            max_recv_token_num=0,
            dispatch_quant_mode=quant_mode,
            dispatch_quant_out_dtype=quant_dtype,
        )
    return buffers[key]


def destroy_mega_moe_buffers() -> None:
    buffers = get_resources().buffers.get(_BUFFER_NAME)
    if buffers:
        torch.npu.synchronize()
        for key in list(buffers):
            buffers[key].destroy()
            del buffers[key]


def prepare_mega_moe_arch22_weights(layer, prefix, weight, *, packed_int4=False) -> None:
    """Build per-expert NZ tensors for arch22 alongside the grouped matmul weights."""
    config = getattr(getattr(layer, "dispatcher", None), "moe_ascend_ops_config", None)
    if config is None or not config.uses_fused_op:
        return

    import torch_npu

    from sglang.srt.hardware_backend.npu.utils import NPUACLFormat, is_npu_arch35

    if is_npu_arch35():
        return

    # Slice the original ND tensor before the grouped matmul path casts the
    # entire stack to NZ. Each expert needs its own NZ storage descriptor.
    name = f"_mega_moe_{prefix}_weights"
    previous = getattr(layer, name, None)
    experts = []
    for expert in weight.unbind(0):
        expert = torch_npu.npu_format_cast(
            expert.contiguous(), int(NPUACLFormat.ACL_FORMAT_FRACTAL_NZ)
        )
        if packed_int4:
            expert = expert.view(torch.int32)
        experts.append(expert)
    if previous is None:
        setattr(layer, name, experts)
    else:
        for dest, source in zip(previous, experts, strict=True):
            dest.copy_(source)


def _arch22_expert_args(layer, prefix, int4, quantized):
    weights = getattr(layer, f"_mega_moe_{prefix}_weights")
    scale = getattr(layer, f"{prefix}_weight_scale", None)
    bias = getattr(layer, f"{prefix}_scale_bias", None)
    if bias is None:
        bias = getattr(layer, f"{prefix}_weight_bias", None)
    scales = None
    if quantized:
        if scale is None:
            raise ValueError(f"Missing {prefix} scales for quantized MegaMoE")
        channels = weights[0].shape[-1] * (8 if int4 else 1)
        scale = scale.flatten(1).contiguous()
        if scale.shape[1] != channels:
            raise ValueError(f"arch22 MegaMoE requires per-channel scales for {prefix}")
        if scale.dtype not in (torch.int64, torch.uint64):
            # CANN consumes one UINT64 per scale, with FP32 bits in the low word.
            scale = scale.float().contiguous().view(torch.int32).to(torch.int64)
            scale = scale & 0xFFFFFFFF
        if int4:
            scale = scale.view(torch.int64) | (1 << 46)
        scales = list(scale.view(torch.uint64).unbind(0))
    if int4 and bias is None:
        raise ValueError(f"arch22 MegaMoE W4A8 requires {prefix} compensation bias")
    if bias is not None and not int4:
        raise ValueError("arch22 MegaMoE supports bias only for W4A8 compensation")
    biases = None
    if bias is not None:
        biases = list(bias.float().flatten(1).contiguous().unbind(0))
    return weights, scales, biases


def _arch35_expert_args(layer, prefix):
    weight = getattr(layer, f"{prefix}_weight")
    scale = getattr(layer, f"{prefix}_weight_scale", None)
    if scale is None:
        scale = getattr(layer, f"{prefix}_weight_scale_inv", None)
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if (
        e8m0 is None
        or scale is None
        or scale.dtype not in (torch.uint8, e8m0)
        or scale.ndim != 4
        or scale.shape[-1] != 2
    ):
        raise ValueError(f"arch35 MegaMoE requires paired E8M0 scales for {prefix}")
    # The grouped matmul path stores transposed views of the original NZ
    # weights and [E, N, K/64, 2] scales. Restore the original view for CANN.
    return [weight.transpose(-1, -2)], [scale.transpose(1, 2).view(e8m0)], None


def _activation_args(config):
    clamp = config.gemm1_clamp_limit
    alpha = config.gemm1_alpha
    if config.activation == "situ":
        params = {"beta": 4.0 if alpha is None else alpha}
        if clamp is not None:
            params["linear_beta"] = clamp
        return {"activation": "situglu", "activation_params": params}
    if config.activation == "npu_swiglu_oai":
        return {
            "activation": "swigluoai",
            "activation_clamp": clamp,
            "activation_params": {"alpha": 1.702 if alpha is None else alpha, "beta": 1.0},
        }
    if config.activation != "silu":
        raise ValueError(f"Unsupported MegaMoE activation {config.activation!r}")
    if clamp is not None:
        return {"activation": "swiglustep", "activation_clamp": clamp}
    limit = config.swiglu_limit
    if limit is not None and limit > 0:
        return {"activation": "swiglu", "activation_clamp": limit}
    return {"activation": "swiglu"}


def _prepare_inputs(x, ids, weights, capacity, is_arch35):
    rows, topk = ids.shape
    ids = ids.to(torch.int32).contiguous()
    weights = weights.float().contiguous()
    active = (ids >= 0).all(dim=-1) & (weights != 0).any(dim=-1)
    # Graph padding can use -1 or repeated expert IDs. The provider requires
    # valid, distinct IDs even for rows with zero routing weights.
    dummy_ids = torch.arange(topk, device=ids.device, dtype=torch.int32)
    ids = torch.where(active[:, None], ids, dummy_ids)
    weights = torch.where(active[:, None], weights, 0)
    if is_arch35:
        x = torch.where(active[:, None], x, 0)
    padded_rows = max(rows, 1) if is_arch35 else capacity
    pad = padded_rows - rows
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
        ids = torch.cat((ids, dummy_ids.expand(pad, topk)), dim=0)
        weights = F.pad(weights, (0, 0, 0, pad))
        active = torch.cat((active, active.new_zeros((pad,))))
    # arch35 accepts the local row count and disallows x_active_mask. Keep one
    # zero-weight dummy on an empty rank so URMA can still run the collective.
    return x.contiguous(), ids, weights, None if is_arch35 else active.to(torch.int8)


def run_mega_moe(
    x: torch.Tensor, config: MoeRunnerConfig, state: _MegaMoEState
) -> torch.Tensor:
    from sgl_kernel_npu.operators.moe import mega_moe

    from sglang.srt.hardware_backend.npu.quantization.dtype_utils import (
        _get_float4_e2m1fn_x2_dtype,
    )
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
        NPUMXFP8MoEMethod,
        NPUUnquantMoEMethod,
        NPUW4A8Int8MoEMethod,
        NPUW4A8MXFP4MoEMethod,
        NPUW8A8Int8MoEMethod,
    )
    from sglang.srt.hardware_backend.npu.utils import is_npu_arch35

    layer = config.layer
    kernel = layer.w2_kernel
    is_arch35 = is_npu_arch35()
    rows = x.shape[0]
    weight_types = {}
    if is_arch35 and isinstance(kernel, (NPUMXFP8MoEMethod, NPUW4A8MXFP4MoEMethod)):
        quant_mode, quant_dtype = 4, torch.float8_e4m3fn
        w13, s13, b13 = _arch35_expert_args(layer, "w13")
        w2, s2, b2 = _arch35_expert_args(layer, "w2")
        if isinstance(kernel, NPUW4A8MXFP4MoEMethod):
            fp4 = _get_float4_e2m1fn_x2_dtype()
            if type(fp4) is not int:
                raise RuntimeError("MegaMoE W4A8 requires the torch_npu FP4 enum")
            weight_types = {"weight1_type": fp4, "weight2_type": fp4}
    elif not is_arch35 and isinstance(
        kernel, (NPUUnquantMoEMethod, NPUW4A8Int8MoEMethod, NPUW8A8Int8MoEMethod)
    ):
        quantized = not isinstance(kernel, NPUUnquantMoEMethod)
        int4 = isinstance(kernel, NPUW4A8Int8MoEMethod)
        quant_mode, quant_dtype = (2, torch.int8) if quantized else (0, None)
        w13, s13, b13 = _arch22_expert_args(layer, "w13", int4, quantized)
        w2, s2, b2 = _arch22_expert_args(layer, "w2", int4, quantized)
    else:
        raise NotImplementedError(
            f"ascend_ops fused_op does not support {type(kernel).__name__} "
            f"on {'arch35' if is_arch35 else 'arch22'}"
        )

    x, ids, weights, mask = _prepare_inputs(
        x, state.topk_ids, state.topk_weights, state.capacity, is_arch35
    )
    output, _ = mega_moe(
        x=x,
        topk_ids=ids,
        topk_weights=weights,
        l1_weights=w13,
        l2_weights=w2,
        sym_buffer=_get_symm_buffer(x, config, state, quant_mode, quant_dtype),
        l1_weights_sf=s13,
        l2_weights_sf=s2,
        l1_bias=b13,
        l2_bias=b2,
        x_active_mask=mask,
        **_activation_args(config),
        **weight_types,
    )
    return output[:rows]
