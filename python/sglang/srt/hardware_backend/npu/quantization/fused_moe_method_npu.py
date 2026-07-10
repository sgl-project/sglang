from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from sglang.srt.hardware_backend.npu.utils import npu_format_cast
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        DeepEPLLDispatchOutput,
        DeepEPNormalDispatchOutput,
        DispatchOutput,
    )
    from sglang.srt.layers.quantization.base_config import QuantizationConfig


def npu_fused_experts_w4a4(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
):
    original_shape = hidden_states.shape
    original_dtype = hidden_states.dtype
    scale_dtype = original_dtype if original_dtype == torch.bfloat16 else torch.float32
    if len(original_shape) == 3:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    num_tokens = hidden_states.shape[0]
    num_experts = w13.shape[0]

    hidden_states, expanded_row_idx, expert_tokens, _ = (
        torch.ops.npu.npu_moe_init_routing_v2(
            hidden_states,
            topk_ids,
            active_num=num_tokens * top_k,
            expert_num=num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, num_experts],
            quant_mode=-1,
        )
    )
    expert_tokens = expert_tokens.to(torch.int64)

    # gmm1: gate_up_proj
    hidden_states, pertoken_scale = torch.ops.npu.npu_dynamic_quant(
        hidden_states, dst_type=torch.quint4x2
    )
    scale_args13 = {
        "scale": [w13_scale],
        "per_token_scale": [pertoken_scale],
    }

    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w13],
        **scale_args13,
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]
    # act_fn: swiglu
    hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
    hidden_states, pertoken_scale = torch.ops.npu.npu_dynamic_quant(hidden_states)

    scale_args2 = {
        "scale": [w2_scale.to(scale_dtype)],
        "per_token_scale": [pertoken_scale],
    }
    # gmm2: down_proj
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        **scale_args2,
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    final_hidden_states = torch.ops.npu.npu_moe_finalize_routing(
        hidden_states,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights,
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids,
        drop_pad_mode=2,
    )
    if len(original_shape) == 3:
        final_hidden_states = final_hidden_states.view(original_shape)
    return final_hidden_states


def npu_fused_experts(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    **kwargs,
):
    w13_offset = kwargs.get("w13_offset", None)
    w2_offset = kwargs.get("w2_offset", None)
    use_wna16 = kwargs.get("use_wna16", False)

    original_shape = hidden_states.shape
    original_dtype = hidden_states.dtype
    scale_dtype = original_dtype if original_dtype == torch.bfloat16 else torch.float32
    if len(original_shape) == 3:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    num_tokens = hidden_states.shape[0]
    num_experts = w13.shape[0]
    row_idx_len = num_tokens * top_k
    row_idx = (
        torch.arange(0, row_idx_len, dtype=torch.int32, device=topk_weights.device)
        .view(top_k, -1)
        .permute(1, 0)
        .contiguous()
    )
    hidden_states, expanded_row_idx, expanded_expert_idx = (
        torch.ops.npu.npu_moe_init_routing(
            hidden_states, row_idx=row_idx, expert_idx=topk_ids, active_num=num_tokens
        )
    )
    expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(
        expanded_expert_idx, num_experts
    )
    expert_tokens = expert_tokens.to(torch.int64)
    # gmm1: gate_up_proj
    if not use_wna16:
        hidden_states, pertoken_scale = torch.ops.npu.npu_dynamic_quant(hidden_states)
        scale_args13 = {
            "scale": [w13_scale.to(scale_dtype)],
            "per_token_scale": [pertoken_scale],
        }
    else:
        scale_args13 = {
            "antiquant_scale": [w13_scale],
            "antiquant_offset": [w13_offset],
        }

    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w13],
        **scale_args13,
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]
    # act_fn: swiglu
    if not use_wna16:
        hidden_states, pertoken_scale = torch.ops.npu.npu_dequant_swiglu_quant(
            hidden_states,
            activate_left=True,
            quant_mode=1,
        )

        scale_args2 = {
            "scale": [w2_scale.to(scale_dtype)],
            "per_token_scale": [pertoken_scale],
        }
    else:
        hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
        scale_args2 = {"antiquant_scale": [w2_scale], "antiquant_offset": [w2_offset]}
    # gmm2: down_proj
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        **scale_args2,
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    final_hidden_states = torch.ops.npu.npu_moe_finalize_routing(
        hidden_states,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights,
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids,
    )
    if len(original_shape) == 3:
        final_hidden_states = final_hidden_states.view(original_shape)
    return final_hidden_states


def npu_fused_experts_w8a8_decode(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    **kwargs,
):
    num_tokens = hidden_states.shape[:-1].numel()
    first_expert_idx = 0
    last_expert_idx = w13.shape[0]
    global_num_experts = w13.shape[0]
    original_shape = hidden_states.shape
    group_list_type = 1

    sorted_hidden_states, expanded_row_idx, expert_tokens, pertoken_scale = (
        torch.ops.npu.npu_moe_init_routing_v2(
            hidden_states,
            topk_ids,
            active_num=num_tokens * top_k,
            expert_num=global_num_experts,
            expert_tokens_num_type=group_list_type,
            expert_tokens_num_flag=True,
            active_expert_range=[first_expert_idx, last_expert_idx],
            quant_mode=1,
        )
    )

    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[sorted_hidden_states],
        weight=[w13],
        scale=[w13_scale],
        per_token_scale=[pertoken_scale],
        group_list=expert_tokens,
        split_item=2,
        group_type=0,
        group_list_type=group_list_type,
        output_dtype=torch.bfloat16,
    )[0]

    # act_fn: swiglu
    hidden_states, swiglu_out_scale = torch.ops.npu.npu_dequant_swiglu_quant(
        hidden_states, quant_mode=1, activate_left=True
    )

    output = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        scale=[w2_scale],
        per_token_scale=[swiglu_out_scale],
        group_list=expert_tokens,
        split_item=2,
        group_type=0,
        group_list_type=group_list_type,
        output_dtype=torch.bfloat16,
    )[0]

    assert original_shape is not None
    final_hidden_states = torch.ops.npu.npu_moe_token_unpermute(
        permuted_tokens=output,
        sorted_indices=torch.abs(expanded_row_idx),
        probs=topk_weights,
    )
    if len(original_shape) == 3:
        final_hidden_states = final_hidden_states.view(original_shape)

    return final_hidden_states


_E8M0_DTYPE = None


def _get_e8m0_dtype():
    """Lazy-resolve float8_e8m0fnu (int 293 on A5). Cached after first lookup."""
    global _E8M0_DTYPE
    if _E8M0_DTYPE is None:
        import torch_npu

        _E8M0_DTYPE = getattr(
            torch_npu,
            "float8_e8m0fnu",
            getattr(torch, "float8_e8m0fnu", None),
        )
        if _E8M0_DTYPE is None:
            raise ImportError(
                "float8_e8m0fnu dtype not found — torch_npu may be too old "
                "(MXFP8 MoE requires Ascend A5 + torch_npu >= 2.9)"
            )
    return _E8M0_DTYPE


def npu_fused_experts_mxfp8(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
):
    """TP-only MXFP8 MoE kernel: routing → act quant → gmm1(swiglu+fused) → gmm2.

    Mirrors vllm-ascend A5 MXFP8 MLP path (npu_grouped_matmul_swiglu_quant_v2
    for gmm1, plain npu_grouped_matmul with e8m0 scales for gmm2). Weights are
    kept as strided transpose views (NO .contiguous()) — the probe confirmed
    this tensor version accepts strided views with identical cos to contiguous.
    """
    _E8M0 = _get_e8m0_dtype()
    original_shape = hidden_states.shape
    original_dtype = hidden_states.dtype
    if len(original_shape) == 3:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    num_tokens = hidden_states.shape[0]
    num_experts = w13.shape[0]

    # Routing: COUNT-type expert_tokens, no in-routing quantization.
    sorted_states, expanded_row_idx, expert_tokens, _ = (
        torch.ops.npu.npu_moe_init_routing_v2(
            hidden_states,
            topk_ids,
            active_num=num_tokens * top_k,
            expert_num=num_experts,
            expert_tokens_num_type=1,  # COUNT
            expert_tokens_num_flag=True,
            active_expert_range=[0, num_experts],
            quant_mode=-1,  # activation quant done separately below
        )
    )
    expert_tokens = expert_tokens.to(torch.int64)

    # Activation: dynamic MXFP8 quant → e4m3 + e8m0 per-token scale.
    qx, x_scale = torch.ops.npu.npu_dynamic_mx_quant(
        sorted_states, dst_type=torch.float8_e4m3fn
    )

    # gmm1: gate_up + swiglu + requantise (fused single kernel).
    group_cumsum = expert_tokens.cumsum(0)
    g1_out, g1_scale = torch.ops.npu.npu_grouped_matmul_swiglu_quant_v2(
        x=qx,
        weight=[w13],
        group_list=group_cumsum,
        weight_scale=[w13_scale],
        x_scale=x_scale,
        dequant_mode=2,
        quant_mode=2,
        dequant_dtype=torch.float32,
        quant_dtype=torch.float8_e4m3fn,
        x_dtype=None,  # e4m3 is implicit — not in QUANT_DTYPES
        weight_dtype=None,
        weight_scale_dtype=_E8M0,
        x_scale_dtype=_E8M0,
    )

    # gmm2: down_proj.
    y = torch.ops.npu.npu_grouped_matmul(
        x=[g1_out],
        weight=[w2],
        scale=[w2_scale],
        bias=None,
        per_token_scale=[g1_scale],
        split_item=2,
        group_list_type=1,  # COUNT
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
        scale_dtype=_E8M0,
        per_token_scale_dtype=_E8M0,
        x_dtype=None,
        weight_dtype=None,
    )[0]

    # Finalize: unpermute back to original token order.
    final_hidden_states = torch.ops.npu.npu_moe_finalize_routing(
        y,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights,
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids,
        drop_pad_mode=2,
    )
    if len(original_shape) == 3:
        final_hidden_states = final_hidden_states.view(original_shape)
    return final_hidden_states


def npu_fused_moe_without_routing_weights_bf16(
    layer, hidden_states, group_list_type, group_list, output_dtype
):
    from sgl_kernel_npu.activation.swiglu_quant import swiglu_quant

    # gmm1: gate_up_proj
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[layer.w13_weight.transpose(1, 2)],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
        output_dtype=output_dtype,
    )[0]
    hidden_states, _ = swiglu_quant(
        hidden_states, group_list, group_list_type, need_quant=False
    )
    # gmm2: down_proj
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[layer.w2_weight.transpose(1, 2)],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
        output_dtype=output_dtype,
    )[0]
    return hidden_states


def fused_moe_npu(
    x,
    w1,
    w2,
    topk_output,
    moe_runner_config,
):
    # TODO: reuse the codes of UnquantizedFusedMoEMethod-forward_npu
    topk_weights, topk_ids, _ = topk_output
    original_dtype = x.dtype
    num_tokens = x.shape[0]
    topk_weights = topk_weights.to(x.dtype)
    topk_ids = topk_ids.to(torch.int32)
    num_experts = w1.shape[0]
    top_k = topk_weights.shape[-1]
    row_idx_len = num_tokens * top_k
    row_idx = (
        torch.arange(0, row_idx_len, dtype=torch.int32, device=topk_weights.device)
        .view(top_k, -1)
        .permute(1, 0)
        .contiguous()
    )

    hidden_states, expanded_row_idx, expanded_expert_idx = (
        torch.ops.npu.npu_moe_init_routing(
            x, row_idx=row_idx, expert_idx=topk_ids, active_num=num_tokens
        )
    )

    expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(
        expanded_expert_idx, num_experts
    )

    expert_tokens = expert_tokens.to(torch.int64)

    # gmm1: gate_up_proj
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w1.permute(0, 2, 1)],
        bias=None,
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    # act_fn:
    if moe_runner_config.activation == "silu":
        hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
    else:
        from sglang.srt.layers.activation import GeluAndMul

        hidden_states = GeluAndMul()(hidden_states)

    # gmm2: down_proj
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2.permute(0, 2, 1)],
        bias=None,
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    final_hidden_states = torch.ops.npu.npu_moe_finalize_routing(
        hidden_states,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights,
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids,
    )
    return final_hidden_states


def maybe_apply_deepep_npu(
    quant_method,
    layer: torch.nn.Module,
    dispatch_output: "DispatchOutput",
) -> Optional["CombineInput"]:
    """Route DeepEP dispatch outputs through the NPU compute path.

    Replaces the deprecated DeepEPMoE.forward_npu wrapper: detects DeepEP
    normal/LL formats, calls ``quant_method.apply_without_routing_weights``,
    and wraps the result in the matching CombineInput. Returns None for
    non-DeepEP formats so the caller falls through to its standard path.
    """
    from sglang.srt.layers.moe.token_dispatcher import (
        DeepEPLLCombineInput,
        DeepEPNormalCombineInput,
    )
    from sglang.srt.layers.moe.token_dispatcher.base import DispatchOutputChecker

    if not dispatch_output.format.is_deepep():
        return None

    # NOTE: Ascend's Dispatch & Combine does not support FP16
    output_dtype = torch.bfloat16
    group_list_type = 1

    if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
        if TYPE_CHECKING:
            assert isinstance(dispatch_output, DeepEPNormalDispatchOutput)
        (
            hidden_states,
            hidden_states_scale,
            _,
            _,
            num_recv_tokens_per_expert,
        ) = dispatch_output
        group_list = torch.tensor(
            num_recv_tokens_per_expert,
            dtype=torch.int64,
            device=hidden_states.device,
        )
        combine_cls = DeepEPNormalCombineInput
    else:
        if TYPE_CHECKING:
            assert isinstance(dispatch_output, DeepEPLLDispatchOutput)
        (
            hidden_states,
            hidden_states_scale,
            _,
            _,
            group_list,
            _,
        ) = dispatch_output
        group_list = group_list.to(torch.int64)
        combine_cls = DeepEPLLCombineInput

    hidden_states = quant_method.apply_without_routing_weights(
        layer,
        hidden_states,
        hidden_states_scale,
        group_list_type,
        group_list,
        output_dtype,
    )

    return combine_cls(
        hidden_states=hidden_states,
        topk_ids=dispatch_output.topk_ids,
        topk_weights=dispatch_output.topk_weights,
    )


def maybe_apply_fuseep_weights(layer: torch.nn.Module) -> bool:
    """Apply the FuseEP weight layout if --moe-a2a-backend is ascend_fuseep.

    Returns True when the FuseEP layout was applied and the caller should
    skip its own ``process_weights_after_loading`` body.
    """
    from sglang.srt.layers.moe import get_moe_a2a_backend

    if not get_moe_a2a_backend().is_ascend_fuseep():
        return False
    from sglang.srt.hardware_backend.npu.moe.fuseep import process_fuseep_weights

    process_fuseep_weights(layer)
    return True


class _NPUFusedMoEMethodBase(FusedMoEMethodBase):

    def __init__(
        self,
        quant_config: Optional["QuantizationConfig"] = None,
    ):
        self.quant_config = quant_config

    def _maybe_apply_deepep(
        self,
        layer: torch.nn.Module,
        dispatch_output: "DispatchOutput",
    ) -> Optional["CombineInput"]:
        return maybe_apply_deepep_npu(self, layer, dispatch_output)

    @staticmethod
    def _maybe_apply_fuseep_weights(layer: torch.nn.Module) -> bool:
        return maybe_apply_fuseep_weights(layer)


class NPUW4A4Int4DynamicMoEMethod(_NPUFusedMoEMethodBase):

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight.data = npu_format_cast(
            layer.w13_weight.data.transpose(1, 2).contiguous()
        )
        layer.w13_weight.data = self._pack_to_int32(
            layer.w13_weight.data.to(torch.int32)
        )

        layer.w2_weight.data = npu_format_cast(
            layer.w2_weight.data.transpose(1, 2).contiguous()
        )

        scale_np = layer.w13_weight_scale.data.cpu().numpy()
        scale_np.dtype = np.uint32
        scale_uint64_tensor = torch.from_numpy(scale_np.astype(np.int64)).npu()

        layer.w13_weight_scale = torch.nn.Parameter(
            scale_uint64_tensor.squeeze(-1), requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            layer.w2_weight_scale.data.squeeze(-1), requires_grad=False
        )

        # Compressed-tensors format doesn't have this field
        if hasattr(layer, "w13_weight_offset"):
            layer.w13_weight_offset = torch.nn.Parameter(
                layer.w13_weight_offset.data.squeeze(-1),
                requires_grad=False,
            )
        if hasattr(layer, "w2_weight_offset"):
            layer.w2_weight_offset = torch.nn.Parameter(
                layer.w2_weight_offset.data.squeeze(-1),
                requires_grad=False,
            )

        # Quantizes in int4 separately from the dispatcher
        # since deep_ep does not support quantization in int4
        # dispatching works in bf16
        if hasattr(layer, "dispatcher"):
            layer.dispatcher.set_quant_config({"dispatcher_output_dtype": "bf16"})

    def _pack_to_int32(self, weight: torch.Tensor):
        # pack 8 int4 to int32, we use a int32 to represent a int4
        assert (
            weight.shape[-1] % 8 == 0
        ), "the last dim of weight needs to be divided by 8"
        new_weight = torch.ops.npu.npu_convert_weight_to_int4pack(weight.flatten(0, 1))
        new_weight = new_weight.view(weight.shape[0], weight.shape[1], -1)
        return new_weight

    def apply(
        self,
        layer,
        dispatch_output: "DispatchOutput",
    ) -> "CombineInput":
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        combine_input = self._maybe_apply_deepep(layer, dispatch_output)
        if combine_input is not None:
            return combine_input

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        topk_weights, topk_ids, _ = topk_output
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(x.dtype)
        output = npu_fused_experts_w4a4(
            hidden_states=x,
            w13=layer.w13_weight,
            w13_scale=layer.w13_weight_scale,
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=topk_ids.shape[1],
        )
        return StandardCombineInput(hidden_states=output)

    def apply_without_routing_weights(
        self,
        layer,
        hidden_states,
        hidden_states_scale,
        group_list_type,
        group_list,
        output_dtype,
    ):
        hidden_states, hidden_states_scale = torch.ops.npu.npu_dynamic_quant(
            hidden_states, dst_type=torch.quint4x2
        )
        # gmm1: up_gate_proj
        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w13_weight],
            scale=[layer.w13_weight_scale],
            per_token_scale=[hidden_states_scale],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=output_dtype,
        )[0]
        # act_fn: swiglu
        hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
        hidden_states, pertoken_scale = torch.ops.npu.npu_dynamic_quant(hidden_states)

        # gmm2: down_proj
        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w2_weight],
            scale=[layer.w2_weight_scale.to(output_dtype)],
            per_token_scale=[pertoken_scale],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=output_dtype,
        )[0]
        return hidden_states


class NPUW8A8Int8DynamicMoEMethod(_NPUFusedMoEMethodBase):

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self._maybe_apply_fuseep_weights(layer):
            return
        layer.w13_weight.data = npu_format_cast(
            layer.w13_weight.data.transpose(1, 2).contiguous()
        )
        layer.w2_weight.data = npu_format_cast(
            layer.w2_weight.data.transpose(1, 2).contiguous()
        )
        layer.w13_weight_scale = torch.nn.Parameter(
            layer.w13_weight_scale.data.squeeze(-1), requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            layer.w2_weight_scale.data.squeeze(-1), requires_grad=False
        )
        layer.w13_weight_scale_bf16 = torch.nn.Parameter(
            layer.w13_weight_scale.data.to(dtype=torch.bfloat16), requires_grad=False
        )
        layer.w2_weight_scale_bf16 = torch.nn.Parameter(
            layer.w2_weight_scale.data.to(dtype=torch.bfloat16), requires_grad=False
        )
        # Compressed-tensors format doesn't have this field
        if hasattr(layer, "w13_weight_offset"):
            layer.w13_weight_offset = torch.nn.Parameter(
                layer.w13_weight_offset.data.squeeze(-1),
                requires_grad=False,
            )
        if hasattr(layer, "w2_weight_offset"):
            layer.w2_weight_offset = torch.nn.Parameter(
                layer.w2_weight_offset.data.squeeze(-1),
                requires_grad=False,
            )

        if hasattr(layer, "dispatcher"):
            layer.dispatcher.set_quant_config({"dispatcher_output_dtype": "int8"})

    def apply(
        self,
        layer,
        dispatch_output: "DispatchOutput",
    ) -> "CombineInput":
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        combine_input = self._maybe_apply_deepep(layer, dispatch_output)
        if combine_input is not None:
            return combine_input

        # release fp32 scale to save memory
        layer.w13_weight_scale = None
        layer.w2_weight_scale = None

        hidden_states = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        topk_weights, topk_ids, _ = topk_output
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(hidden_states.dtype)

        # prefill
        if not torch.npu.is_current_stream_capturing():
            output = npu_fused_experts(
                hidden_states=hidden_states,
                w13=layer.w13_weight,
                w13_scale=layer.w13_weight_scale_bf16,
                w2=layer.w2_weight,
                w2_scale=layer.w2_weight_scale_bf16,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                top_k=topk_ids.shape[1],
            )
        # decode
        else:
            output = npu_fused_experts_w8a8_decode(
                hidden_states=hidden_states,
                w13=layer.w13_weight,
                w13_scale=layer.w13_weight_scale_bf16,
                w2=layer.w2_weight,
                w2_scale=layer.w2_weight_scale_bf16,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                top_k=topk_ids.shape[1],
            )

        return StandardCombineInput(hidden_states=output)

    def apply_without_routing_weights(
        self,
        layer,
        hidden_states,
        hidden_states_scale,
        group_list_type,
        group_list,
        output_dtype,
    ):
        # gmm1: gate_up_proj
        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w13_weight],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=torch.int32,
        )[0]

        # act_fn: swiglu
        hidden_states, swiglu_out_scale = torch.ops.npu.npu_dequant_swiglu_quant(
            x=hidden_states,
            weight_scale=layer.w13_weight_scale,
            activation_scale=hidden_states_scale,
            bias=None,
            quant_scale=None,
            quant_offset=None,
            group_index=group_list,
            activate_left=True,
            quant_mode=1,
        )

        # gmm2: down_proj
        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w2_weight],
            scale=[layer.w2_weight_scale_bf16],
            per_token_scale=[swiglu_out_scale],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=output_dtype,
        )[0]
        return hidden_states


class NPUW4A8Int8DynamicMoEMethod(_NPUFusedMoEMethodBase):

    def _process_scale(
        self, weight: torch.Tensor, scale, per_group_scale, is_per_channel_weight
    ):
        scale = scale.transpose(1, 2).contiguous()

        if is_per_channel_weight:
            scale_np = scale.cpu().numpy()
            scale_np.dtype = np.uint32
            scale_uint64_tensor = torch.from_numpy(scale_np.astype(np.int64)).npu()
            return scale_uint64_tensor, None

        per_group_scale = per_group_scale.transpose(1, 2).contiguous()
        group_num, k, n = weight.shape
        # the weight of the new version is reduced by half by pack n, so it needs to be restored
        n = n * 2
        per_group_scale = per_group_scale.reshape(group_num, -1, n)
        group_num, quantgroup_num, n = per_group_scale.shape
        bias = None

        scale_fp32 = (scale * per_group_scale).to(torch.float16).to(torch.float32)
        scale_fp32_np = scale_fp32.cpu().numpy()
        scale_fp32_np.dtype = np.uint32
        sscale_uint64 = np.zeros((group_num, quantgroup_num, n * 2), dtype=np.uint32)

        sscale_uint64[..., ::2] = scale_fp32_np

        sscale_uint64_buffer = np.frombuffer(
            sscale_uint64.tobytes(), dtype=np.int64
        ).copy()
        sscale_uint64_tensor = torch.from_numpy(sscale_uint64_buffer).reshape(
            group_num, quantgroup_num, n
        )
        sscale_uint64_tensor = sscale_uint64_tensor.npu()
        return sscale_uint64_tensor, bias

    def _update_bias(self, layer, w13_bias, w2_bias):
        layer.w13_scale_bias.data = (
            layer.w13_scale_bias.data.transpose(1, 2).contiguous().sum(axis=1)
        )
        layer.w2_scale_bias.data = (
            layer.w2_scale_bias.data.transpose(1, 2).contiguous().sum(axis=1)
        )

    def _pack_to_int32(self, weight: torch.Tensor):
        # pack 4 int8(int4*2) to int32, because in pytorch, we need to use int32 to represent int4
        assert (
            weight.shape[-1] % 4 == 0
        ), "the last dim of weight needs to be divided by 4"
        return weight.view(torch.int32).contiguous()

    def process_weights_after_loading(
        self, layer: torch.nn.Module, is_per_channel_weight, activation_use_clip
    ) -> None:
        if not activation_use_clip:
            self._process_weights_without_clip(layer, is_per_channel_weight)
        else:
            self._process_weights_with_clip(layer)

        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight.data.transpose(1, 2).contiguous(), requires_grad=False
        )
        layer.w2_weight = torch.nn.Parameter(
            layer.w2_weight.data.transpose(1, 2).contiguous(), requires_grad=False
        )

        layer.w13_weight.data = npu_format_cast(layer.w13_weight.data)
        layer.w2_weight.data = npu_format_cast(layer.w2_weight.data)

        layer.w13_weight.data = self._pack_to_int32(layer.w13_weight.data)
        layer.w2_weight.data = self._pack_to_int32(layer.w2_weight.data)

        if hasattr(layer, "dispatcher"):
            layer.dispatcher.set_quant_config({"dispatcher_output_dtype": "int8"})

    def _process_weights_without_clip(
        self, layer: torch.nn.Module, is_per_channel_weight
    ) -> None:
        w13_weight_scale_second = (
            layer.w13_weight_scale_second.data
            if hasattr(layer, "w13_weight_scale_second")
            else None
        )
        w2_weight_scale_second = (
            layer.w2_weight_scale_second.data
            if hasattr(layer, "w2_weight_scale_second")
            else None
        )
        layer.w13_weight_scale.data, w13_bias = self._process_scale(
            layer.w13_weight,
            layer.w13_weight_scale.data,
            w13_weight_scale_second,
            is_per_channel_weight,
        )
        layer.w2_weight_scale.data, w2_bias = self._process_scale(
            layer.w2_weight,
            layer.w2_weight_scale.data,
            w2_weight_scale_second,
            is_per_channel_weight,
        )
        if hasattr(layer, "w13_weight_scale_second"):
            # scale_second is no longer used, release this part of the memory
            del layer.w13_weight_scale_second
            del layer.w2_weight_scale_second
            del layer.w13_weight_offset_second
            del layer.w2_weight_offset_second

        self._update_bias(layer, w13_bias, w2_bias)

    def _process_weights_with_clip(self, layer: torch.nn.Module) -> None:
        w13_weight_scale = (
            layer.w13_weight_scale.data.squeeze(-1).contiguous().unsqueeze(1)
        )
        w2_weight_scale = (
            layer.w2_weight_scale.data.squeeze(-1).contiguous().unsqueeze(1)
        )
        layer.w13_weight_scale = torch.nn.Parameter(
            w13_weight_scale, requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(w2_weight_scale, requires_grad=False)
        layer.w13_scale_bias = layer.w13_bias
        layer.w2_scale_bias = layer.w2_bias

    def apply(
        self,
        layer,
        dispatch_output: "DispatchOutput",
    ) -> "CombineInput":
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        combine_input = self._maybe_apply_deepep(layer, dispatch_output)
        if combine_input is not None:
            return combine_input

        hidden_states = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        topk_weights, topk_ids, _ = topk_output
        top_k = topk_ids.shape[1]
        group_list_type = 1
        original_shape = hidden_states.shape
        topk_weights = topk_weights

        num_tokens = hidden_states.shape[:-1].numel()

        first_expert_idx = 0
        last_expert_idx = layer.num_experts
        global_num_experts = layer.num_experts

        sorted_hidden_states, expanded_row_idx, expert_tokens, pertoken_scale = (
            torch.ops.npu.npu_moe_init_routing_v2(
                hidden_states,
                topk_ids,
                active_num=num_tokens * top_k,
                expert_num=global_num_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[first_expert_idx, last_expert_idx],
                quant_mode=1,
            )
        )

        expert_tokens = expert_tokens.to(torch.int64)

        bias1 = [layer.w13_scale_bias]
        bias2 = [layer.w2_scale_bias]
        w1_scale = [layer.w13_weight_scale]
        w2_scale = [layer.w2_weight_scale]
        _output_dtype = torch.bfloat16

        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[sorted_hidden_states],
            weight=[layer.w13_weight],
            scale=w1_scale,
            bias=bias1,
            per_token_scale=[pertoken_scale],
            group_list=expert_tokens,
            split_item=2,
            group_type=0,
            group_list_type=group_list_type,
            output_dtype=_output_dtype,
        )[0]

        # act_fn: swiglu
        hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
        hidden_states, swiglu_out_scale = torch.ops.npu.npu_dynamic_quant(hidden_states)

        output = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w2_weight],
            scale=w2_scale,
            bias=bias2,
            per_token_scale=[swiglu_out_scale],
            group_list=expert_tokens,
            split_item=2,
            group_type=0,
            group_list_type=group_list_type,
            output_dtype=_output_dtype,
        )[0]

        assert original_shape is not None
        final_hidden_states = torch.ops.npu.npu_moe_token_unpermute(
            permuted_tokens=output,
            sorted_indices=torch.abs(expanded_row_idx),
            probs=topk_weights,
        )
        if len(original_shape) == 3:
            final_hidden_states = final_hidden_states.view(original_shape)

        return StandardCombineInput(hidden_states=final_hidden_states)

    def apply_without_routing_weights(
        self,
        layer,
        hidden_states,
        hidden_states_scale,
        group_list_type,
        group_list,
        output_dtype,
    ):
        from sgl_kernel_npu.activation.swiglu_quant import swiglu_quant

        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w13_weight],
            scale=[layer.w13_weight_scale],
            bias=[layer.w13_scale_bias],
            per_token_scale=[hidden_states_scale],
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=group_list_type,
            output_dtype=output_dtype,
        )[0]

        hidden_states, swiglu_out_scale = swiglu_quant(
            hidden_states, group_list, group_list_type
        )

        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w2_weight],
            scale=[layer.w2_weight_scale],
            bias=[layer.w2_scale_bias],
            per_token_scale=[swiglu_out_scale],
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=group_list_type,
            output_dtype=output_dtype,
        )[0]

        return hidden_states


class NPUW4A16Int4DynamicMoEMethod(_NPUFusedMoEMethodBase):

    def _pack_to_int32(self, weight: torch.Tensor):
        assert weight.dim() == 3
        if weight.dtype == torch.int32:
            # pack 8 int4 to int32, we use a int32 to represent a int4
            assert (
                weight.shape[-1] % 8 == 0
            ), "the last dim of weight needs to be divided by 8"
            new_weight = torch.ops.npu.npu_convert_weight_to_int4pack(
                weight.flatten(0, 1)
            )
            new_weight = new_weight.view(weight.shape[0], weight.shape[1], -1)
        elif weight.dtype == torch.int8:
            # pack 4 int8(int4*2) to int32, because in pytorch, we need to use int32 to represent int4
            assert (
                weight.shape[-1] % 4 == 0
            ), "the last dim of weight needs to be divided by 4"
            new_weight = weight.view(torch.int32).contiguous()
        else:
            raise ValueError(f"{weight.dtype=} is not supported !")
        return new_weight

    def _unpack_from_int32(
        self,
        value: torch.Tensor,
        num_bits: int,
        shape: torch.Size = None,
        packed_dim=1,
    ) -> torch.Tensor:
        """
        Unpacks a tensor of packed int32 weights into individual int8s, maintaining the
        original bit range.

        Return tensors in int8

        :param value: tensor to unpack
        :param num_bits: number of bits to unpack each data point into
        :param shape: shape to unpack into, used to remove padding
        :returns: unpacked int8 tensor
        """
        if value.dtype is not torch.int32:
            raise ValueError(
                f"Expected {torch.int32} but got {value.dtype}, Aborting unpack."
            )

        if num_bits > 8:
            raise ValueError("Unpacking is only supported for less than 8 bits")

        pack_factor = 32 // num_bits

        # unpack
        mask = (1 << num_bits) - 1

        if packed_dim == 1:
            unpacked = torch.zeros(
                (value.shape[0], value.shape[1] * pack_factor),
                device=value.device,
                dtype=torch.int32,
            )
            for i in range(pack_factor):
                unpacked[:, i::pack_factor] = (value >> (num_bits * i)) & mask

            # remove padding
            if shape is not None:
                original_row_size = int(shape[1])
                unpacked = unpacked[:, :original_row_size]
        else:
            unpacked = torch.zeros(
                (value.shape[0] * pack_factor, value.shape[1]),
                device=value.device,
                dtype=torch.int32,
            )
            for i in range(pack_factor):
                unpacked[i::pack_factor, :] = (value >> (num_bits * i)) & mask

            # remove padding
            original_row_size = int(shape[0])
            unpacked = unpacked[:original_row_size, :]

        # bits are packed in unsigned format, reformat to signed
        # update the value range from unsigned to signed
        offset = pow(2, num_bits) // 2
        unpacked = (unpacked - offset).to(torch.int8)

        return unpacked

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w13_weight_scale = layer.w13_weight_scale.data.transpose(-1, -2).contiguous()
        w2_weight_scale = layer.w2_weight_scale.data.transpose(-1, -2).contiguous()
        layer.w13_weight_scale = torch.nn.Parameter(
            w13_weight_scale, requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(w2_weight_scale, requires_grad=False)

        layer.w13_weight_offset = torch.nn.Parameter(
            layer.w13_weight_offset.data.transpose(-1, -2).contiguous(),
            requires_grad=False,
        )
        layer.w2_weight_offset = torch.nn.Parameter(
            layer.w2_weight_offset.data.transpose(-1, -2).contiguous(),
            requires_grad=False,
        )

        # w = [n, k // 8]  --> [k, n // 8]
        # w13_weight = layer.w13_weight.data.transpose(1, 2).contiguous()
        # w2_weight = layer.w2_weight.data.transpose(1, 2).contiguous()
        unpacked_w13_weight = (
            self._unpack_from_int32(layer.w13_weight.data.flatten(0, 1), 4)
            .view(layer.w13_weight.data.shape[0], layer.w13_weight.data.shape[1], -1)
            .transpose(1, 2)
            .contiguous()
            .int()
        )
        unpacked_w2_weight = (
            self._unpack_from_int32(layer.w2_weight.data.flatten(0, 1), 4)
            .view(layer.w2_weight.data.shape[0], layer.w2_weight.data.shape[1], -1)
            .transpose(1, 2)
            .contiguous()
            .int()
        )

        w13_weight = self._pack_to_int32(unpacked_w13_weight)
        w2_weight = self._pack_to_int32(unpacked_w2_weight)

        layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)

        if hasattr(layer, "dispatcher"):
            layer.dispatcher.set_quant_config({"dispatcher_output_dtype": "bf16"})

    def apply(
        self,
        layer,
        dispatch_output: "DispatchOutput",
    ) -> "CombineInput":
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        combine_input = self._maybe_apply_deepep(layer, dispatch_output)
        if combine_input is not None:
            return combine_input

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        topk_weights, topk_ids, _ = topk_output
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(x.dtype)
        output = npu_fused_experts(
            hidden_states=x,
            w13=layer.w13_weight,
            w13_scale=layer.w13_weight_scale,
            w13_offset=layer.w13_weight_offset,
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
            w2_offset=layer.w2_weight_offset,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=topk_ids.shape[1],
            use_wna16=True,
        )
        return StandardCombineInput(hidden_states=output)

    def apply_without_routing_weights(
        self,
        layer,
        hidden_states,
        hidden_states_scale,
        group_list_type,
        group_list,
        output_dtype,
    ):
        if hidden_states_scale is None:
            # gmm1: gate_up_proj
            hidden_states = torch.ops.npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=[layer.w13_weight],
                antiquant_scale=[layer.w13_weight_scale],
                antiquant_offset=[layer.w13_weight_offset],
                split_item=2,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=output_dtype,
            )[0]

            # act_fn: swiglu
            hidden_states = torch.ops.npu.npu_swiglu(hidden_states)

            # gmm2: down_proj
            out_hidden = torch.ops.npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=[layer.w2_weight],
                antiquant_scale=[layer.w2_weight_scale],
                antiquant_offset=[layer.w2_weight_offset],
                split_item=2,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=output_dtype,
            )[0]
        else:
            raise ValueError(
                "when weight is int4, hidden_states only supports non-quant dtype!"
            )

        return out_hidden


class NPUMXFP8FusedMoEMethod(_NPUFusedMoEMethodBase):
    """Online MXFP8 FusedMoE method (``--quantization mxfp8`` on Ascend A5).

    Weights loaded as BF16/FP16 are quantised to MXFP8 (float8_e4m3fn + e8m0
    block scales) in ``process_weights_after_loading``. The forward path
    mirrors vllm-ascend's A5 MXFP8 MoE MLP: init_routing_v2 → dynamic MX
    activation quant → npu_grouped_matmul_swiglu_quant_v2 (gmm1) →
    npu_grouped_matmul (gmm2).

    Weights are kept as **strided transpose views** — NO ``.contiguous()``
    after ``.transpose(1,2)`` — because the grouped-matmul kernels accept
    non-contiguous views and ``.contiguous()`` would tank HBM bandwidth
    (matches dense ``NPUMXFP8LinearMethod`` and vllm-ascend's
    ``AscendW8A8MXFP8DynamicFusedMoEMethod``).
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.utils.common import set_weight_attrs

        # Fused gate_up_proj: [E, 2*intermediate, hidden]
        w13_up_dim = 2 * intermediate_size_per_partition
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, w13_up_dim, hidden_size, dtype=params_dtype
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj: [E, hidden, intermediate]
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, hidden_size, intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

    def create_moe_runner(
        self,
        layer: torch.nn.Module,
        moe_runner_config: "MoeRunnerConfig",
    ) -> None:
        """NPU does its own grouped matmul in ``apply``, not via ``MoeRunner``."""
        self.moe_runner_config = moe_runner_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from torch.nn import Parameter

        w13 = layer.w13_weight.data  # [E, 2I, H] (online bf16) or [E, 2I, H] fp8 (offline)
        w2 = layer.w2_weight.data    # [E, H,  I]

        if w13.dtype == torch.float8_e4m3fn:
            # ------- offline (ModelSlim) path -------
            # Weights are pre-quantised float8_e4m3fn; scales are uint8
            # [E, N, K//32] loaded by the scheme. Only re-layout.
            w13_scale = layer.w13_weight_scale.data  # [E, 2I, H//32] uint8
            w2_scale = layer.w2_weight_scale.data    # [E, H,  I//32] uint8

            num_experts = w13.shape[0]

            # Reshape 2D scales to 3D pair-split, then transpose to kernel layout.
            # w13_scale: [E, 2I, H//32] -> [E, 2I, H//64, 2] -> [E, H//64, 2I, 2]
            s13 = w13_scale.reshape(num_experts, -1, w13_scale.shape[-1] // 2, 2)
            layer.w13_weight_scale = Parameter(
                s13.transpose(1, 2), requires_grad=False
            )
            # w2_scale: [E, H, I//32] -> [E, H, I//64, 2] -> [E, I//64, H, 2]
            s2 = w2_scale.reshape(num_experts, -1, w2_scale.shape[-1] // 2, 2)
            layer.w2_weight_scale = Parameter(
                s2.transpose(1, 2), requires_grad=False
            )

            # Strided transpose views — DO NOT call .contiguous().
            layer.w13_weight = Parameter(
                w13.transpose(1, 2), requires_grad=False
            )
            layer.w2_weight = Parameter(
                w2.transpose(1, 2), requires_grad=False
            )
            return

        # ------- online path -------
        w13_bf = w13
        w2_bf = w2

        # Move to NPU if needed (cpu offload may have moved them back).
        if not w13_bf.is_npu:
            w13_bf = w13_bf.to(
                torch.device(f"npu:{torch.npu.current_device()}")
            )
        if not w2_bf.is_npu:
            w2_bf = w2_bf.to(
                torch.device(f"npu:{torch.npu.current_device()}")
            )

        # Cast to bf16 if needed.
        if w13_bf.dtype not in (torch.float16, torch.bfloat16):
            w13_bf = w13_bf.to(torch.bfloat16)
        if w2_bf.dtype not in (torch.float16, torch.bfloat16):
            w2_bf = w2_bf.to(torch.bfloat16)

        # Online MXFP8 quant — probe confirms 3D input [E,N,K] accepted.
        # qw: [E, N, K] float8_e4m3fn, scale: [E, N, K//64, 2] uint8.
        qw13, s13 = torch.ops.npu.npu_dynamic_mx_quant(
            w13_bf, dst_type=torch.float8_e4m3fn
        )
        qw2, s2 = torch.ops.npu.npu_dynamic_mx_quant(
            w2_bf, dst_type=torch.float8_e4m3fn
        )

        # Strided transpose views — DO NOT call .contiguous().
        # w13:      [E,  H,    2I] float8_e4m3fn
        # w13_scale:[E,  H/64, 2I, 2] uint8
        # w2:       [E,  I,    H ] float8_e4m3fn
        # w2_scale: [E,  I/64, H,  2] uint8
        layer.w13_weight = Parameter(
            qw13.transpose(1, 2), requires_grad=False
        )
        layer.w2_weight = Parameter(
            qw2.transpose(1, 2), requires_grad=False
        )
        layer.w13_weight_scale = Parameter(
            s13.transpose(1, 2), requires_grad=False
        )
        layer.w2_weight_scale = Parameter(
            s2.transpose(1, 2), requires_grad=False
        )

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "DispatchOutput",
    ) -> "CombineInput":
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        combine_input = self._maybe_apply_deepep(layer, dispatch_output)
        if combine_input is not None:
            return combine_input

        hidden_states = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        topk_weights, topk_ids, _ = topk_output
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(hidden_states.dtype)

        output = npu_fused_experts_mxfp8(
            hidden_states=hidden_states,
            w13=layer.w13_weight,
            w13_scale=layer.w13_weight_scale,
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=topk_ids.shape[1],
        )
        return StandardCombineInput(hidden_states=output)

    def apply_without_routing_weights(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor | None,
        group_list_type: int,
        group_list: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """MXFP8 path for DeepEP dispatch outputs."""
        _E8M0 = _get_e8m0_dtype()

        if hidden_states_scale is None:
            hidden_states, pertoken_scale = torch.ops.npu.npu_dynamic_mx_quant(
                hidden_states, dst_type=torch.float8_e4m3fn
            )
        else:
            pertoken_scale = hidden_states_scale

        # gmm1: gate_up + swiglu + requantise. COUNT → cumulative.
        if group_list_type == 1:
            g1_group_list = group_list.cumsum(0)
        else:
            g1_group_list = group_list

        g1_out, g1_scale = torch.ops.npu.npu_grouped_matmul_swiglu_quant_v2(
            x=hidden_states,
            weight=[layer.w13_weight],
            group_list=g1_group_list,
            weight_scale=[layer.w13_weight_scale],
            x_scale=pertoken_scale,
            dequant_mode=2,
            quant_mode=2,
            dequant_dtype=torch.float32,
            quant_dtype=torch.float8_e4m3fn,
            x_dtype=None,
            weight_dtype=None,
            weight_scale_dtype=_E8M0,
            x_scale_dtype=_E8M0,
        )

        # gmm2: down_proj
        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[g1_out],
            weight=[layer.w2_weight],
            scale=[layer.w2_weight_scale],
            bias=None,
            per_token_scale=[g1_scale],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=output_dtype,
            scale_dtype=_E8M0,
            per_token_scale_dtype=_E8M0,
            x_dtype=None,
            weight_dtype=None,
        )[0]
        return hidden_states
