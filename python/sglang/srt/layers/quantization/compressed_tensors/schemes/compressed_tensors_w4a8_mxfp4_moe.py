from __future__ import annotations

import logging
from typing import Optional, Callable, TYPE_CHECKING

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.utils import RoutingMethodType, get_moe_runner_backend
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMoEScheme,
)
from sglang.srt.layers.quantization.fp8_utils import is_blackwell_supported
from sglang.srt.layers.quantization.utils import (
    prepare_static_weights_for_trtllm_fp4_moe,
    reorder_w1w3_to_w3w1,
    replace_parameter,
    swizzle_blockscale,
)
from sglang.srt.utils import next_power_of_2, set_weight_attrs

import torch_npu

from sglang.srt.layers.activation import SituAndMul
from sgl_kernel_npu.activation.situ import situ_and_mul

logger = logging.getLogger(__name__)

__all__ = ["NPUCompressedTensorsW4A8mxfp4MoE"]

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )


def _npu_swiglu(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.npu.npu_swiglu(x)


class NPUCompressedTensorsW4A8mxfp4MoE(CompressedTensorsMoEScheme):

    def __init__(self):
        self.group_size = 32
        self.act_fn: Callable = _npu_swiglu

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        layer.params_dtype = params_dtype

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // 2,
                requires_grad=False,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Weight Scales
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # From packed to weight
        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w13_weight_packed")

        layer.w2_weight = torch.nn.Parameter(
            layer.w2_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w2_weight_packed")

        # Skip NZ format cast when MoE DRAM offload is enabled.
        # NZ format is incompatible with CPU round-trip (clone/copy_/
        # npu_format_cast(→0) all fail on internal format). For offload,
        # weights are stored in ND format and converted to NZ at forward
        # time in w4a8_mxfp4_gmm_npu (is_nd_format flag).
        _skip_nz_cast = getattr(layer, "moe_dram_offload", False)

        # If weights are on CPU (DRAM offload with _force_cpu_allocation),
        # move to NPU first — npu_format_cast requires NPU backend.
        if not _skip_nz_cast:
            if layer.w13_weight.data.device.type == "cpu":
                layer.w13_weight.data = layer.w13_weight.data.npu()
                layer.w2_weight.data = layer.w2_weight.data.npu()
            if layer.w13_weight_scale.data.device.type == "cpu":
                layer.w13_weight_scale.data = layer.w13_weight_scale.data.npu()
                layer.w2_weight_scale.data = layer.w2_weight_scale.data.npu()

            layer.w13_weight.data = torch_npu.npu_format_cast(
                layer.w13_weight.data, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
            )
            layer.w2_weight.data = torch_npu.npu_format_cast(
                layer.w2_weight.data, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
            )
            layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2)
            layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2)
        else:
            # ND format: no transpose for weights — register_layer_batch
            # will cast to NZ on original shape [E, N, K_packed] directly,
            # avoiding a redundant transpose(1,2).contiguous() round-trip
            # that costs ~22s per weight on 4.7GB tensors.
            pass

        g, n, k = layer.w13_weight_scale.shape
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.reshape(g, n, k // 2, 2).transpose(-3, -2)
        g, n, k = layer.w2_weight_scale.shape
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.reshape(g, n, k // 2, 2).transpose(-3, -2)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        if self.moe_runner_config.activation == "situ":
            # self.act_fn = SituAndMul(
            #     beta=self.moe_runner_config.activation_situ_beta,
            #     linear_beta=self.moe_runner_config.activation_situ_linear_beta,
            # )
            self.act_fn = situ_and_mul

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        combine_input = npu_apply_w4a8_mxfp4_moe_deepep(layer, dispatch_output, act_fn=self.act_fn)
        if combine_input is not None:
            return combine_input

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        hidden_states = dispatch_output.hidden_states
        topk_weights, topk_ids, _ = dispatch_output.topk_output
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(hidden_states.dtype)
        top_k = (
            self.moe_runner_config.top_k
            if self.moe_runner_config is not None
            else topk_ids.shape[1]
        )

        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale

        # NZ storage path (acc_offload + transfer_weight): weight is already
        # NZ format in HBM buffer. No format_cast or expert extraction needed.
        # Skip torch.unique (graph-safe, all experts loaded in HBM buffer).
        is_nz_stored = getattr(layer, "_is_nz_stored", False)
        is_nd_format = getattr(layer, "_dram_offload_enabled", False) and not is_nz_stored

        # Legacy ND path: extract only selected experts before NZ conversion
        # to avoid converting entire [num_experts, ...] tensor (OOM).
        # torch.unique is incompatible with graph capture (dynamic output
        # shape). Skip during capture; GMM processes all experts.
        if is_nd_format and not torch.npu.is_current_stream_capturing():
            unique_ids, inverse_indices = torch.unique(
                topk_ids, return_inverse=True
            )
            w13 = w13[unique_ids]
            w2 = w2[unique_ids]
            w13_scale = w13_scale[unique_ids]
            w2_scale = w2_scale[unique_ids]
            topk_ids = inverse_indices.to(torch.int32).view_as(topk_ids)

        output = npu_fused_experts_w4a8_mxfp4(
            hidden_states,
            w13,
            w13_scale,
            w2,
            w2_scale,
            topk_weights,
            topk_ids,
            top_k,
            act_fn=self.act_fn,
            is_nd_format=is_nd_format,
            is_nz_stored=is_nz_stored,
        )
        return StandardCombineInput(hidden_states=output)


def _permute_scale(scale: torch.Tensor) -> torch.Tensor:
    """Reshape and permute MXFP4 scale from [E, N, K/2, 2] to [E, K/64, N, 2]."""
    if scale.dim() == 3:
        num_experts, n, k32 = scale.shape
        if k32 % 2 != 0:
            raise ValueError(
                "MXFP4 scale K dimension must be divisible by 2 for "
                "[E, K/64, N, 2] layout."
            )
        scale = scale.view(num_experts, n, k32 // 2, 2).transpose(1, 2)
    return scale


def npu_fused_experts_w4a8_mxfp4(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    act_fn: Callable = _npu_swiglu,
    is_nd_format: bool = False,
    is_nz_stored: bool = False,
):
    if torch.npu.is_current_stream_capturing():
        return npu_fused_experts_w4a8_mxfp4_decode(
            hidden_states=hidden_states,
            w13=w13,
            w13_weight_scale=w13_weight_scale,
            w2=w2,
            w2_weight_scale=w2_weight_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=top_k,
            act_fn=act_fn,
            is_nd_format=is_nd_format,
            is_nz_stored=is_nz_stored,
        )

    original_shape = hidden_states.shape
    original_dtype = hidden_states.dtype
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
            hidden_states,
            row_idx=row_idx,
            expert_idx=topk_ids,
            active_num=num_tokens,
        )
    )
    expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(
        expanded_expert_idx, num_experts
    )
    expert_tokens = expert_tokens.to(torch.int64)

    rows = hidden_states.shape[0]
    row_ids = torch.arange(rows, device=hidden_states.device, dtype=torch.int64)
    valid_mask = row_ids < expert_tokens[-1]
    valid_mask_2d = valid_mask.unsqueeze(1)

    hidden_states = w4a8_mxfp4_gmm_npu(
        input=hidden_states,
        input_scale=None,
        weight=w13,
        weight_scale=w13_weight_scale,
        group_list_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
        is_nd_format=is_nd_format,
        is_nz_stored=is_nz_stored,
    )
    hidden_states = act_fn(hidden_states, expert_tokens, 0)
    hidden_states = w4a8_mxfp4_gmm_npu(
        input=hidden_states,
        input_scale=None,
        weight=w2,
        weight_scale=w2_weight_scale,
        group_list_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
        is_nd_format=is_nd_format,
        is_nz_stored=is_nz_stored,
    )

    hidden_states = hidden_states * valid_mask_2d.to(hidden_states.dtype)

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


def npu_fused_experts_w4a8_mxfp4_decode(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    act_fn: Callable = _npu_swiglu,
    is_nd_format: bool = False,
    is_nz_stored: bool = False,
):
    num_tokens = hidden_states.shape[:-1].numel()
    global_num_experts = w13.shape[0]
    original_shape = hidden_states.shape
    original_dtype = hidden_states.dtype
    group_list_type = 1

    hidden_states, expanded_row_idx, expert_tokens, _ = (
        torch.ops.npu.npu_moe_init_routing_v2(
            hidden_states,
            topk_ids,
            active_num=num_tokens * top_k,
            expert_num=global_num_experts,
            expert_tokens_num_type=group_list_type,
            expert_tokens_num_flag=True,
            active_expert_range=[0, global_num_experts],
            quant_mode=-1,
        )
    )
    expert_tokens = expert_tokens.to(torch.int64)

    if logger.isEnabledFor(logging.WARNING):
        logger.warning(
            f"[decode] num_tokens={num_tokens} top_k={top_k} "
            f"global_num_experts={global_num_experts} "
            f"expert_tokens.shape={expert_tokens.shape} "
            f"w13.shape={w13.shape} topk_ids.shape={topk_ids.shape} "
            f"capturing={torch.npu.is_current_stream_capturing()}"
        )

    hidden_states = w4a8_mxfp4_gmm_npu(
        input=hidden_states,
        input_scale=None,
        weight=w13,
        weight_scale=w13_weight_scale,
        group_list_type=group_list_type,
        group_list=expert_tokens,
        output_dtype=original_dtype,
        is_nd_format=is_nd_format,
        is_nz_stored=is_nz_stored,
    )
    hidden_states = act_fn(hidden_states, expert_tokens, group_list_type)
    hidden_states = w4a8_mxfp4_gmm_npu(
        input=hidden_states,
        input_scale=None,
        weight=w2,
        weight_scale=w2_weight_scale,
        group_list_type=group_list_type,
        group_list=expert_tokens,
        output_dtype=original_dtype,
        is_nd_format=is_nd_format,
        is_nz_stored=is_nz_stored,
    )

    final_hidden_states = torch.ops.npu.npu_moe_token_unpermute(
        permuted_tokens=hidden_states,
        sorted_indices=torch.abs(expanded_row_idx),
        probs=topk_weights,
    )

    if len(original_shape) == 3:
        final_hidden_states = final_hidden_states.view(original_shape)
    return final_hidden_states

def npu_apply_w4a8_mxfp4_moe_deepep(
    layer: torch.nn.Module,
    dispatch_output: "DispatchOutput",
    act_fn: Callable = _npu_swiglu,
) -> Optional["CombineInput"]:
    from sglang.srt.layers.moe.token_dispatcher import (
        DeepEPLLCombineInput,
        DeepEPNormalCombineInput,
    )
    from sglang.srt.layers.moe.token_dispatcher.base import DispatchOutputChecker

    if not dispatch_output.format.is_deepep():
        return None

    output_dtype = torch.bfloat16
    group_list_type = 1

    if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
        hidden_states, hidden_states_scale, _, _, num_recv_tokens_per_expert = (
            dispatch_output
        )
        group_list = torch.tensor(
            num_recv_tokens_per_expert,
            dtype=torch.int64,
            device=hidden_states.device,
        )
        combine_cls = DeepEPNormalCombineInput
    else:
        hidden_states, hidden_states_scale, _, _, group_list, _ = dispatch_output
        group_list = group_list.to(torch.int64)
        combine_cls = DeepEPLLCombineInput

    # Early return when this rank received no tokens.
    # In DeepEP, some ranks may receive 0 tokens for certain layers. Running
    # the CANN kernel with 0 tokens still requires group_list size == weight
    # dim 0, but the weight may be stale (e.g., [num_active, ...] from a
    # previous decode). Skip the kernel entirely — there's nothing to compute.
    # Note: hidden_states.shape[0] == 0 is equivalent to group_list.sum() == 0
    # (DeepEP guarantees group_list.sum() == num_recv_tokens), so we avoid
    # the D2H sync from group_list.sum().item() and check shape only.
    if hidden_states.shape[0] == 0:
        return combine_cls(
            hidden_states=hidden_states,
            topk_ids=dispatch_output.topk_ids,
            topk_weights=dispatch_output.topk_weights,
        )

    # Decode DRAM offload: use group_pack_copy to load active expert weights
    # from DRAM and compact group_list, entirely on-device (no D2H sync).
    # The kernel copies only non-zero group_list entries to the front of the
    # weight buffer and outputs packedGroupList. CANN skips experts with 0
    # tokens in the tail. Prefill uses the shared buffer (pre-loaded by
    # _load_experts_on_demand).
    if (
        getattr(layer, "_dram_offload_enabled", False)
        and layer._expert_weight_store is not None
        and layer._expert_weight_store._is_decode_mode
    ):
        store = layer._expert_weight_store
        sample_key = (layer.layer_id, 0)
        # Guard against missing dram_store entry (e.g., skip-layers or
        # initialization timing). group_pack_copy_active_weights has its
        # own defense, but we need weight_names before calling it.
        if sample_key in store.dram_store:
            weight_names = list(store.dram_store[sample_key].keys())
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    f"[deepep] before group_pack_copy: "
                    f"group_list.shape={group_list.shape} "
                    f"layer_id={layer.layer_id} "
                    f"capturing={torch.npu.is_current_stream_capturing()}"
                )
            compact_weights, group_list = store.group_pack_copy_active_weights(
                layer.layer_id, group_list, weight_names
            )
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    f"[deepep] after group_pack_copy: "
                    f"group_list.shape={group_list.shape} "
                    f"w13.shape={compact_weights.get('w13_weight', None).shape if 'w13_weight' in compact_weights else 'N/A'}"
                )
            for name, tensor in compact_weights.items():
                setattr(layer, name, tensor)
            # Mark NZ-stored: both acc_offload (group_pack_copy) and
            # transfer_weight paths store NZ bytes in HBM buffers.
            layer._is_nz_stored = True

    # Determine weight storage format for this layer:
    # - is_nz_stored=True: weight is NZ format in HBM (acc_offload or
    #   transfer_weight path). No forward-time format_cast needed.
    # - is_nd_format=True (and not is_nz_stored): weight is ND from DRAM
    #   (legacy H2D path). Forward-time ND→NZ conversion required.
    # - Both False: HBM-resident weight (already NZ from process_weights).
    _dram_offload = getattr(layer, "_dram_offload_enabled", False)
    _is_nz = getattr(layer, "_is_nz_stored", False)
    _is_nd = _dram_offload and not _is_nz

    hidden_states = npu_apply_without_routing_weights_w4a8_mxfp4(
        layer,
        hidden_states,
        hidden_states_scale,
        group_list_type,
        group_list,
        output_dtype,
        act_fn=act_fn,
        is_nd_format=_is_nd,
        is_nz_stored=_is_nz,
    )
    return combine_cls(
        hidden_states=hidden_states,
        topk_ids=dispatch_output.topk_ids,
        topk_weights=dispatch_output.topk_weights,
    )


def npu_apply_without_routing_weights_w4a8_mxfp4(
    layer,
    hidden_states,
    hidden_states_scale,
    group_list_type,
    group_list,
    output_dtype,
    act_fn: Callable = _npu_swiglu,
    is_nd_format: bool = False,
    is_nz_stored: bool = False,
):
    hidden_states = w4a8_mxfp4_gmm_npu(
        input=hidden_states,
        input_scale=hidden_states_scale,
        weight=layer.w13_weight,
        weight_scale=layer.w13_weight_scale,
        group_list_type=group_list_type,
        group_list=group_list,
        output_dtype=output_dtype,
        is_nd_format=is_nd_format,
        is_nz_stored=is_nz_stored,
    )
    # In the DRAM offload prefill path, _load_experts_on_demand loads
    # [num_local_experts, ...] ND tensors into shared HBM buffers. The NZ
    # conversion inside w4a8_mxfp4_gmm_npu creates additional tensors.
    # Without dropping the layer reference here, w13 ND + w13 NZ + w2 ND
    # + w2 NZ all coexist in HBM, causing OOM during prefill.
    # NOTE: In the decode path, weights point to _shared_decode_buffers
    # (reused across steps); setting to None here only drops the layer's
    # reference, not the underlying HBM. This is harmless — decode M is
    # small so peak pressure is lower.
    if is_nd_format and not is_nz_stored:
        layer.w13_weight_scale = None
    hidden_states = act_fn(hidden_states, group_list, group_list_type)
    hidden_states = w4a8_mxfp4_gmm_npu(
        input=hidden_states,
        input_scale=None,
        weight=layer.w2_weight,
        weight_scale=layer.w2_weight_scale,
        group_list_type=group_list_type,
        group_list=group_list,
        output_dtype=output_dtype,
        is_nd_format=is_nd_format,
        is_nz_stored=is_nz_stored,
    )
    # Release w2 compact weights after GMM to reduce HBM peak.
    if is_nd_format and not is_nz_stored:
        layer.w2_weight = None
        layer.w2_weight_scale = None
    return hidden_states


def w4a8_mxfp4_gmm_npu(
    input: torch.Tensor,
    input_scale: Optional[torch.Tensor],
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_list_type: int = 1,
    group_list: Optional[torch.Tensor] = None,
    output_dtype=torch.bfloat16,
    is_nd_format: bool = False,
    is_nz_stored: bool = False,
) -> torch.Tensor:
    if group_list is None:
        raise ValueError("group_list must be provided to w4a8_mxfp4_gmm_npu")
    group_list = group_list.to(torch.int64)

    if input_scale is None:
        x, x_scale = torch.ops.npu.npu_dynamic_mx_quant(
            input,
            axis=1,
            round_mode="rint",
            dst_type=torch.float8_e4m3fn,
            block_size=32,
            scale_alg=None,
        )
    else:
        x, x_scale = input, input_scale

    # Weight format conversion logic:
    # - is_nz_stored=True: weight is pre-allocated as NZ format in HBM
    #   by _ensure_shared_global_buffers (format_cast on [E, N, K_packed]
    #   + transpose(1,2) → [E, K_packed, N] NZ). DRAM stores the same NZ
    #   layout (cast on [E, N, K_packed] + transpose). Both HBM and DRAM
    #   have NZ blocks for N×K (K=2*K_packed=hidden, unpacked from fp4).
    #   No forward-time format_cast needed — graph-safe path.
    # - is_nd_format=True (and not is_nz_stored): weight is ND from DRAM.
    #   DRAM stores pre-transpose [E, N, K_packed] (register_layer_batch
    #   undid process_weights_after_loading's transpose before storage).
    #   Cast to NZ format, then transpose(1,2) to get [E, K_packed, N]
    #   transposed view matching HBM-resident path's final state.
    if is_nz_stored:
        pass
    elif is_nd_format:
        # ND storage (PyTorch H2D path): DRAM stores pre-transpose format
        # [E, N, K_packed] (register_layer_batch undid process_weights_after_loading's
        # transpose(1,2) before storage). Cast to NZ format, then transpose(1,2)
        # to get [E, K_packed, N] transposed view (is_contiguous=False) matching
        # HBM-resident path's final state. CANN requires weight to be transposed.
        weight = torch_npu.npu_format_cast(
            weight.contiguous().view(torch.uint8),
            29,
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=torch_npu.float4_e2m1fn_x2,
        )
        weight = weight.transpose(1, 2)

    # Scale: restore transposed state to match weight's transpose state.
    #
    # For is_nz_stored AND is_nd_format: weight is transposed (transpose(1,2)
    # applied). Scale from DRAM is [E, N, K//64, 2] (pre-transpose, because
    # register_layer_batch undid process_weights_after_loading's transpose
    # before storage). Re-apply transpose(-3, -2) to get [E, K//64, N, 2]
    # transposed view (is_contiguous=False) matching HBM-resident path.
    #
    # For HBM-resident (neither flag): scale already transposed from
    # process_weights_after_loading, no action needed.
    if is_nz_stored or is_nd_format:
        if weight_scale.is_contiguous():
            weight_scale = weight_scale.transpose(-3, -2)

    return torch.ops.npu.npu_grouped_matmul(
        [x],
        [weight],
        antiquant_scale=[weight_scale],
        scale_dtype=None,
        scale=None,
        per_token_scale=[x_scale],
        split_item=2,
        group_type=0,
        group_list=group_list,
        group_list_type=group_list_type,
        output_dtype=output_dtype,
        x_dtype=torch_npu.float8_e4m3fn,
        weight_dtype=torch_npu.float4_e2m1fn_x2,
        per_token_scale_dtype=torch_npu.float8_e8m0fnu,
    )[0]