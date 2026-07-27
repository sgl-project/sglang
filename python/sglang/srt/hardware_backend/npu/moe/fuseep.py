"""Ascend FuseEP fused dispatch+GEMM+combine forward path.

Follows the mega_moe shape: a free-function bypass invoked from
``FusedMoE.forward`` when ``--moe-a2a-backend ascend_fuseep`` is set, plus a
weight-postprocess helper that NPU quant_methods call from their
``process_weights_after_loading`` when the same backend is selected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.distributed import get_moe_ep_group
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.utils import FusedMoEMode, npu_format_cast
from sglang.srt.layers.dp_attention import (
    get_dp_global_num_tokens,
    is_dp_attention_enabled,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPBuffer
from sglang.srt.layers.moe.utils import DeepEPMode
from sglang.srt.runtime_context import get_exec

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.topk import TopKOutput


_PARAMS_BYTES = 2  # bf16 — Ascend's Dispatch & Combine does not support fp16


def _get_fuseep_buffer(layer: FusedMoE, normal_mode: bool = False):
    if normal_mode:
        DeepEPBuffer.set_dispatch_mode_as_normal()
    else:
        DeepEPBuffer.set_dispatch_mode_as_low_latency()
    return DeepEPBuffer.get_deepep_buffer(
        get_moe_ep_group().device_group,
        layer.hidden_size,
        _PARAMS_BYTES,
        DeepEPMode.AUTO,
        envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get(),
        layer.num_experts,
    )


def forward_fuseep(
    layer: FusedMoE,
    hidden_states: torch.Tensor,
    topk_output: TopKOutput,
    m3_fuseep_normal: bool = False,
    m3_fuseep_num_input_tokens: Optional[int] = None,
) -> torch.Tensor:
    if m3_fuseep_normal:
        if envs.SGLANG_NPU_FUSED_MOE_MODE.get() != FusedMoEMode.DISPATCH_FFN_COMBINE.value:
            raise RuntimeError("MiniMax-M3 FuseEP prefill requires SGLANG_NPU_FUSED_MOE_MODE=2")
        buf = _get_fuseep_buffer(layer, normal_mode=True)
        # The M3 operator's workspace must hold all routed tokens that could arrive
        # at one EP rank. This conservative bound is safe for skewed router output.
        num_input_tokens = (
            m3_fuseep_num_input_tokens
            if m3_fuseep_num_input_tokens is not None
            else hidden_states.shape[0]
        )
        if is_dp_attention_enabled():
            # Idle DP ranks have no local tokens, but can receive routed tokens
            # from another DP rank through the shared EP16 collective.
            global_num_tokens = get_dp_global_num_tokens()
            if global_num_tokens is not None:
                num_input_tokens = max(num_input_tokens, sum(global_num_tokens))
        num_output_tokens = hidden_states.shape[0]
        is_idle_dp_rank = num_output_tokens == 0
        if is_idle_dp_rank:
            # The custom operator does not accept an empty input tensor. A zero
            # weighted dummy route keeps every EP rank in the collective; discard
            # its zero output before returning to the idle scheduler.
            hidden_states = hidden_states.new_zeros((1, layer.hidden_size))
            topk_ids = torch.zeros(
                (1, topk_output.topk_ids.shape[1]),
                dtype=topk_output.topk_ids.dtype,
                device=hidden_states.device,
            )
            topk_weights = torch.ones(
                (1, topk_output.topk_weights.shape[1]),
                dtype=topk_output.topk_weights.dtype,
                device=hidden_states.device,
            )
        else:
            topk_ids = topk_output.topk_ids
            topk_weights = topk_output.topk_weights
        # Padded decode tokens can carry -1 expert ids with zero gate weights.
        # DispatchFFNCombineM3 indexes routing buffers with every id, so replace
        # the sentinel with a valid expert while retaining the zero contribution.
        topk_ids = topk_ids.masked_fill(topk_ids < 0, 0)
        if is_dp_attention_enabled():
            # DP MLP gather returns a view into a rank-padded buffer. The custom
            # OPP requires a standalone base allocation for its input tensor.
            hidden_states = hidden_states.clone()
            topk_ids = topk_ids.contiguous()
            topk_weights = topk_weights.contiguous()
        normal_decode = m3_fuseep_num_input_tokens is None
        if normal_decode and hidden_states.shape[0] < 128:
            # DispatchFFNCombineM3's normal-mode tiles require a non-trivial
            # M dimension. Pad decode to one full M tile, then
            # discard these dummy routes before returning to the model.
            pad_tokens = 128 - hidden_states.shape[0]
            hidden_states = torch.cat(
                (hidden_states, hidden_states.new_zeros((pad_tokens, layer.hidden_size)))
            )
            topk_ids = torch.cat(
                (
                    topk_ids,
                    torch.zeros(
                        (pad_tokens, topk_ids.shape[1]),
                        dtype=topk_ids.dtype,
                        device=topk_ids.device,
                    ),
                )
            )
            topk_weights = torch.cat(
                (topk_weights, topk_weights.new_ones((pad_tokens, topk_weights.shape[1])))
            )
        max_output_size = max(num_input_tokens, 1) * topk_ids.shape[1]
        if not is_dp_attention_enabled():
            max_output_size *= get_tp_group().device_group.size()
        if normal_decode:
            max_output_size = max(
                max_output_size,
                hidden_states.shape[0]
                * topk_ids.shape[1]
                * get_tp_group().device_group.size(),
            )
        hidden_states, _ = buf.dispatch_ffn_combine_m3(
            hidden_states,
            topk_idx=topk_ids,
            topk_weights=topk_weights,
            weight1=layer.w13_weight,
            scale1=layer.w13_weight_scale,
            weight2=layer.w2_weight,
            scale2=layer.w2_weight_scale,
            max_output_size=max_output_size,
            num_experts=layer.num_experts,
        )
        return hidden_states[:num_output_tokens]

    is_idle_dp_rank = is_dp_attention_enabled() and hidden_states.shape[0] == 0
    if is_idle_dp_rank:
        # All EP ranks must enter the low-latency collective. Use a valid
        # expert-0 route and discard its result for idle DP ranks.
        hidden_states = hidden_states.new_zeros((1, layer.hidden_size))
        topk_ids = torch.zeros(
            (1, topk_output.topk_ids.shape[1]),
            dtype=topk_output.topk_ids.dtype,
            device=hidden_states.device,
        )
        topk_weights = torch.ones(
            (1, topk_output.topk_weights.shape[1]),
            dtype=topk_output.topk_weights.dtype,
            device=hidden_states.device,
        )
    else:
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights
    # Low-latency routing also indexes every expert id, including masked tokens.
    topk_ids = topk_ids.masked_fill(topk_ids < 0, 0)
    buf = _get_fuseep_buffer(layer)
    hidden_states, _ = buf.fused_deep_moe(
        hidden_states,
        topk_idx=topk_ids,
        topk_weights=topk_weights,
        gmm1_permuted_weight=layer.w13_weight,
        gmm1_permuted_weight_scale=layer.w13_weight_scale,
        gmm2_weight=layer.w2_weight,
        gmm2_weight_scale=layer.w2_weight_scale,
        num_max_dispatch_tokens_per_rank=(
            envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        ),
        num_experts=layer.num_experts,
        fuse_mode=get_exec().moe.fuseep_mode,
    )
    return hidden_states[:0] if is_idle_dp_rank else hidden_states


def _permute_w13_weight_scale(w: torch.Tensor, tile_n: int) -> torch.Tensor:
    if tile_n % 2 != 0:
        raise ValueError(f"tile_n must be even, got {tile_n}")

    *dims, n = w.shape
    if n % tile_n != 0:
        raise ValueError(f"Last dimension {n} must be divisible by tile_n {tile_n}")

    w_reshaped = w.reshape(*dims, 2, n // tile_n, tile_n // 2)
    perm_order = list(range(len(dims))) + [-2, -3, -1]
    return w_reshaped.permute(perm_order).reshape(*dims, n)


def _reshape_w13_weight(
    weight: torch.Tensor, dim: int, chunk_size: int = 64
) -> torch.Tensor:
    # Achieving greater computing power through reshape on Ascend.
    original_shape = weight.shape
    if dim < 0:
        dim += len(original_shape)

    if original_shape[dim] % (2 * chunk_size) != 0:
        raise ValueError(
            f"Dimension {dim} size {original_shape[dim]} must be divisible by "
            f"{2 * chunk_size}"
        )

    new_shape = (
        *original_shape[:dim],
        2,
        original_shape[dim] // (2 * chunk_size),
        chunk_size,
        *original_shape[dim + 1 :],
    )

    weight = weight.view(new_shape)
    weight = weight.transpose(dim, dim + 1).contiguous()
    return weight.view(*original_shape[:dim], -1, *original_shape[dim + 1 :])


def _release_weight_cache(weight: torch.Tensor) -> torch.Tensor:
    # .contiguous() introduces additional memory overhead; release with resize_(0)
    origin_weight = weight.data.transpose(1, 2)
    new_weight = origin_weight.contiguous()
    origin_weight.untyped_storage().resize_(0)
    return new_weight


def _scale_from_float_to_int64(scale: torch.Tensor) -> torch.nn.Parameter:
    import numpy as np

    converted = torch.from_numpy(
        np.frombuffer(
            scale.cpu().to(torch.float32).numpy().tobytes(), dtype=np.int32
        ).astype(np.int64)
    ).reshape(scale.shape).to(scale.device)
    return torch.nn.Parameter(converted, requires_grad=False)


def process_fuseep_weights(layer: torch.nn.Module, weight_prefix: str) -> None:
    """Apply the Ascend FuseEP-specific weight layout for a single weight group.

    Invoked by ``maybe_apply_fuseep_weights`` for both ``"w13"`` and ``"w2"``.
    """
    if get_exec().moe.fuseep_mode == 1:
        # -- The fused MoE optimization mode "1": dispatch_gmm_combine_decode --
        if weight_prefix == "w13":
            cpu_w13 = layer.w13_weight.data.transpose(1, 2).cpu()
            layer.w13_weight.data = _reshape_w13_weight(cpu_w13, -1).npu()
            w13_scale = layer.w13_weight_scale.data.squeeze(-1).contiguous()
            w13_scale = _permute_w13_weight_scale(w13_scale, 128)
            layer.w13_weight_scale = torch.nn.Parameter(
                w13_scale.to(torch.float32), requires_grad=False
            )
            layer.w13_weight.data = npu_format_cast(layer.w13_weight.data)
        else:  # weight_prefix == "w2"
            layer.w2_weight.data = npu_format_cast(layer.w2_weight.data)
            w2_scale = layer.w2_weight_scale.data.squeeze(-1).contiguous()
            layer.w2_weight_scale = torch.nn.Parameter(
                w2_scale.to(torch.float32), requires_grad=False
            )
    elif get_exec().moe.fuseep_mode == 2:
        # -- The fused MoE optimization mode "2": dispatch_ffn_combine --
        if weight_prefix == "w13":
            w13_weight = _release_weight_cache(layer.w13_weight)
            layer.w13_weight.data = npu_format_cast(w13_weight)
            layer.w13_weight_scale.data = layer.w13_weight_scale.data.view(
                layer.w13_weight_scale.data.shape[0], -1
            )
            layer.w13_weight_scale = _scale_from_float_to_int64(
                layer.w13_weight_scale.data
            )
        else:  # weight_prefix == "w2"
            w2_weight = _release_weight_cache(layer.w2_weight)
            layer.w2_weight.data = npu_format_cast(w2_weight)
            w2_scale = layer.w2_weight_scale.data.squeeze(-1).contiguous()
            layer.w2_weight_scale = torch.nn.Parameter(
                w2_scale.to(torch.float32), requires_grad=False
            )
            layer.w2_weight_scale = _scale_from_float_to_int64(
                layer.w2_weight_scale.data
            )

    # -- offsets (exist or not, same logic for both prefixes) ---------------
    offset_attr = f"{weight_prefix}_weight_offset"
    if hasattr(layer, offset_attr):
        setattr(
            layer,
            offset_attr,
            torch.nn.Parameter(
                getattr(layer, offset_attr).data.squeeze(-1).contiguous(),
                requires_grad=False,
            ),
        )
