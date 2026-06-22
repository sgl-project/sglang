"""Ascend FuseEP fused dispatch+GEMM+combine forward path.

Follows the mega_moe shape: a free-function bypass invoked from
``FusedMoE.forward`` when ``--moe-a2a-backend ascend_fuseep`` is set, plus a
weight-postprocess helper that NPU quant_methods call from their
``process_weights_after_loading`` when the same backend is selected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.utils import FusedMoEMode, npu_format_cast
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPBuffer
from sglang.srt.layers.moe.utils import DeepEPMode

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.topk import TopKOutput


_PARAMS_BYTES = 2  # bf16 — Ascend's Dispatch & Combine does not support fp16


def _get_fuseep_buffer(layer: FusedMoE):
    DeepEPBuffer.set_dispatch_mode_as_low_latency()
    return DeepEPBuffer.get_deepep_buffer(
        get_tp_group().device_group,
        layer.hidden_size,
        _PARAMS_BYTES,
        DeepEPMode.LOW_LATENCY,
        envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get(),
        layer.num_experts,
    )


def forward_fuseep(
    layer: FusedMoE,
    hidden_states: torch.Tensor,
    topk_output: TopKOutput,
) -> torch.Tensor:
    buf = _get_fuseep_buffer(layer)
    hidden_states, _ = buf.fused_deep_moe(
        hidden_states,
        topk_idx=topk_output.topk_ids,
        topk_weights=topk_output.topk_weights,
        gmm1_permuted_weight=layer.w13_weight,
        gmm1_permuted_weight_scale=layer.w13_weight_scale,
        gmm2_weight=layer.w2_weight,
        gmm2_weight_scale=layer.w2_weight_scale,
        num_max_dispatch_tokens_per_rank=(
            envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        ),
        num_experts=layer.num_experts,
        fuse_mode=envs.SGLANG_NPU_FUSED_MOE_MODE.get(),
    )
    return hidden_states


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
    ).to(scale.device)
    return torch.nn.Parameter(converted, requires_grad=False)


def process_fuseep_weights(layer: torch.nn.Module, weight_prefix: str) -> None:
    """Apply the Ascend FuseEP-specific weight layout for a single weight group.

    Invoked by ``maybe_apply_fuseep_weights`` for both ``"w13"`` and ``"w2"``.
    """
    if envs.SGLANG_NPU_FUSED_MOE_MODE.get() == FusedMoEMode.DISPATCH_FFN_COMBINE.value:
        # -- branch 1: DISPATCH_FFN_COMBINE ---------------------------------
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
    else:
        # -- branch 2: other modes -----------------------------------------
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
