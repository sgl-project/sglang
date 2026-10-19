"""SM90 NVFP4 W4A8 MegaMoE through flashinfer.megamoe.

ModelOpt NVFP4 expert tensors are converted at load into the input FlashInfer's
transform_nvfp4_weights_for_mega_moe expects (Marlin byte order, local scales the
decoder can hold, one global scale per expert). The forward quantizes activations to
FP8 (group 128) with DeepGEMM's SM90 pre-dispatch kernel straight into the FlashInfer
symmetric buffer, then calls flashinfer.megamoe.nvfp4_mega_moe (fused up to
FAMILY_THRESHOLD tokens per rank, split above).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    get_dp_global_num_tokens,
    is_dp_attention_enabled,
)
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.runtime_context import get_exec

logger = logging.getLogger(__name__)

_FI_SYMM_BUFFERS: dict = {}
_DISPATCH_GROUP_SIZE = 128
# The SM90 decoder turns FP4 * local scale into one E4M3 byte: 6 * scale <= 448.
_MAX_LOCAL_SCALE = 448.0 / 6.0


def is_sm90_fi_nvfp4_mega_moe_available(experts) -> bool:
    return experts._mega_moe_sm90_fi_nvfp4


def _require_flashinfer_megamoe() -> None:
    try:
        from flashinfer.megamoe import MegaMoESymmBuffer
    except ImportError as e:
        raise RuntimeError(
            "--moe-a2a-backend megamoe on SM90 with NVFP4 experts needs a FlashInfer "
            "build with flashinfer.megamoe (flashinfer-ai/flashinfer#6343 and #6517); "
            "the installed FlashInfer does not provide it."
        ) from e
    if not hasattr(MegaMoESymmBuffer, "commit_staged"):
        raise RuntimeError(
            "SM90 NVFP4 MegaMoE needs FlashInfer with flashinfer-ai/flashinfer#6517 "
            "(MegaMoESymmBuffer.prewarm / commit_staged)."
        )


# ---------------------------------------------------------------------------
# Weights: ModelOpt NVFP4 checkpoint -> flashinfer.megamoe input
# ---------------------------------------------------------------------------


def _low_nibble_first_to_marlin(packed: torch.Tensor) -> torch.Tensor:
    # Checkpoints store K[2i] in the low nibble of byte i; FlashInfer expects
    # K[b] (high) | K[b + 4] (low) in byte b of each four-byte word.
    words = packed.reshape(*packed.shape[:-1], -1, 4)
    codes = torch.stack((words & 15, words >> 4), dim=-1).flatten(-2)
    return ((codes[..., :4] << 4) | codes[..., 4:]).reshape(packed.shape)


def _rebalance_scales(
    scale: torch.Tensor, global_scale: torch.Tensor, gated: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per expert, move a power of two from the E4M3 local scales into the FP32
    global scale until 6 * local <= 448; gate / up globals fold into one."""
    local = scale.view(torch.float8_e4m3fn).float()
    global_scale = global_scale.float()
    if gated and global_scale.dim() == 2:
        merged = global_scale.amax(dim=1)
        ratio = global_scale / merged[:, None]
        n = local.shape[1] // 2
        local = torch.cat(
            (
                local[:, :n] * ratio[:, 0, None, None],
                local[:, n:] * ratio[:, 1, None, None],
            ),
            dim=1,
        )
        global_scale = merged
    shift = torch.exp2(
        torch.ceil(torch.log2(local.amax(dim=(1, 2)) / _MAX_LOCAL_SCALE)).clamp_min(0)
    )
    local = local / shift[:, None, None]
    encoded = local.to(torch.float8_e4m3fn)
    if not torch.equal(encoded.float(), local):
        raise ValueError(
            "NVFP4 expert scales cannot be rebalanced into the SM90 MegaMoE decoder "
            "range exactly; the per-expert scale range is too wide"
        )
    return encoded.view(torch.uint8), (global_scale * shift).contiguous()


def convert_nvfp4_experts_for_fi(w13, s13, g13, w2, s2, g2):
    """Returns (l1_weights, l2_weights, l1_global_scales, l2_global_scales)."""
    from flashinfer.megamoe import transform_nvfp4_weights_for_mega_moe

    s13, g13 = _rebalance_scales(s13, g13, gated=True)
    s2, g2 = _rebalance_scales(s2, g2, gated=False)
    l1_w, l2_w = transform_nvfp4_weights_for_mega_moe(
        (_low_nibble_first_to_marlin(w13.view(torch.uint8)), s13),
        (_low_nibble_first_to_marlin(w2.view(torch.uint8)), s2),
    )
    return l1_w, l2_w, g13, g2


def build_sm90_fi_nvfp4_mega_moe_experts_weights(experts) -> None:
    if experts._mega_moe_weights_built:
        return
    if get_exec().moe.enable_eplb:
        # The packed weights live outside the expert parameters EPLB rebalances.
        raise ValueError(
            "SM90 NVFP4 MegaMoE (--moe-a2a-backend megamoe) does not support EPLB; "
            "disable --enable-eplb."
        )
    _require_flashinfer_megamoe()
    l1_w, l2_w, l1_g, l2_g = convert_nvfp4_experts_for_fi(
        experts.w13_weight.data,
        experts.w13_weight_scale.data,
        experts.w13_weight_scale_2.data,
        experts.w2_weight.data,
        experts.w2_weight_scale.data,
        experts.w2_weight_scale_2.data,
    )
    experts.mega_l1_weights = l1_w
    experts.mega_l2_weights = l2_w
    experts.mega_l1_global_scales = l1_g
    experts.mega_l2_global_scales = l2_g
    # MegaMoE is the only consumer of the expert weights from here on.
    for param in (
        experts.w13_weight,
        experts.w2_weight,
        experts.w13_weight_scale,
        experts.w2_weight_scale,
    ):
        param.data = torch.empty(0, dtype=torch.uint8, device=l1_w.device)
    experts._mega_moe_sm90_fi_nvfp4 = True
    experts._mega_moe_weights_built = True


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def _get_fi_symm_buffer(
    group, *, num_experts, cap, top_k, hidden, intermediate, activation_clamp
):
    key = (id(group), num_experts, cap, top_k, hidden, intermediate, activation_clamp)
    buf = _FI_SYMM_BUFFERS.get(key)
    if buf is None:
        from flashinfer.megamoe import MegaMoESymmBuffer

        buf = MegaMoESymmBuffer(
            group,
            num_experts=num_experts,
            num_max_tokens_per_rank=cap,
            num_topk=top_k,
            hidden=hidden,
            intermediate_hidden=intermediate,
            activation_clamp=activation_clamp,
            fast_math=True,
        )
        if envs.SGLANG_FLASHINFER_MEGAMOE_SM90_PREWARM.get():
            num_modules = buf.prewarm(
                kernel_family=envs.SGLANG_FLASHINFER_MEGAMOE_SM90_KERNEL_FAMILY.get()
            )
            logger.info("FI_MEGAMOE_PREWARM built %d modules", num_modules)
        _FI_SYMM_BUFFERS[key] = buf
    return buf


def _rank_invariant_tokens(num_tokens: int) -> int:
    # Plan/family selection keys on M, so every EP rank must launch with the same M.
    # Decode graphs are captured and replayed at one padded batch size on all DP ranks.
    if get_is_capture_mode() or not is_dp_attention_enabled():
        return num_tokens
    global_num_tokens = get_dp_global_num_tokens()
    if not global_num_tokens:
        raise RuntimeError(
            "FI MegaMoE under DP attention needs the gathered global_num_tokens"
        )
    m = max(global_num_tokens)
    if num_tokens > m:
        raise RuntimeError(f"local M={num_tokens} exceeds gathered max M={m}")
    return m


def run_sm90_fi_nvfp4_mega_routed(
    experts,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_tokens: int,
    *,
    ep_group,
    num_max_tokens_per_rank: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    activation_clamp: Optional[float],
    routed_scaling_factor: float,
) -> torch.Tensor:
    import deep_gemm
    import flashinfer.megamoe as fm

    if activation_clamp is None:
        # The kernel always clamps SwiGLU; there is no unclamped variant.
        raise ValueError("SM90 NVFP4 MegaMoE needs the model's swiglu_limit")
    buf = _get_fi_symm_buffer(
        ep_group,
        num_experts=experts.num_experts,
        cap=num_max_tokens_per_rank,
        top_k=top_k,
        hidden=hidden_size,
        intermediate=intermediate_size,
        activation_clamp=float(activation_clamp),
    )
    m = _rank_invariant_tokens(num_tokens)
    if m == 0:
        return hidden_states.new_empty((0, hidden_size))

    x = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
    ids = topk_ids.to(torch.int32)
    weights = topk_weights.to(torch.float32)
    if m > num_tokens:
        # Padded rows route nowhere (-1) and carry zero weight; FlashInfer masks -1 routes.
        pad_x = x.new_zeros((m, hidden_size))
        pad_ids = ids.new_full((m, top_k), -1)
        pad_w = weights.new_zeros((m, top_k))
        if num_tokens:
            pad_x[:num_tokens].copy_(x)
            pad_ids[:num_tokens].copy_(ids)
            pad_w[:num_tokens].copy_(weights)
        x, ids, weights = pad_x, pad_ids, pad_w

    deep_gemm.mega_moe_pre_dispatch_sm90(
        x,
        ids,
        weights,
        buf.x,
        buf.x_sf,
        buf.topk_idx,
        buf.topk_weights,
        num_tokens=m,
        group_size=_DISPATCH_GROUP_SIZE,
        routed_scaling_factor=routed_scaling_factor,
    )
    buf.commit_staged(m)
    y = torch.empty((m, hidden_size), dtype=torch.bfloat16, device=hidden_states.device)
    fm.nvfp4_mega_moe(
        y,
        experts.mega_l1_weights,
        experts.mega_l2_weights,
        buf,
        num_tokens=m,
        l1_global_scales=experts.mega_l1_global_scales,
        l2_global_scales=experts.mega_l2_global_scales,
        kernel_family=envs.SGLANG_FLASHINFER_MEGAMOE_SM90_KERNEL_FAMILY.get(),
    )
    return y[:num_tokens]
