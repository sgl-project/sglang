"""Optional Muse Spark MegaMoE adapter for sequence-sharded BF16 prefill.

The caller selects active layernorm SP. Decode continues to use the canonical
TRTLLM parameters. DeepGEMM's ``weight_layout="trtllm"`` reads the same
final weight and scale storage directly. Both kernels apply Muse's
per-expert norm before the route-weighted sum, using DeepGEMM.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_MEGAMOE_SYMM_BUFFERS: dict = {}


def _dimensions(layer):
    if hasattr(layer, "global_num_experts"):
        return (
            layer.global_num_experts,
            layer.hidden_size,
            layer.intermediate_size_per_partition,
        )
    return layer.num_experts, layer.expert_hidden_size, layer.intermediate_size


def _gemm_shapes(hidden, intermediate):
    """(mn, k) for L1 and L2."""
    return (2 * intermediate, hidden), (hidden, intermediate)


def _check_canonical_weights(layer, mma_type):
    experts, hidden, intermediate = _dimensions(layer)
    local = layer.num_local_experts
    packed = 2 if mma_type == "nvfp4xnvfp4" else 1
    (mn1, k1), (mn2, k2) = _gemm_shapes(hidden, intermediate)
    if (
        tuple(layer.gate_up_proj.shape) != (local, mn1, k1 // packed)
        or tuple(layer.down_proj.shape) != (local, mn2, k2 // packed)
        or local <= 0
        or experts % local
    ):
        raise ValueError("Muse MegaMoE requires canonical local expert weight shapes.")
    if getattr(layer, "load_up_proj_weight_first", False):
        raise ValueError("Muse MegaMoE requires canonical [gate; up] weight order.")
    if getattr(layer, "gemm1_clamp_limit", None) is not None:
        raise ValueError(
            "Muse MegaMoE does not support this checkpoint's SwiGLU clamp."
        )


def _same_weight_storage(view, parameter):
    return (
        view.is_contiguous()
        and parameter.is_contiguous()
        and view.untyped_storage().data_ptr() == parameter.untyped_storage().data_ptr()
        and view.data_ptr() == parameter.data_ptr()
        and view.numel() * view.element_size()
        == parameter.numel() * parameter.element_size()
    )


@torch.no_grad()
def bind_megamoe_shared_weights(layer, mma_type) -> None:
    """Bind exact final TRT-LLM byte ranges; views never repack weights.

    Run after TRT-LLM's final row shuffle (and BF16 BlockMajorK conversion).
    The logical 3D view describes dimensions only; the kernel knows the physical
    TRT layout. Repeated preparation verifies that no weight was replaced.
    """
    if not getattr(layer, "_megamoe_metadata_prepared", False):
        raise RuntimeError("Prepare Muse MegaMoE metadata before TRT alignment.")
    _, hidden, intermediate = _dimensions(layer)
    local = layer.num_local_experts
    if mma_type not in ("nvfp4xnvfp4", "bf16xbf16"):
        raise ValueError("Unsupported Muse MegaMoE weight type.")
    packed = 2 if mma_type == "nvfp4xnvfp4" else 1
    (mn1, k1), (mn2, k2) = _gemm_shapes(hidden, intermediate)
    if getattr(layer, "_megamoe_weights_built", False):
        l1, l2 = layer.megamoe_l1_weights, layer.megamoe_l2_weights
        if isinstance(l1, tuple):
            l1, l2 = l1[0], l2[0]
    else:
        l1, l2 = layer.gate_up_proj.detach(), layer.down_proj.detach()
        if packed == 2:
            l1, l2 = l1.view(torch.int8), l2.view(torch.int8)
        l1 = l1.view(local, mn1, k1 // packed)
        l2 = l2.view(local, mn2, k2 // packed)
    if not all(
        _same_weight_storage(view, parameter)
        for view, parameter in ((l1, layer.gate_up_proj), (l2, layer.down_proj))
    ):
        raise RuntimeError("Muse MegaMoE must alias the final TRT-LLM weights.")
    extra = []
    if packed == 2:
        # TRT-LLM's own scale tensors, one UE4M3 word per 64 elements of k: its
        # tensor maps carry TRT's axis order, so nothing is copied or permuted,
        # and a padded one fails the view.
        sf1 = (
            layer.gate_up_proj_weight_scale.detach()
            .reshape(local, -1)
            .view(torch.int32)
        )
        sf2 = layer.down_proj_weight_scale.detach().reshape(local, -1).view(torch.int32)
        l1 = (l1, sf1.view(local, mn1, k1 // 64))
        l2 = (l2, sf2.view(local, mn2, k2 // 64))
        extra = [
            layer.megamoe_l1_alphas,
            layer.megamoe_l2_alphas,
            layer.megamoe_l2_act_scales,
        ]
    layer.megamoe_l1_weights, layer.megamoe_l2_weights = l1, l2
    layer._megamoe_mma_type = mma_type
    allocations = {
        t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
        for t in extra
        if t is not None
    }
    layer.megamoe_weight_bytes = sum(allocations.values())
    layer._megamoe_weights_built = True
    logger.debug(
        "Muse %s MegaMoE shares TRT weights; extra metadata %.3f GiB",
        mma_type,
        layer.megamoe_weight_bytes / (1 << 30),
    )


@torch.no_grad()
def build_nvfp4_megamoe_weights(layer) -> None:
    """Capture accumulator alphas before TRT-LLM alignment rewrites their source.

    Scales are not captured: `bind_megamoe_shared_weights` reads TRT-LLM's own.
    Dequant alphas stay on their own accumulator, since folding small GEMM1
    alphas into FP8 block scales can flush values to zero.
    """
    if getattr(layer, "_megamoe_metadata_prepared", False):
        return

    mma_type = "nvfp4xnvfp4"
    _check_canonical_weights(layer, mma_type)
    w13, w2 = layer.gate_up_proj.detach(), layer.down_proj.detach()
    sf13, sf2 = (
        layer.gate_up_proj_weight_scale.detach(),
        layer.down_proj_weight_scale.detach(),
    )
    if (
        w13.dtype not in (torch.uint8, torch.int8)
        or w2.dtype not in (torch.uint8, torch.int8)
        or sf13.dtype != torch.float8_e4m3fn
        or sf2.dtype != torch.float8_e4m3fn
    ):
        raise ValueError("Muse MegaMoE expects packed NVFP4 and E4M3 block scales.")
    local = layer.num_local_experts
    s13 = layer.gate_up_proj_weight_scale_2.detach().float()
    if s13.ndim == 1:
        s13 = s13[:, None].expand(-1, 2)
    if s13.shape != (local, 2):
        raise ValueError(
            "Muse MegaMoE requires one gate/up alpha pair per local expert."
        )
    s13 = s13.clone().contiguous()
    s2 = layer.down_proj_weight_scale_2.detach().float().flatten().clone()
    # Read the original calibration, not w2_input_scale_quant: ModelOpt may
    # neutralize the latter when per-token activation quantization is enabled.
    in2_inv = layer.down_proj_input_scale.detach().float().flatten()
    global_experts, _, _ = _dimensions(layer)
    if in2_inv.numel() == global_experts and global_experts != local:
        offset = layer.local_expert_offset
        in2_inv = in2_inv[offset : offset + local]
    if s2.numel() != local or in2_inv.numel() != local:
        raise ValueError("Muse MegaMoE requires one GEMM2 scale per local expert.")
    act_scales = torch.where(in2_inv > 0, in2_inv.reciprocal(), 1.0)
    if bool(torch.all(act_scales == 1)):
        act_scales = None
    else:
        act_scales = act_scales.contiguous()
        s2 = s2 / act_scales

    layer.megamoe_l1_alphas = s13
    layer.megamoe_l2_alphas = s2.contiguous()
    layer.megamoe_l2_act_scales = act_scales
    layer._megamoe_metadata_prepared = True


@torch.no_grad()
def build_bf16_megamoe_weights(layer) -> None:
    """Validate canonical BF16 experts before TRT-LLM shuffles them in place."""
    if getattr(layer, "_megamoe_metadata_prepared", False):
        return
    _check_canonical_weights(layer, "bf16xbf16")
    if (
        layer.gate_up_proj.dtype != torch.bfloat16
        or layer.down_proj.dtype != torch.bfloat16
    ):
        raise ValueError("Muse BF16 MegaMoE requires BF16 checkpoint expert weights.")
    layer._megamoe_metadata_prepared = True


def _is_capturing():
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _is_weight_first(norm):
    if norm is None:
        return False
    from sglang.srt.layers.muse_spark.muse_spark_layers import (
        MuseSpark1xRMSNormWithInputScale,
    )

    return isinstance(norm, MuseSpark1xRMSNormWithInputScale)


@torch.no_grad()
def prepare_megamoe_norm(layer, post_expert_norm) -> None:
    """Cache FP32 effective gain after norm weights load, outside graph capture.

    May be called from post-load preparation or the first eager SP prefill.
    Cache by norm module, not a temporary gain tensor: zero-centered norms can
    return a fresh effective-weight tensor on every call. Explicitly call this
    helper again after any later norm-weight update.
    """
    if _is_capturing():
        raise RuntimeError("Prepare Muse MegaMoE norm before CUDA graph capture.")
    weight = None
    if post_expert_norm is not None:
        weight = post_expert_norm._get_effective_weight()
        if weight is not None:
            _, hidden, _ = _dimensions(layer)
            if weight.numel() != hidden:
                raise ValueError(
                    "Muse MegaMoE norm gain must match the expert hidden size."
                )
            weight = (
                weight.detach()
                .to(device=layer.gate_up_proj.device, dtype=torch.float32)
                .reshape(-1)
                .clone()
                .contiguous()
            )
    # A tuple keeps the existing norm from registering as another child module.
    layer._megamoe_norm_cache = (
        post_expert_norm,
        weight,
        float(getattr(post_expert_norm, "eps", 1e-8)),
        _is_weight_first(post_expert_norm),
    )


def _get_ep_group():
    from sglang.srt.runtime_context import get_parallel

    return get_parallel().moe_ep_group.device_group


def _token_capacity():
    from sglang.srt.environ import envs

    return envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()


def _get_symm_buffer(layer, group, capacity, device):
    import deep_gemm

    experts, hidden, intermediate = _dimensions(layer)
    if (
        layer.num_local_experts * group.size() != experts
        or layer.local_expert_offset != group.rank() * layer.num_local_experts
    ):
        raise ValueError(
            "Muse MegaMoE requires contiguous expert ownership in its EP group."
        )
    key = (
        id(group),
        group.size(),
        group.rank(),
        device,
        layer._megamoe_mma_type,
        experts,
        capacity,
        layer.top_k,
        hidden,
        intermediate,
    )
    buf = _MEGAMOE_SYMM_BUFFERS.get(key)
    if buf is None:
        if _is_capturing():
            raise RuntimeError(
                "Warm Muse MegaMoE symmetric buffers before graph capture."
            )
        # Capacity is a common configured bound, never this rank's current row
        # count. All ranks must rendezvous with the same allocation shape.
        buf = deep_gemm.get_symm_buffer_for_mega_moe(
            group,
            experts,
            capacity,
            layer.top_k,
            hidden,
            intermediate,
            mma_type=layer._megamoe_mma_type,
            activation="swiglu",
        )
        _MEGAMOE_SYMM_BUFFERS[key] = buf
    return buf


@torch.no_grad()
def forward_megamoe(layer, hidden_states, topk_scores, topk_ids, post_expert_norm):
    """Dispatch sequence shards, run both GEMMs, normalize experts, then combine.

    Inputs and return value are BF16 sequence shards. Every EP rank calls even
    with zero local tokens. This does not select a backend or all-reduce output;
    the caller must invoke it only for active layernorm-SP prefill.
    """
    import deep_gemm

    if not getattr(layer, "_megamoe_weights_built", False):
        raise RuntimeError("Prepare Muse MegaMoE weights before SP prefill.")
    _, hidden, _ = _dimensions(layer)
    if (
        hidden_states.ndim != 2
        or hidden_states.dtype != torch.bfloat16
        or hidden_states.shape[1] != hidden
    ):
        raise ValueError(
            "Muse MegaMoE requires a BF16 sequence shard of expert hidden width."
        )
    n = hidden_states.shape[0]
    if (
        topk_ids.shape != (n, layer.top_k)
        or topk_scores.shape != topk_ids.shape
        or topk_ids.dtype not in (torch.int32, torch.int64)
        or topk_ids.device != hidden_states.device
        or topk_scores.device != hidden_states.device
    ):
        raise ValueError(
            "Muse MegaMoE requires matching per-token top-k ids and scores."
        )
    capacity = _token_capacity()
    if capacity <= 0 or n > capacity:
        raise ValueError(
            f"Muse MegaMoE sequence shard has {n} tokens, exceeding the per-rank "
            f"capacity {capacity}; size the configured bound before SP prefill."
        )
    cached = getattr(layer, "_megamoe_norm_cache", None)
    if cached is None or cached[0] is not post_expert_norm:
        prepare_megamoe_norm(layer, post_expert_norm)
        cached = layer._megamoe_norm_cache
    buf = _get_symm_buffer(layer, _get_ep_group(), capacity, hidden_states.device)
    mma_type = layer._megamoe_mma_type
    if mma_type == "nvfp4xnvfp4" and n:
        # Quantization also invalidates every padded routing row, including
        # rows left by a previous larger batch in this shared symmetric buffer.
        deep_gemm.mega_moe_pre_dispatch(
            hidden_states.contiguous(),
            topk_ids.to(torch.int32).contiguous(),
            topk_scores.float().contiguous(),
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            n,
            group_size=16,
            mma_type=mma_type,
            buf_x_scales=buf.x_scales,
        )
    else:
        # DeepGEMM's quantizing pre-dispatch API explicitly rejects BF16 MMA.
        # copy_ handles route dtype conversion and noncontiguous input shards.
        if n:
            buf.x[:n].copy_(hidden_states)
            buf.topk_idx[:n].copy_(topk_ids)
            buf.topk_weights[:n].copy_(topk_scores)
        buf.topk_idx[n:].fill_(-1)
        buf.topk_weights[n:].zero_()
    # The binding requires a non-null output pointer, even for an idle rank.
    y = torch.empty(
        (max(n, 1), hidden), dtype=torch.bfloat16, device=hidden_states.device
    )
    kwargs = dict(
        activation="swiglu",
        weight_layout="trtllm",
        fast_math=True,
        post_expert_norm=post_expert_norm is not None,
        post_expert_norm_weight=cached[1],
        post_expert_norm_eps=cached[2],
        post_expert_norm_weight_first=cached[3],
    )
    if mma_type == "nvfp4xnvfp4":
        kwargs.update(
            use_x_scales=True,
            l1_alphas=layer.megamoe_l1_alphas,
            l2_alphas=layer.megamoe_l2_alphas,
            l2_act_scales=layer.megamoe_l2_act_scales,
        )
        kernel = deep_gemm.nvfp4_mega_moe
    else:
        kernel = deep_gemm.bf16_mega_moe
    kernel(y, layer.megamoe_l1_weights, layer.megamoe_l2_weights, buf, **kwargs)
    return y[:n]
