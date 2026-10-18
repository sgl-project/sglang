"""Opt-in Muse Spark model glue for the spec-decode verify Helion megakernels."""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any

import torch

from sglang.kernels.ops.muse_spark.fused_norm_residual import (
    FusedPostNormResidualPreNorm,
)
from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.layers.cp.utils import is_cp_active
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.muse_spark.muse_spark_expert_tp import VERIFY_TOKENS
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer
from sglang.srt.utils import log_info_on_rank0

logger = logging.getLogger(__name__)

DERIVED_WEIGHT_CACHE_ERROR = (
    "Online weight updates are not supported with the Muse Spark Helion verify "
    "megakernels: their weight banks and the CUDA graphs that read them are "
    "derived from the loaded weights. Restart with expert parallelism "
    "(--ep-size equal to --tp-size) to update weights online."
)


@functools.cache
def _log_engaged(tokens: int) -> None:
    log_info_on_rank0(
        logger,
        f"Muse Spark T={tokens} verify runs on the Helion TP4 NVFP4 megakernels",
    )


@functools.cache
def require_helion_features() -> None:
    """Fail early unless Helion is 1.5.0 or newer and has the features the
    verify megakernels use: cross-loop pipelines and explicit PDL."""
    import helion
    import helion.language as hl
    from packaging.version import Version

    from sglang.kernels.ops.muse_spark.helion_runtime import _base_package_version

    helion_version = _base_package_version("helion")
    missing = [
        f"hl.{name}"
        for name in ("pdl_wait", "pdl_launch_dependents")
        if not hasattr(hl, name)
    ]
    if "cross_loop_pipeline" not in inspect.signature(helion.Config).parameters:
        missing.append("Config(cross_loop_pipeline=...)")
    if helion_version is None or Version(helion_version) < Version("1.5.0"):
        missing.insert(
            0, f"version >=1.5.0 (found {helion_version or 'not installed'})"
        )
    if missing:
        raise RuntimeError(
            "the Muse Spark Helion verify megakernels need a Helion build with "
            + ", ".join(missing)
        )


def megakernel_verify_enabled(layer, forward_batch, tensor: torch.Tensor) -> bool:
    """Return whether this call runs the verify layer on the Helion
    megakernels; the model binds a verify bank only on eligible servers."""

    return (
        layer.mlp.experts.helion_verify_bank is not None
        and tensor.ndim == 2
        and tensor.shape[0] in VERIFY_TOKENS
        and tensor.dtype == torch.bfloat16
        and forward_batch is not None
        and forward_batch.forward_mode.is_target_verify()
        and forward_batch.batch_size == 1
        and not get_attn_backend().forward_metadata.is_ragged_verify
    )


def _attention_inputs(attention: Any, positions: torch.Tensor, forward_batch: Any):
    from sglang.kernels.ops.muse_spark.helion_megakernels.megakernel_verify import (
        AttentionInputs,
    )

    backend = get_attn_backend()
    pool = get_token_to_kv_pool()
    # The attention megakernel serves the sliding-window layers only and
    # hardcodes the unit-scale FP8 KV layout, the 128-token page and the
    # 2048-token window.
    if not (
        isinstance(backend, TRTLLMHAAttnBackend)
        and not backend.is_xqa_impl
        and backend.data_type == torch.float8_e4m3fn
        and pool.dtype == backend.data_type
        and backend.page_size == 128
        and attention.attn.k_scale is None
        and attention.attn.v_scale is None
        and attention.use_rope
        and attention.attn.sliding_window_size == 2047
        and not getattr(attention, "use_attn_temperature_tuning", False)
        and forward_batch.dcp_kv_mask is None
        and not is_cp_active(forward_batch)
    ):
        raise RuntimeError(
            "the Helion attention megakernel requires TRTLLM-GEN with an "
            "unscaled FP8 KV cache, page size 128 and a 2048-token sliding "
            "window with RoPE"
        )
    k_cache = pool.get_key_buffer(attention.layer_idx)
    v_cache = pool.get_value_buffer(attention.layer_idx)
    page_width = backend.page_size * attention.num_kv_heads * attention.head_dim
    if k_cache.numel() % page_width or v_cache.numel() % page_width:
        raise RuntimeError(
            "Helion verify received a KV cache that cannot be represented as "
            f"whole page rows of width {page_width}"
        )
    return AttentionInputs(
        k_cache=k_cache.view(-1, page_width),
        v_cache=v_cache.view(-1, page_width),
        cache_loc=backend._get_layer_cache_loc(attention.attn, forward_batch),
        block_tables=backend._get_layer_page_table(attention.attn, forward_batch),
        seq_lens=backend.forward_metadata.cache_seqlens_int32,
        positions=positions,
        cos_sin_cache=attention.rotary_emb.cos_sin_cache,
    )


def _check_model_dims(layer: Any) -> None:
    """The megakernels specialize Muse Spark 1.2's per-rank TP4 shapes; any other
    shape must fail here rather than compute garbage."""
    from sglang.kernels.ops.muse_spark.helion_megakernels import (
        attention_boundary_verify_megakernel as boundary_mk,
    )
    from sglang.kernels.ops.muse_spark.helion_megakernels import (
        attention_verify_megakernel as attention_mk,
    )

    attn, mlp = layer.self_attn, layer.mlp
    got = {
        "q heads per rank": attn.num_heads,
        "kv heads per rank": attn.num_kv_heads,
        "head_dim": attn.head_dim,
        "hidden_size": mlp.hidden_size,
        "moe_hidden_size": mlp.expert_hidden_size,
        "moe_intermediate_size": mlp.intermediate_size,
        "num_local_experts": mlp.num_experts,
        "num_experts_per_tok": mlp.top_k,
    }
    want = {
        "q heads per rank": attention_mk.NUM_Q_HEADS,
        "kv heads per rank": attention_mk.NUM_KV_HEADS,
        "head_dim": attention_mk.HEAD_DIM,
        "hidden_size": boundary_mk.HIDDEN,
        "moe_hidden_size": boundary_mk.EXPERT_HIDDEN,
        "moe_intermediate_size": boundary_mk.EXPERT_HIDDEN,
        "num_local_experts": boundary_mk.NUM_EXPERTS,
        "num_experts_per_tok": boundary_mk.TOPK,
    }
    bad = [f"{k} {got[k]} (needs {want[k]})" for k in want if got[k] != want[k]]
    if bad:
        raise RuntimeError(
            "the Helion verify megakernels are specialized for Muse Spark 1.2 on "
            "TP4; this model has " + ", ".join(bad)
        )


def _layer_scalars(layer: Any):
    from sglang.kernels.ops.muse_spark.helion_megakernels.megakernel_verify import (
        LayerScalars,
    )

    mlp = layer.mlp
    experts = mlp.experts
    parallel = get_parallel()
    # moe_tp_rank resolves the MoE-TP group, so read it only under expert TP.
    if not (
        parallel.tp_size == 4
        and parallel.moe_ep_size == 1
        and parallel.moe_tp_size == 4
        and parallel.moe_tp_rank == get_parallel().tp_group.rank_in_group
        and not is_dp_attention_enabled()
        and mlp.pre_expert_proj is not None
        and mlp.pre_expert_norm is not None
        and experts.post_expert_norm is not None
        and experts.quant_config.use_per_token_activation
    ):
        raise RuntimeError(
            "the Helion verify megakernels require TP4 with expert TP4 (EP1), "
            "per-token NVFP4 activation scales and no DP attention"
        )
    _check_model_dims(layer)
    attn_boundary = layer._fused_attn_boundary
    # The last layer has no next pre-attn norm; its attention-input output is
    # discarded, so any [hidden] weight stands in.
    ffn_boundary = layer._fused_ffn_boundary or FusedPostNormResidualPreNorm(
        layer.post_feedforward_layernorm,
        layer.post_feedforward_residual_gate,
        layer.input_layernorm,
    )
    for op in (attn_boundary, ffn_boundary):
        op._ensure_cached_weights(layer.operator_dtype)
    scalars = LayerScalars(
        attn_scale=layer.self_attn.attn.scaling,
        attn_norm_eps=layer.self_attn.norm_eps,
        post_attn_eps=attn_boundary._post_norm.eps,
        pre_ffn_eps=attn_boundary._pre_norm.eps,
        pre_expert_eps=mlp.pre_expert_norm.eps,
        router_multiplier=mlp.router_multiplier,
        route_scale=mlp.route_scale,
        route_eps=mlp.route_eps,
        post_expert_eps=experts.post_expert_norm.eps,
        post_ffn_eps=ffn_boundary._post_norm.eps,
        next_pre_attn_eps=ffn_boundary._pre_norm.eps,
    )
    layer._derived_weight_cache_error = DERIVED_WEIGHT_CACHE_ERROR
    return attn_boundary, ffn_boundary, scalars


def _layer_weights(layer: Any, runner: Any, attn_boundary: Any, ffn_boundary: Any):
    from sglang.kernels.ops.muse_spark.helion_megakernels.megakernel_verify import (
        MoeVerifyWeights,
    )

    mlp = layer.mlp
    experts = mlp.experts
    experts.set_post_expert_norm(experts.post_expert_norm)
    if not experts._post_expert_norm_weight_first:
        raise RuntimeError(
            "the Helion verify MoE megakernel fuses only the weight-first "
            "post-expert norm"
        )
    rank, world = get_parallel().tp_group.rank_in_group, get_parallel().tp_size
    boundary = runner.attention_boundary.prepare_weights(
        o_proj=layer.self_attn.o_proj.weight,
        pre_expert=mlp.pre_expert_proj.weight,
        router=mlp.gate.weight,
        alpha=attn_boundary._cc_alpha,
        beta=attn_boundary._cc_beta,
        pre_ffn_norm=attn_boundary._cc_pre_weight,
        pre_expert_norm=experts._pre_norm_weight(mlp.pre_expert_norm),
        router_bias=mlp.gate.e_score_correction_bias,
        rank=rank,
        world=world,
    )
    nxt = layer._next_layer
    fuse_qkv = (
        layer._fused_ffn_boundary is not None
        and nxt is not None
        and nxt._megakernel_capable
        and nxt.mlp.experts.helion_verify_bank is not None
    )
    unpool_rows = mlp.hidden_size // world
    moe = MoeVerifyWeights(
        bank=experts.helion_verify_bank,
        output1_scale_scalar=experts.g1_scale_c,
        output1_scale_gate_scalar=experts.g1_alphas,
        output2_scale_scalar=experts.g2_alphas,
        post_expert_norm=experts._post_expert_norm_weight,
        unpool=mlp.post_expert_proj.weight[
            rank * unpool_rows : (rank + 1) * unpool_rows
        ],
        alpha=ffn_boundary._cc_alpha,
        beta=ffn_boundary._cc_beta,
        post_weight=ffn_boundary._cc_post_weight,
        pre_weight=ffn_boundary._cc_pre_weight,
        next_qkv=nxt.self_attn.qkv_proj.weight if fuse_qkv else None,
    )
    return boundary, moe


def _runner(scalars: Any, attn_input: torch.Tensor):
    from sglang.kernels.ops.muse_spark.helion_megakernels.megakernel_verify import (
        get_megakernel_verify,
    )

    return get_megakernel_verify(
        device=attn_input.device,
        group=get_parallel().tp_group.device_group,
        scalars=scalars,
        tokens=attn_input.shape[0],
    )


def try_helion_megakernel_forward(
    layer: Any,
    positions: torch.Tensor,
    stream: torch.Tensor,
    attn_input: torch.Tensor,
    forward_batch: Any,
) -> tuple[torch.Tensor, torch.Tensor | None] | None:
    """Run an eligible verify layer; return ``None`` for the stock path."""
    if not megakernel_verify_enabled(layer, forward_batch, attn_input):
        return None
    require_helion_features()
    params = getattr(layer, "_megakernel_verify_params", None)
    if params is None:
        attn_boundary, ffn_boundary, scalars = _layer_scalars(layer)
        runner = _runner(scalars, attn_input)
        params = layer._megakernel_verify_params = (
            scalars,
            *_layer_weights(layer, runner, attn_boundary, ffn_boundary),
        )
        _log_engaged(attn_input.shape[0])
    scalars, boundary, moe = params
    runner = _runner(scalars, attn_input)
    # The previous layer's tail computed this layer's QKV in this same forward.
    pending, layer._verify_qkv = layer._verify_qkv, None
    qkv = pending[1] if pending is not None and pending[0] is forward_batch else None
    if qkv is None:
        qkv, _ = layer.self_attn.qkv_proj(attn_input)
    if layer.self_attn.use_rope:
        gated = runner.sliding_attention(
            qkv=qkv,
            attention=_attention_inputs(layer.self_attn, positions, forward_batch),
        )
    else:
        # Global layers keep the stock attention: its KV-split decode scales with
        # the context, while the megakernel's one task per SM falls behind past
        # ~16K keys (B200, 80K keys: 69.6 us vs 32.8 us).
        gated = layer.self_attn.gated_attention(positions, qkv, forward_batch)
    stream, next_attn_input, next_qkv = runner.layer(
        gated=gated, stream=stream, attention_boundary=boundary, moe=moe
    )
    if (capturer := get_global_experts_capturer()) is not None:
        capturer.capture(layer_id=layer.layer_idx, topk_indices=runner.topk_ids)
    if next_qkv is not None:
        layer._next_layer._verify_qkv = (forward_batch, next_qkv)
    if layer._fused_ffn_boundary is None:
        return stream, None
    return stream, next_attn_input


__all__ = [
    "DERIVED_WEIGHT_CACHE_ERROR",
    "megakernel_verify_enabled",
    "require_helion_features",
    "try_helion_megakernel_forward",
]
