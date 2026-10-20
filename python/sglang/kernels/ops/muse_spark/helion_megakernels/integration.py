"""Opt-in Muse Spark model glue for the spec-decode verify Helion megakernels."""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any

import torch

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
    scalars = LayerScalars(
        attn_scale=layer.self_attn.attn.scaling,
        attn_norm_eps=layer.self_attn.norm_eps,
        pre_expert_eps=mlp.pre_expert_norm.eps,
        router_multiplier=mlp.router_multiplier,
        route_scale=mlp.route_scale,
        route_eps=mlp.route_eps,
        post_expert_eps=experts.post_expert_norm.eps,
    )
    layer._derived_weight_cache_error = DERIVED_WEIGHT_CACHE_ERROR
    return scalars


def _layer_weights(layer: Any, runner: Any):
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
        pre_expert=mlp.pre_expert_proj.weight,
        router=mlp.gate.weight,
        pre_expert_norm=experts._pre_norm_weight(mlp.pre_expert_norm),
        router_bias=mlp.gate.e_score_correction_bias,
        rank=rank,
        world=world,
    )
    moe = MoeVerifyWeights(
        bank=experts.helion_verify_bank,
        output1_scale_scalar=experts.g1_scale_c,
        output1_scale_gate_scalar=experts.g1_alphas,
        output2_scale_scalar=experts.g2_alphas,
        post_expert_norm=experts._post_expert_norm_weight,
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
        scalars = _layer_scalars(layer)
        runner = _runner(scalars, attn_input)
        params = layer._megakernel_verify_params = (
            scalars,
            *_layer_weights(layer, runner),
        )
        _log_engaged(attn_input.shape[0])
    scalars, boundary, moe = params
    runner = _runner(scalars, attn_input)
    attention = layer.self_attn
    qkv, _ = attention.qkv_proj(attn_input)
    if attention.use_rope:
        gated = runner.sliding_attention(
            qkv=qkv,
            attention=_attention_inputs(attention, positions, forward_batch),
        )
    else:
        # Global layers keep the stock attention: its KV-split decode scales with
        # the context, while the megakernel's one task per SM falls behind past
        # ~16K keys (B200, 80K keys: 69.6 us vs 32.8 us).
        gated = attention.gated_attention(positions, qkv, forward_batch)
    # The stock O-projection (its GEMM streams the weights in under PDL while the
    # attention runs), all-reduce and post-attention / pre-FFN norms.
    attn_output, _ = attention.o_proj(gated)
    ffn_input, stream, _ = layer._fused_attn_boundary(
        stream, attn_output, branch_dtype=layer.operator_dtype
    )
    expert_output = runner.experts(
        ffn_in=ffn_input, attention_boundary=boundary, moe=moe
    )
    if (capturer := get_global_experts_capturer()) is not None:
        capturer.capture(layer_id=layer.layer_idx, topk_indices=runner.topk_ids)
    # Expert TP already reduced the routes; the stock unpooler and FFN boundary
    # follow (the unpooler GEMM streams its weights in while the MoE finishes).
    ffn_output = layer.mlp._reduce_and_unpool(expert_output, already_reduced=True)
    if ffn_output.shape[-1] != stream.shape[-1]:
        next_attn_input, stream = layer._fused_ffn_boundary.forward_with_ag(
            stream, ffn_output, get_parallel().tp_group.ca_comm
        )
        return stream, next_attn_input
    if layer._fused_ffn_boundary is None:
        return layer.post_feedforward_residual_gate(
            stream, layer.post_feedforward_layernorm(ffn_output)
        ), None
    next_attn_input, stream, _ = layer._fused_ffn_boundary(
        stream, ffn_output, branch_dtype=layer.operator_dtype
    )
    return stream, next_attn_input


__all__ = [
    "DERIVED_WEIGHT_CACHE_ERROR",
    "megakernel_verify_enabled",
    "require_helion_features",
    "try_helion_megakernel_forward",
]
