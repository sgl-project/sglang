"""DeepSeek-V4.1's span: the whole layer through the fused mHC boundary, as the K1 + K2 launches. It takes and returns
exactly what ``forward_hc_pre_from_prev_fused_boundary`` does: (hidden, next pre, pending post); a deferred post stays
pending for the next layer's launch."""

from __future__ import annotations

import torch

# the SGLang names this module reads (pinned by test_mono_contract)
SGLANG_READS = (
    "sglang.srt.model_executor.forward_context.get_attn_backend",
    "sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc.forward_hc_pre_from_prev_fused_boundary",
)


def run_layer(
    route,
    layer,
    positions,
    hidden_states,
    input_ids,
    forward_batch,
    input_ids_global,
    prev_pre,
    pending_post,
    defer_post: bool,
):
    if route.checker is not None:
        return _checked(
            route,
            layer,
            positions,
            hidden_states,
            input_ids,
            forward_batch,
            input_ids_global,
            prev_pre,
            pending_post,
            defer_post,
        )
    if pending_post is None:
        pending_post = (None, hidden_states, None, None)
    return _whole(layer, route, forward_batch, prev_pre, pending_post, defer_post)


def _checked(
    route,
    layer,
    positions,
    hidden_states,
    input_ids,
    forward_batch,
    input_ids_global,
    prev_pre,
    pending_post,
    defer_post: bool,
):
    """Check mode: the module path on copies of the layer's inputs, then the kernels; logs the layer output's and the
    next pre's max |diff| (a deferred post is applied to compare)."""
    from sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc import (
        forward_hc_pre_from_prev_fused_boundary,
    )

    def copy(t):
        return t.clone() if isinstance(t, torch.Tensor) else t

    ref = forward_hc_pre_from_prev_fused_boundary(
        layer,
        positions,
        copy(hidden_states),
        input_ids,
        forward_batch,
        input_ids_global,
        copy(prev_pre),
        None if pending_post is None else tuple(copy(t) for t in pending_post),
        defer_post,
    )
    out = run_layer(
        route._replace(checker=None),
        layer,
        positions,
        hidden_states,
        input_ids,
        forward_batch,
        input_ids_global,
        prev_pre,
        pending_post,
        defer_post,
    )

    def hidden(result):
        h, _, pending = result
        return layer.hc_post(*(copy(t) for t in pending)) if pending is not None else h

    route.checker.compare(
        layer.layer_id,
        route.step.rows,
        {"hidden": (hidden(out), hidden(ref)), "pre": (out[1], ref[1])},
    )
    return out


def _whole(layer, route, forward_batch, prev_pre, pending_post, defer_post: bool):
    """The whole layer (attention seam .. MoE all-reduce) as the mono K1 + K2 launches; the kernels write the
    layer's KV themselves. A materialized residual (no pending post) enters as the identity fold: x = 0, post = 0,
    comb = I."""
    from sglang.kernels.ops.mono.deepseek_v41_gfx950.spec import whole_layer_meta

    kernel, step, cache = route.kernel, route.step, route.cache
    x, residual, post, comb = pending_post
    m, hc = residual.shape[0], residual.shape[1]
    if x is None:
        x = residual.new_zeros(m, residual.shape[-1])
        post = torch.zeros(m, hc, dtype=torch.float32, device=residual.device)
        comb = (
            torch.eye(hc, dtype=torch.float32, device=residual.device)
            .expand(m, hc, hc)
            .contiguous()
        )
    if prev_pre is None:
        # layer 0: SGLang's boundary collapses to stream 0 without a previous pre; a one-hot pre is that, exactly
        prev_pre = torch.zeros(m, hc, dtype=torch.float32, device=residual.device)
        prev_pre[:, 0] = 1.0
    ratio = kernel.packed.weights[layer.layer_id].attn.ratio
    meta = whole_layer_meta(
        layer.layer_id,
        ratio,
        forward_batch,
        m,
        cache,
        kernel.packed.runner.record,
        kernel.packed.runner.comp_record,
        swa_width=kernel.packed.runner.swa_width,
    )
    between = None
    if layer.layer_id in kernel.packed.between:
        between = _indexer_between(
            layer, kernel.packed.runner, forward_batch, m, ratio, cache
        )
    out, res_out, post_out, comb_out, pre_out = kernel.launch_whole(
        step,
        layer.layer_id,
        x.contiguous(),
        residual,
        post.reshape(m, hc, 1).contiguous(),
        comb.contiguous(),
        prev_pre.contiguous(),
        forward_batch.positions[:m].contiguous(),
        meta,
        between=between,
    )
    post_out = post_out.view(m, -1)
    if defer_post:
        return None, pre_out, (out, res_out, post_out, comb_out)
    return layer.hc_post(out, res_out, post_out, comb_out), pre_out, None


def _indexer_between(layer, runner, forward_batch, m: int, ratio: int, cache: dict):
    """The layer's compressor and indexer between the whole-layer launches, in SGLang's serial order, on the bf16
    attention input and q_lora K1 exported; returns the fresh top-k the key table is rebuilt from. The compressor is
    SGLang's; the indexer is SGLang's decode indexer with the K2i scorer (``indexer_hook``) unless K2A=0."""
    from sglang.kernels.ops.mono.deepseek_v41_gfx950 import indexer_hook
    from sglang.srt.model_executor.forward_context import get_attn_backend

    attn = layer.self_attn

    def between():
        backend = get_attn_backend()
        x, q_lora, positions = runner.xn[:m], runner.ql[:m], forward_batch.positions[:m]
        k2i = indexer_hook.enabled() and attn.indexer is not None
        backend.forward_low_ratio_sources(
            layer=attn,
            x=x,
            q_lora=q_lora,
            positions=positions,
            forward_batch=forward_batch,
            run_indexer=not k2i,
        )
        if k2i:
            if "k2i_plans" not in cache:
                cache["k2i_plans"] = indexer_hook.step_plans(backend, forward_batch)
            pos = getattr(backend.forward_metadata, "low_ratio_pos_i64", None)
            if pos is None or pos.shape[0] != m:
                pos = positions
            indexer_hook.index_topk_decode(
                backend, attn, x, q_lora, pos, forward_batch, cache["k2i_plans"]
            )
        return backend.forward_metadata.core_attn_metadata.sparse_raw_indices(ratio)[:m]

    return between
