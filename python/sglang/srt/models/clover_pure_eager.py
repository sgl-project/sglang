# SPDX-License-Identifier: Apache-2.0
# Adapted from Diffuspace/Clover-35B-A3B-Preview and the Qwen/Hugging Face implementation.
"""BF16 eager reference arithmetic over the SGLang TP=1 weight and state pools."""

import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear.clover_pure_memory import bd_gdn_local_memory
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_req_to_token_pool,
    get_token_to_kv_pool,
)


def norm(x, module, eps, gate=None):
    y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    w = module.weight.float()
    if gate is None:
        y = y * (1 + w)
    else:
        y = y * w * F.silu(gate.float())
    return y.to(x.dtype)


def rotary(x, positions, config):
    d = int(config.head_dim * config.partial_rotary_factor)
    inv = 1.0 / config.rope_parameters["rope_theta"] ** (
        torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d
    )
    phase = positions.float().view(1, 1, -1, 1) * inv
    cos, sin = phase.cos().to(x.dtype), phase.sin().to(x.dtype)
    a, b = x[..., :d].chunk(2, -1)
    return torch.cat((a * cos - b * sin, b * cos + a * sin, x[..., d:]), -1)


def full_attention(layer, x, batch, preview, padded_prefix=None):
    c = layer.config
    q_weight, k_weight, v_weight = layer.qkv_proj.weight.split(
        (2 * layer.q_size, layer.kv_size, layer.kv_size)
    )
    q_gate = F.linear(x, q_weight).view(1, -1, layer.num_heads, 2 * c.head_dim)
    q, gate = q_gate.chunk(2, -1)
    k = F.linear(x, k_weight).view(1, -1, layer.num_kv_heads, c.head_dim)
    v = F.linear(x, v_weight).view_as(k).transpose(1, 2)
    q = rotary(
        norm(q, layer.q_norm, c.rms_norm_eps).transpose(1, 2), batch.positions, c
    )
    k = rotary(
        norm(k, layer.k_norm, c.rms_norm_eps).transpose(1, 2), batch.positions, c
    )
    pool = get_token_to_kv_pool()
    prefix = batch.extend_prefix_lens_cpu[0]
    if not preview:
        pool.set_kv_buffer(
            layer.attn,
            KVWriteLoc.for_layer(batch, layer.attn),
            k[0].transpose(0, 1).contiguous(),
            v[0].transpose(0, 1).contiguous(),
        )
    if padded_prefix is not None:
        # Repeat a valid prefix page for padding. The mask excludes those keys from the softmax. Positions use the true prefix.
        pages, prefix_valid = padded_prefix
        kb, vb = pool.get_kv_buffer(layer.layer_id)
        k = torch.cat((kb[pages].transpose(0, 1)[None], k), 2)
        v = torch.cat((vb[pages].transpose(0, 1)[None], v), 2)
        mask = torch.cat((prefix_valid, prefix_valid.new_ones(x.shape[1])))[
            None, None, None
        ]
    else:
        if prefix:
            kb, vb = pool.get_kv_buffer(layer.layer_id)
            indices = (
                get_req_to_token_pool()
                .req_to_token[batch.req_pool_indices[0], :prefix]
                .long()
            )
            k = torch.cat((kb[indices].transpose(0, 1)[None], k), 2)
            v = torch.cat((vb[indices].transpose(0, 1)[None], v), 2)
        mask = torch.ones(
            (1, 1, x.shape[1], k.shape[2]), device=x.device, dtype=torch.bool
        )
    if not preview:
        mask &= torch.arange(k.shape[2], device=x.device) <= batch.positions.view(-1, 1)
    repeats = layer.num_heads // layer.num_kv_heads
    y = F.scaled_dot_product_attention(
        q,
        k.repeat_interleave(repeats, 1),
        v.repeat_interleave(repeats, 1),
        attn_mask=mask,
    )
    y = (
        y.transpose(1, 2).reshape(1, x.shape[1], -1)
        * gate.reshape(1, x.shape[1], -1).sigmoid()
    )
    return F.linear(y, layer.o_proj.weight)


def _causal_scan(q, k, v, g, beta, state):
    values = []
    for i in range(q.shape[1]):
        decayed = state * g[:, i].exp()[..., None, None]
        remembered = torch.einsum("bhvk,bhk->bhv", decayed, k[:, i])
        delta = (v[:, i].float() - remembered) * beta[:, i].float()[..., None]
        state = decayed + delta[..., None] * k[:, i, :, None, :]
        values.append(torch.einsum("bhvk,bhk->bhv", state, q[:, i]))
    return torch.stack(values, 1).to(v.dtype), state


class CausalReplayGraph:
    """One fixed-shape recurrence graph reused across the model's GDN layers."""

    def __init__(self):
        self.graph = None

    def run(self, *inputs):
        if self.graph is None:
            self.buffers = tuple(torch.empty_like(x) for x in inputs)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = _causal_scan(*self.buffers)
        for dst, src in zip(self.buffers, inputs):
            dst.copy_(src)
        self.graph.replay()
        return self.output


def linear_attention(layer, x, batch, preview, slots=None, replay_graph=None):
    t = x.shape[1]
    width = layer.conv_kernel_size
    qw, zw = layer.in_proj_qkvz.weight.split((layer.conv_dim, layer.value_dim))
    bw, aw = layer.in_proj_ba.weight.chunk(2)
    raw = F.linear(x, qw)
    cache = get_req_to_token_pool().mamba2_layer_cache(layer.layer_id)
    if slots is None:
        slots = get_attn_backend().linear_attn_backend.forward_metadata.mamba_cache_indices.long()
    conv = cache.conv[0][slots]
    state = cache.temporal[slots]
    if batch.extend_prefix_lens_cpu[0] == 0:
        conv.zero_()
        state.zero_()
    joined = torch.cat((conv, raw.transpose(1, 2)), -1)
    weights = layer.conv1d.weight
    if preview:
        mixed = F.conv1d(joined.float(), weights.float(), groups=layer.conv_dim)
        for d in range(1, min(width, t)):
            right = layer.conv_right_weight[:, 0, d - 1].float()
            mixed[:, :, :-d] += (
                raw[:, d:].transpose(1, 2).float() * right[None, :, None]
            )
    else:
        # Contiguous [B,C,T] matches causal_conv_silu and its downstream GDN reductions.
        mixed = torch.zeros(
            (1, layer.conv_dim, t), device=x.device, dtype=torch.float32
        )
        for tap in range(width):
            mixed.add_(
                (joined[..., tap : tap + t] * weights[:, 0, tap][None, :, None]).float()
            )
        cache.conv[0][slots] = joined[..., -(width - 1) :]
    q, k, v = (
        F.silu(mixed)
        .to(x.dtype)
        .transpose(1, 2)
        .split((layer.key_dim, layer.key_dim, layer.value_dim), -1)
    )
    q = q.reshape(1, t, layer.num_k_heads, layer.head_k_dim)
    k = k.reshape_as(q)
    v = v.reshape(1, t, layer.num_v_heads, layer.head_v_dim)
    g = -layer.A_log.float().exp() * F.softplus(
        F.linear(x, aw).float() + layer.dt_bias.float()
    )
    if preview:
        y = bd_gdn_local_memory(
            q,
            k,
            v,
            g,
            clean_ssm_states=state,
            cache_indices=torch.zeros_like(slots),
            active_valid_mask=torch.ones(t, device=x.device, dtype=torch.bool),
            scale=layer.head_k_dim**-0.5,
            softmax_logit_scale=layer.bd_softmax_logit_scale,
            softmax_history_bias=layer.bd_softmax_history_bias,
            softmax_history_c=layer.bd_softmax_history_c,
        )
    else:
        beta = F.linear(x, bw).sigmoid()
        groups = layer.num_v_heads // layer.num_k_heads
        q = q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)
        k = k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)
        q = q.repeat_interleave(groups, 2) / layer.head_k_dim**0.5
        k = k.repeat_interleave(groups, 2)
        if replay_graph is None:
            y, state = _causal_scan(q, k, v, g, beta, state)
        else:
            y, state = replay_graph.run(q, k, v, g, beta, state)
        cache.temporal[slots] = state
    z = F.linear(x, zw).reshape_as(y)
    y = norm(y, layer.norm, layer.config.rms_norm_eps, z)
    return F.linear(y.reshape(1, t, -1), layer.out_proj.weight)


def route(layer, x, preview):
    from sglang.srt.models.clover_moe import _block_coreset_reference

    logits = F.linear(x, layer.gate.weight)
    top_k = layer.topk.topk_config.top_k
    if preview and layer.clover_coreset:
        logits = _block_coreset_reference(logits, x.shape[0], top_k, layer.gate.top_p)
    ids = logits.float().argsort(dim=-1, descending=True, stable=True)[:, :top_k]
    weights = logits.float().softmax(-1).gather(-1, ids)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    return ids, weights


def moe(layer, x, preview):
    x = x[0]
    ids, weights = route(layer, x, preview)
    top_k = ids.shape[-1]
    # Group all routes once. Stable sorting preserves the reference GEMM row order.
    flat = ids.flatten()
    order = flat.argsort(stable=True)
    counts = torch.zeros(layer.gate.weight.shape[0], device=x.device, dtype=torch.int32)
    counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=counts.dtype))
    groups = [(e, n) for e, n in enumerate(counts.tolist()) if n]
    inputs = x[order // top_k]
    sizes = [n for _, n in groups]
    gate_up = torch.cat(
        [
            F.linear(rows, layer.experts.w13_weight[e])
            for (e, _), rows in zip(groups, inputs.split(sizes))
        ]
    )
    gate, up = gate_up.chunk(2, -1)
    hidden = (F.silu(gate.float()) * up.float()).to(x.dtype)
    down = torch.cat(
        [
            torch.mm(rows, layer.experts.w2_weight[e].T, out_dtype=torch.float32)
            for (e, _), rows in zip(groups, hidden.split(sizes))
        ]
    )
    weighted = (down * weights.flatten()[order, None]).to(x.dtype)
    contributions = torch.empty_like(weighted)
    contributions[order] = weighted
    contributions = contributions.view(x.shape[0], top_k, -1)
    sorted_slots = ids.argsort(dim=-1)
    contributions = contributions.gather(
        1, sorted_slots[..., None].expand_as(contributions)
    )
    out = contributions[:, 0].float()
    for slot in range(1, top_k):
        out = out + contributions[:, slot].float()
    return (out.to(x.dtype) + shared_expert(layer, x))[None]


def shared_expert(layer, x):
    shared = layer.shared_expert
    gate_w, up_w = shared.gate_up_proj.weight.chunk(2)
    gate = F.linear(x, gate_w)
    up = F.linear(x, up_w)
    shared_out = F.linear(
        (F.silu(gate.float()) * up.float()).to(x.dtype), shared.down_proj.weight
    )
    return shared_out * layer.shared_expert_gate(x).sigmoid()


def preview_linear_graph(layer, x, batch, eps, pool):
    # We only enter with a committed prefix. All variable request data enters
    # through the two persistent buffers, including the clean-state pool slot.
    indices = (
        get_attn_backend().linear_attn_backend.forward_metadata.mamba_cache_indices
    )
    captured = getattr(layer, "_pure_linear_graph", None)
    if captured is None:
        residual = torch.empty_like(x)
        slots = torch.empty_like(indices, dtype=torch.long)
        residual.copy_(x)
        slots.copy_(indices)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            h = norm(residual, layer.input_layernorm, eps)
            y = residual + linear_attention(layer.linear_attn, h, batch, True, slots)
            mlp_input = norm(y, layer.post_attention_layernorm, eps)
        captured = (graph, residual, slots, y, mlp_input)
        layer._pure_linear_graph = captured
    graph, residual, slots, y, mlp_input = captured
    residual.copy_(x)
    slots.copy_(indices)
    graph.replay()
    return y, mlp_input


def forward_eager(model, input_ids, batch):
    preview = batch.forward_mode.is_dllm_extend()
    if batch.batch_size != 1:
        raise ValueError("Clover pure eager forward requires exactly one active row")
    if (
        preview
        and batch.extend_prefix_lens_cpu[0]
        and envs.SGLANG_CLOVER_PURE_FUSED_PREVIEW.get()
    ):
        from sglang.srt.models.clover_pure_fused import preview_forward

        return preview_forward(model, input_ids, batch)
    x = F.embedding(input_ids[None], model.model.embed_tokens.weight)
    eps = model.config.rms_norm_eps
    replay_graph = None
    if (
        not preview
        and not envs.SGLANG_CLOVER_PURE_DISABLE_GRAPH.get()
        and input_ids.numel() == model.config.block_size
    ):
        if not hasattr(model, "_pure_replay_graph"):
            model._pure_replay_graph = CausalReplayGraph()
        replay_graph = model._pure_replay_graph
    elif (
        not preview
        and not envs.SGLANG_CLOVER_PURE_DISABLE_GRAPH.get()
        and 8 <= input_ids.numel() < model.config.block_size
    ):
        # One short-prefill shape plus the permanent full-block replay shape.
        length = input_ids.numel()
        cached = getattr(model, "_pure_prefill_graph", None)
        if cached is None or cached[0] != length:
            cached = model._pure_prefill_graph = (length, CausalReplayGraph())
        replay_graph = cached[1]
    graph_linear = (
        not envs.SGLANG_CLOVER_PURE_DISABLE_GRAPH.get()
        and preview
        and batch.extend_prefix_lens_cpu[0] > 0
    )
    if graph_linear and not hasattr(model, "_pure_preview_pool"):
        model._pure_preview_pool = torch.cuda.graph_pool_handle()
    for layer in model.model.layers:
        if graph_linear and hasattr(layer, "linear_attn"):
            x, h = preview_linear_graph(layer, x, batch, eps, model._pure_preview_pool)
        else:
            h = norm(x, layer.input_layernorm, eps)
            if hasattr(layer, "linear_attn"):
                x = x + linear_attention(
                    layer.linear_attn, h, batch, preview, replay_graph=replay_graph
                )
            else:
                x = x + full_attention(layer, h, batch, preview)
            h = norm(x, layer.post_attention_layernorm, eps)
        x = x + moe(layer.mlp, h, preview)
    return norm(x, model.model.norm, eps)[0]
