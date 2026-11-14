import math
from contextlib import nullcontext
from functools import wraps
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch._guards import active_fake_mode
from torch._subclasses.fake_tensor import FakeTensorMode


class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        return torch.gather(
            rearrange(input, "b ... -> b (...)"),
            0,
            repeat(indices, "z -> z d", d=second_dim),
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        grad_input.scatter_(
            0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output
        )
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(
            first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype
        )
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        grad_values = grad_output[indices]
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply


def unpad_input(hidden_states, attention_mask, unused_mask=None):
    all_masks = (
        (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    )
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    in_fake_mode = active_fake_mode() is not None
    if not in_fake_mode:
        indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max().item()
    else:
        # torch.nonzero and .item() are not supported in FakeTensorMode
        batch_size, seqlen = attention_mask.shape
        indices = torch.arange(batch_size * seqlen, device=hidden_states.device)
        max_seqlen_in_batch = seqlen
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        used_seqlens_in_batch,
    )


def pad_input(hidden_states, indices, batch, seqlen):
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def generate_random_padding_mask(
    max_seqlen, batch_size, device, mode="random", zero_lengths=False, min_seqlen=None
):
    assert mode in ["full", "random", "third"]
    min_seqlen = min_seqlen if min_seqlen is not None else 0 if zero_lengths else 1
    if mode == "full":
        lengths = torch.full(
            (batch_size, 1), max_seqlen, device=device, dtype=torch.int32
        )
    elif mode == "random":
        lengths = torch.randint(
            max(min_seqlen, max_seqlen - 20),
            max_seqlen + 1,
            (batch_size, 1),
            device=device,
        )
    else:
        lengths = torch.randint(
            max(min_seqlen, max_seqlen // 3),
            max_seqlen + 1,
            (batch_size, 1),
            device=device,
        )

    if zero_lengths:
        for i in range(batch_size):
            if i % 5 == 0:
                lengths[i] = 0
        lengths[-1] = 0
    padding_mask = (
        repeat(torch.arange(max_seqlen, device=device), "s -> b s", b=batch_size)
        < lengths
    )
    return padding_mask


def generate_qkv(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    qv=None,
    kvpacked=False,
    qkvpacked=False,
    query_unused_mask=None,
    key_unused_mask=None,
):
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen_q, nheads, d = q.shape
    d_v = v.shape[-1]
    _, seqlen_k, nheads_k, _ = k.shape
    assert k.shape == (batch_size, seqlen_k, nheads_k, d)
    assert v.shape == (batch_size, seqlen_k, nheads_k, d_v)
    if query_unused_mask is not None or key_unused_mask is not None:
        assert not kvpacked
        assert not qkvpacked

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, seqused_q = unpad_input(
            q, query_padding_mask, query_unused_mask
        )
        output_pad_fn = lambda output_unpad: pad_input(
            output_unpad, indices_q, batch_size, seqlen_q
        )
        qv_unpad = (
            rearrange(qv, "b s ... -> (b s) ...")[indices_q] if qv is not None else None
        )
    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = torch.arange(
            0,
            (batch_size + 1) * seqlen_q,
            step=seqlen_q,
            dtype=torch.int32,
            device=q_unpad.device,
        )
        seqused_q = None
        max_seqlen_q = seqlen_q
        output_pad_fn = lambda output_unpad: rearrange(
            output_unpad, "(b s) h d -> b s h d", b=batch_size
        )
        qv_unpad = rearrange(qv, "b s ... -> (b s) ...") if qv is not None else None

    if key_padding_mask is not None:
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k, seqused_k = unpad_input(
            k, key_padding_mask, key_unused_mask
        )
        v_unpad, *_ = unpad_input(v, key_padding_mask, key_unused_mask)
    else:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")
        cu_seqlens_k = torch.arange(
            0,
            (batch_size + 1) * seqlen_k,
            step=seqlen_k,
            dtype=torch.int32,
            device=k_unpad.device,
        )
        seqused_k = None
        max_seqlen_k = seqlen_k

    if qkvpacked:
        assert (query_padding_mask == key_padding_mask).all()
        assert nheads == nheads_k
        qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
        qkv = torch.stack([q, k, v], dim=2)
        if query_padding_mask is not None:
            dqkv_pad_fn = lambda dqkv_unpad: pad_input(
                dqkv_unpad, indices_q, batch_size, seqlen_q
            )
        else:
            dqkv_pad_fn = lambda dqkv_unpad: rearrange(
                dqkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            qkv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            max_seqlen_q,
            qkv.detach().requires_grad_(),
            output_pad_fn,
            dqkv_pad_fn,
        )
    elif kvpacked:
        kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
        kv = torch.stack([k, v], dim=2)
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dkv_pad_fn = lambda dkv_unpad: pad_input(
                dkv_unpad, indices_k, batch_size, seqlen_k
            )
        else:
            dkv_pad_fn = lambda dkv_unpad: rearrange(
                dkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            q_unpad.detach().requires_grad_(),
            kv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            kv.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dkv_pad_fn,
        )
    else:
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dk_pad_fn = lambda dk_unpad: pad_input(
                dk_unpad, indices_k, batch_size, seqlen_k
            )
        else:
            dk_pad_fn = lambda dk_unpad: rearrange(
                dk_unpad, "(b s) h d -> b s h d", b=batch_size
            )
        return (
            q_unpad.detach().requires_grad_(),
            k_unpad.detach().requires_grad_(),
            v_unpad.detach().requires_grad_(),
            qv_unpad.detach() if qv is not None else None,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            k.detach().requires_grad_(),
            v.detach().requires_grad_(),
            qv.detach() if qv is not None else None,
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        )


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(None, None),
    sink_token_length=0,
    query_padding_mask=None,
    key_padding_mask=None,
    key_leftpad=None,
    device=None,
):
    row_idx = rearrange(
        torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1"
    )
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] is None:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        if window_size[1] is None:
            local_mask_left = col_idx > sk
        else:
            local_mask_left = col_idx > torch.minimum(
                row_idx + sk - sq + window_size[1], sk
            )
        return torch.logical_or(
            local_mask_left,
            torch.logical_and(
                col_idx < row_idx + sk - sq - window_size[0],
                col_idx >= sink_token_length,
            ),
        )


def construct_chunk_mask(
    seqlen_q,
    seqlen_k,
    attention_chunk,
    query_padding_mask=None,
    key_padding_mask=None,
    key_leftpad=None,
    device=None,
):
    row_idx = rearrange(
        torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1"
    )
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
    col_limit_left_chunk = row_idx + sk - sq - (row_idx + sk - sq) % attention_chunk
    return torch.logical_or(
        col_idx < col_limit_left_chunk,
        col_idx >= col_limit_left_chunk + attention_chunk,
    )


def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    key_leftpad=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(None, None),
    attention_chunk=0,
    sink_token_length=0,
    learnable_sink: Optional[torch.Tensor] = None,
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    intermediate_dtype=None,
    return_lse=False,
    gather_kv_indices=None,
    rel_bias: Optional[torch.Tensor] = None,  # [b, seqlen_q, h, rel_extent]
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
):
    assert v is not None
    has_qk = q is not None and k is not None
    assert has_qk or qv is not None
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = v.dtype
    q_shape = q.shape if q is not None else qv.shape
    if upcast:
        q, k, v, qv = [t.float() if t is not None else None for t in (q, k, v, qv)]
    if q_descale is not None:
        q_descale = repeat(q_descale, "b h -> b 1 (h g) 1", g=q_shape[2] // v.shape[2])
        q, qv = [
            (t.float() * q_descale).to(t.dtype) if t is not None else None
            for t in (q, qv)
        ]
    if k_descale is not None:
        k = (k.float() * rearrange(k_descale, "b h -> b 1 h 1")).to(dtype=k.dtype)
    if v_descale is not None:
        v = (v.float() * rearrange(v_descale, "b h -> b 1 h 1")).to(dtype=v.dtype)
    seqlen_q, seqlen_k = q_shape[1], v.shape[1]
    k, v = [
        (
            repeat(t, "b s h d -> b s (h g) d", g=q_shape[2] // t.shape[2])
            if t is not None
            else None
        )
        for t in (k, v)
    ]
    d = q_shape[-1]  # == dv for qv
    dv = v.shape[-1]
    softmax_scale = 1.0 / math.sqrt(d if qv is None or q is None else d + dv)
    if has_qk:
        scores = torch.einsum(
            "bthd,bshd->bhts",
            q if reorder_ops else q * softmax_scale,
            k * softmax_scale if reorder_ops else k,
        )
    if qv is not None:
        qv_scores = torch.einsum(
            "bthd,bshd->bhts",
            qv if reorder_ops else qv * softmax_scale,
            v * softmax_scale if reorder_ops else v,
        )
        scores = qv_scores if not has_qk else scores + qv_scores
    if softcap > 0:
        scores = torch.tanh(scores / softcap) * softcap
    if key_padding_mask is not None:
        scores.masked_fill_(
            rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf")
        )
    local_mask = None
    if window_size[0] is not None or window_size[1] is not None:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            sink_token_length,
            query_padding_mask,
            key_padding_mask,
            key_leftpad=key_leftpad,
            device=v.device,
        )
    if attention_chunk > 0:
        chunk_mask = construct_chunk_mask(
            seqlen_q,
            seqlen_k,
            attention_chunk,
            query_padding_mask,
            key_padding_mask,
            key_leftpad=key_leftpad,
            device=v.device,
        )
        local_mask = (
            torch.logical_or(local_mask, chunk_mask)
            if local_mask is not None
            else chunk_mask
        )
    if gather_kv_indices is not None:
        batch = q_shape[0]
        topk_len = gather_kv_indices.shape[2]
        if topk_len < seqlen_k:
            topk_index_mask = torch.full(
                (batch, seqlen_q, seqlen_k), False, device="cuda"
            ).scatter_(-1, gather_kv_indices, True)
            scores.masked_fill_(
                rearrange(~topk_index_mask, "b t s -> b 1 t s"), float("-inf")
            )
    if local_mask is not None:
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    if rel_bias is not None:
        # Reference for Inkling sheared bias: gather rel_bias[i, h, i - j] into the [t, s] score grid.
        rel_extent = rel_bias.shape[-1]
        if cu_seqlens_q is not None:
            seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        elif seqused_q is not None:
            seqlens_q = seqused_q
        else:
            seqlens_q = torch.full(
                (q.shape[0],), seqlen_q, device=q.device, dtype=torch.long
            )
        if cu_seqlens_k is not None:
            seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        elif seqused_k is not None:
            seqlens_k = seqused_k
        else:
            seqlens_k = torch.full(
                (q.shape[0],), seqlen_k, device=q.device, dtype=torch.long
            )
        seqlen_offset = (seqlens_k - seqlens_q).to(torch.long)  # [b]
        q_idx = torch.arange(seqlen_q, device=q.device, dtype=torch.long)
        kv_idx = torch.arange(seqlen_k, device=q.device, dtype=torch.long)
        rel_dist = (
            q_idx.unsqueeze(1) - kv_idx.unsqueeze(0) + seqlen_offset.view(-1, 1, 1)
        )  # [b, seqlen_q, seqlen_k]
        safe_dist = rel_dist.clamp(0, rel_extent - 1)
        is_within_window = (rel_dist >= 0) & (rel_dist < rel_extent)
        idx = safe_dist.unsqueeze(2).expand(-1, -1, rel_bias.shape[2], -1)
        abs_bias = rel_bias.gather(dim=-1, index=idx)  # [b, seqlen_q, h, seqlen_k]
        abs_bias = rearrange(abs_bias, "b t h s -> b h t s")
        abs_bias = abs_bias.masked_fill(
            rearrange(~is_within_window, "b t s -> b 1 t s"), 0.0
        )
        scores = scores + abs_bias
    # After all masks are applied, before softmax:
    # scores shape: [b, h, t, s]
    lse = torch.logsumexp(scores, dim=-1)  # [b, h, t]
    if learnable_sink is None:
        attention = torch.softmax(scores, dim=-1).to(v.dtype)
    else:
        scores_fp32 = scores.to(torch.float32)
        logits_max = torch.amax(scores_fp32, dim=-1, keepdim=True)
        learnable_sink = rearrange(learnable_sink, "h -> h 1 1")
        logits_or_sinks_max = torch.maximum(learnable_sink, logits_max)
        unnormalized_scores = torch.exp(scores_fp32 - logits_or_sinks_max)
        normalizer = unnormalized_scores.sum(dim=-1, keepdim=True) + torch.exp(
            learnable_sink - logits_or_sinks_max
        )
        # LSE with sink: log(Z) = log(normalizer) + max
        lse = (torch.log(normalizer.squeeze(-1)) + logits_or_sinks_max.squeeze(-1)).to(
            dtype_og
        )
        attention = (unnormalized_scores / normalizer).to(v.dtype)
    if query_padding_mask is not None:
        attention = attention.masked_fill(
            rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0
        )
    if key_padding_mask is not None:
        attention = attention.masked_fill(
            rearrange(~key_padding_mask, "b s -> b 1 1 s"), 0.0
        )
    if local_mask is not None:
        attention = attention.masked_fill(
            torch.all(local_mask, dim=-1, keepdim=True), 0.0
        )
    dropout_scaling = 1.0 / (1 - dropout_p)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    if intermediate_dtype is not None:
        attention_drop = attention_drop.to(intermediate_dtype).to(attention_drop.dtype)
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    if return_lse:
        return output.to(dtype_og), attention.to(dtype_og), lse.to(dtype_og)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


def maybe_fake_tensor_mode(fake: bool = True):
    """
    One way to populate/pre-compile cache is to use torch fake tensor mode,
    which does not allocate actual GPU tensors but retains tensor shape/dtype
    metadata for cute.compile.
    """

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with FakeTensorMode() if fake else nullcontext():
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def is_fake_mode() -> bool:
    return active_fake_mode() is not None
