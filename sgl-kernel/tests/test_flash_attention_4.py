# Adapted from https://github.com/Dao-AILab/flash-attention/blob/8ecf128f683266735ba68e3c106ff67a2611886e/tests/cute/test_flash_attn.py

# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

import itertools
import math
from functools import partial

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from flash_attn.layers.rotary import apply_rotary_emb
except ImportError:
    apply_rotary_emb = None

from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from sgl_kernel.testing.rotary_embedding import _apply_rotary_emb as apply_rotary_emb

# from utils import is_hopper  # Not used in this test

# Force sgl_kernel.flash_attn wrappers to use FA4 (Cute-DSL) implementations.
# The wrappers accept a superset of args; for FA4, extra args are ignored.
flash_attn_varlen_func = partial(flash_attn_varlen_func, ver=4)
flash_attn_with_kvcache = partial(flash_attn_with_kvcache, ver=4)

# Skip this test on Hopper machine
skip_condition = torch.cuda.get_device_capability() < (10, 0)


def unpad_input(hidden_states, attention_mask, unused_mask=None):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
        unused_mask: (batch, seqlen), bool / int, 1 means the element is allocated but unused.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask + unused_mask.
        indices: (total_nnz), the indices of masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
        seqused: (batch), returns the number of tokens selected in attention_mask + unused_mask.
    """
    all_masks = (
        (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    )
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices.
    return (
        rearrange(hidden_states, "b s ... -> (b s) ...")[indices],
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        used_seqlens_in_batch,
    )


def pad_input(hidden_states, indices, batch, seqlen):
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    dim = hidden_states.shape[1:]
    output = torch.zeros(
        (batch * seqlen), *dim, device=hidden_states.device, dtype=hidden_states.dtype
    )
    output[indices] = hidden_states
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def generate_random_padding_mask(
    max_seqlen, batch_size, device, mode="random", zero_lengths=False
):
    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full(
            (batch_size, 1), max_seqlen, device=device, dtype=torch.int32
        )
    elif mode == "random":
        lengths = torch.randint(
            max(0 if zero_lengths else 1, max_seqlen - 20),
            max_seqlen + 1,
            (batch_size, 1),
            device=device,
        )
    elif mode == "third":
        lengths = torch.randint(
            max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device
        )
    else:
        # This should never happen due to the assertion above, but for linter
        lengths = torch.full(
            (batch_size, 1), max_seqlen, device=device, dtype=torch.int32
        )

    if zero_lengths:
        # Generate zero-lengths every 5 batches and the last batch.
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
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, d)
        k: (batch_size, seqlen_k, nheads_k, d)
        v: (batch_size, seqlen_k, nheads_k, d_v)
        query_padding_mask: (batch_size, seqlen), bool
        key_padding_mask: (batch_size, seqlen), bool
    """
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
        v_unpad, *rest = unpad_input(v, key_padding_mask, key_unused_mask)
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
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
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
    # Subtract remainder instead of divide and then multiply to take care of negative values
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
    learnable_sink=None,
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    intermediate_dtype=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads, head_dim)
        v: (batch_size, seqlen_k, nheads, head_dim_v)
        qv: (batch_size, seqlen_q, nheads, head_dim_v)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling k, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim_v)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
        qv = qv.float() if qv is not None else None
    if q_descale is not None:
        q_descale = repeat(q_descale, "b h -> b 1 (h g) 1", g=q.shape[2] // k.shape[2])
        q = (q.float() * q_descale).to(q.dtype)
        qv = (qv.float() * q_descale).to(qv.dtype) if qv is not None else None
    if k_descale is not None:
        k = (k.float() * rearrange(k_descale, "b h -> b 1 h 1")).to(dtype=k.dtype)
    if v_descale is not None:
        v = (v.float() * rearrange(v_descale, "b h -> b 1 h 1")).to(dtype=v.dtype)
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    dv = v.shape[-1]
    softmax_scale = 1.0 / math.sqrt(d if qv is None else d + dv)
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q * softmax_scale, k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
    if qv is not None:
        scores = scores + torch.einsum("bthd,bshd->bhts", qv * softmax_scale, v)
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
            device=q.device,
        )
    if attention_chunk > 0:
        chunk_mask = construct_chunk_mask(
            seqlen_q,
            seqlen_k,
            attention_chunk,
            query_padding_mask,
            key_padding_mask,
            key_leftpad=key_leftpad,
            device=q.device,
        )
        local_mask = (
            torch.logical_or(local_mask, chunk_mask)
            if local_mask is not None
            else chunk_mask
        )
    if local_mask is not None:
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
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
        attention = (unnormalized_scores / normalizer).to(v.dtype)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(
            rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0
        )
    # Without this we might get NaN in dv
    if key_padding_mask is not None:
        attention = attention.masked_fill(
            rearrange(~key_padding_mask, "b s -> b 1 1 s"), 0.0
        )
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if local_mask is not None:
        attention = attention.masked_fill(
            torch.all(local_mask, dim=-1, keepdim=True), 0.0
        )
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    if intermediate_dtype is not None:
        attention_drop = attention_drop.to(intermediate_dtype).to(attention_drop.dtype)
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


@pytest.mark.skipif(
    skip_condition, reason="FA4 Requires compute capability of 10 or above."
)
# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("mha_type", ["mqa"])
@pytest.mark.parametrize("has_learnable_sink", [False, True])
# @pytest.mark.parametrize("has_learnable_sink", [False])
# @pytest.mark.parametrize("has_qv", [False, True])
@pytest.mark.parametrize("has_qv", [False])
# @pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("deterministic", [False])
# @pytest.mark.parametrize("softcap", [0.0, 15.0])
@pytest.mark.parametrize("softcap", [0.0])
# @pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [False])
# @pytest.mark.parametrize("add_unused_qkv", [False, True])
@pytest.mark.parametrize("add_unused_qkv", [False])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192, 256])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128])
# @pytest.mark.parametrize("d", [64, 96, 128])
@pytest.mark.parametrize("d", [128, 192])
# @pytest.mark.parametrize("d", [192])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        # (1, 1),
        # (1, 3),
        # (2, 1),
        (511, 1),
        (3, 513),
        (64, 128),
        (128, 128),
        (256, 256),
        # (113, 203),
        # (128, 217),
        # (113, 211),
        # (108, 256),
        # (256, 512),
        (307, 256),
        (640, 128),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
    ],
)
def test_flash_attn_varlen_output(
    seqlen_q,
    seqlen_k,
    d,
    add_unused_qkv,
    causal,
    local,
    softcap,
    deterministic,
    has_qv,
    has_learnable_sink,
    mha_type,
    dtype,
):
    if (
        causal or local
    ):  # Right now we only support causal attention with seqlen_k == seqlen_q
        seqlen_k = seqlen_q
    device = "cuda"
    # set seed
    torch.random.manual_seed(seqlen_q + seqlen_k + d + int(causal) * 2 + int(local))
    batch_size = 49 if seqlen_q <= 1024 else 7
    nheads = 6
    # batch_size = 1
    # nheads = 1
    nheads_kv = nheads if mha_type == "mha" else (3 if mha_type == "gqa" else 1)
    dtype_ref = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    # dv_vals = [128, d] if d > 128 and d <= 192 else ([256, 512, d] if d <= 64 else [d])
    dv_vals = [128] if d == 192 else ([d] if d != 128 else [64, d])
    if dtype == torch.float8_e4m3fn:
        dv_vals = [d]
    # attention_chunk_vals = [torch.randint(1, seqlen_k * 2, (1,)).item(), 0] if seqlen_q <= seqlen_k else [0]
    attention_chunk_vals = [0]
    for dv, attention_chunk in itertools.product(dv_vals, attention_chunk_vals):
        q_ref = torch.randn(
            batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_ref
        )
        if softcap > 0.0:
            # Ensure the values of qk are at least within softcap range.
            q_ref = (q_ref * softcap / 4).detach().requires_grad_()
        q_ref = q_ref.to(dtype).to(dtype_ref).requires_grad_()
        k_ref = (
            torch.randn(
                batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype_ref
            )
            .to(dtype)
            .to(dtype_ref)
            .requires_grad_()
        )
        v_ref = (
            torch.randn(
                batch_size, seqlen_k, nheads_kv, dv, device=device, dtype=dtype_ref
            )
            .to(dtype)
            .to(dtype_ref)
            .requires_grad_()
        )
        if has_qv:
            qv_ref = (
                torch.randn(
                    batch_size, seqlen_q, nheads, dv, device=device, dtype=dtype_ref
                )
                .to(dtype)
                .to(dtype_ref)
            )
        else:
            qv_ref = None
        # Put window_size after QKV randn so that window_size changes from test to test
        window_size = (
            (None, None) if not local else torch.randint(0, seqlen_k, (2,)).tolist()
        )
        if has_learnable_sink:
            learnable_sink = torch.randn(nheads, dtype=torch.bfloat16, device=device)
        else:
            learnable_sink = None
        if dtype == torch.float8_e4m3fn:
            q_descale, k_descale, v_descale = [
                torch.rand(batch_size, nheads_kv, device=device, dtype=torch.float32)
                * 2
                for _ in range(3)
            ]
        else:
            q_descale, k_descale, v_descale = None, None, None
        q, k, v = [x.detach().requires_grad_() for x in (q_ref, k_ref, v_ref)]
        qv = qv_ref.detach() if has_qv else None
        query_padding_mask = generate_random_padding_mask(
            seqlen_q, batch_size, device, mode="random", zero_lengths=False
        )
        # TODO: test zero_lengths
        key_padding_mask = generate_random_padding_mask(
            # seqlen_k, batch_size, device, mode="random", zero_lengths=True
            seqlen_k,
            batch_size,
            device,
            mode="random",
            zero_lengths=False,
        )

        def _gen_unused_masks(padding_mask, add_unused, max_seq_len, bs, device):
            if add_unused:
                another_mask = generate_random_padding_mask(max_seq_len, bs, device)
                attn_mask = torch.logical_and(padding_mask, another_mask)
                unused_mask = torch.logical_xor(
                    torch.logical_or(padding_mask, another_mask), attn_mask
                )
            else:
                attn_mask = padding_mask
                unused_mask = None
            return attn_mask, unused_mask

        query_padding_mask, query_unused_mask = _gen_unused_masks(
            query_padding_mask, add_unused_qkv, seqlen_q, batch_size, q.device
        )
        # query_padding_mask[:] = True
        # query_unused_mask = None
        key_padding_mask, key_unused_mask = _gen_unused_masks(
            key_padding_mask, add_unused_qkv, seqlen_k, batch_size, k.device
        )

        if causal or local:
            key_padding_mask = query_padding_mask

        result = generate_qkv(
            q,
            k,
            v,
            query_padding_mask,
            key_padding_mask,
            qv=qv,
            kvpacked=False,
            query_unused_mask=query_unused_mask,
            key_unused_mask=key_unused_mask,
        )
        (
            q_unpad,  # 0
            k_unpad,  # 1
            v_unpad,  # 2
            qv_unpad,  # 3
            cu_seqlens_q,  # 4
            cu_seqlens_k,  # 5
            seqused_q,  # 6
            seqused_k,  # 7
            max_seqlen_q,  # 8
            max_seqlen_k,  # 9
            q,  # 10
            k,  # 11
            v,  # 12
            qv,  # 13
            output_pad_fn,  # 14
            dq_pad_fn,  # 15
            dk_pad_fn,  # 16
        ) = result
        q_unpad, k_unpad, v_unpad = [
            x.detach().to(dtype).requires_grad_() for x in (q_unpad, k_unpad, v_unpad)
        ]
        out_ref, attn_ref = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            qv=qv_ref,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            window_size=window_size,
            attention_chunk=attention_chunk,
            learnable_sink=learnable_sink,
            softcap=softcap,
        )
        out_pt, attn_pt = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            qv=qv_ref,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            window_size=window_size,
            attention_chunk=attention_chunk,
            learnable_sink=learnable_sink,
            softcap=softcap,
            upcast=False,
            reorder_ops=True,
            intermediate_dtype=dtype if dtype == torch.float8_e4m3fn else None,
        )

        print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
        print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

        if query_unused_mask is not None:
            q_zero_masking = rearrange(query_unused_mask, "b s -> b s 1 1")

        # Numerical error if we just do any arithmetic on out_ref
        fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
        rtol = 2 if softcap == 0.0 else 3

        pack_gqa_vals = [False, True, None]
        # num_splits_vals = [1, 3]
        num_splits_vals = [1]
        for pack_gqa, num_splits in itertools.product(pack_gqa_vals, num_splits_vals):
            out_unpad, lse = flash_attn_varlen_func(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                # max_seqlen_q and max_seqlen_k not needed for FA4
                seqused_q=seqused_q,
                seqused_k=seqused_k,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                sinks=learnable_sink,  # FA4 uses learnable_sink, not sinks
                pack_gqa=pack_gqa,
                return_softmax_lse=True,
                ver=4,  # Use FA4
            )
            out = output_pad_fn(out_unpad)
            if query_unused_mask is not None:
                out.masked_fill_(q_zero_masking, 0.0)
            print(f"Output max diff: {(out - out_ref).abs().max().item()}")
            print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
            # if not causal:
            #     print(f"LSE max diff: {(lse - lse_ref).abs().max().item()}")
            # breakpoint()

            # Check that FlashAttention's numerical error is at most 3x the numerical error
            # of a Pytorch implementation.
            assert (out - out_ref).abs().max().item() <= rtol * (
                out_pt - out_ref
            ).abs().max().item() + fwd_atol

        if (
            dtype != torch.float8_e4m3fn
            and not has_qv
            and not dv > 256
            and not attention_chunk != 0
            and dv == d
            and not has_learnable_sink
            and False
        ):
            g_unpad = torch.randn_like(out_unpad)
            do_o = ((g_unpad.float() * out_unpad.float()).sum(-1)).transpose(-1, -2)
            # import flash_attn_3_cuda
            # dq_unpad, dk_unpad, dv_unpad, softmax_d, dq_accum, lse_log2 = flash_attn_3_cuda.bwd_varlen(
            #     g_unpad,
            #     q_unpad,
            #     k_unpad,
            #     v_unpad,
            #     out_unpad,
            #     lse,
            #     None,
            #     None,
            #     None,
            #     cu_seqlens_q,
            #     cu_seqlens_k,
            #     None, None,
            #     max_seqlen_q,
            #     max_seqlen_k,
            #     d ** (-0.5),
            #     causal,
            #     window_size[0], window_size[1],
            #     softcap,
            #     deterministic,
            #     0,  # sm_margin
            # )
            dq_unpad, dk_unpad, dv_unpad = torch.autograd.grad(
                out_unpad, (q_unpad, k_unpad, v_unpad), g_unpad
            )
            dq = dq_pad_fn(dq_unpad)
            dk = dk_pad_fn(dk_unpad)
            dv = dk_pad_fn(dv_unpad)
            if key_unused_mask is not None:
                k_zero_masking = rearrange(key_unused_mask, "b s -> b s 1 1")
                dk.masked_fill_(k_zero_masking, 0.0)
                dv.masked_fill_(k_zero_masking, 0.0)
            if query_unused_mask is not None:
                dq.masked_fill_(q_zero_masking, 0.0)
            # print(f"dO_O max diff: {(softmax_d - do_o).abs().max().item()}")
            # assert (softmax_d - do_o).abs().max().item() <= 1e-5
            # assert dq_accum.abs().max().item() == 0.0
            g = output_pad_fn(g_unpad)

            # qk = torch.einsum('bthd,bshd->bhts', q / (d ** 0.5), k).float()
            # qk = torch.masked_fill(qk, rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
            # dS = torch.einsum('bthd,bshd->bhts', g.float(), v.float())
            # P = torch.softmax(qk, -1)
            # dP = P * (dS - (g.float() * out.float()).sum(-1).transpose(1, 2).unsqueeze(-1))
            # dQ = torch.einsum('bhts,bshd->bthd', dP, k.float())
            # dV = torch.einsum('bhts,bthd->bshd', P, g.float())
            # dK = torch.einsum('bhts,bthd->bshd', dP, q.float())

            # dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
            dq_ref, dk_ref, dv_ref = torch.autograd.grad(
                out_ref, (q_ref, k_ref, v_ref), g
            )
            dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_ref, k_ref, v_ref), g)
            print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
            print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
            print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
            print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
            print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
            print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
            print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
            print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
            print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
            print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
            print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
            print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")
            # breakpoint()
            dq_atol = 2 * (dq_ref + 0.3 - 0.3 - dq_ref).abs().max().item() + (
                0 if softcap == 0 else 3e-4
            )
            assert (dq - dq_ref).abs().max().item() <= rtol * (
                dq_pt - dq_ref
            ).abs().max().item() + dq_atol
            dk_atol = 2 * (dk_ref + 0.3 - 0.3 - dk_ref).abs().max().item() + (
                0 if softcap == 0 else 3e-4
            )
            assert (dk - dk_ref).abs().max().item() <= rtol * (
                dk_pt - dk_ref
            ).abs().max().item() + dk_atol
            dv_atol = 2 * (dv_ref + 0.3 - 0.3 - dv_ref).abs().max().item() + (
                0 if softcap == 0 else 3e-4
            )
            assert (dv - dv_ref).abs().max().item() <= rtol * (
                dv_pt - dv_ref
            ).abs().max().item() + dv_atol


@pytest.mark.skipif(
    skip_condition, reason="FA4 Requires compute capability of 10 or above."
)
# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("mha_type", ["mha"])
@pytest.mark.parametrize("has_learnable_sink", [False, True])
# @pytest.mark.parametrize("has_learnable_sink", [False])
# @pytest.mark.parametrize("new_kv", [False, True])
@pytest.mark.parametrize("new_kv", [False])
# @pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("local", [False])
# @pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("causal", [True])
# @pytest.mark.parametrize("seqlen_new_eq_seqlen_q", [True, False])
@pytest.mark.parametrize("seqlen_new_eq_seqlen_q", [False])
# @pytest.mark.parametrize("has_rotary_seqlens", [False, True])
@pytest.mark.parametrize("has_rotary_seqlens", [False])
# @pytest.mark.parametrize("rotary_interleaved", [False, True])
@pytest.mark.parametrize("rotary_interleaved", [True])
# @pytest.mark.parametrize("rotary_fraction", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("rotary_fraction", [0.0])
# @pytest.mark.parametrize("page_size", [None] + ([1, 4, 128]))
# @pytest.mark.parametrize("page_size", [None, 128])
@pytest.mark.parametrize("page_size", [128])
# @pytest.mark.parametrize("has_leftpad", [False, True])
@pytest.mark.parametrize("has_leftpad", [False])
# @pytest.mark.parametrize("has_batch_idx", [False, True])
@pytest.mark.parametrize("has_batch_idx", [False])
# @pytest.mark.parametrize("varlen_q", [False, True])
@pytest.mark.parametrize("varlen_q", [False])
# @pytest.mark.parametrize("d", [32, 59, 64, 80, 128, 256])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
# @pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("d", [64])
# @pytest.mark.parametrize("d", [192])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 128),
        (1, 339),
        (3, 1024),
        (64, 800),
        (64, 256),
        (3, 799),
        (64, 2048),
        (16, 20000),
        # # (1, 128 * 1024),
        # # (16, 128 * 1024),
        # (128, 128),
        # (256, 512),  # To test appending KV with more than 1 block
        # (2048, 3577),  # Enough tile to test persistent scheduler
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(256, 128)])
def test_flash_attn_kvcache(
    seqlen_q,
    seqlen_k,
    d,
    varlen_q,
    has_batch_idx,
    has_leftpad,
    page_size,
    rotary_fraction,
    rotary_interleaved,
    has_rotary_seqlens,
    seqlen_new_eq_seqlen_q,
    causal,
    local,
    new_kv,
    has_learnable_sink,
    mha_type,
    dtype,
):
    if page_size is not None and seqlen_k % page_size != 0:
        pytest.skip()
    if seqlen_q > seqlen_k and new_kv:
        pytest.skip()
    if not new_kv and rotary_fraction > 0.0:
        pytest.skip()
    if rotary_fraction == 0.0 and has_rotary_seqlens:
        pytest.skip()
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 5
    # batch_size = 1
    batch_size_cache = batch_size if not has_batch_idx else batch_size * 2
    nheads = 6
    # nheads = 1
    # rotary_dim must be a multiple of 16, and must be <= d
    rotary_dim = math.floor(int(rotary_fraction * d) / 16) * 16
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0
    dtype_ref = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    # dv_vals = [128, d] if d > 128 and d <= 192 else ([256, 512, d] if d <= 64 else [d])
    dv_vals = [d]
    if dtype == torch.float8_e4m3fn:
        dv_vals = [d]
    # attention_chunk_vals = [torch.randint(1, seqlen_k * 2, (1,)).item(), 0] if (causal or local) else [0]
    attention_chunk_vals = [0]
    for dv, attention_chunk in itertools.product(dv_vals, attention_chunk_vals):
        # has_qv = d == 64 and dv >= 256
        has_qv = False
        q = (
            torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_ref)
            .to(dtype)
            .to(dtype_ref)
        )
        if has_qv:
            qv = (
                torch.randn(
                    batch_size, seqlen_q, nheads, dv, device=device, dtype=dtype_ref
                )
                .to(dtype)
                .to(dtype_ref)
            )
        else:
            qv = None
        if varlen_q:
            query_padding_mask = generate_random_padding_mask(
                seqlen_q, batch_size, device, mode="random"
            )
            q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, *rest = unpad_input(
                q, query_padding_mask
            )
            output_pad_fn = lambda output_unpad: pad_input(
                output_unpad, indices_q, batch_size, seqlen_q
            )
            qv_unpad = (
                rearrange(qv, "b s ... -> (b s) ...")[indices_q] if has_qv else None
            )
        else:
            query_padding_mask = None
            q_unpad = q
            qv_unpad = qv
            cu_seqlens_q, max_seqlen_q = None, None
        # Put window_size after QKV randn so that window_size changes from test to test
        window_size = (
            (None, None) if not local else torch.randint(0, seqlen_k, (2,)).tolist()
        )
        if has_learnable_sink:
            learnable_sink = torch.randn(nheads, dtype=torch.bfloat16, device=device)
        else:
            learnable_sink = None

        seqlen_new = (
            seqlen_q
            if seqlen_new_eq_seqlen_q
            else torch.randint(1, seqlen_q + 1, (1,)).item()
        )
        cu_seqlens_k_new = None
        key_new_padding_mask = None
        if new_kv:
            k = (
                torch.randn(
                    batch_size, seqlen_new, nheads_k, d, device=device, dtype=dtype_ref
                )
                .to(dtype)
                .to(dtype_ref)
            )
            v = (
                torch.randn(
                    batch_size, seqlen_new, nheads_k, dv, device=device, dtype=dtype_ref
                )
                .to(dtype)
                .to(dtype_ref)
            )
            if varlen_q:  # k & v are also varlen
                key_new_padding_mask = generate_random_padding_mask(
                    seqlen_new, batch_size, device, mode="random"
                )
                k_unpad, indices_k, cu_seqlens_k_new, *rest = unpad_input(
                    k, key_new_padding_mask
                )
                v_unpad, *rest = unpad_input(v, key_new_padding_mask)
            else:
                k_unpad, v_unpad = k, v
        else:
            k, v, k_unpad, v_unpad = None, None, None, None
        if page_size is None:
            k_cache = (
                torch.randn(
                    batch_size_cache,
                    seqlen_k,
                    nheads_k,
                    d,
                    device=device,
                    dtype=dtype_ref,
                )
                .to(dtype)
                .to(dtype_ref)
            )
            v_cache = (
                torch.randn(
                    batch_size_cache,
                    seqlen_k,
                    nheads_k,
                    dv,
                    device=device,
                    dtype=dtype_ref,
                )
                .to(dtype)
                .to(dtype_ref)
            )
            page_table = None
            num_blocks = None
        else:
            (
                k_cache,
                v_cache,
                page_table,
                k_cache_paged,
                v_cache_paged,
                num_blocks,
            ) = _generate_block_kvcache(
                seqlen_k,
                page_size,
                batch_size_cache,
                nheads_k,
                d,
                dv,
                device,
                dtype,
                dtype_ref,
            )
        cache_seqlens = torch.randint(
            0 if new_kv else 1,
            # If we don't use seqlen_q in the case of causal and rotary, cos/sin won't be long enough
            (
                (
                    seqlen_k
                    - (seqlen_q if (causal or local) and rotary_dim > 1 else seqlen_new)
                    + 1
                )
                if new_kv
                else (seqlen_k + 1)
            ),
            (batch_size,),
            dtype=torch.int32,
            device=device,
        )
        if has_leftpad:
            cache_leftpad = torch.cat(
                [
                    (
                        torch.randint(
                            0,
                            cache_seqlens[i].item(),
                            (1,),
                            dtype=torch.int32,
                            device=device,
                        )
                        if cache_seqlens[i].item() > 0
                        else torch.zeros(1, dtype=torch.int32, device=device)
                    )
                    for i in range(batch_size)
                ]
            )
        else:
            cache_leftpad = None
        if has_batch_idx:
            cache_batch_idx = torch.randperm(
                batch_size_cache, dtype=torch.int32, device=device
            )[:batch_size]
        else:
            cache_batch_idx = None
        arange = rearrange(torch.arange(seqlen_k, device=device), "s -> 1 s")
        cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
        if not new_kv:
            key_padding_mask = arange < cache_seqlens_expanded
        else:
            k_new_seqlens = (
                key_new_padding_mask.sum(-1, keepdims=True) if varlen_q else seqlen_new
            )
            key_padding_mask = arange < cache_seqlens_expanded + k_new_seqlens
        if has_leftpad:
            key_padding_mask = torch.logical_and(
                key_padding_mask,
                arange >= cache_leftpad.unsqueeze(-1).expand(-1, seqlen_k),
            )
        # cache_seqlens = torch.tensor([64], dtype=torch.int32, device=device)
        rotary_seqlens = cache_seqlens if not has_rotary_seqlens else cache_seqlens // 2
        if rotary_dim > 0:
            angle = (
                torch.rand(
                    seqlen_k if page_size is None else num_blocks * page_size,
                    rotary_dim // 2,
                    device=device,
                )
                * 2
                * math.pi
            )
            cos = torch.cos(angle).to(dtype=dtype_ref).to(dtype).to(dtype_ref)
            sin = torch.sin(angle).to(dtype=dtype_ref).to(dtype).to(dtype_ref)
            if causal or local:
                q_ro = apply_rotary_emb(
                    q,
                    cos,
                    sin,
                    seqlen_offsets=rotary_seqlens,
                    interleaved=rotary_interleaved,
                )
            else:
                q_ro = rearrange(
                    apply_rotary_emb(
                        rearrange(q, "b s h d -> b 1 (s h) d"),
                        cos,
                        sin,
                        seqlen_offsets=rotary_seqlens,
                        interleaved=rotary_interleaved,
                    ),
                    "b 1 (s h) d -> b s h d",
                    s=seqlen_q,
                )
            # q_ro = q
            k_ro = apply_rotary_emb(
                k,
                cos,
                sin,
                seqlen_offsets=rotary_seqlens,
                interleaved=rotary_interleaved,
            )
        else:
            cos, sin = None, None
            q_ro, k_ro = q, k
        # k_cache[:, 64:] = -1
        k_cache_ref = (
            k_cache if not has_batch_idx else k_cache[cache_batch_idx]
        ).clone()
        v_cache_ref = (
            v_cache if not has_batch_idx else v_cache[cache_batch_idx]
        ).clone()
        if new_kv:
            update_mask = torch.logical_and(
                cache_seqlens_expanded <= arange,
                arange < cache_seqlens_expanded + k_new_seqlens,
            )
            k_to_update = rearrange(k_ro, "b s ... -> (b s) ...")
            v_to_update = rearrange(v, "b s ... -> (b s) ...")
            if varlen_q:
                k_to_update = k_to_update[indices_k]
                v_to_update = v_to_update[indices_k]
            k_cache_ref[update_mask] = k_to_update
            v_cache_ref[update_mask] = v_to_update
        k_cache_rep = repeat(
            k_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k
        )
        v_cache_rep = repeat(
            v_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k
        )
        out_ref, _ = attention_ref(
            q_ro,
            k_cache_rep,
            v_cache_rep,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            qv=qv,
            window_size=window_size,
            learnable_sink=learnable_sink,
            attention_chunk=attention_chunk,
            key_leftpad=cache_leftpad,
        )
        out_pt, _ = attention_ref(
            q_ro,
            k_cache_rep,
            v_cache_rep,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            qv=qv,
            window_size=window_size,
            learnable_sink=learnable_sink,
            attention_chunk=attention_chunk,
            upcast=False,
            reorder_ops=True,
            key_leftpad=cache_leftpad,
            intermediate_dtype=dtype if dtype == torch.float8_e4m3fn else None,
        )
        q = q.to(dtype)
        q_unpad = q_unpad.to(dtype) if varlen_q else None
        k_cache = k_cache.to(dtype)
        v_cache = v_cache.to(dtype)
        k_cache_paged = k_cache_paged.to(dtype) if page_size is not None else None
        v_cache_paged = v_cache_paged.to(dtype) if page_size is not None else None
        k = k.to(dtype) if k is not None else None
        v = v.to(dtype) if v is not None else None
        k_unpad = k_unpad.to(dtype) if k_unpad is not None else None
        v_unpad = v_unpad.to(dtype) if v_unpad is not None else None
        qv = qv.to(dtype) if qv is not None else None
        qv_unpad = qv_unpad.to(dtype) if (varlen_q and qv is not None) else None
        cos = cos.to(dtype) if cos is not None else None
        sin = sin.to(dtype) if sin is not None else None
        k_cache_saved = k_cache.clone() if page_size is None else k_cache_paged.clone()
        v_cache_saved = v_cache.clone() if page_size is None else v_cache_paged.clone()
        # num_splits_vals = [1, 0]
        num_splits_vals = [1]
        # precompute_metadata_vals = [False, True]
        precompute_metadata_vals = [False]
        for num_splits, precompute_metadata in itertools.product(
            num_splits_vals, precompute_metadata_vals
        ):
            # if precompute_metadata:
            #     scheduler_metadata = get_scheduler_metadata(
            #         batch_size, max_seqlen_q if varlen_q else seqlen_q, seqlen_k, nheads, nheads_k, d,
            #         cache_seqlens, q.dtype, headdim_v=dv, cu_seqlens_q=cu_seqlens_q,
            #         cu_seqlens_k_new=cu_seqlens_k_new, cache_leftpad=cache_leftpad,
            #         max_seqlen_k_new=seqlen_new, page_size=page_size,
            #         causal=causal, window_size=window_size, attention_chunk=attention_chunk,
            #         num_splits=num_splits
            #     )
            # else:
            #     scheduler_metadata = None
            scheduler_metadata = None
            # Repeat to test metadata reuse
            for _ in range(1 if not precompute_metadata else 2):
                if page_size is None:
                    k_cache.copy_(k_cache_saved)
                    v_cache.copy_(v_cache_saved)
                else:
                    k_cache_paged.copy_(k_cache_saved)
                    v_cache_paged.copy_(v_cache_saved)
                # For FA4, use flash_attn_varlen_func directly instead of flash_attn_with_kvcache
                # This matches the pattern from the original FA4 test
                out, lse = flash_attn_varlen_func(
                    q if not varlen_q else q_unpad,
                    k_cache if page_size is None else k_cache_paged,
                    v_cache if page_size is None else v_cache_paged,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=None,  # FA4 doesn't use cu_seqlens_k for KV cache
                    # max_seqlen_q and max_seqlen_k not needed for FA4
                    seqused_k=cache_seqlens,  # Use cache_seqlens as seqused_k
                    page_table=page_table,
                    causal=causal,
                    window_size=window_size,
                    sinks=learnable_sink,  # FA4 uses learnable_sink, not sinks
                    softcap=0.0,
                    pack_gqa=None,
                    return_softmax_lse=True,
                    ver=4,  # Use FA4
                )
                if varlen_q:
                    out = output_pad_fn(out)
                # out = flash_attn_with_kvcache(
                #     q, k_cache, v_cache, cache_seqlens=cache_seqlens, causal=causal, window_size=window_size
                # )
                # out = flash_attn_with_kvcache(q, k_cache, v_cache, causal=causal, window_size=window_size)
                # qk = torch.einsum("bqhd,bkhd->bhqk", q, k_cache_ref)
                # m = qk.amax(-1, keepdim=True)
                # s_tmp = torch.exp((qk - m) / math.sqrt(d))
                # o1 = torch.einsum('bhst,bthd->bshd', s_tmp, v_cache_ref)
                # lse_ref = torch.logsumexp(qk / math.sqrt(d), -1)
                # probs = torch.softmax(qk, dim=-1)
                print(f"Output max diff: {(out - out_ref).abs().max().item()}")
                print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
                print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
                print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
                # breakpoint()

                # Check that FlashAttention's numerical error is at most twice the numerical error
                # of a Pytorch implementation.
                if new_kv:
                    if page_size is None:
                        k_cache_select = (
                            k_cache.to(dtype_ref)
                            if not has_batch_idx
                            else k_cache.to(dtype_ref)[cache_batch_idx]
                        )
                        v_cache_select = (
                            v_cache.to(dtype_ref)
                            if not has_batch_idx
                            else v_cache.to(dtype_ref)[cache_batch_idx]
                        )
                    else:
                        k_cache_select = rearrange(
                            k_cache_paged.to(dtype_ref)[
                                (
                                    page_table
                                    if not has_batch_idx
                                    else page_table[cache_batch_idx]
                                ).flatten()
                            ],
                            "(b nblocks) block_size ... -> b (nblocks block_size) ...",
                            b=batch_size,
                        )[:, :seqlen_k].to(dtype_ref)
                        v_cache_select = rearrange(
                            v_cache_paged.to(dtype_ref)[
                                (
                                    page_table
                                    if not has_batch_idx
                                    else page_table[cache_batch_idx]
                                ).flatten()
                            ],
                            "(b nblocks) block_size ... -> b (nblocks block_size) ...",
                            b=batch_size,
                        )[:, :seqlen_k].to(dtype_ref)
                    k_cache_ref = k_cache_ref.to(dtype).to(dtype_ref)
                    v_cache_ref = v_cache_ref.to(dtype).to(dtype_ref)
                    if dtype is not torch.float8_e4m3fn:
                        assert torch.equal(v_cache_select, v_cache_ref)
                    else:
                        assert torch.allclose(
                            v_cache_select, v_cache_ref, rtol=1e-3, atol=1e-3
                        )
                    # breakpoint()
                    # if rotary_dim == 0 and dtype is not torch.float8_e4m3fn:
                    if rotary_dim == 0:
                        assert torch.equal(k_cache_select, k_cache_ref)
                    else:
                        # if not torch.allclose(k_cache_select, k_cache_ref, rtol=1e-3, atol=1e-3):
                        #     breakpoint()
                        if dtype is not torch.float8_e4m3fn:
                            assert torch.allclose(
                                k_cache_select, k_cache_ref, rtol=1e-3, atol=1e-3
                            )
                        else:
                            assert torch.allclose(
                                k_cache_select, k_cache_ref, rtol=1e-1, atol=1e-1
                            )
                mult = 4 if dtype == torch.float8_e4m3fn else 2
                assert (out - out_ref).abs().max().item() <= mult * (
                    out_pt - out_ref
                ).abs().max().item() + 1e-5
                mult_mean = 3 if dtype == torch.float8_e4m3fn else 1.5
                assert (out - out_ref).abs().mean().item() <= mult_mean * (
                    out_pt - out_ref
                ).abs().mean().item()


def _generate_block_kvcache(
    seqlen_k, page_size, batch_size, nheads_k, d, dv, device, dtype, dtype_ref
):
    num_blocks = math.ceil(seqlen_k / page_size) * batch_size * 3
    k_cache_paged = (
        torch.randn(num_blocks, page_size, nheads_k, d, device=device, dtype=dtype_ref)
        .to(dtype)
        .to(dtype_ref)
    )
    v_cache_paged = (
        torch.randn(num_blocks, page_size, nheads_k, dv, device=device, dtype=dtype_ref)
        .to(dtype)
        .to(dtype_ref)
    )
    page_table = rearrange(
        torch.randperm(num_blocks, dtype=torch.int32, device=device),
        "(b nblocks) -> b nblocks",
        b=batch_size,
    )
    k_cache = rearrange(
        k_cache_paged[page_table.flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    v_cache = rearrange(
        v_cache_paged[page_table.flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    return k_cache, v_cache, page_table, k_cache_paged, v_cache_paged, num_blocks


if __name__ == "__main__":
    pytest.main([__file__])
