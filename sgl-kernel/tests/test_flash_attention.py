# Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/hopper/test_flash_attn.py
import itertools
import math
import os

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

apply_rotary_emb = None


def is_fa3_supported(device=None) -> bool:
    # FA3 can fail without a enough shared memory for a some shapes, currently
    #  only 8.0 and 8.7 have enough shared memory for all shapes
    #  https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-8-x
    #  now sgl-kernel only build fa3 for sm90a && cuda >= 12.4
    return (
        (torch.cuda.get_device_capability(device)[0] >= 9)
        and (torch.version.cuda >= "12.4")
        # or torch.cuda.get_device_capability(device) == (8, 0)
        # or torch.cuda.get_device_capability(device) == (8, 7)
    )


DISABLE_BACKWARD = True
# For CI test, we close them to True.
# DISABLE_SPLIT = os.getenv("FLASH_ATTENTION_DISABLE_SPLIT", "FALSE") == "TRUE"
# DISABLE_PAGEDKV = os.getenv("FLASH_ATTENTION_DISABLE_PAGEDKV", "FALSE") == "TRUE"
# DISABLE_APPENDKV = os.getenv("FLASH_ATTENTION_DISABLE_APPENDKV", "FALSE") == "TRUE"
# DISABLE_LOCAL = os.getenv("FLASH_ATTENTION_DISABLE_LOCAL", "FALSE") == "TRUE"
# DISABLE_SOFTCAP = os.getenv("FLASH_ATTENTION_DISABLE_SOFTCAP", "FALSE") == "TRUE"
# DISABLE_PACKGQA = os.getenv("FLASH_ATTENTION_DISABLE_PACKGQA", "FALSE") == "TRUE"
# DISABLE_FP16 = os.getenv("FLASH_ATTENTION_DISABLE_FP16", "FALSE") == "TRUE"
# DISABLE_FP8 = (
#     os.getenv("FLASH_ATTENTION_DISABLE_FP8", "FALSE") == "TRUE"
#     or torch.cuda.get_device_capability("cuda")[0] < 9
# )

DISABLE_SPLIT = True
DISABLE_PAGEDKV = True
DISABLE_APPENDKV = True
DISABLE_LOCAL = True
DISABLE_SOFTCAP = True
DISABLE_PACKGQA = True
DISABLE_FP16 = True
DISABLE_FP8 = True


# Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/hopper/padding.py
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


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
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
    if window_size[0] < 0:
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
    window_size=(-1, -1),  # -1 means infinite window size
    sink_token_length=0,
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
    if window_size[0] >= 0 or window_size[1] >= 0:
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
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
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
    if window_size[0] >= 0 or window_size[1] >= 0:
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
    not is_fa3_supported(),
    reason="flash_attn at sgl-kernel is only supported on sm90 and above",
)
# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16] + ([torch.float8_e4m3fn] if not DISABLE_FP8 else [])
)
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
# @pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("mha_type", ["mha"])
@pytest.mark.parametrize("new_kv", [False] + ([True] if not DISABLE_APPENDKV else []))
# @pytest.mark.parametrize("new_kv", [True])
# @pytest.mark.parametrize(
#     "causal,local",
#     [(False, False), (True, False)] + ([(False, True)] if not DISABLE_LOCAL else []),
# )
# @pytest.mark.parametrize("causal,local", [(False, False), (True, False)])
@pytest.mark.parametrize("causal,local", [(False, False)])
@pytest.mark.parametrize(
    "seqlen_new_eq_seqlen_q", [True, False] if not DISABLE_APPENDKV else [True]
)
# @pytest.mark.parametrize("seqlen_new_eq_seqlen_q", [True])
# @pytest.mark.parametrize("has_rotary_seqlens", [False, True])
@pytest.mark.parametrize("has_rotary_seqlens", [False])
@pytest.mark.parametrize(
    "rotary_interleaved", [False, True] if not DISABLE_APPENDKV else [False]
)
# @pytest.mark.parametrize("rotary_interleaved", [True])
@pytest.mark.parametrize(
    "rotary_fraction",
    (
        [0.0, 0.5, 1.0]
        if (not DISABLE_APPENDKV) and (apply_rotary_emb is not None)
        else [0.0]
    ),
)
# @pytest.mark.parametrize("rotary_fraction", [0.0])
@pytest.mark.parametrize(
    "page_size", [None] + ([1, 4, 128] if not DISABLE_PAGEDKV else [])
)
# @pytest.mark.parametrize("page_size", [None])
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
        # (1, 128 * 1024),
        # (16, 128 * 1024),
        (128, 128),
        (256, 512),  # To test appending KV with more than 1 block
        (2048, 3577),  # Enough tile to test persistent scheduler
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
    mha_type,
    dtype,
):
    from sgl_kernel.flash_attn import flash_attn_with_kvcache

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
    dv_vals = [128, d] if d > 128 and d <= 192 else ([256, 512, d] if d <= 64 else [d])
    if dtype == torch.float8_e4m3fn:
        dv_vals = [d]
    for dv in dv_vals:
        has_qv = d == 64 and dv >= 256
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
        window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))

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
        num_splits_vals = [1, 0] if not DISABLE_SPLIT else [1]
        precompute_metadata_vals = [False]
        for num_splits, precompute_metadata in itertools.product(
            num_splits_vals, precompute_metadata_vals
        ):
            scheduler_metadata = None
            # Repeat to test metadata reuse
            for _ in range(1 if not precompute_metadata else 2):
                if page_size is None:
                    k_cache.copy_(k_cache_saved)
                    v_cache.copy_(v_cache_saved)
                else:
                    k_cache_paged.copy_(k_cache_saved)
                    v_cache_paged.copy_(v_cache_saved)
                out, lse, *rest = flash_attn_with_kvcache(
                    q if not varlen_q else q_unpad,
                    k_cache if page_size is None else k_cache_paged,
                    v_cache if page_size is None else v_cache_paged,
                    k if not new_kv or not varlen_q else k_unpad,
                    v if not new_kv or not varlen_q else v_unpad,
                    qv=qv if not varlen_q else qv_unpad,
                    rotary_cos=cos,
                    rotary_sin=sin,
                    cache_seqlens=cache_seqlens,
                    cache_batch_idx=cache_batch_idx,
                    cache_leftpad=cache_leftpad,
                    page_table=page_table,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k_new,
                    max_seqlen_q=max_seqlen_q,
                    rotary_seqlens=rotary_seqlens,
                    causal=causal,
                    window_size=window_size,
                    rotary_interleaved=rotary_interleaved,
                    scheduler_metadata=scheduler_metadata,
                    num_splits=num_splits,
                    return_softmax_lse=True,
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
