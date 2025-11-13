import math
import random
from typing import Optional, Tuple

import pytest
import torch
import triton
from sgl_kernel.flash_mla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
    get_mla_metadata,
)

# ================ prefill usage ================ #
S_Q_PREFILL = [1, 62]
KV_TOPK_PREFILL = [
    # Regular shapes
    (128, 128),
    (256, 256),
    (512, 512),
    # Irregular shapes
    (592, 128),
    (1840, 256),
    (1592, 384),
    (1521, 512),
    # Irregular shapes with OOB TopK
    (95, 128),
    (153, 256),
    (114, 384),
]

# ================= decode usage ================= #
B_DECODE = [1, 2, 6, 64]
S_Q_DECODE = [1, 2, 4]
S_K_DECODE = [20, 140, 4096]
IS_VARLEN = [False, True]
CAUSAL_TOPK = [(True, None), (False, None), (False, 128), (False, 2048)]
DTYPE = [torch.float16, torch.bfloat16]


def quantize_k_cache(
    input_k_cache: torch.Tensor,  # (num_blocks, block_size, h_k, d)
    dv: int,
    tile_size: int = 128,
) -> torch.Tensor:
    """
    Quantize the k-cache
    Return a tensor with shape (num_blocks, block_size, h_k, dv + 4(dv/tile_size) + t(d-dv)) of dtype uint8_t, where t = input_k_cache.element_size()
    For more detail about the layout of K/V, please refer to comments in flash_mla_interface.py or README.md
    """
    assert dv % tile_size == 0
    num_tiles = dv // tile_size
    num_blocks, block_size, h_k, d = input_k_cache.shape
    assert h_k == 1
    input_k_cache = input_k_cache.squeeze(2)  # [num_blocks, block_size, d]
    input_elem_size = input_k_cache.element_size()

    result = torch.empty(
        (num_blocks, block_size, dv + num_tiles * 4 + input_elem_size * (d - dv)),
        dtype=torch.float8_e4m3fn,
        device=input_k_cache.device,
    )
    result_k_nope_part = result[..., :dv]
    result_k_scale_factor = result[..., dv : dv + num_tiles * 4].view(torch.float32)
    result_k_rope_part = result[..., dv + num_tiles * 4 :].view(input_k_cache.dtype)
    result_k_rope_part[:] = input_k_cache[..., dv:]

    for tile_idx in range(0, num_tiles):
        cur_scale_factors_inv = (
            torch.abs(
                input_k_cache[..., tile_idx * tile_size : (tile_idx + 1) * tile_size]
            )
            .max(dim=-1)
            .values
            / 448.0
        )  # [num_blocks, block_size]
        result_k_scale_factor[:, :, tile_idx] = cur_scale_factors_inv

        cur_scale_factors_inv.unsqueeze_(-1)  # [num_blocks, block_size, 1]
        cur_quantized_nope = (
            input_k_cache[
                ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
            ].float()
            / cur_scale_factors_inv.float()
        ).to(torch.float8_e4m3fn)
        result_k_nope_part[..., tile_idx * tile_size : (tile_idx + 1) * tile_size] = (
            cur_quantized_nope
        )

    result = result.view(num_blocks, block_size, 1, -1)
    return result


def dequantize_k_cache(
    quant_k_cache: torch.Tensor,  # (num_blocks, block_size, 1, bytes_per_token)
    dv: int = 512,
    tile_size: int = 128,
    d: int = 576,
) -> torch.Tensor:
    """
    De-quantize the k-cache
    """
    assert dv % tile_size == 0
    num_tiles = dv // tile_size
    num_blocks, block_size, h_k, _ = quant_k_cache.shape
    assert h_k == 1
    result = torch.empty(
        (num_blocks, block_size, d), dtype=torch.bfloat16, device=quant_k_cache.device
    )

    quant_k_cache = quant_k_cache.view(num_blocks, block_size, -1)

    input_nope = quant_k_cache[..., :dv]
    input_scale = quant_k_cache[..., dv : dv + num_tiles * 4].view(torch.float32)
    input_rope = quant_k_cache[..., dv + num_tiles * 4 :].view(torch.bfloat16)
    result[..., dv:] = input_rope

    for tile_idx in range(0, num_tiles):
        cur_nope = input_nope[
            ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
        ].to(torch.float32)
        cur_scales = input_scale[..., tile_idx].unsqueeze(-1)
        result[..., tile_idx * tile_size : (tile_idx + 1) * tile_size] = (
            cur_nope * cur_scales
        )

    result = result.view(num_blocks, block_size, 1, d)
    return result


def cdiv(x: int, y: int):
    return (x + y - 1) // y


def get_window_size(causal, window):
    if window > 0:
        window_size = (window - 1, 0) if causal else (window - 1, window - 1)
    else:
        window_size = (-1, -1)
    return window_size


def get_attn_bias(s_q, s_k, causal, window):
    attn_bias = torch.zeros(s_q, s_k, dtype=torch.float32, device="cuda")
    if causal:
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool, device="cuda").tril(
            diagonal=s_k - s_q
        )
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    if window > 0:
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool, device="cuda").tril(
            diagonal=s_k - s_q - window
        )
        attn_bias.masked_fill_(temp_mask, float("-inf"))
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool, device="cuda").tril(
            diagonal=s_k - s_q + window - 1
        )
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    return attn_bias


def sdpa(query, key, value, attn_bias, softmax_scale=None):
    query = query.float().transpose(-3, -2)
    key = key.float().transpose(-3, -2)
    value = value.float().transpose(-3, -2)
    key = key.repeat_interleave(h // h_k, dim=-3)
    value = value.repeat_interleave(h // h_k, dim=-3)
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** (-0.5)
    attn_weight = (query @ key.transpose(-2, -1)) * softmax_scale
    attn_weight += attn_bias
    lse = attn_weight.logsumexp(dim=-1)
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    return attn_weight.to(query.dtype) @ value, lse


def sdpa_checkpoint(*args, **kwargs):
    return checkpoint(sdpa, *args, use_reentrant=False, **kwargs)


def reference_torch_prefill(
    s_q, s_kv, topk, indices, q, kv, sm_scale: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def log2sumexp2(a: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.logsumexp(a * math.log(2), dim=dim) * math.log2(math.e)

    indices = indices[0, :, 0, :]  # [s_q, topk]
    invalid_indices_mask = (indices < 0) | (indices >= s_kv)
    qs = q[0, :, :, :].float()  # [s_q, h_q, d_qk]
    kvs = kv[0, :, 0, :].float()  # [s_kv, d_qk]

    kvs = torch.index_select(
        kvs, 0, indices.masked_fill(invalid_indices_mask, 0).flatten()
    ).view(
        s_q, topk, 576
    )  # [s_q, topk, d_qk]
    attn_score = qs @ kvs.transpose(1, 2)  # [s_q, h_q, topk]
    attn_score.masked_fill_(invalid_indices_mask.unsqueeze(1), float("-inf"))
    attn_score *= sm_scale * math.log2(math.e)
    max_logits = torch.max(attn_score, dim=-1)[0]  # [s_q, h_q]
    lse = log2sumexp2(attn_score, dim=-1)  # [s_q, h_q]
    attn_score = torch.exp2(attn_score - lse.unsqueeze(-1))  # [s_q, h_q, topk]
    result = attn_score @ kvs[:, :, :512]
    return (max_logits, lse, result)


def reference_torch_decode(
    cache_seqlens: torch.Tensor,  # [batch_size]
    block_table: torch.Tensor,  # [batch_size, ?]
    q: torch.Tensor,  # [batch_size, s_q, h_q, d]
    blocked_k: torch.Tensor,  # [?, block_size, h_kv, d]
    dv: int,
    is_causal: bool,
    indices: Optional[torch.Tensor] = None,  # [batch_size, s_q, topk]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A reference implementation in PyTorch
    """

    def get_topk_attn_mask(s_q: int, s_k: int, indices: torch.Tensor):
        mask = torch.zeros(s_q, s_k, dtype=torch.bool, device="cuda")
        for i in range(s_q):
            cur_indices = indices[i]
            valid_indices = cur_indices[cur_indices != -1]
            mask[i, valid_indices] = True
        return mask

    def scaled_dot_product_attention(
        batch_idx: int,
        query: torch.Tensor,  # [h_q, s_q, d]
        kv: torch.Tensor,  # [h_kv, s_k, d]
        dv: int,
        is_causal,
        indices: Optional[torch.Tensor],  # [s_q, topk]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_q = query.size(0)
        h_kv = kv.size(0)
        s_q = query.shape[-2]
        s_k = kv.shape[-2]
        query = query.float()
        kv = kv.float()
        if h_kv != 1:
            kv = kv.repeat_interleave(h_q // h_kv, dim=0)
        kv[kv != kv] = 0.0
        attn_weight = query @ kv.transpose(-2, -1)  # [h_q, s_q, s_k]
        if (is_causal and query.size(1) > 1) or indices is not None:
            mask = torch.ones(s_q, s_k, dtype=torch.bool, device="cuda")
            if is_causal:
                assert indices is None
                mask = mask.tril(diagonal=s_k - s_q)
            if indices is not None:
                mask &= get_topk_attn_mask(s_q, s_k, indices)
            attn_bias = torch.zeros(s_q, s_k, dtype=torch.float, device="cuda")
            attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
            attn_weight += attn_bias.to(q.dtype)
        attn_weight /= math.sqrt(query.size(-1))
        lse = attn_weight.logsumexp(dim=-1)  # [h_q, s_q]
        attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
        output = attn_weight @ kv[..., :dv]  # [h_q, s_q, dv]
        # Correct for q tokens which has no attendable k
        lonely_q_mask = lse == float("-inf")
        output[lonely_q_mask.unsqueeze(-1).broadcast_to(h_q, s_q, dv)] = 0.0
        lse[lonely_q_mask] = float("+inf")

        return output, lse

    b, s_q, h_q, d = q.size()
    block_size = blocked_k.size(1)
    h_kv = blocked_k.size(2)
    cache_seqlens_cpu = cache_seqlens.cpu()
    out_ref = torch.empty(b, s_q, h_q, dv, dtype=torch.float32, device="cuda")
    lse_ref = torch.empty(b, h_q, s_q, dtype=torch.float32, device="cuda")
    for i in range(b):
        cur_len = cache_seqlens_cpu[i].item()
        cur_num_blocks = cdiv(cur_len, block_size)
        cur_block_indices = block_table[i][0:cur_num_blocks]
        cur_kv = blocked_k[cur_block_indices].view(-1, h_kv, d)[:cur_len, ...]
        cur_out, cur_lse = scaled_dot_product_attention(
            i,
            q[i].transpose(0, 1),
            cur_kv.transpose(0, 1),
            dv,
            is_causal,
            indices[i] if indices is not None else None,
        )
        out_ref[i] = cur_out.transpose(0, 1)
        lse_ref[i] = cur_lse
    out_ref = out_ref.to(torch.bfloat16)
    return out_ref, lse_ref


@pytest.mark.parametrize("s_q", S_Q_PREFILL)
@pytest.mark.parametrize("kv_topk", KV_TOPK_PREFILL)
@torch.inference_mode()
def test_flashmla_prefill(
    s_q: int,
    kv_topk: Tuple[int, int],
):

    torch.cuda.empty_cache()

    q = torch.randn((1, s_q, 128, 576), dtype=torch.bfloat16, device="cuda") / 10
    kv = torch.randn((1, kv_topk[0], 1, 576), dtype=torch.bfloat16, device="cuda") / 10

    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)

    indices = torch.full(
        (1, s_q, 1, kv_topk[1]), kv_topk[0], dtype=torch.int32, device="cuda"
    )
    for s in range(s_q):
        # NOTE We use the following method to generate indices so that most indices lies within [s_kv-20000, s_kv), which is more realistic for sparse attention
        near_mask = (
            torch.randint(0, 32, (min(kv_topk[1], kv_topk[0]),), device="cuda") < 31
        )
        cur_indices = torch.randperm(kv_topk[0], device="cuda")[: kv_topk[1]]
        cur_indices[near_mask] = torch.randint(
            max(0, kv_topk[0] - 20000),
            kv_topk[0] - 1,
            (near_mask.sum().item(),),
            device="cuda",
        )
        if len(cur_indices) < kv_topk[1]:
            cur_indices = torch.cat(
                [
                    cur_indices,
                    torch.full(
                        (kv_topk[1] - len(cur_indices),), 2147480000, device="cuda"
                    ),
                ]
            )
        cur_indices = cur_indices[torch.randperm(kv_topk[1], device="cuda")]
        indices[0, s, 0] = cur_indices
    indices = indices.to(q.device)

    sm_scale = 1 / math.sqrt(576)
    torch.cuda.synchronize()

    ans_out, ans_max_logits, ans_lse = flash_mla_sparse_fwd(
        q.squeeze(0), kv.squeeze(0), indices.squeeze(0), sm_scale=sm_scale
    )

    ans_out, ans_max_logits, ans_lse = (
        ans_out.float(),
        ans_max_logits.float(),
        ans_lse.float(),
    )

    torch.cuda.synchronize()
    ref_max_logits, ref_lse, ref_out = reference_torch_prefill(
        s_q, kv_topk[0], kv_topk[1], indices, q, kv, sm_scale
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(ans_out, ref_out, atol=8e-4, rtol=2.01 / 128)
    torch.testing.assert_close(
        ans_max_logits,
        ref_max_logits,
        atol=1e-6,
        rtol=2.01 / 65536,
    )
    torch.testing.assert_close(ans_lse, ref_lse, atol=1e-6, rtol=2.01 / 65536)


@pytest.mark.parametrize("b", B_DECODE)
@pytest.mark.parametrize("s_q", S_Q_DECODE)
@pytest.mark.parametrize("s_k", S_K_DECODE)
@pytest.mark.parametrize("is_varlen", IS_VARLEN)
@pytest.mark.parametrize("causal_topk", CAUSAL_TOPK)
@pytest.mark.parametrize("dtype", DTYPE)
@torch.inference_mode()
def test_flash_mla_decode(
    b: int,
    s_q: int,
    s_k: int,
    is_varlen: bool,
    causal_topk: Tuple[bool, Optional[int]],
    dtype: torch.dtype,
):
    d = 576
    dv = 512
    block_size = 64
    h_q = 128
    h_kv = 1
    is_causal = causal_topk[0]
    topk = causal_topk[1]

    # Generating test data
    torch.cuda.synchronize()

    cache_seqlens_cpu = torch.full((b,), s_k, dtype=torch.int32, device="cpu")
    if is_varlen:
        for i in range(b):
            cache_seqlens_cpu[i] = max(random.normalvariate(s_k, s_k / 2), s_q)

    max_seqlen = cache_seqlens_cpu.max().item()
    max_seqlen_pad = cdiv(max_seqlen, 256) * 256
    cache_seqlens = cache_seqlens_cpu.cuda()

    q = torch.randn(b, s_q, 128, d, dtype=torch.bfloat16, device="cuda")
    q.clamp_(min=-1.0, max=1.0)

    block_table = torch.arange(
        b * max_seqlen_pad // block_size, dtype=torch.int32, device="cuda"
    ).view(b, max_seqlen_pad // block_size)
    block_table = block_table.view(-1)[torch.randperm(block_table.numel())].view(b, -1)
    blocked_k = (
        torch.randn(
            block_table.numel(),
            block_size,
            h_kv,
            d,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 10
    )
    blocked_k.clamp_(min=-1.0, max=1.0)

    if topk is None:
        for i in range(b):
            cur_len = cache_seqlens_cpu[i].item()
            cur_num_blocks = cdiv(cur_len, block_size)
            blocked_k[block_table[i][cur_num_blocks:]] = float("nan")
            if cur_len % block_size != 0:
                blocked_k[block_table[i][cur_num_blocks - 1]][
                    cur_len % block_size :
                ] = float("nan")
            block_table[i][cur_num_blocks:] = 2147480000
        abs_indices = None
        indices_in_kvcache = None
    else:
        block_table_cpu = block_table.cpu()
        abs_indices = torch.empty(b, s_q, topk, dtype=torch.int32, device="cpu")
        indices_in_kvcache = torch.empty(b, s_q, topk, dtype=torch.int32, device="cpu")
        for i in range(b):
            # Generate indices
            for j in range(s_q):
                cur_abs_indices = torch.randperm(
                    int(cache_seqlens_cpu[i].item()), device="cpu"
                )[:topk]
                cur_blocked_indices = block_table_cpu[
                    i, cur_abs_indices // block_size
                ] * block_size + (cur_abs_indices % block_size)
                if len(cur_abs_indices) < topk:
                    pad_len = topk - len(cur_abs_indices)
                    cur_abs_indices = torch.cat(
                        [cur_abs_indices, torch.full((pad_len,), -1, device="cpu")]
                    )
                    cur_blocked_indices = torch.cat(
                        [cur_blocked_indices, torch.full((pad_len,), -1, device="cpu")]
                    )

                # Mask KV
                perm = torch.randperm(topk, device="cpu")
                cur_abs_indices = cur_abs_indices[perm]
                cur_blocked_indices = cur_blocked_indices[perm]

                abs_indices[i, j, :] = cur_abs_indices
                indices_in_kvcache[i, j, :] = cur_blocked_indices

        # Mask nonused KV as NaN
        all_indices = indices_in_kvcache.flatten().tolist()
        all_indices = list(set(all_indices))
        if -1 in all_indices:
            all_indices.remove(-1)
        all_indices = torch.tensor(all_indices, dtype=torch.int32, device="cpu")

        blocked_k = blocked_k.view(-1, h_kv, d)
        nonused_indices_mask = torch.ones(
            blocked_k.size(0) * blocked_k.size(1), dtype=torch.bool, device="cpu"
        )
        nonused_indices_mask[all_indices] = False
        blocked_k[nonused_indices_mask, :, :] = float("nan")
        blocked_k = blocked_k.view(-1, block_size, h_kv, d)

        abs_indices = abs_indices.to(q.device)
        indices_in_kvcache = indices_in_kvcache.to(q.device)

    is_fp8 = topk is not None
    if is_fp8:
        # The quantization error may be too large to be distinguished from wrong kernels
        # So we quantize and de-quantize kv-cache here to mitigate quantization error
        blocked_k_quantized = quantize_k_cache(blocked_k, dv, 128)
        blocked_k_dequantized = dequantize_k_cache(blocked_k_quantized)
        blocked_k = blocked_k_dequantized

    # Get schedule metadata
    torch.cuda.synchronize()
    tile_scheduler_metadata, num_splits = get_mla_metadata(
        cache_seqlens, s_q * h_q // h_kv, h_kv, h_q, is_fp8, topk
    )
    torch.cuda.synchronize()

    out_ans, lse_ans = flash_mla_with_kvcache(
        q,
        blocked_k if not is_fp8 else blocked_k_quantized,  # type: ignore
        block_table,
        cache_seqlens,
        dv,
        tile_scheduler_metadata,
        num_splits,
        causal=is_causal,
        is_fp8_kvcache=is_fp8,
        indices=indices_in_kvcache,
    )

    out_ref, lse_ref = reference_torch_decode(
        cache_seqlens, block_table, q, blocked_k, dv, is_causal, abs_indices
    )
    torch.testing.assert_close(out_ans, out_ref, atol=8e-4, rtol=2.01 / 128)
    torch.testing.assert_close(lse_ans, lse_ref, atol=1e-6, rtol=8.01 / 65536)


@pytest.mark.parametrize("b", [128])
@pytest.mark.parametrize("s_q", [1, 2])
@pytest.mark.parametrize("mean_sk", [4096, 8192, 16384])
@pytest.mark.parametrize("h_q", [16, 32, 64, 128])
@pytest.mark.parametrize("h_kv", [1])
@pytest.mark.parametrize("d", [576])
@pytest.mark.parametrize("dv", [512])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("torch_dtype", [torch.float8_e4m3fn])
@torch.inference_mode()
def test_flash_mla_fp8(
    b, s_q, mean_sk, h_q, h_kv, d, dv, block_size, causal, varlen, torch_dtype
):
    device = torch.device("cuda:0")
    init_dtype = torch.bfloat16 if torch_dtype == torch.float8_e4m3fn else torch_dtype
    torch.set_default_dtype(init_dtype)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    random.seed(0)

    use_fp8 = torch_dtype == torch.float8_e4m3fn
    cache_seqlens = torch.full((b,), mean_sk, dtype=torch.int32)
    if varlen:
        for i in range(b):
            cache_seqlens[i] = max(random.normalvariate(mean_sk, mean_sk / 2), s_q)
    total_seqlens = cache_seqlens.sum().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256

    q = torch.randn(b, s_q, h_q, d)
    block_table = torch.arange(
        b * max_seqlen_pad // block_size, dtype=torch.int32
    ).view(b, max_seqlen_pad // block_size)
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d)
    for i in range(b):
        blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item() :] = (
            float("nan")
        )
    blocked_v = blocked_k[..., :dv]

    tile_scheduler_metadata, num_splits = get_mla_metadata(
        cache_seqlens, s_q * h_q // h_kv, h_kv, is_fp8_kvcache=use_fp8
    )

    init_dtype = q.dtype
    if use_fp8:
        fp8_dtype = torch.float8_e4m3fn
        descale_q = torch.ones((1), dtype=torch.float32)
        descale_k = torch.ones((1), dtype=torch.float32)

        q = q.to(fp8_dtype)
        blocked_k = blocked_k.to(fp8_dtype)
        blocked_v = blocked_v.to(fp8_dtype)
    else:
        descale_q = None
        descale_k = None

    def flash_mla():
        return flash_mla_with_kvcache(
            q,
            blocked_k,
            block_table,
            cache_seqlens,
            dv,
            tile_scheduler_metadata,
            num_splits,
            causal=causal,
            descale_q=descale_q,
            descale_k=descale_k,
        )

    def scaled_dot_product_attention(query, key, value, is_causal=False):
        query = query.float()
        key = key.float()
        value = value.float()
        key = key.repeat_interleave(h_q // h_kv, dim=0)
        value = value.repeat_interleave(h_q // h_kv, dim=0)
        attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        if is_causal:
            s_q = query.shape[-2]
            s_k = key.shape[-2]
            attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
            temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)
            attn_weight += attn_bias
        lse = attn_weight.logsumexp(dim=-1)
        attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
        return attn_weight @ value, lse

    def ref_mla():
        q_ = (q.to(torch.float) * descale_q).to(init_dtype) if use_fp8 else q
        blocked_k_ = (
            (blocked_k.to(torch.float) * descale_k).to(init_dtype)
            if use_fp8
            else blocked_k
        )
        blocked_v_ = (
            (blocked_v.to(torch.float) * descale_k).to(init_dtype)
            if use_fp8
            else blocked_v
        )
        out = torch.empty(b, s_q, h_q, dv, dtype=torch.float32)
        lse = torch.empty(b, h_q, s_q, dtype=torch.float32)
        for i in range(b):
            begin = i * max_seqlen_pad
            end = begin + cache_seqlens[i]
            out_i, lse_i = scaled_dot_product_attention(
                q_[i].transpose(0, 1),
                blocked_k_.view(-1, h_kv, d)[begin:end].transpose(0, 1),
                blocked_v_.view(-1, h_kv, dv)[begin:end].transpose(0, 1),
                is_causal=causal,
            )
            out[i] = out_i.transpose(0, 1)
            lse[i] = lse_i
        return out, lse

    def cal_diff(
        x: torch.Tensor, y: torch.Tensor, name: str, use_fp8: bool = False
    ) -> None:
        x, y = x.double(), y.double()
        cos_diff = 1 - 2 * (x * y).sum().item() / max(
            (x * x + y * y).sum().item(), 1e-12
        )
        if use_fp8:
            assert cos_diff < 1e-4
        else:
            assert cos_diff < 1e-5

    out_flash, lse_flash = flash_mla()
    out_torch, lse_torch = ref_mla()
    cal_diff(out_flash, out_torch, "out", use_fp8)
    cal_diff(lse_flash, lse_torch, "lse")


if __name__ == "__main__":
    pytest.main([__file__])
