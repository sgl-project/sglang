import math
import random
from typing import Optional, Tuple

import pytest
import torch
from sgl_kernel import flash_mla_sparse_fwd, flash_mla_with_kvcache, get_mla_metadata

cc_major, _ = torch.cuda.get_device_capability()
IS_SM100 = cc_major == 10


def cdiv(x: int, y: int):
    return (x + y - 1) // y


# ============================================================================
# Decode Tests (KV Cache)
# ============================================================================
def generate_decode_test_data(
    b: int,
    s_q: int,
    s_k: int,
    h_q: int,
    h_kv: int,
    d: int,
    block_size: int,
    is_varlen: bool = False,
    topk: Optional[int] = None,
    have_zero_seqlen_k: bool = False,
    is_all_indices_invalid: bool = False,
    seed: int = 0,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """
    Generate test data for decode phase
    Return: [cache_seqlens, q, block_table, blocked_k, abs_indices, indices_in_kvcache]
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    assert h_q % h_kv == 0

    cache_seqlens_cpu = torch.full((b,), s_k, dtype=torch.int32, device="cpu")
    if is_varlen:
        for i in range(b):
            cache_seqlens_cpu[i] = max(random.normalvariate(s_k, s_k / 2), s_q)

    if have_zero_seqlen_k:
        zeros_mask = torch.randn(b, dtype=torch.float32, device="cpu") > 0
        cache_seqlens_cpu[zeros_mask] = 0

    max_seqlen = cache_seqlens_cpu.max().item()
    max_seqlen_pad = cdiv(max_seqlen, 256) * 256
    cache_seqlens = cache_seqlens_cpu.cuda()

    q = torch.randn(b, s_q, h_q, d)
    q.clamp_(min=-1.0, max=1.0)

    block_table = torch.arange(
        b * max_seqlen_pad // block_size, dtype=torch.int32
    ).view(b, max_seqlen_pad // block_size)
    block_table = block_table.view(-1)[torch.randperm(block_table.numel())].view(b, -1)
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d) / 10
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
        return cache_seqlens, q, block_table, blocked_k, None, None
    else:
        block_table_cpu = block_table.cpu()
        abs_indices = torch.empty(b, s_q, topk, dtype=torch.int32, device="cpu")
        indices_in_kvcache = torch.empty(b, s_q, topk, dtype=torch.int32, device="cpu")
        for i in range(b):
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

                perm = torch.randperm(topk, device="cpu")
                cur_abs_indices = cur_abs_indices[perm]
                cur_blocked_indices = cur_blocked_indices[perm]

                if is_all_indices_invalid:
                    cur_abs_indices.fill_(-1)
                    cur_blocked_indices.fill_(-1)

                abs_indices[i, j, :] = cur_abs_indices
                indices_in_kvcache[i, j, :] = cur_blocked_indices

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

        return cache_seqlens, q, block_table, blocked_k, abs_indices, indices_in_kvcache


def reference_decode_torch(
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    q: torch.Tensor,
    blocked_k: torch.Tensor,
    dv: int,
    is_causal: bool,
    indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A reference implementation in PyTorch for decode phase
    """

    def get_topk_attn_mask(s_q: int, s_k: int, indices: torch.Tensor):
        mask = torch.zeros(s_q, s_k, dtype=torch.bool)
        for i in range(s_q):
            cur_indices = indices[i]
            valid_indices = cur_indices[cur_indices != -1]
            mask[i, valid_indices] = True
        return mask

    def scaled_dot_product_attention(
        batch_idx: int,
        query: torch.Tensor,
        kv: torch.Tensor,
        dv: int,
        is_causal,
        indices: Optional[torch.Tensor],
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
        attn_weight = query @ kv.transpose(-2, -1)
        if (is_causal and query.size(1) > 1) or indices is not None:
            mask = torch.ones(s_q, s_k, dtype=torch.bool)
            if is_causal:
                assert indices is None
                mask = mask.tril(diagonal=s_k - s_q)
            if indices is not None:
                mask &= get_topk_attn_mask(s_q, s_k, indices)
            attn_bias = torch.zeros(s_q, s_k, dtype=torch.float)
            attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
            attn_weight += attn_bias.to(q.dtype)
        attn_weight /= math.sqrt(query.size(-1))
        lse = attn_weight.logsumexp(dim=-1)
        attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
        output = attn_weight @ kv[..., :dv]
        lonely_q_mask = lse == float("-inf")
        output[lonely_q_mask.unsqueeze(-1).broadcast_to(h_q, s_q, dv)] = 0.0
        lse[lonely_q_mask] = float("+inf")

        return output, lse

    b, s_q, h_q, d = q.size()
    block_size = blocked_k.size(1)
    h_kv = blocked_k.size(2)
    cache_seqlens_cpu = cache_seqlens.cpu()
    out_ref = torch.empty(b, s_q, h_q, dv, dtype=torch.float32)
    lse_ref = torch.empty(b, h_q, s_q, dtype=torch.float32)
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


# ============================================================================
# Prefill Tests (Sparse Attention)
# ============================================================================


def generate_prefill_test_data(
    b: int,
    s_q: int,
    s_kv: int,
    topk: int,
    h_q: int,
    h_kv: int,
    d_qk: int,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate test data for prefill phase
    Return: [q, kv, indices]
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    q = torch.randn((b, s_q, h_q, d_qk), dtype=torch.bfloat16) / 10
    kv = torch.randn((b, s_kv, h_kv, d_qk), dtype=torch.bfloat16) / 10

    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)

    indices = torch.full((b, s_q, h_kv, topk), s_kv, dtype=torch.int32)
    for b_idx in range(b):
        for s in range(s_q):
            for h in range(h_kv):
                # Generate indices with most lying in recent positions
                near_mask = torch.randint(0, 32, (min(topk, s_kv),)) < 31
                cur_indices = torch.randperm(s_kv)[:topk]
                cur_indices[near_mask] = torch.randint(
                    max(0, s_kv - 20000), s_kv - 1, (near_mask.sum().item(),)
                )
                if len(cur_indices) < topk:
                    cur_indices = torch.cat(
                        [
                            cur_indices,
                            torch.full((topk - len(cur_indices),), 2147480000),
                        ]
                    )
                cur_indices = cur_indices[torch.randperm(topk)]
                indices[b_idx, s, h] = cur_indices
    indices = indices.to(q.device)

    return q, kv, indices


def reference_prefill_torch(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A reference implementation in PyTorch for prefill phase
    """

    def log2sumexp2(a: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.logsumexp(a * math.log(2), dim=dim) * math.log2(math.e)

    b, s_q, h_q, d_qk = q.shape
    _, s_kv, h_kv, _ = kv.shape
    topk = indices.shape[-1]

    assert b == 1, "Reference implementation only supports batch_size=1"

    indices_squeezed = indices[0, :, 0, :]  # [s_q, topk]
    invalid_indices_mask = (indices_squeezed < 0) | (indices_squeezed >= s_kv)
    qs = q[0, :, :, :].float()  # [s_q, h_q, d_qk]
    kvs = kv[0, :, 0, :].float()  # [s_kv, d_qk]

    kvs = torch.index_select(
        kvs, 0, indices_squeezed.masked_fill(invalid_indices_mask, 0).flatten()
    ).view(s_q, topk, d_qk)
    attn_score = qs @ kvs.transpose(1, 2)  # [s_q, h_q, topk]
    attn_score.masked_fill_(invalid_indices_mask.unsqueeze(1), float("-inf"))
    attn_score *= sm_scale * math.log2(math.e)
    max_logits = torch.max(attn_score, dim=-1)[0]  # [s_q, h_q]
    lse = log2sumexp2(attn_score, dim=-1)  # [s_q, h_q]
    attn_score = torch.exp2(attn_score - lse.unsqueeze(-1))  # [s_q, h_q, topk]
    result = attn_score @ kvs[:, :, :d_v]

    return max_logits, lse, result


# ============================================================================
# Varlen Flash Attention Tests
# ============================================================================


def get_window_size(causal: bool, window: int) -> Tuple[int, int]:
    """Get window size tuple for Flash Attention"""
    if window > 0:
        window_size = (window - 1, 0) if causal else (window - 1, window - 1)
    else:
        window_size = (-1, -1)
    return window_size


def get_attn_bias(s_q: int, s_k: int, causal: bool, window: int) -> torch.Tensor:
    """Generate attention bias matrix"""
    attn_bias = torch.zeros(s_q, s_k, dtype=torch.float32)
    if causal:
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    if window > 0:
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(
            diagonal=s_k - s_q - window
        )
        attn_bias.masked_fill_(temp_mask, float("-inf"))
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(
            diagonal=s_k - s_q + window - 1
        )
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    return attn_bias


def sdpa(query, key, value, attn_bias, h, h_k, softmax_scale=None):
    """Scaled dot product attention reference implementation"""
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
    """Checkpointed version of SDPA"""
    return checkpoint(sdpa, *args, use_reentrant=False, **kwargs)


def generate_varlen_test_data(
    b: int,
    mean_sq: int,
    mean_sk: int,
    h: int,
    h_k: int,
    d: int,
    dv: int,
    varlen: bool = False,
    seed: int = 0,
):
    """Generate test data for varlen flash attention"""
    torch.manual_seed(seed)
    random.seed(seed)

    seqlens_q = torch.full((b,), mean_sq, dtype=torch.int32)
    seqlens_k = torch.full((b,), mean_sk, dtype=torch.int32)

    if varlen:
        for i in range(b):
            seqlens_q[i] = max(random.normalvariate(mean_sq, mean_sq / 2), 1)
        for i in range(b):
            seqlens_k[i] = max(
                random.normalvariate(mean_sk, mean_sk / 2), seqlens_q[i].item()
            )

    cu_seqlens_q = torch.cumsum(
        torch.nn.functional.pad(seqlens_q, (1, 0)), 0, dtype=torch.int32
    )
    cu_seqlens_k = torch.cumsum(
        torch.nn.functional.pad(seqlens_k, (1, 0)), 0, dtype=torch.int32
    )
    total_q = seqlens_q.sum().item()
    total_k = seqlens_k.sum().item()
    max_seqlen_q = seqlens_q.max().item()
    max_seqlen_k = seqlens_k.max().item()

    q = torch.randn(total_q, h, d) / 10
    k = torch.randn(total_k, h_k, d) / 10
    v = torch.randn(total_k, h_k, dv) / 10

    return (
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        seqlens_q,
        seqlens_k,
    )


def reference_varlen_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seqlens_q: torch.Tensor,
    seqlens_k: torch.Tensor,
    h: int,
    h_k: int,
    causal: bool,
    window: int,
    softmax_scale: float,
):
    """Reference implementation for varlen flash attention"""
    b = len(seqlens_q)
    out = []
    lse = []
    for i in range(b):
        OUT, LSE = sdpa_checkpoint(
            q[cu_seqlens_q[i].item() : cu_seqlens_q[i + 1].item()],
            k[cu_seqlens_k[i].item() : cu_seqlens_k[i + 1].item()],
            v[cu_seqlens_k[i].item() : cu_seqlens_k[i + 1].item()],
            attn_bias=get_attn_bias(
                seqlens_q[i].item(), seqlens_k[i].item(), causal, window
            ),
            h=h,
            h_k=h_k,
            softmax_scale=softmax_scale,
        )
        out.append(OUT.transpose(-3, -2))
        lse.append(LSE.transpose(-2, -1))
    out = torch.cat(out)
    lse = torch.cat(lse)
    return out, lse


# ============================================================================
# Decode Phase Tests
# ============================================================================


@pytest.mark.parametrize("b", [1, 2, 6, 64])
@pytest.mark.parametrize("s_q", [1, 2, 4])
@pytest.mark.parametrize("s_k", [20, 140, 4096])
@pytest.mark.parametrize("is_varlen", [False, True])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("is_fp8", [False, True])
@pytest.mark.parametrize("topk", [None, 128, 2048])
def test_flash_mla_decode_correctness(b, s_q, s_k, is_varlen, is_causal, is_fp8, topk):
    """Test decode phase correctness"""
    if is_causal and topk is not None:
        pytest.skip("Causal attention with topk is not supported")
    if not is_fp8 and topk is not None:
        pytest.skip("Non-FP8 with topk is not tested in correctness cases")

    cc_major, cc_minor = torch.cuda.get_device_capability()
    if cc_major == 10 and not (is_fp8 and topk is not None):
        pytest.skip("Only FP8 with topk is supported on this device")

    h_q = 128
    h_kv = 1
    d = 576
    dv = 512
    block_size = 64

    cache_seqlens, q, block_table, blocked_k, abs_indices, indices_in_kvcache = (
        generate_decode_test_data(
            b, s_q, s_k, h_q, h_kv, d, block_size, is_varlen, topk
        )
    )

    if is_fp8:
        blocked_k_quantized = quant.quantize_k_cache(blocked_k, dv, 128)
        blocked_k_dequantized = quant.dequantize_k_cache(blocked_k_quantized)
        blocked_k = blocked_k_dequantized

    tile_scheduler_metadata, num_splits = get_mla_metadata(
        cache_seqlens, s_q * h_q // h_kv, h_kv, h_q, is_fp8, topk
    )

    out_ans, lse_ans = flash_mla_with_kvcache(
        q,
        blocked_k if not is_fp8 else blocked_k_quantized,
        block_table,
        cache_seqlens,
        dv,
        tile_scheduler_metadata,
        num_splits,
        causal=is_causal,
        is_fp8_kvcache=is_fp8,
        indices=indices_in_kvcache,
    )

    out_ref, lse_ref = reference_decode_torch(
        cache_seqlens, block_table, q, blocked_k, dv, is_causal, abs_indices
    )

    torch.testing.assert_close(out_ans, out_ref, rtol=2.01 / 128, atol=8e-4)
    torch.testing.assert_close(lse_ans, lse_ref, rtol=8.01 / 65536, atol=1e-6)


@pytest.mark.parametrize("topk", [128, 2048, 4096])
def test_flash_mla_decode_invalid_indices(topk):
    """Test decode phase with all invalid topk indices"""
    b = 128
    s_q = 2
    s_k = 4096
    h_q = 128
    h_kv = 1
    d = 576
    dv = 512
    block_size = 64

    cache_seqlens, q, block_table, blocked_k, abs_indices, indices_in_kvcache = (
        generate_decode_test_data(
            b,
            s_q,
            s_k,
            h_q,
            h_kv,
            d,
            block_size,
            is_varlen=True,
            topk=topk,
            is_all_indices_invalid=True,
        )
    )

    blocked_k_quantized = quant.quantize_k_cache(blocked_k, dv, 128)
    blocked_k_dequantized = quant.dequantize_k_cache(blocked_k_quantized)
    blocked_k = blocked_k_dequantized

    tile_scheduler_metadata, num_splits = get_mla_metadata(
        cache_seqlens, s_q * h_q // h_kv, h_kv, h_q, True, topk
    )

    out_ans, lse_ans = flash_mla_with_kvcache(
        q,
        blocked_k_quantized,
        block_table,
        cache_seqlens,
        dv,
        tile_scheduler_metadata,
        num_splits,
        causal=False,
        is_fp8_kvcache=True,
        indices=indices_in_kvcache,
    )

    out_ref, lse_ref = reference_decode_torch(
        cache_seqlens, block_table, q, blocked_k, dv, False, abs_indices
    )

    torch.testing.assert_close(out_ans, out_ref, rtol=2.01 / 128, atol=8e-4)
    torch.testing.assert_close(lse_ans, lse_ref, rtol=8.01 / 65536, atol=1e-6)


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("is_fp8", [False, True])
@pytest.mark.parametrize("topk", [None, 128, 2048])
def test_flash_mla_decode_zero_seqlen(is_causal, is_fp8, topk):
    """Test decode phase with some zero-length kv caches"""
    if is_causal and topk is not None:
        pytest.skip("Causal attention with topk is not supported")
    if not is_fp8 and topk is not None:
        pytest.skip("Non-FP8 with topk is not tested")

    b = 128
    s_q = 2
    s_k = 4096
    h_q = 128
    h_kv = 1
    d = 576
    dv = 512
    block_size = 64

    cache_seqlens, q, block_table, blocked_k, abs_indices, indices_in_kvcache = (
        generate_decode_test_data(
            b,
            s_q,
            s_k,
            h_q,
            h_kv,
            d,
            block_size,
            is_varlen=True,
            topk=topk,
            have_zero_seqlen_k=True,
        )
    )

    if is_fp8:
        blocked_k_quantized = quant.quantize_k_cache(blocked_k, dv, 128)
        blocked_k_dequantized = quant.dequantize_k_cache(blocked_k_quantized)
        blocked_k = blocked_k_dequantized

    tile_scheduler_metadata, num_splits = get_mla_metadata(
        cache_seqlens, s_q * h_q // h_kv, h_kv, h_q, is_fp8, topk
    )

    out_ans, lse_ans = flash_mla_with_kvcache(
        q,
        blocked_k if not is_fp8 else blocked_k_quantized,
        block_table,
        cache_seqlens,
        dv,
        tile_scheduler_metadata,
        num_splits,
        causal=is_causal,
        is_fp8_kvcache=is_fp8,
        indices=indices_in_kvcache,
    )

    out_ref, lse_ref = reference_decode_torch(
        cache_seqlens, block_table, q, blocked_k, dv, is_causal, abs_indices
    )

    torch.testing.assert_close(out_ans, out_ref, rtol=2.01 / 128, atol=8e-4)
    torch.testing.assert_close(lse_ans, lse_ref, rtol=8.01 / 65536, atol=1e-6)


# ============================================================================
# Prefill Phase Tests
# ============================================================================


@pytest.mark.parametrize("s_q", [1, 62])
@pytest.mark.parametrize(
    "s_kv,topk",
    [
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
    ],
)
def test_flash_mla_prefill_correctness(s_q, s_kv, topk):
    """Test prefill phase correctness"""
    b = 1
    h_q = 128
    h_kv = 1
    d_qk = 576
    d_v = 512

    q, kv, indices = generate_prefill_test_data(b, s_q, s_kv, topk, h_q, h_kv, d_qk)
    sm_scale = 1 / math.sqrt(d_qk)

    ans_out, ans_max_logits, ans_lse = flash_mla_sparse_fwd(
        q.squeeze(0), kv.squeeze(0), indices.squeeze(0), sm_scale=sm_scale
    )

    ref_max_logits, ref_lse, ref_out = reference_prefill_torch(
        q, kv, indices, sm_scale, d_v
    )

    torch.testing.assert_close(ans_out, ref_out, rtol=2.01 / 128, atol=8e-4)
    torch.testing.assert_close(
        ans_max_logits, ref_max_logits, rtol=2.01 / 65536, atol=1e-6
    )
    torch.testing.assert_close(ans_lse, ref_lse, rtol=2.01 / 65536, atol=1e-6)


@pytest.mark.parametrize("s_q", [1, 1024])
@pytest.mark.parametrize(
    "s_kv,topk",
    [
        (32, 2048),
        (64, 8192),
    ],
)
def test_flash_mla_prefill_corner_cases(s_q, s_kv, topk):
    """Test prefill phase corner cases where some blocks may not have valid topk indices"""
    b = 1
    h_q = 128
    h_kv = 1
    d_qk = 576
    d_v = 512

    q, kv, indices = generate_prefill_test_data(b, s_q, s_kv, topk, h_q, h_kv, d_qk)
    sm_scale = 1 / math.sqrt(d_qk)

    ans_out, ans_max_logits, ans_lse = flash_mla_sparse_fwd(
        q.squeeze(0), kv.squeeze(0), indices.squeeze(0), sm_scale=sm_scale
    )

    ref_max_logits, ref_lse, ref_out = reference_prefill_torch(
        q, kv, indices, sm_scale, d_v
    )

    torch.testing.assert_close(ans_out, ref_out, rtol=2.01 / 128, atol=8e-4)
    torch.testing.assert_close(
        ans_max_logits, ref_max_logits, rtol=2.01 / 65536, atol=1e-6
    )
    torch.testing.assert_close(ans_lse, ref_lse, rtol=2.01 / 65536, atol=1e-6)


# ============================================================================
# Varlen Flash Attention Tests On SM100
# ============================================================================
@pytest.mark.skipif(
    not IS_SM100, reason="Only supports SM100 (compute capability 10.x)"
)
@pytest.mark.parametrize("mean_sq,mean_sk", [(4096, 4096), (8192, 8192)])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("h,h_k", [(128, 128), (32, 4)])
@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128)])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_varlen_fwd(mean_sq, mean_sk, varlen, h, h_k, d, dv, causal):
    """Test varlen flash attention forward pass"""
    # Skip large tests for GQA case or large sequence lengths
    if mean_sq == 8192 and mean_sk == 8192 and h == 128:
        pytest.skip("Skip large test case for memory constraints")

    b = 2
    window = 0

    (
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        seqlens_q,
        seqlens_k,
    ) = generate_varlen_test_data(b, mean_sq, mean_sk, h, h_k, d, dv, varlen)

    softmax_scale = (d + 100) ** (-0.5)

    q1 = q.clone().requires_grad_(False)
    k1 = k.clone().requires_grad_(False)
    v1 = v.clone().requires_grad_(False)

    q2 = q.clone().requires_grad_(False)
    k2 = k.clone().requires_grad_(False)
    v2 = v.clone().requires_grad_(False)

    kwargs = {}
    if causal:
        kwargs["causal"] = causal
    if window != 0:
        kwargs["window_size"] = get_window_size(causal, window)

    out_flash, lse_flash = flash_attn_varlen_func(
        q1,
        k1,
        v1,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=softmax_scale,
        is_varlen=varlen,
        **kwargs
    )

    out_torch, lse_torch = reference_varlen_torch(
        q2,
        k2,
        v2,
        cu_seqlens_q,
        cu_seqlens_k,
        seqlens_q,
        seqlens_k,
        h,
        h_k,
        causal,
        window,
        softmax_scale,
    )

    torch.testing.assert_close(out_flash, out_torch, rtol=8.01 / 128, atol=1e-3)
    torch.testing.assert_close(lse_flash, lse_torch, rtol=2.01 / 65536, atol=1e-6)

    # Test determinism
    for _ in range(3):
        out_test, lse_test = flash_attn_varlen_func(
            q1,
            k1,
            v1,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale=softmax_scale,
            is_varlen=varlen,
            **kwargs
        )
        assert torch.equal(out_test, out_flash), "out deterministic check failed!"
        assert torch.equal(lse_test, lse_flash), "lse deterministic check failed!"


if __name__ == "__main__":
    pytest.main([__file__])
