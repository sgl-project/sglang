"""Fused paged FP8 indexer scoring for the DSV4 C4 indexer (portable Triton).

Drop-in for ``fp8_paged_mqa_logits_torch_sm120`` — same signature, same
semantics, one difference: gather, FP8 decode, per-head dot, ReLU, weighting,
head reduction and the per-token scale all happen in registers. The
``[B, S, num_heads]`` intermediate the torch path materialises (and reads
back twice) never exists, and the KV is never expanded to a bf16 tensor.

The mechanics are lifted from the sparse-decode kernel in
``flash_mla_sm120_triton.py`` rather than reinvented:

- KV bytes are read as int32 (4 bytes per load) and q is split into the four
  matching byte lanes, with one dot per lane summed into the accumulator. A
  dot product does not care about the order of the contracted dimension, so
  applying the same permutation to both sides is exactly equivalent.
- E4M3 is decoded with bit arithmetic (``_e4m3_to_f32`` is copied verbatim;
  keep the two in sync). The two NaN encodings (0x7F/0xFF) decode to +-480
  rather than NaN — the same trade-off the sparse-decode kernel makes.
- The int64 page base is added to the pointer first to get a per-token
  pointer vector, then the int32 inner offsets broadcast against it.

Cache layout (from ``indexer.py``'s torch path): each 8448-byte page is
segmented page-level, not per-token — [8192 B: 64 tokens x 128 FP8 values]
followed by [256 B: 64 fp32 scales]. As int32 words: 2112 words per page,
value words of token (p, t) start at ``p * 2112 + t * 32``, its scale word is
``p * 2112 + 2048 + t`` (bitcast to fp32).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _e4m3_to_f32(b):
    # Copied verbatim from flash_mla_sm120_triton.py; keep the two in sync.
    e = (b >> 3) & 0xF
    m = b & 0x7
    mag = (((e + 120) << 23) | (m << 20)).to(tl.float32, bitcast=True)
    mag = tl.where(e == 0, m.to(tl.float32) * 1.953125e-3, mag)  # 2^-9
    return tl.where((b >> 7) & 1 == 1, -mag, mag)


@triton.autotune(
    configs=[
        triton.Config({"BN": 32}, num_warps=4, num_stages=2),
        triton.Config({"BN": 64}, num_warps=4, num_stages=2),
        triton.Config({"BN": 64}, num_warps=8, num_stages=2),
        triton.Config({"BN": 64}, num_warps=4, num_stages=3),
        triton.Config({"BN": 128}, num_warps=8, num_stages=2),
    ],
    # The key deliberately excludes the padded length: chunked prefill hands
    # this kernel a different padded length on nearly every call, and keying on
    # it re-runs the autotune benchmark per length — measured as a net
    # slowdown on a 32K needle run. The config choice is insensitive to the
    # padded length (the grid adapts), so one tune per head count is enough.
    key=["H"],
)
@triton.jit
def _fused_paged_indexer_kernel(
    CACHE_I32,
    Q4,
    W,
    PT,
    OUT,
    s_q4_b,
    s_q4_j,
    s_q4_h,
    s_w_b,
    s_pt_b,
    s_out_b,
    PADDED_EFF,
    H: tl.constexpr,
    BN: tl.constexpr,
):
    # program_id is int32 and Triton does not promote int multiplies: at
    # 1M-context prefill shapes the row base pid_b * s_out_b crosses 2^31
    # (bisected: first IMA exactly one step past B*msl = 2^31), so every
    # per-row base below is computed in int64. Same class of bug as the
    # vLLM-side indexer overflow (vllm#50576).
    pid_b = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    n_mask = offs_n < PADDED_EFF

    page_idx = offs_n // 64
    tok = offs_n % 64
    pids = tl.load(PT + pid_b * s_pt_b + page_idx, mask=n_mask, other=0)
    pids = tl.maximum(pids, 0).to(tl.int64)  # clamp(min=0), as the torch path does

    # Value section: [BN, 32] int32, 4 bytes per load.
    w_ptrs = CACHE_I32 + (pids * 2112 + tok.to(tl.int64) * 32)
    kv_w = tl.load(
        w_ptrs[:, None] + tl.arange(0, 32)[None, :], mask=n_mask[:, None], other=0
    )
    b0 = kv_w & 0xFF
    b1 = (kv_w >> 8) & 0xFF
    b2 = (kv_w >> 16) & 0xFF
    b3 = (kv_w >> 24) & 0xFF
    kv0 = _e4m3_to_f32(b0).to(tl.bfloat16)
    kv1 = _e4m3_to_f32(b1).to(tl.bfloat16)
    kv2 = _e4m3_to_f32(b2).to(tl.bfloat16)
    kv3 = _e4m3_to_f32(b3).to(tl.bfloat16)

    # q is pre-arranged as [B, 4, H, 32]: byte j of word w is channel 4w+j,
    # which lives at q4[b, j, h, w].
    offs_h = tl.arange(0, H)
    offs_k = tl.arange(0, 32)
    q_base = Q4 + pid_b * s_q4_b
    q0 = tl.load(q_base + 0 * s_q4_j + offs_h[:, None] * s_q4_h + offs_k[None, :])
    q1 = tl.load(q_base + 1 * s_q4_j + offs_h[:, None] * s_q4_h + offs_k[None, :])
    q2 = tl.load(q_base + 2 * s_q4_j + offs_h[:, None] * s_q4_h + offs_k[None, :])
    q3 = tl.load(q_base + 3 * s_q4_j + offs_h[:, None] * s_q4_h + offs_k[None, :])

    acc = tl.dot(kv0, tl.trans(q0))  # [BN, H], fp32 accumulation throughout
    acc += tl.dot(kv1, tl.trans(q1))
    acc += tl.dot(kv2, tl.trans(q2))
    acc += tl.dot(kv3, tl.trans(q3))

    # Per-head ReLU before the weighted head sum — this is the operator; the
    # head dimension cannot be folded through it (see the #33271 discussion).
    acc = tl.maximum(acc, 0.0)
    wv = tl.load(W + pid_b * s_w_b + offs_h).to(tl.float32)
    red = tl.sum(acc * wv[None, :], axis=1)  # [BN]

    sc_i = tl.load(
        CACHE_I32 + (pids * 2112 + 2048 + tok.to(tl.int64)), mask=n_mask, other=0
    )
    red = red * sc_i.to(tl.float32, bitcast=True)  # scale after the head sum
    tl.store(OUT + pid_b * s_out_b + offs_n, red, mask=n_mask)


def fused_paged_mqa_logits(
    q_fp8,
    kvcache_fp8,
    weight,
    seq_lens,
    page_table,
    deep_gemm_metadata,
    max_seq_len,
    clean_logits=True,
):
    """Signature and semantics match ``fp8_paged_mqa_logits_torch_sm120``."""
    _ = deep_gemm_metadata
    assert clean_logits is False
    batch_size, _, num_heads, head_dim = q_fp8.shape
    assert head_dim == 128 and kvcache_fp8.shape[1:] == (64, 1, 132)
    device = q_fp8.device
    max_pages = page_table.shape[1]
    padded_eff = min(max_pages * 64, max_seq_len)

    cache_i32 = kvcache_fp8.view(-1).view(torch.int32)
    q4 = (
        q_fp8[:, 0]
        .to(torch.bfloat16)
        .view(batch_size, num_heads, 32, 4)
        .permute(0, 3, 1, 2)
        .contiguous()
    )  # [B, 4, H, 32]
    w32 = weight if weight.dtype == torch.float32 else weight.float()

    # empty, not full(-inf): consumers bound reads by seq_lens (see the
    # torch-path comment); a full-width fill is pure ceiling tax per call.
    logits = torch.empty(
        (batch_size, max_seq_len), dtype=torch.float32, device=device
    )
    grid = lambda meta: (batch_size, triton.cdiv(padded_eff, meta["BN"]))
    _fused_paged_indexer_kernel[grid](
        cache_i32,
        q4,
        w32,
        page_table,
        logits,
        q4.stride(0),
        q4.stride(1),
        q4.stride(2),
        w32.stride(0),
        page_table.stride(0),
        logits.stride(0),
        padded_eff,
        H=num_heads,
    )
    # Tail exactly as the torch path: every invalid position becomes -inf.
    positions = torch.arange(max_seq_len, device=device)
    invalid = positions[None, :] >= seq_lens.reshape(batch_size)[:, None]
    logits.masked_fill_(invalid, float("-inf"))
    return logits
