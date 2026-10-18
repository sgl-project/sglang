"""Muse Spark 1.2 spec-decode verify attention as one persistent Helion
megakernel (TP4 rank-local shapes: 32 Q heads, 4 KV heads, head_dim 64, FP8
E4M3 paged KV with page size 128, one request of T = 8 or 16 query tokens).

Replaces three production launches that follow the QKVG GEMM:

    FusedQKNorm._forward_triton     per-head RMSNorm (+ interleaved RoPE on
                                    SWA layers), FP8 Q, FP8 K/V store
    FI trtllm_batch_decode          FP8 paged causal decode, q_len_per_req=T
    rms_norm_gate (CUDA)            per-head RMSNorm + sigmoid gate

Roots, in source order:

    ATTEND  [4 KV heads, splits]   each task recomputes its T x 8 Q rows,
                                   attends its 64-key cache tiles; the last
                                   split also attends and stores the T new
                                   K/V rows from registers
    MERGE   [4 KV heads, T]        merges every split of one (KV head, token)
                                   and applies norm + gate

MERGE reads every split of its KV head, so Helion's tile-dependency scheduler
orders it after them: no counters or atomics here.

Only the KV length, positions, cache_loc and page-table contents are dynamic;
every shape is specialized.
"""

from __future__ import annotations

import helion
import helion.language as hl
import torch

from sglang.kernels.ops.muse_spark.helion_norm_gate import (
    _exp2_approx_ftz_f32 as exp2_approx_ftz_f32,
)
from sglang.kernels.ops.muse_spark.helion_norm_gate import (
    _rcp_approx_ftz_f32 as rcp_approx_ftz_f32,
)
from sglang.kernels.ops.muse_spark.helion_norm_gate import (
    _rsqrt_approx_ftz_f32 as rsqrt_approx_ftz_f32,
)

NUM_Q_HEADS = 32

NUM_KV_HEADS = 4

HEAD_DIM = 64

HEADS_PER_KV = 8

PAGE_SIZE = 128

Q_WIDTH = NUM_Q_HEADS * HEAD_DIM

KV_WIDTH = NUM_KV_HEADS * HEAD_DIM

QKV_WIDTH = 2 * Q_WIDTH + 2 * KV_WIDTH

# Muse Spark 1.2 defaults for standalone use; the served model passes its own.
BMM1_SCALE = 43.7840518911 / HEAD_DIM

EPS = 1e-5

KEY_TILE = 64
LOG2E = 1.4426950408889634


def _fp8_clamp(value: torch.Tensor) -> torch.Tensor:
    """Production Triton's ordered clamp tl.minimum(tl.maximum(v, -448), 448):
    NaN -> -448, which torch.clamp would propagate."""

    value = torch.where(value > -448.0, value, -448.0)
    return torch.where(value < 448.0, value, 448.0)


def _cvt_rn_satfinite_bf16(value: torch.Tensor) -> torch.Tensor:
    """TRTLLM-GEN's saturating FP32 -> BF16 epilogue conversion."""

    return hl.inline_asm_elementwise(
        "cvt.rn.satfinite.bf16.f32 $0, $1;",
        "=h,f",
        [value],
        dtype=torch.bfloat16,
        is_pure=True,
        pack=1,
    )


def _e4m3_to_f16(value: torch.Tensor, pinned: bool = False) -> torch.Tensor:
    """Exact FP8 -> FP16. Triton sinks a pure cast into the dot operand layout;
    pinned (non-pure) keeps it in the load layout, before the operand shuffle."""
    return hl.inline_asm_elementwise(
        """
        {
            .reg .b16 lo, hi;
            mov.b32 { lo, hi }, $2;
            cvt.rn.f16x2.e4m3x2 $0, lo;
            cvt.rn.f16x2.e4m3x2 $1, hi;
        }
        """,
        "=r,=r,r",
        [value],
        dtype=torch.float16,
        is_pure=not pinned,
        pack=4,
    )


def _qk_prep(
    raw: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rows: int,
    head_dim: int,
    eps: float,
    use_rope: bool,
) -> torch.Tensor:
    """FusedQKNorm FP8-path value for Q/K rows, before the FP8 cast."""

    raw_fp32 = raw.to(torch.float32)
    variance = torch.sum(raw_fp32 * raw_fp32, dim=-1, keepdim=True) / head_dim
    normalized = (raw_fp32 * torch.rsqrt(variance + eps)).to(torch.bfloat16)
    if use_rope:
        even, odd = hl.split(
            normalized.to(torch.float32).reshape(rows, head_dim // 2, 2)
        )
        values = hl.join(
            even * cos - odd * sin,
            even * sin + odd * cos,
        ).reshape(rows, head_dim)
        values = values.to(torch.bfloat16).to(torch.float32)
    else:
        values = normalized.to(torch.float32)
    return _fp8_clamp(values)


def _cache_tiles(
    cache_end: torch.Tensor,
    split: int,
    key_tile: int,
    num_splits: int,
    window_left: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(first, lo, hi): split's 64-key cache tiles, counted from the tile holding
    the first cached key the oldest query sees. Low splits take the extras."""

    if window_left < 0:
        first_key = cache_end * 0
    else:
        first_key = torch.clamp_min(cache_end - window_left, 0)
    first = first_key & -key_tile
    num_tiles = (cache_end - first + key_tile - 1) >> (key_tile.bit_length() - 1)
    base = num_tiles // num_splits
    extra = num_tiles - base * num_splits
    split = cache_end * 0 + split
    lo = split * base + torch.minimum(split, extra)
    return first, lo, lo + base + (split < extra).to(torch.int32)


def _softmax_step(
    q_tile: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid: torch.Tensor,
    run_max: torch.Tensor,
    run_sum: torch.Tensor,
    acc: torch.Tensor,
    scaled_log2: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Online-softmax update of [rows] state with FP8 keys and FP16 values."""

    scores = hl.dot(q_tile, key.transpose(-2, -1), out_dtype=torch.float32)
    scores = torch.where(valid, scores * scaled_log2, float("-inf"))
    tile_max = torch.amax(scores, dim=-1)
    new_max = torch.maximum(run_max, tile_max)
    correction = exp2_approx_ftz_f32(run_max - new_max)
    prob = exp2_approx_ftz_f32(scores - new_max[:, None])
    acc = hl.dot(
        prob.to(torch.float16),
        value,
        acc=acc * correction[:, None],
        out_dtype=torch.float32,
    )
    run_sum = run_sum * correction + torch.sum(prob, dim=-1)
    # A fresh node: returning new_max would clobber the carried name early.
    return torch.maximum(tile_max, run_max), run_sum, acc


def _attention_verify_megakernel(
    qkv: torch.Tensor,  # [T, QKV_WIDTH] bf16, [q | k | v | gate]
    positions: torch.Tensor,  # [T] int64
    cos_sin_cache: torch.Tensor,  # [P, D] fp32, cos then sin halves
    k_cache: torch.Tensor,  # flat FP8 page storage
    v_cache: torch.Tensor,  # flat FP8 page storage
    cache_loc: torch.Tensor,  # [T] int64, 0 = padding
    block_tables: torch.Tensor,  # [B, pages] int32, row 0 is the request
    seq_lens: torch.Tensor,  # [B] int32, including the T new tokens
    att_ws: torch.Tensor,  # [Hkv, S, T, Hq / Hkv, D] fp32
    lse_ws: torch.Tensor,  # [Hkv, S, T, Hq / Hkv] fp32, log2 units
    out: torch.Tensor,  # [T, Hq * D] bf16
    page_size: hl.constexpr,
    k_page_stride: hl.constexpr,
    v_page_stride: hl.constexpr,
    scaled_log2: hl.constexpr,
    eps: hl.constexpr,
    use_rope: hl.constexpr,
    window_left: hl.constexpr,
    key_tile: hl.constexpr,
) -> torch.Tensor:
    """ATTEND: (kv_head, split) tasks store a partial per (token, head) row;
    MERGE: (kv_head, token) tasks merge the splits and apply norm + gate."""

    hl.specialize(qkv.size())
    hl.specialize(qkv.stride())
    hl.specialize(positions.size())
    hl.specialize(cos_sin_cache.size())
    hl.specialize(cos_sin_cache.stride())
    hl.specialize(k_cache.size())
    hl.specialize(k_cache.stride())
    hl.specialize(v_cache.size())
    hl.specialize(v_cache.stride())
    hl.specialize(cache_loc.size())
    hl.specialize(block_tables.size())
    hl.specialize(block_tables.stride())
    hl.specialize(seq_lens.size())
    hl.specialize(att_ws.size())
    hl.specialize(att_ws.stride())
    hl.specialize(lse_ws.size())
    hl.specialize(lse_ws.stride())
    hl.specialize(out.size())
    hl.specialize(out.stride())

    num_kv_heads, num_splits, tokens, heads_per_kv, head_dim = att_ws.size()
    att_rows = tokens * heads_per_kv
    qkv_width = qkv.size(1)
    kv_width = num_kv_heads * head_dim
    q_width = num_kv_heads * heads_per_kv * head_dim
    page_shift = page_size.bit_length() - 1
    qkv_flat = qkv.view(-1)
    table_flat = block_tables.view(-1)
    cos_sin_flat = cos_sin_cache.view(-1)
    out_flat = out.view(-1)
    # Rows r = token * heads_per_kv + head.
    att_rows_ws = att_ws.view(num_kv_heads, num_splits, att_rows, head_dim)
    lse_rows_ws = lse_ws.view(num_kv_heads, num_splits, att_rows)

    hl.pdl_wait()
    # Trigger at entry: the next kernel's CTAs land on SMs this grid frees early,
    # and its own wait still orders it after this grid.
    hl.pdl_launch_dependents()
    # ATTEND. Stock: FusedQKNorm (Q/K norm, RoPE, FP8 Q, FP8 K/V store) and the
    # KV-split part of FI trtllm_batch_decode.
    for a_g, a_s in hl.tile([num_kv_heads, num_splits], block_size=[1, 1]):
        g = a_g.begin
        split = a_s.begin
        seq_len = seq_lens[0]
        # Cached keys end where the new tokens begin; those come from registers.
        cache_end = seq_len - tokens
        first, lo, hi = _cache_tiles(
            cache_end, split, key_tile, num_splits, window_left
        )
        rows = hl.arange(att_rows)
        cols = hl.arange(head_dim)
        q_tok = rows // heads_per_kv
        q_head = g * heads_per_kv + rows % heads_per_kv
        q_idx = (q_tok * qkv_width + q_head * head_dim)[:, None] + cols[None, :]
        if use_rope:
            half = hl.arange(head_dim // 2)[None, :]
            q_pos = positions[q_tok][:, None] * head_dim
            q_cos = cos_sin_flat[q_pos + half]
            q_sin = cos_sin_flat[q_pos + head_dim // 2 + half]
        else:
            q_cos = hl.zeros([att_rows, head_dim // 2], dtype=torch.float32)
            q_sin = hl.zeros([att_rows, head_dim // 2], dtype=torch.float32)
        q_tile = _qk_prep(
            qkv_flat[q_idx], q_cos, q_sin, att_rows, head_dim, eps, use_rope
        ).to(torch.float8_e4m3fn)

        run_max = hl.full([att_rows], -1e30, dtype=torch.float32)
        run_sum = hl.zeros([att_rows], dtype=torch.float32)
        acc = hl.zeros([att_rows, head_dim], dtype=torch.float32)
        q_last = cache_end + q_tok
        keys = hl.arange(key_tile)
        for tile in hl.grid(lo, hi):
            tile_begin = first + tile * key_tile
            token = tile_begin + keys
            page = hl.load(table_flat, [(tile_begin >> page_shift) + hl.arange(1)])
            in_page = (
                ((tile_begin & (page_size - 1)) + keys) * kv_width + g * head_dim
            )[:, None] + cols[None, :]
            # Rows past cache_end may be this launch's new rows: never read them.
            in_cache = token < cache_end
            key = hl.load(
                k_cache,
                [page.to(torch.int64)[:, None] * k_page_stride + in_page],
                extra_mask=in_cache[:, None],
            )
            value = hl.load(
                v_cache,
                [page.to(torch.int64)[:, None] * v_page_stride + in_page],
                extra_mask=in_cache[:, None],
            )
            valid = in_cache[None, :] & (q_last[:, None] >= token[None, :])
            if window_left >= 0:
                valid = valid & (q_last[:, None] - token[None, :] <= window_left)
            run_max, run_sum, acc = _softmax_step(
                q_tile,
                key,
                _e4m3_to_f16(value),
                valid,
                run_max,
                run_sum,
                acc,
                scaled_log2,
            )

        # The last split has the fewest cache tiles; it owns the new rows.
        if split == num_splits - 1:
            new_tok = hl.arange(tokens)
            k_idx = (new_tok * qkv_width + q_width + g * head_dim)[:, None] + cols[
                None, :
            ]
            if use_rope:
                k_half = hl.arange(head_dim // 2)[None, :]
                k_pos = positions[new_tok][:, None] * head_dim
                k_cos = cos_sin_flat[k_pos + k_half]
                k_sin = cos_sin_flat[k_pos + head_dim // 2 + k_half]
            else:
                k_cos = hl.zeros([tokens, head_dim // 2], dtype=torch.float32)
                k_sin = hl.zeros([tokens, head_dim // 2], dtype=torch.float32)
            k_new = _qk_prep(
                qkv_flat[k_idx], k_cos, k_sin, tokens, head_dim, eps, use_rope
            ).to(torch.float8_e4m3fn)
            v_new = _fp8_clamp(qkv_flat[k_idx + kv_width].to(torch.float32)).to(
                torch.float8_e4m3fn
            )
            loc = cache_loc[new_tok]
            live = loc != 0
            dead = loc == 0
            # Slot 0 is the padding sentinel: nothing is written, so a padded
            # token attends the cache row its position already maps to.
            new_pos = cache_end + new_tok
            new_page = hl.load(table_flat, [new_pos >> page_shift], extra_mask=dead).to(
                torch.int64
            )
            new_in_page = ((new_pos & (page_size - 1)) * kv_width + g * head_dim)[
                :, None
            ] + cols[None, :]
            old_k = hl.load(
                k_cache,
                [new_page[:, None] * k_page_stride + new_in_page],
                extra_mask=dead[:, None],
            )
            old_v = hl.load(
                v_cache,
                [new_page[:, None] * v_page_stride + new_in_page],
                extra_mask=dead[:, None],
            )
            loc_in_page = ((loc % page_size) * kv_width + g * head_dim)[:, None] + cols[
                None, :
            ]
            hl.store(
                k_cache,
                [(loc // page_size)[:, None] * k_page_stride + loc_in_page],
                k_new,
                extra_mask=live[:, None],
            )
            hl.store(
                v_cache,
                [(loc // page_size)[:, None] * v_page_stride + loc_in_page],
                v_new,
                extra_mask=live[:, None],
            )
            # Assumes cache_loc[t] is the slot of position cache_end + t.
            new_key = torch.where(
                live[:, None], k_new.to(torch.float32), old_k.to(torch.float32)
            )
            new_value = torch.where(
                live[:, None], v_new.to(torch.float16), old_v.to(torch.float16)
            )
            run_max, run_sum, acc = _softmax_step(
                q_tile,
                new_key.to(torch.float8_e4m3fn),
                new_value,
                q_tok[:, None] >= new_tok[None, :],
                run_max,
                run_sum,
                acc,
                scaled_log2,
            )

        # A split with no visible key publishes weight 0: ex2(-inf) is +0.
        seen = run_sum > 0.0
        att_rows_ws[g, split, :, :] = acc / torch.where(seen, run_sum, 1.0)[:, None]
        lse_rows_ws[g, split, :] = torch.where(
            seen, run_max + torch.log2(run_sum), float("-inf")
        )

    # MERGE reads every split of its KV head, so it runs after all of them.
    # MERGE. Stock: trtllm_batch_decode's split merge, then rms_norm_gate.
    for m_g, m_t in hl.tile([num_kv_heads, tokens], block_size=[1, 1]):
        mg = m_g.begin
        mt = m_t.begin
        m_cols = hl.arange(head_dim)
        heads = mg * heads_per_kv + hl.arange(heads_per_kv)
        head_idx = (heads * head_dim)[:, None] + m_cols[None, :]
        gate = qkv_flat[mt * qkv_width + q_width + 2 * kv_width + head_idx]
        gate_factor = rcp_approx_ftz_f32(
            1.0 + exp2_approx_ftz_f32(gate.to(torch.float32) * -LOG2E)
        )
        m_lse = lse_ws[mg, :, mt, :]
        weight = exp2_approx_ftz_f32(m_lse - torch.amax(m_lse, dim=0)[None, :])
        merged = torch.sum(att_ws[mg, :, mt, :, :] * weight[:, :, None], dim=0)
        attn = _cvt_rn_satfinite_bf16(merged / torch.sum(weight, dim=0)[:, None]).to(
            torch.float32
        )
        variance = torch.sum(attn * attn, dim=-1) / head_dim
        normed = attn * rsqrt_approx_ftz_f32(variance + eps)[:, None]
        out_flat[mt * q_width + head_idx] = (normed * gate_factor).to(torch.bfloat16)

    return out


def make_config(
    *,
    num_warps: int = 8,
    num_stages: int = 1,
    pipeline: str = "static",
    **extra: object,
) -> helion.Config:
    # Tile dependencies need persistent_blocked; no root has a tunable block.
    return helion.Config(
        block_sizes=[],
        cross_loop_pipeline=pipeline,
        indexing="pointer",
        num_stages=num_stages,
        num_warps=num_warps,
        pid_type="persistent_blocked",
        **extra,
    )


def default_config(**overrides: object) -> helion.Config:
    return make_config(**overrides)


def build_kernel(config: helion.Config) -> object:
    return helion.kernel(
        _attention_verify_megakernel,
        config=config,
        static_shapes=True,
        backend="triton",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )


class MuseAttentionVerifyMegakernel:
    """Holds workspaces for one layer kind (SWA+RoPE or global, no RoPE)."""

    def __init__(
        self,
        device: torch.device,
        *,
        tokens: int,
        sliding_window: bool,
        bmm1_scale: float = BMM1_SCALE,
        eps: float = EPS,
        config: helion.Config | None = None,
    ):
        self.config = config or default_config()
        self.kernel = build_kernel(self.config)
        self.tokens = tokens
        self.use_rope = sliding_window
        self.window_left = 2047 if sliding_window else -1
        # One ATTEND task per SM (37 splits on B200): a 2047-token window spans
        # at most 33 cache tiles of 64 keys. Measured on B200: 37 beats 32.
        num_splits = (
            torch.cuda.get_device_properties(device).multi_processor_count
            // NUM_KV_HEADS
        )
        rows = (NUM_KV_HEADS, num_splits, tokens, HEADS_PER_KV)
        # Every ATTEND task stores its slice each launch, empty splits included.
        self.att_ws = torch.empty(*rows, HEAD_DIM, dtype=torch.float32, device=device)
        self.lse_ws = torch.empty(*rows, dtype=torch.float32, device=device)
        self.no_positions = torch.zeros(tokens, dtype=torch.int64, device=device)
        self.no_cos_sin = torch.zeros(1, HEAD_DIM, dtype=torch.float32, device=device)
        self.constants = (
            bmm1_scale * LOG2E,
            eps,
            self.use_rope,
            self.window_left,
            KEY_TILE,
        )

    def __call__(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor | None,
        cos_sin_cache: torch.Tensor | None,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_loc: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Gated [T, 2048] bf16; also stores the T new K/V rows."""
        if qkv.shape != (self.tokens, QKV_WIDTH) or not qkv.is_contiguous():
            raise RuntimeError(
                f"the verify attention megakernel needs [{self.tokens}, QKV] qkv"
            )
        if (
            k_cache.ndim != 2
            or v_cache.ndim != 2
            or k_cache.size(1) != PAGE_SIZE * KV_WIDTH
            or v_cache.size(1) != PAGE_SIZE * KV_WIDTH
            or k_cache.stride(1) != 1
            or v_cache.stride(1) != 1
        ):
            raise RuntimeError(
                "the verify attention megakernel requires K/V caches as "
                "dense-within-page row views"
            )
        k_page_stride = k_cache.stride(0)
        v_page_stride = v_cache.stride(0)
        k_storage = torch.as_strided(
            k_cache,
            ((k_cache.size(0) - 1) * k_page_stride + k_cache.size(1),),
            (1,),
        )
        v_storage = torch.as_strided(
            v_cache,
            ((v_cache.size(0) - 1) * v_page_stride + v_cache.size(1),),
            (1,),
        )
        if not self.use_rope:
            positions, cos_sin_cache = self.no_positions, self.no_cos_sin
        out = torch.empty(self.tokens, Q_WIDTH, dtype=torch.bfloat16, device=qkv.device)
        return self.kernel(
            qkv,
            positions,
            cos_sin_cache,
            k_storage,
            v_storage,
            cache_loc,
            block_tables,
            seq_lens,
            self.att_ws,
            self.lse_ws,
            out,
            PAGE_SIZE,
            k_page_stride,
            v_page_stride,
            *self.constants,
        )


__all__ = [
    "KEY_TILE",
    "MuseAttentionVerifyMegakernel",
    "default_config",
    "make_config",
]
