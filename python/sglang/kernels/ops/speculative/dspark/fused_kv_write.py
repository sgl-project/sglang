from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_kv_norm_rope_write_kernel(
    kv_ptr,
    meta_ptr,
    knw_ptr,
    cos_sin_ptr,
    pos_ptr,
    loc_ptr,
    commit_lens_ptr,
    locs_row_width,
    KV: tl.constexpr,
    D: tl.constexpr,
    NH: tl.constexpr,
    L: tl.constexpr,
    EPS: tl.constexpr,
    HAS_COMMIT_LENS: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    l = tl.program_id(1).to(tl.int64)
    if HAS_COMMIT_LENS:
        row_b = t // locs_row_width
        col = t - row_b * locs_row_width
        num_commit = tl.load(commit_lens_ptr + row_b).to(tl.int64)
        if col >= num_commit:
            return
    loc = tl.load(loc_ptr + t).to(tl.int64)
    if loc < 0:
        return
    pos = tl.load(pos_ptr + t).to(tl.int64)

    HALF: tl.constexpr = D // 2
    half_ar = tl.arange(0, HALF)
    d_ar = tl.arange(0, D)
    cos = tl.load(cos_sin_ptr + pos * D + half_ar).to(tl.float32)
    sin = tl.load(cos_sin_ptr + pos * D + HALF + half_ar).to(tl.float32)
    knw1 = tl.load(knw_ptr + l * D + half_ar).to(tl.float32)
    knw2 = tl.load(knw_ptr + l * D + HALF + half_ar).to(tl.float32)

    k_buf = tl.load(meta_ptr + l * 4 + 0).to(tl.pointer_type(tl.bfloat16))
    v_buf = tl.load(meta_ptr + l * 4 + 1).to(tl.pointer_type(tl.bfloat16))
    ks0 = tl.load(meta_ptr + l * 4 + 2)
    vs0 = tl.load(meta_ptr + l * 4 + 3)

    row = kv_ptr + t * (L * 2 * KV) + l * (2 * KV)
    for h in tl.static_range(NH):
        k = tl.load(row + h * D + d_ar).to(tl.float32)
        ms = tl.sum(k * k, 0) / D
        inv = 1.0 / tl.sqrt(ms + EPS)
        k1 = tl.load(row + h * D + half_ar).to(tl.float32) * inv * knw1
        k2 = tl.load(row + h * D + HALF + half_ar).to(tl.float32) * inv * knw2
        k1 = k1.to(tl.bfloat16).to(tl.float32)
        k2 = k2.to(tl.bfloat16).to(tl.float32)
        o1 = k1 * cos - k2 * sin
        o2 = k2 * cos + k1 * sin
        tl.store(k_buf + loc * ks0 + h * D + half_ar, o1.to(tl.bfloat16))
        tl.store(k_buf + loc * ks0 + h * D + HALF + half_ar, o2.to(tl.bfloat16))

        v = tl.load(row + KV + h * D + d_ar)
        tl.store(v_buf + loc * vs0 + h * D + d_ar, v)


def fused_kv_norm_rope_write(
    kv: torch.Tensor,
    meta: torch.Tensor,
    k_norm_weights: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    locs: torch.Tensor,
    num_layers: int,
    kv_size: int,
    head_dim: int,
    eps: float,
    commit_lens: Optional[torch.Tensor] = None,
    locs_row_width: Optional[int] = None,
) -> None:
    """Write per-layer normed+roped K and raw V rows into the KV pools.

    Rows with loc < 0 are skipped. When commit_lens is given, locs is the
    flattened [bs, locs_row_width] verify window and only the first
    commit_lens[b] columns of each row are written — the in-kernel
    replacement for masking the tail columns to -1 on the host.
    """
    T = kv.shape[0]
    if T == 0:
        return
    has_commit_lens = commit_lens is not None
    if has_commit_lens != (locs_row_width is not None):
        raise ValueError(
            "commit_lens and locs_row_width must be passed together, got "
            f"commit_lens={'set' if has_commit_lens else None}, "
            f"locs_row_width={locs_row_width}."
        )
    if has_commit_lens:
        if commit_lens.numel() * locs_row_width != locs.numel():
            raise ValueError(
                f"locs must be a flattened [{commit_lens.numel()}, "
                f"{locs_row_width}] window, got numel={locs.numel()}."
            )
        commit_lens_arg = commit_lens.contiguous()
    else:
        locs_row_width = 1
        commit_lens_arg = locs
    grid = (T, num_layers)
    _fused_kv_norm_rope_write_kernel[grid](
        kv,
        meta,
        k_norm_weights,
        cos_sin_cache,
        positions.to(torch.int64).contiguous(),
        locs.to(torch.int64).contiguous(),
        commit_lens_arg,
        locs_row_width,
        KV=kv_size,
        D=head_dim,
        NH=kv_size // head_dim,
        L=num_layers,
        EPS=eps,
        HAS_COMMIT_LENS=has_commit_lens,
    )
