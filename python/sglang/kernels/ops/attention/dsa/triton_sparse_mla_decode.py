"""Triton sparse MLA decode with FP8 and BF16 KV cache support.

Decode-tuned launcher for the fused / split-K kernels of ``triton_sparse_mla``
on DSA shapes:
  q:       [bs, H, DIM]     fp8/bf16 (DIM=576 = D_V+D_TAIL)
  kv:      [num_pages, 1, DIM]  fp8/bf16
  indices: [bs, 1, topk]    int32
  output:  [1, bs, H, D_V]  bf16
"""

import torch

from sglang.kernels.ops.attention.dsa.triton_sparse_mla import (
    _FP8_MAX,
    _LOG2E,
    _PREFERRED_BLOCK_K,
    _cu_count,
    _gfx950_sparse_mla_kv_splits,
    _gfx950_sparse_mla_num_warps,
    _is_gfx950_sparse_mla_fp8,
    _kv_splits_heuristic,
    _next_pow2,
    _no_async_copy,
    _page_offsets_fit_i32,
    _reduce_d_chunk,
    _row_strides,
    _sparse_mla_block_k,
    _sparse_mla_fused_kernel,
    _sparse_mla_reduce_kernel,
    _sparse_mla_split_k_kernel,
    _validate_input_dtypes,
)

_SplitKWorkspace = list[tuple[torch.Tensor, torch.Tensor]]
_splitk_bufs: dict[tuple[torch.device, int], _SplitKWorkspace] = {}


def _gfx950_sparse_mla_decode_tile_config(
    base_ctas: int, topk: int, block_k: int, max_kv_splits: int
) -> tuple[int, int]:
    if topk == 2048 and base_ctas <= 2:
        block_k = 32
        if base_ctas == 1:
            max_kv_splits = 64
    return block_k, max_kv_splits


def _get_splitk_bufs(
    bs: int,
    kv_splits: int,
    h_padded: int,
    d_v: int,
    device: torch.device,
    workspace: _SplitKWorkspace | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if workspace is None:
        stream_id = int(torch.cuda.current_stream(device).cuda_stream)
        workspace = _splitk_bufs.setdefault((device, stream_id), [])

    needed_lse = bs * kv_splits * h_padded
    needed_acc = bs * kv_splits * h_padded * d_v
    for lse_buf, acc_buf in reversed(workspace):
        if lse_buf.numel() >= needed_lse and acc_buf.numel() >= needed_acc:
            lse = lse_buf[:needed_lse].view(bs, kv_splits, h_padded)
            acc = acc_buf[:needed_acc].view(bs, kv_splits, h_padded, d_v)
            return lse, acc

    # Keep old allocations alive because captured graphs retain their pointers;
    # round up so a growing shape sequence adds O(log range) buffers, not one each.
    capacity_lse = _next_pow2(max(needed_lse, 2 * _cu_count(device) * 16))
    lse_buf = torch.empty(capacity_lse, dtype=torch.float32, device=device)
    acc_buf = torch.empty(capacity_lse * d_v, dtype=torch.bfloat16, device=device)
    workspace.append((lse_buf, acc_buf))
    lse = lse_buf[:needed_lse].view(bs, kv_splits, h_padded)
    acc = acc_buf[:needed_acc].view(bs, kv_splits, h_padded, d_v)
    return lse, acc


def triton_sparse_mla_decode_splitk(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    kv_splits: int | None = None,
    workspace: _SplitKWorkspace | None = None,
) -> torch.Tensor:
    """Split-K Triton sparse MLA decode (DSv4 pattern).

    q_nope:  [bs, H, d_v] fp8/bf16
    q_rope:  [bs, H, d_tail] fp8/bf16
    kv:      [num_pages, 1, DIM] fp8/bf16
    indices: [bs, 1, topk] int32
    workspace: backend-owned grow-only split-K buffers for graph-safe reuse
    returns: [1, bs, H, d_v] bf16
    """
    is_fp8 = _validate_input_dtypes(q_nope, q_rope, kv)
    use_fp8_dot = is_fp8
    bs, H, d_v_in = q_nope.shape
    assert d_v_in == d_v
    d_tail = q_rope.shape[-1]
    kv_dim = kv.shape[-1]
    topk = indices.shape[-1]
    idx_flat = indices.squeeze(1).contiguous()
    # The kernels address q by explicit row strides, so a packed [N, H, D] layout
    # is not required -- only a unit-stride last dim. Callers that pass an
    # already-concatenated q (dsa_backend, GLM-5.2 path) hand us two strided views
    # of one [N, H, D_V + D_TAIL] buffer; copying those would cost two extra
    # device kernels per layer per forward for nothing.
    q_nope, stride_qn_t, stride_qn_h = _row_strides(q_nope)
    q_rope, stride_qr_t, stride_qr_h = _row_strides(q_rope)

    BLOCK_H = 16
    BLOCK_K = _sparse_mla_block_k(kv)
    n_head_blocks = (H + BLOCK_H - 1) // BLOCK_H
    h_padded = n_head_blocks * BLOCK_H

    assert d_v % 128 == 0, f"d_v must be divisible by 128, got {d_v}"
    num_groups = d_v // 128

    # Keep the number of split partials independent of a smaller LDS-safe
    # BLOCK_K. This retains the existing reduction cost on 64 KiB devices.
    max_kv_splits = max(1, topk // _PREFERRED_BLOCK_K)
    num_cu = _cu_count(q_nope.device)
    base_ctas = max(1, bs * n_head_blocks)
    optimize_gfx950_fp8 = H == 16 and _is_gfx950_sparse_mla_fp8(
        kv.dtype, H, d_v, d_tail, kv_dim
    )
    if optimize_gfx950_fp8:
        BLOCK_K, max_kv_splits = _gfx950_sparse_mla_decode_tile_config(
            base_ctas, topk, BLOCK_K, max_kv_splits
        )
    if kv_splits is None:
        # Very sparse BF16 launches benefit from enough split-K work to queue
        # two workgroups per CU. Once token/head parallelism is less sparse, keep the
        # one-wave target to avoid paying extra partial-output reduction cost.
        target_wg_per_cu = 1.0
        if not use_fp8_dot and base_ctas <= max(1, num_cu // 16):
            target_wg_per_cu = 2.0
        kv_splits = min(
            _kv_splits_heuristic(
                bs,
                H,
                BLOCK_H,
                num_cu=num_cu,
                target_wg_per_cu=target_wg_per_cu,
                max_kv_splits=max_kv_splits,
            ),
            max_kv_splits,
        )
        if optimize_gfx950_fp8:
            kv_splits = _gfx950_sparse_mla_kv_splits(
                base_ctas,
                topk,
                BLOCK_K,
                num_cu,
                kv_splits,
                max_kv_splits,
            )
    else:
        kv_splits = min(kv_splits, max_kv_splits)

    qk_scale = float(sm_scale) * _LOG2E

    if kv_splits == 1:
        fused_num_warps = 4
        if optimize_gfx950_fp8:
            fused_num_warps = _gfx950_sparse_mla_num_warps(base_ctas, 1, num_cu)
        out = torch.empty(bs, H, d_v, device=q_nope.device, dtype=torch.bfloat16)
        with _no_async_copy():
            _sparse_mla_fused_kernel[(bs, n_head_blocks)](
                q_nope,
                q_rope,
                kv,
                idx_flat,
                idx_flat,
                out,
                qk_scale,
                _FP8_MAX,
                topk=topk,
                H=H,
                KV_DIM=kv_dim,
                D_V=d_v,
                D_TAIL=d_tail,
                NUM_GROUPS=num_groups,
                STRIDE_QN_T=stride_qn_t,
                STRIDE_QN_H=stride_qn_h,
                STRIDE_QR_T=stride_qr_t,
                STRIDE_QR_H=stride_qr_h,
                USE_FP8_DOT=use_fp8_dot,
                USE_TOPK_LENGTH=False,
                BLOCK_H=BLOCK_H,
                BLOCK_K=BLOCK_K,
                USE_I64_PAGE=not (
                    optimize_gfx950_fp8
                    and fused_num_warps == 2
                    and topk >= 2048
                    and _page_offsets_fit_i32(kv.shape[0], kv_dim)
                ),
                num_warps=fused_num_warps,
                num_stages=2,
            )
        return out.unsqueeze(0)

    tiles_per_split = (topk + kv_splits * BLOCK_K - 1) // (kv_splits * BLOCK_K)
    active_splits = (topk + tiles_per_split * BLOCK_K - 1) // (
        tiles_per_split * BLOCK_K
    )
    active_splits = min(active_splits, kv_splits)
    split_num_warps = 4
    if optimize_gfx950_fp8:
        split_num_warps = _gfx950_sparse_mla_num_warps(base_ctas, active_splits, num_cu)

    lse_partial, acc_partial = _get_splitk_bufs(
        bs, kv_splits, h_padded, d_v, q_nope.device, workspace
    )
    out = torch.empty(bs, H, d_v, device=q_nope.device, dtype=torch.bfloat16)

    grid_split = (bs, n_head_blocks, kv_splits)
    with _no_async_copy():
        _sparse_mla_split_k_kernel[grid_split](
            q_nope,
            q_rope,
            kv,
            idx_flat,
            lse_partial,
            acc_partial,
            qk_scale,
            _FP8_MAX,
            topk=topk,
            H=H,
            KV_DIM=kv_dim,
            D_V=d_v,
            D_TAIL=d_tail,
            NUM_GROUPS=num_groups,
            STRIDE_QN_T=stride_qn_t,
            STRIDE_QN_H=stride_qn_h,
            STRIDE_QR_T=stride_qr_t,
            STRIDE_QR_H=stride_qr_h,
            USE_FP8_DOT=use_fp8_dot,
            KV_SPLITS=kv_splits,
            BLOCK_H=BLOCK_H,
            BLOCK_K=BLOCK_K,
            num_warps=split_num_warps,
            num_stages=2,
        )

    D_CHUNK = _reduce_d_chunk(active_splits, bs * H) if optimize_gfx950_fp8 else 64
    grid_reduce = (bs, H, (d_v + D_CHUNK - 1) // D_CHUNK)
    _sparse_mla_reduce_kernel[grid_reduce](
        lse_partial,
        acc_partial,
        out,
        H=H,
        D_V=d_v,
        KV_SPLITS=kv_splits,
        ACTIVE_SPLITS=active_splits,
        ACTIVE_SPLITS_POW2=_next_pow2(active_splits),
        D_CHUNK=D_CHUNK,
        BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return out.unsqueeze(0)


# ---------------------------------------------------------------------------
# Convenience: auto-select best variant
# ---------------------------------------------------------------------------


def triton_sparse_mla_decode(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> torch.Tensor:
    return triton_sparse_mla_decode_splitk(q_nope, q_rope, kv, indices, sm_scale, d_v)
