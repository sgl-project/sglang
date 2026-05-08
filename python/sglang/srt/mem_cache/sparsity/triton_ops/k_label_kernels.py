"""K_label gather kernels for Double Sparsity.

Writes a per-token, per-KV-head channel summary `K_label[loc, h, :]` by
gathering S calibrated channels from the freshly projected K. Used in both
prefill (`attention_end` after extend) and decode (`attention_end` after
single-step decode); the kernel is the same — only the count of new tokens
differs.

Shapes:
- K              : [N, num_kv_heads_local, head_dim]   bf16/fp16
- channel_idx    : [num_kv_heads_local, S]              int32 (TP-sliced, layer-specific)
- out_cache_loc  : [N]                                  int64 (physical token ids in KV pool)
- K_label        : [num_tokens_in_pool, num_kv_heads_local, S]  bf16/fp32

CUDA-graph correctness: the kernel uses static grid sizes parameterized by
`max_n` for decode-append (=batch size, padded). For extend it is launched
with the actual N at extend time (extend is not under graph capture in
SGLang's piecewise model). All scratch is preallocated upstream; this kernel
only writes into K_label.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _ds_k_label_write_kernel(
    K_ptr,
    channel_idx_ptr,
    out_cache_loc_ptr,
    K_label_ptr,
    N,
    H_kv: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    K_stride_n: tl.constexpr,
    K_stride_h: tl.constexpr,
    KL_stride_t: tl.constexpr,
    KL_stride_h: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Gather S channels from K[h, :] and scatter to K_label[loc, h, :].

    Grid: (ceil(N / BLOCK_N), H_kv).
    """
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N
    s_off = tl.arange(0, S)

    # Channel indices for this KV head: [S]
    chans = tl.load(channel_idx_ptr + pid_h * S + s_off)

    # Gather K[n, pid_h, chans]: [BLOCK_N, S]
    k_offs = n_offsets[:, None] * K_stride_n + pid_h * K_stride_h + chans[None, :]
    k_vals = tl.load(K_ptr + k_offs, mask=n_mask[:, None], other=0)

    # Lookup destination physical-token ids: [BLOCK_N]
    locs = tl.load(out_cache_loc_ptr + n_offsets, mask=n_mask, other=0).to(tl.int64)

    # Scatter to K_label[locs, pid_h, :]: [BLOCK_N, S]
    out_offs = locs[:, None] * KL_stride_t + pid_h * KL_stride_h + s_off[None, :]
    tl.store(K_label_ptr + out_offs, k_vals, mask=n_mask[:, None])


def ds_compute_k_label_write(
    k: torch.Tensor,
    channel_idx: torch.Tensor,
    out_cache_loc: torch.Tensor,
    k_label: torch.Tensor,
    *,
    block_n: int = 64,
) -> None:
    """Write K_label rows for `k.shape[0]` tokens.

    Args:
      k:             `[N, H_kv, D]` freshly projected key tensor.
      channel_idx:   `[H_kv, S]` int32 channel indices for this layer.
      out_cache_loc: `[N]` int64 physical token ids in the KV pool.
      k_label:       `[T, H_kv, S]` side cache, mutated in place.

    All tensors must be CUDA. CPU is not supported (selection kernels and
    FA3 are CUDA-only). For CPU testing use `ds_compute_k_label_torch_ref`.
    """
    if not k.is_cuda:
        raise RuntimeError(
            "ds_compute_k_label_write requires CUDA tensors; "
            "use ds_compute_k_label_torch_ref for CPU testing."
        )
    n = k.shape[0]
    if n == 0:
        return
    h_kv, d = k.shape[1], k.shape[2]
    s = channel_idx.shape[1]

    if channel_idx.shape[0] != h_kv:
        raise ValueError(
            f"channel_idx[0]={channel_idx.shape[0]} mismatches H_kv={h_kv}"
        )
    if k_label.shape[1] != h_kv or k_label.shape[2] != s:
        raise ValueError(
            f"k_label shape {tuple(k_label.shape)} incompatible with "
            f"H_kv={h_kv} S={s}"
        )

    grid = (triton.cdiv(n, block_n), h_kv)
    _ds_k_label_write_kernel[grid](
        k,
        channel_idx,
        out_cache_loc,
        k_label,
        n,
        H_kv=h_kv,
        D=d,
        S=s,
        K_stride_n=k.stride(0),
        K_stride_h=k.stride(1),
        KL_stride_t=k_label.stride(0),
        KL_stride_h=k_label.stride(1),
        BLOCK_N=block_n,
    )


def ds_compute_k_label_torch_ref(
    k: torch.Tensor,
    channel_idx: torch.Tensor,
    out_cache_loc: torch.Tensor,
    k_label: torch.Tensor,
) -> None:
    """Pure-torch reference for `ds_compute_k_label_write`. Mutates `k_label`.

    Equivalent to:
        K_label[out_cache_loc[n], h, s] = K[n, h, channel_idx[h, s]]
    Used by tests (CPU and GPU) and as a fallback when Triton is unavailable.
    """
    n = k.shape[0]
    if n == 0:
        return
    # gather: [N, H_kv, S]
    chans = channel_idx.to(torch.long)
    h_kv = k.shape[1]
    s = chans.shape[1]

    # Expand index for torch.gather: shape [N, H_kv, S]
    gather_idx = chans.unsqueeze(0).expand(n, h_kv, s)
    gathered = torch.gather(k, 2, gather_idx)  # [N, H_kv, S]

    # Cast to k_label dtype (e.g. bf16 / fp32)
    if gathered.dtype != k_label.dtype:
        gathered = gathered.to(k_label.dtype)

    locs = out_cache_loc.to(torch.long)
    k_label.index_copy_(0, locs, gathered)
