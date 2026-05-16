"""Helper kernels for the DSv4 sparse-prefill path that routes through
``flash_mla.flash_mla_sparse_fwd``.

The sparse-decode kernel (``flash_mla_with_kvcache``) takes the SWA cache and
the optional extra compressed cache as separate paged tensors. The sparse-
prefill kernel takes a *single* flat KV tensor. To use it for layers with
``compress_ratio in {4, 128}`` we need to:

  1. Dequantize both caches to BF16 and concatenate them along the kv-token
     axis: ``kv = [swa_bf16; extra_bf16]``. The extra region starts at
     offset ``s_kv_swa = num_swa_blocks * swa_block_size``.

  2. Merge each query token's swa and extra topk indices into one flat index
     array, shifting the extra indices by ``s_kv_swa`` so they point into the
     concatenated buffer. Per-token valid lengths are summed.

``combine_swa_extra_indices`` implements step 2. The dequant+concat in step 1
is plain PyTorch and lives in the caller.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl

# Alignment required by ``flash_mla_sparse_fwd`` along the topk dimension.
# Covers both ``h_q=64`` (B_TOPK=64) and ``h_q=128`` (B_TOPK=128) kernel
# variants; FlashMLA decode asserts ``extra_topk % B_TOPK == 0`` so the
# combined topk must be a multiple of 128 to be safe across head counts.
_TOPK_ALIGNMENT = 128


@triton.jit
def _combine_swa_extra_indices_kernel(
    combined_ptr,
    combined_stride_token,
    combined_lens_ptr,
    swa_indices_ptr,
    swa_stride_token,
    extra_indices_ptr,
    extra_stride_token,
    swa_topk_len_ptr,
    extra_topk_len_ptr,
    s_kv_swa,
    TOPK_SWA: tl.constexpr,
    TOPK_EXTRA: tl.constexpr,
    COMBINED_TOPK: tl.constexpr,
):
    """One program per query token.

    Reads ``swa_indices[token_idx, 0, :]`` and ``extra_indices[token_idx, 0, :]``
    along with their per-token valid lengths, then writes a packed layout
    ``[swa_valid; extra_valid + s_kv_swa; -1 padding]`` into
    ``combined[token_idx, 0, :]``.

    The output buffer is assumed to be pre-initialized to -1, so the kernel
    only needs to fill the valid prefix.
    """
    token_idx = tl.program_id(0)

    swa_len = tl.load(swa_topk_len_ptr + token_idx)
    extra_len = tl.load(extra_topk_len_ptr + token_idx)

    # --- Copy swa valid prefix unchanged ---
    swa_off = tl.arange(0, TOPK_SWA)
    swa_mask = swa_off < swa_len
    swa_vals = tl.load(
        swa_indices_ptr + token_idx * swa_stride_token + swa_off,
        mask=swa_mask,
        other=-1,
    )
    tl.store(
        combined_ptr + token_idx * combined_stride_token + swa_off,
        swa_vals,
        mask=swa_mask,
    )

    # --- Copy extra valid prefix shifted by s_kv_swa, written after swa ---
    extra_off = tl.arange(0, TOPK_EXTRA)
    extra_mask = extra_off < extra_len
    extra_vals = tl.load(
        extra_indices_ptr + token_idx * extra_stride_token + extra_off,
        mask=extra_mask,
        other=-1,
    )
    extra_vals = extra_vals + s_kv_swa

    write_pos = swa_len + extra_off
    write_mask = extra_mask & (write_pos < COMBINED_TOPK)
    tl.store(
        combined_ptr + token_idx * combined_stride_token + write_pos,
        extra_vals,
        mask=write_mask,
    )

    tl.store(combined_lens_ptr + token_idx, swa_len + extra_len)


def combine_swa_extra_indices(
    swa_indices: torch.Tensor,
    swa_topk_lens: torch.Tensor,
    extra_indices: torch.Tensor,
    extra_topk_lens: torch.Tensor,
    s_kv_swa: int,
    alignment: int = _TOPK_ALIGNMENT,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Combine per-token swa and extra topk indices into a single fixed-size
    index tensor suitable for ``flash_mla_sparse_fwd``.

    Args:
        swa_indices:        ``[num_tokens, 1, topk_swa]`` int32. Global indices
                            into the flat swa KV view.
        swa_topk_lens:      ``[num_tokens]`` int32. Per-token count of valid
                            entries in the swa prefix.
        extra_indices:      ``[num_tokens, 1, topk_extra]`` int32. Global
                            indices into the flat extra KV view (will be
                            shifted by ``s_kv_swa`` so they index into the
                            concatenated buffer).
        extra_topk_lens:    ``[num_tokens]`` int32. Per-token valid count.
        s_kv_swa:           ``num_swa_blocks * swa_block_size``. The size of
                            the swa region in the concatenated KV buffer;
                            added to every extra index.
        alignment:          The topk dim of the output is padded up to a
                            multiple of this value (default 128, matching
                            sparse_fwd's kernel B_TOPK).

    Returns:
        combined_indices: ``[num_tokens, 1, combined_topk]`` int32. Padded
                          with -1 on the right.
        combined_lens:    ``[num_tokens]`` int32. ``swa_len + extra_len`` per
                          token.
    """
    assert (
        swa_indices.dim() == 3 and swa_indices.shape[1] == 1
    ), f"swa_indices must be [num_tokens, 1, topk_swa]; got {swa_indices.shape}"
    assert (
        extra_indices.dim() == 3 and extra_indices.shape[1] == 1
    ), f"extra_indices must be [num_tokens, 1, topk_extra]; got {extra_indices.shape}"
    assert swa_indices.shape[0] == extra_indices.shape[0], (
        f"swa and extra token counts must match: "
        f"{swa_indices.shape[0]} vs {extra_indices.shape[0]}"
    )
    assert swa_topk_lens.shape[0] == extra_topk_lens.shape[0] == swa_indices.shape[0]
    assert swa_indices.dtype == torch.int32 and extra_indices.dtype == torch.int32
    assert swa_indices.is_contiguous() and extra_indices.is_contiguous()

    num_tokens = swa_indices.shape[0]
    topk_swa = swa_indices.shape[-1]
    topk_extra = extra_indices.shape[-1]

    raw_combined = topk_swa + topk_extra
    combined_topk = (raw_combined + alignment - 1) // alignment * alignment

    combined_indices = torch.full(
        (num_tokens, 1, combined_topk),
        fill_value=-1,
        dtype=torch.int32,
        device=swa_indices.device,
    )
    combined_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=swa_indices.device
    )

    # Triton requires the per-program tl.arange size to be a power of 2.
    # topk_{swa,extra} are typically multiples of 64 (asserted in the
    # backend), which are already power of 2 for the values we see
    # (512, 1024, 2048, ...). next_power_of_2 here is a safety net.
    _combine_swa_extra_indices_kernel[(num_tokens,)](
        combined_indices,
        combined_indices.stride(0),
        combined_lens,
        swa_indices,
        swa_indices.stride(0),
        extra_indices,
        extra_indices.stride(0),
        swa_topk_lens,
        extra_topk_lens,
        s_kv_swa,
        TOPK_SWA=triton.next_power_of_2(topk_swa),
        TOPK_EXTRA=triton.next_power_of_2(topk_extra),
        COMBINED_TOPK=combined_topk,
    )
    return combined_indices, combined_lens
