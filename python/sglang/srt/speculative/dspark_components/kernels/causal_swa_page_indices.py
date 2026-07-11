from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.speculative.dspark_components.kernels.dispatch import (
    inputs_on_cuda,
)


class BuildCausalSwaPageIndices:
    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        req_to_token: torch.Tensor,
        full_to_swa_mapping: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        seq_lens_casual: torch.Tensor,
        swa_window: int,
        page_index_aligned_size: int,
    ) -> torch.Tensor:
        return build_causal_swa_page_indices(
            req_to_token=req_to_token,
            full_to_swa_mapping=full_to_swa_mapping,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            swa_window=swa_window,
            page_index_aligned_size=page_index_aligned_size,
        )

    @classmethod
    def triton(
        cls,
        *,
        req_to_token: torch.Tensor,
        full_to_swa_mapping: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        seq_lens_casual: torch.Tensor,
        swa_window: int,
        page_index_aligned_size: int,
    ) -> torch.Tensor:
        return build_causal_swa_page_indices_triton(
            req_to_token=req_to_token,
            full_to_swa_mapping=full_to_swa_mapping,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            swa_window=swa_window,
            page_index_aligned_size=page_index_aligned_size,
        )


def build_causal_swa_page_indices(
    *,
    req_to_token: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    req_pool_indices_repeated: torch.Tensor,
    seq_lens_casual: torch.Tensor,
    swa_window: int,
    page_index_aligned_size: int,
) -> torch.Tensor:
    device = seq_lens_casual.device
    pos_causal = seq_lens_casual - 1
    num_qo_tokens = seq_lens_casual.size(0)
    offsets = pos_causal.unsqueeze(1) - torch.arange(
        swa_window, dtype=torch.int32, device=device
    ).unsqueeze(0)
    invalid_offset_mask = offsets < 0
    offsets.masked_fill_(invalid_offset_mask, 0)
    raw_indices = req_to_token[req_pool_indices_repeated[:, None], offsets]
    assert raw_indices.shape == (num_qo_tokens, swa_window)
    raw_indices.masked_fill_(invalid_offset_mask, -1)
    swa_indices = full_to_swa_mapping[raw_indices]
    swa_indices = swa_indices.to(torch.int32)

    padded_width = (
        (swa_window + page_index_aligned_size - 1) // page_index_aligned_size
    ) * page_index_aligned_size
    if padded_width == swa_window:
        return swa_indices
    return torch.nn.functional.pad(
        swa_indices, (0, padded_width - swa_window), value=-1
    )


@triton.jit
def _causal_swa_page_indices_kernel(
    req_to_token_ptr,
    full_to_swa_ptr,
    req_pool_ptr,
    seq_lens_ptr,
    out_ptr,
    rt_stride,
    swa_window,
    padded_width,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    pos = tl.load(seq_lens_ptr + row).to(tl.int64) - 1
    rp = tl.load(req_pool_ptr + row).to(tl.int64)
    base = req_to_token_ptr + rp * rt_stride
    out_base = out_ptr + row.to(tl.int64) * padded_width

    for k0 in range(0, padded_width, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        kmask = k < padded_width
        off = pos - k.to(tl.int64)
        valid = (k < swa_window) & (off >= 0) & kmask
        full_loc = tl.load(base + tl.where(valid, off, 0), mask=valid, other=-1).to(
            tl.int64
        )
        swa = tl.load(full_to_swa_ptr + full_loc, mask=valid, other=-1).to(tl.int32)
        tl.store(out_base + k, tl.where(valid, swa, -1), mask=kmask)


def build_causal_swa_page_indices_triton(
    *,
    req_to_token: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    req_pool_indices_repeated: torch.Tensor,
    seq_lens_casual: torch.Tensor,
    swa_window: int,
    page_index_aligned_size: int,
) -> torch.Tensor:
    num_qo_tokens = seq_lens_casual.size(0)
    padded_width = (
        (swa_window + page_index_aligned_size - 1) // page_index_aligned_size
    ) * page_index_aligned_size
    out = torch.empty(
        (num_qo_tokens, padded_width),
        dtype=torch.int32,
        device=seq_lens_casual.device,
    )
    BLOCK_K = 256
    _causal_swa_page_indices_kernel[(num_qo_tokens,)](
        req_to_token,
        full_to_swa_mapping,
        req_pool_indices_repeated,
        seq_lens_casual,
        out,
        req_to_token.stride(0),
        swa_window,
        padded_width,
        BLOCK_K=BLOCK_K,
    )
    return out
