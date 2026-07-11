from __future__ import annotations

from typing import Optional

import msgspec
import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_EXPAND_PREFILL.get()


class ExpandPrefillCausallyResult(msgspec.Struct):
    seq_lens_casual: torch.Tensor
    req_pool_indices_repeated: torch.Tensor


class ExpandPrefillCausally:
    @classmethod
    def execute(cls, *args, **kwargs) -> ExpandPrefillCausallyResult:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        extend_start_loc: Optional[torch.Tensor],
        seq_lens_cpu: Optional[list[int]],
        extend_seq_lens_cpu: Optional[list[int]],
        num_tokens: int,
        padded_num_tokens: Optional[int],
    ) -> ExpandPrefillCausallyResult:
        return expand_prefill_causally(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
            extend_start_loc=extend_start_loc,
            seq_lens_cpu=seq_lens_cpu,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            num_tokens=num_tokens,
            padded_num_tokens=padded_num_tokens,
        )

    @classmethod
    def triton(
        cls,
        *,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        extend_start_loc: Optional[torch.Tensor],
        seq_lens_cpu: Optional[list[int]],
        extend_seq_lens_cpu: Optional[list[int]],
        num_tokens: int,
        padded_num_tokens: Optional[int],
    ) -> ExpandPrefillCausallyResult:
        return expand_prefill_causally_triton(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
            num_tokens=num_tokens,
            padded_num_tokens=padded_num_tokens,
        )


def expand_prefill_causally(
    *,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_start_loc: Optional[torch.Tensor],
    seq_lens_cpu: Optional[list[int]],
    extend_seq_lens_cpu: Optional[list[int]],
    num_tokens: int,
    padded_num_tokens: Optional[int],
) -> ExpandPrefillCausallyResult:
    device = req_pool_indices.device
    cuda_int32_kwargs = {"dtype": torch.int32, "device": device}

    if extend_start_loc is not None:
        repeats = extend_seq_lens.to(torch.int64)
        req_pool_indices_repeated = torch.repeat_interleave(
            req_pool_indices, repeats, output_size=num_tokens
        )
        start_positions = seq_lens.to(torch.int32) - extend_seq_lens.to(torch.int32) + 1
        start_positions_repeated = torch.repeat_interleave(
            start_positions, repeats, output_size=num_tokens
        )
        start_locs_repeated = torch.repeat_interleave(
            extend_start_loc.to(torch.int32), repeats, output_size=num_tokens
        )
        token_offsets = (
            torch.arange(num_tokens, **cuda_int32_kwargs) - start_locs_repeated
        )
        seq_lens_casual = start_positions_repeated + token_offsets

        if padded_num_tokens is not None and padded_num_tokens > num_tokens:
            pad_size = padded_num_tokens - num_tokens
            seq_lens_casual = torch.nn.functional.pad(
                seq_lens_casual, (0, pad_size), value=1
            )
            req_pool_indices_repeated = torch.cat(
                (
                    req_pool_indices_repeated,
                    req_pool_indices_repeated[-1:].expand(pad_size),
                )
            )
        return ExpandPrefillCausallyResult(
            seq_lens_casual=seq_lens_casual,
            req_pool_indices_repeated=req_pool_indices_repeated,
        )

    assert seq_lens_cpu is not None and extend_seq_lens_cpu is not None
    seq_lens_casual = torch.empty(num_tokens, **cuda_int32_kwargs)
    idx_to_req_repeated = torch.empty(num_tokens, **cuda_int32_kwargs)
    offset = 0
    for i, (kv_len, qo_len) in enumerate(zip(seq_lens_cpu, extend_seq_lens_cpu)):
        out = seq_lens_casual[offset : offset + qo_len]
        offset += qo_len
        torch.arange(kv_len - qo_len + 1, kv_len + 1, out=out)
        idx_to_req_repeated[offset - qo_len : offset].fill_(i)

    assert offset == num_tokens
    req_pool_indices_repeated = req_pool_indices[idx_to_req_repeated]

    if padded_num_tokens is not None and padded_num_tokens > num_tokens:
        pad_size = padded_num_tokens - num_tokens
        seq_lens_casual = torch.nn.functional.pad(
            seq_lens_casual, (0, pad_size), value=1
        )
        req_pool_indices_repeated = torch.nn.functional.pad(
            req_pool_indices_repeated,
            (0, pad_size),
            value=req_pool_indices_repeated[-1].item(),
        )
    return ExpandPrefillCausallyResult(
        seq_lens_casual=seq_lens_casual,
        req_pool_indices_repeated=req_pool_indices_repeated,
    )


@triton.jit
def _expand_prefill_causally_kernel(
    req_pool_ptr,
    seq_lens_ptr,
    extend_seq_lens_ptr,
    seq_lens_casual_ptr,
    req_pool_repeated_ptr,
    bs,
    num_tokens,
    total_tokens,
    BLOCK: tl.constexpr,
    BS_P2: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_tokens

    b = tl.arange(0, BS_P2)
    bmask = b < bs
    extend = tl.load(extend_seq_lens_ptr + b, mask=bmask, other=0).to(tl.int32)
    start_locs = tl.cumsum(extend, axis=0) - extend

    is_real = offs < num_tokens
    t = tl.where(is_real, offs, 0).to(tl.int32)
    started = (start_locs[None, :] <= t[:, None]) & bmask[None, :]
    r = tl.sum(started.to(tl.int32), axis=1) - 1
    r = tl.where(is_real, r, bs - 1).to(tl.int64)

    seq_len = tl.load(seq_lens_ptr + r, mask=mask, other=0).to(tl.int32)
    ext = tl.load(extend_seq_lens_ptr + r, mask=mask, other=0).to(tl.int32)
    start_loc = tl.sum(tl.where(started, extend[None, :], 0).to(tl.int32), axis=1) - ext
    causal = (seq_len - ext + 1) + (t - start_loc)
    causal = tl.where(is_real, causal, 1)

    rp = tl.load(req_pool_ptr + r, mask=mask, other=0)
    tl.store(seq_lens_casual_ptr + offs, causal, mask=mask)
    tl.store(req_pool_repeated_ptr + offs, rp, mask=mask)


def expand_prefill_causally_triton(
    *,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    num_tokens: int,
    padded_num_tokens: Optional[int],
) -> ExpandPrefillCausallyResult:
    bs = req_pool_indices.shape[0]
    device = req_pool_indices.device
    total_tokens = (
        padded_num_tokens
        if padded_num_tokens is not None and padded_num_tokens > num_tokens
        else num_tokens
    )

    seq_lens_casual = torch.empty(total_tokens, dtype=torch.int32, device=device)
    req_pool_indices_repeated = torch.empty(
        total_tokens, dtype=req_pool_indices.dtype, device=device
    )
    BLOCK = 256
    _expand_prefill_causally_kernel[(triton.cdiv(total_tokens, BLOCK),)](
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        seq_lens_casual,
        req_pool_indices_repeated,
        bs,
        num_tokens,
        total_tokens,
        BLOCK=BLOCK,
        BS_P2=triton.next_power_of_2(max(bs, 1)),
    )
    return ExpandPrefillCausallyResult(
        seq_lens_casual=seq_lens_casual,
        req_pool_indices_repeated=req_pool_indices_repeated,
    )
