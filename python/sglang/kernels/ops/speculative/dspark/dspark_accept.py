from __future__ import annotations

from typing import Optional

import msgspec
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.speculative.dspark.dispatch import inputs_on_cuda
from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import (
    _get_or_create_chain_verify_buffers,
    build_dflash_verify_target_probs,
    compute_dflash_correct_drafts_and_bonus,
)


class AcceptSampling:
    @classmethod
    def execute(
        cls, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        candidates: torch.Tensor,
        target_logits: torch.Tensor,
        draft_probs: torch.Tensor,
        sampling_info,
        draft_input: DFlashDraftInputV2,
        gamma: int,
        verify_num_draft_tokens: int,
        cutoff_verify_lens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return accept_sampling(
            candidates=candidates,
            target_logits=target_logits,
            draft_probs=draft_probs,
            sampling_info=sampling_info,
            draft_input=draft_input,
            gamma=gamma,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_verify_lens=cutoff_verify_lens,
        )

    @classmethod
    def triton(
        cls,
        *,
        candidates: torch.Tensor,
        target_logits: torch.Tensor,
        draft_probs: torch.Tensor,
        sampling_info,
        draft_input: DFlashDraftInputV2,
        gamma: int,
        verify_num_draft_tokens: int,
        cutoff_verify_lens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return accept_sampling_triton(
            candidates=candidates,
            target_logits=target_logits,
            draft_probs=draft_probs,
            sampling_info=sampling_info,
            draft_input=draft_input,
            gamma=gamma,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_verify_lens=cutoff_verify_lens,
        )


def _accept_sampling_core(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    draft_probs: torch.Tensor,
    sampling_info,
    draft_input: DFlashDraftInputV2,
    gamma: int,
    verify_num_draft_tokens: int,
    cutoff_verify_lens: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bs = candidates.shape[0]
    device = candidates.device
    if not sampling_info.need_top_k_sampling and not sampling_info.need_top_p_sampling:
        target_probs = SoftmaxTemp.execute(
            logits=target_logits,
            temperatures=sampling_info.temperatures,
            rows_per_request=verify_num_draft_tokens,
        ).view(bs, verify_num_draft_tokens, -1)
    else:
        target_probs = build_dflash_verify_target_probs(
            next_token_logits=target_logits,
            sampling_info=sampling_info,
            draft_token_num=verify_num_draft_tokens,
            bs=bs,
            max_top_k=draft_input.max_top_k,
            uniform_top_k_value=draft_input.uniform_top_k_value,
        )
    (
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        predicts,
        accept_index,
        accept_token_num,
    ) = _get_or_create_chain_verify_buffers(
        bs=bs,
        draft_token_num=verify_num_draft_tokens,
        device=device,
    )
    uniform_samples = torch.rand((bs, gamma), dtype=torch.float32, device=device)
    uniform_samples_final = torch.rand((bs,), dtype=torch.float32, device=device)
    chain_speculative_sampling_triton(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrieve_index,
        retrive_next_token=retrieve_next_token,
        retrive_next_sibling=retrieve_next_sibling,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_final,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )
    correct_len = accept_token_num
    if cutoff_verify_lens is not None:
        correct_len, cap_trim_lens = CapCorrectLen.execute(
            correct_len=correct_len, verify_lens=cutoff_verify_lens
        )
    else:
        cap_trim_lens = torch.zeros_like(correct_len)
    return correct_len, cap_trim_lens, accept_index, predicts


def accept_sampling(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    draft_probs: torch.Tensor,
    sampling_info,
    draft_input: DFlashDraftInputV2,
    gamma: int,
    verify_num_draft_tokens: int,
    cutoff_verify_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bs = candidates.shape[0]
    device = candidates.device
    correct_len, cap_trim_lens, accept_index, predicts = _accept_sampling_core(
        candidates=candidates,
        target_logits=target_logits,
        draft_probs=draft_probs,
        sampling_info=sampling_info,
        draft_input=draft_input,
        gamma=gamma,
        verify_num_draft_tokens=verify_num_draft_tokens,
        cutoff_verify_lens=cutoff_verify_lens,
    )
    row_ids = torch.arange(bs, dtype=torch.long, device=device)
    accept_pos = accept_index[row_ids, correct_len.to(torch.long)].to(torch.long)
    bonus = predicts[accept_pos].to(torch.int64)
    return correct_len, bonus, cap_trim_lens


@triton.jit
def _gather_two_level_bonus_kernel(
    accept_index_ptr,
    predicts_ptr,
    correct_len_ptr,
    out_ptr,
    cols,
    n,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    cl = tl.load(correct_len_ptr + offs, mask=mask, other=0).to(tl.int64)
    accept_pos = tl.load(accept_index_ptr + offs * cols + cl, mask=mask, other=0).to(
        tl.int64
    )
    bonus = tl.load(predicts_ptr + accept_pos, mask=mask, other=0)
    tl.store(out_ptr + offs, bonus.to(tl.int64), mask=mask)


def gather_two_level_bonus_triton(
    *,
    accept_index: torch.Tensor,
    predicts: torch.Tensor,
    correct_len: torch.Tensor,
) -> torch.Tensor:
    bs, cols = accept_index.shape
    accept_index = accept_index.contiguous()
    predicts = predicts.contiguous()
    correct_len = correct_len.contiguous()
    out = torch.empty(bs, dtype=torch.int64, device=accept_index.device)
    BLOCK = 256
    grid = (triton.cdiv(bs, BLOCK),)
    _gather_two_level_bonus_kernel[grid](
        accept_index, predicts, correct_len, out, cols, bs, BLOCK=BLOCK
    )
    return out


def accept_sampling_triton(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    draft_probs: torch.Tensor,
    sampling_info,
    draft_input: DFlashDraftInputV2,
    gamma: int,
    verify_num_draft_tokens: int,
    cutoff_verify_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    correct_len, cap_trim_lens, accept_index, predicts = _accept_sampling_core(
        candidates=candidates,
        target_logits=target_logits,
        draft_probs=draft_probs,
        sampling_info=sampling_info,
        draft_input=draft_input,
        gamma=gamma,
        verify_num_draft_tokens=verify_num_draft_tokens,
        cutoff_verify_lens=cutoff_verify_lens,
    )
    bonus = gather_two_level_bonus_triton(
        accept_index=accept_index, predicts=predicts, correct_len=correct_len
    )
    return correct_len, bonus, cap_trim_lens


try:
    from flashinfer.sampling import softmax as _flashinfer_softmax
except ImportError:
    _flashinfer_softmax = None


class SoftmaxTemp:
    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if not inputs_on_cuda(*args, **kwargs):
            return cls.torch(*args, **kwargs)
        if _flashinfer_softmax is not None:
            return cls.flashinfer(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        rows_per_request: int,
    ) -> torch.Tensor:
        return softmax_temp(
            logits=logits,
            temperatures=temperatures,
            rows_per_request=rows_per_request,
        )

    @classmethod
    def triton(
        cls,
        *,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        rows_per_request: int,
    ) -> torch.Tensor:
        return softmax_temp_triton(
            logits=logits,
            temperatures=temperatures,
            rows_per_request=rows_per_request,
        )

    @classmethod
    def flashinfer(
        cls,
        *,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        rows_per_request: int,
    ) -> torch.Tensor:
        return softmax_temp_flashinfer(
            logits=logits,
            temperatures=temperatures,
            rows_per_request=rows_per_request,
        )


def softmax_temp(
    *,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    rows_per_request: int,
) -> torch.Tensor:
    num_rows = logits.shape[0]
    bs = num_rows // rows_per_request
    assert (
        bs * rows_per_request == num_rows
    ), f"num_rows {num_rows} not divisible by rows_per_request {rows_per_request}"
    temp_per_row = torch.repeat_interleave(
        temperatures.reshape(bs).to(torch.float32), rows_per_request, dim=0
    )
    scaled = logits.to(torch.float32) / temp_per_row[:, None]
    return torch.softmax(scaled, dim=-1)


@triton.jit
def _softmax_temp_kernel(
    logits_ptr,
    temp_ptr,
    out_ptr,
    vocab,
    rows_per_request,
    logits_row_stride,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    temp = tl.load(temp_ptr + row // rows_per_request).to(tl.float32)
    base = logits_ptr + row.to(tl.int64) * logits_row_stride
    out_base = out_ptr + row.to(tl.int64) * vocab

    row_max = -float("inf")
    for v0 in range(0, vocab, BLOCK_V):
        offs = v0 + tl.arange(0, BLOCK_V)
        vmask = offs < vocab
        x = tl.load(base + offs, mask=vmask, other=-float("inf")).to(tl.float32)
        x = x / temp
        row_max = tl.maximum(row_max, tl.max(x, axis=0))

    sum_exp = 0.0
    for v0 in range(0, vocab, BLOCK_V):
        offs = v0 + tl.arange(0, BLOCK_V)
        vmask = offs < vocab
        x = tl.load(base + offs, mask=vmask, other=-float("inf")).to(tl.float32)
        x = x / temp
        e = tl.exp(x - row_max)
        e = tl.where(vmask, e, 0.0)
        sum_exp += tl.sum(e, axis=0)

    for v0 in range(0, vocab, BLOCK_V):
        offs = v0 + tl.arange(0, BLOCK_V)
        vmask = offs < vocab
        x = tl.load(base + offs, mask=vmask, other=-float("inf")).to(tl.float32)
        x = x / temp
        e = tl.exp(x - row_max)
        tl.store(out_base + offs, e / sum_exp, mask=vmask)


def softmax_temp_triton(
    *,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    rows_per_request: int,
) -> torch.Tensor:
    num_rows, vocab = logits.shape[0], logits.shape[-1]
    bs = num_rows // rows_per_request
    assert (
        bs * rows_per_request == num_rows
    ), f"num_rows {num_rows} not divisible by rows_per_request {rows_per_request}"
    temperatures = temperatures.reshape(bs).to(torch.float32).contiguous()
    out = torch.empty((num_rows, vocab), dtype=torch.float32, device=logits.device)
    BLOCK_V = 4096
    _softmax_temp_kernel[(num_rows,)](
        logits,
        temperatures,
        out,
        vocab,
        rows_per_request,
        logits.stride(0),
        BLOCK_V=BLOCK_V,
    )
    return out


def softmax_temp_flashinfer(
    *,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    rows_per_request: int,
) -> torch.Tensor:
    if _flashinfer_softmax is None:
        raise RuntimeError(
            "softmax_temp_flashinfer requires flashinfer.sampling.softmax, "
            "which is unavailable in this environment"
        )
    num_rows, vocab = logits.shape[0], logits.shape[-1]
    bs = num_rows // rows_per_request
    assert (
        bs * rows_per_request == num_rows
    ), f"num_rows {num_rows} not divisible by rows_per_request {rows_per_request}"
    temp_per_row = torch.repeat_interleave(
        temperatures.reshape(bs).to(torch.float32), rows_per_request, dim=0
    ).contiguous()
    logits_2d = logits.to(torch.float32).contiguous()
    return _flashinfer_softmax(logits=logits_2d, temperature=temp_per_row)


class MixedAcceptSelectResult(msgspec.Struct):
    correct_len: torch.Tensor
    bonus: torch.Tensor
    cap_trim_lens: torch.Tensor


class SelectMixedAccept:
    @classmethod
    def execute(cls, *args, **kwargs) -> MixedAcceptSelectResult:
        if inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        greedy_mask: torch.Tensor,
        greedy_len: torch.Tensor,
        greedy_bonus: torch.Tensor,
        greedy_trim: torch.Tensor,
        sampling_len: torch.Tensor,
        sampling_bonus: torch.Tensor,
        sampling_trim: torch.Tensor,
    ) -> MixedAcceptSelectResult:
        return select_mixed_accept(
            greedy_mask=greedy_mask,
            greedy_len=greedy_len,
            greedy_bonus=greedy_bonus,
            greedy_trim=greedy_trim,
            sampling_len=sampling_len,
            sampling_bonus=sampling_bonus,
            sampling_trim=sampling_trim,
        )

    @classmethod
    def triton(
        cls,
        *,
        greedy_mask: torch.Tensor,
        greedy_len: torch.Tensor,
        greedy_bonus: torch.Tensor,
        greedy_trim: torch.Tensor,
        sampling_len: torch.Tensor,
        sampling_bonus: torch.Tensor,
        sampling_trim: torch.Tensor,
    ) -> MixedAcceptSelectResult:
        return select_mixed_accept_triton(
            greedy_mask=greedy_mask,
            greedy_len=greedy_len,
            greedy_bonus=greedy_bonus,
            greedy_trim=greedy_trim,
            sampling_len=sampling_len,
            sampling_bonus=sampling_bonus,
            sampling_trim=sampling_trim,
        )


def select_mixed_accept(
    *,
    greedy_mask: torch.Tensor,
    greedy_len: torch.Tensor,
    greedy_bonus: torch.Tensor,
    greedy_trim: torch.Tensor,
    sampling_len: torch.Tensor,
    sampling_bonus: torch.Tensor,
    sampling_trim: torch.Tensor,
) -> MixedAcceptSelectResult:
    correct_len = torch.where(
        greedy_mask, greedy_len.to(sampling_len.dtype), sampling_len
    )
    bonus = torch.where(greedy_mask, greedy_bonus, sampling_bonus)
    cap_trim_lens = torch.where(
        greedy_mask, greedy_trim.to(sampling_trim.dtype), sampling_trim
    )
    return MixedAcceptSelectResult(
        correct_len=correct_len, bonus=bonus, cap_trim_lens=cap_trim_lens
    )


@triton.jit
def _mixed_accept_select_kernel(
    greedy_mask_ptr,
    greedy_len_ptr,
    greedy_bonus_ptr,
    greedy_trim_ptr,
    sampling_len_ptr,
    sampling_bonus_ptr,
    sampling_trim_ptr,
    correct_len_ptr,
    bonus_ptr,
    cap_trim_ptr,
    bs,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < bs
    is_greedy = tl.load(greedy_mask_ptr + offs, mask=mask, other=0) != 0

    g_len = tl.load(greedy_len_ptr + offs, mask=mask, other=0)
    s_len = tl.load(sampling_len_ptr + offs, mask=mask, other=0)
    tl.store(correct_len_ptr + offs, tl.where(is_greedy, g_len, s_len), mask=mask)

    g_bonus = tl.load(greedy_bonus_ptr + offs, mask=mask, other=0)
    s_bonus = tl.load(sampling_bonus_ptr + offs, mask=mask, other=0)
    tl.store(bonus_ptr + offs, tl.where(is_greedy, g_bonus, s_bonus), mask=mask)

    g_trim = tl.load(greedy_trim_ptr + offs, mask=mask, other=0)
    s_trim = tl.load(sampling_trim_ptr + offs, mask=mask, other=0)
    tl.store(cap_trim_ptr + offs, tl.where(is_greedy, g_trim, s_trim), mask=mask)


def select_mixed_accept_triton(
    *,
    greedy_mask: torch.Tensor,
    greedy_len: torch.Tensor,
    greedy_bonus: torch.Tensor,
    greedy_trim: torch.Tensor,
    sampling_len: torch.Tensor,
    sampling_bonus: torch.Tensor,
    sampling_trim: torch.Tensor,
) -> MixedAcceptSelectResult:
    bs = greedy_mask.shape[0]
    device = greedy_mask.device

    correct_len = torch.empty(bs, dtype=sampling_len.dtype, device=device)
    bonus = torch.empty(bs, dtype=sampling_bonus.dtype, device=device)
    cap_trim_lens = torch.empty(bs, dtype=sampling_trim.dtype, device=device)
    BLOCK = 256
    _mixed_accept_select_kernel[(triton.cdiv(bs, BLOCK),)](
        greedy_mask,
        greedy_len,
        greedy_bonus,
        greedy_trim,
        sampling_len,
        sampling_bonus,
        sampling_trim,
        correct_len,
        bonus,
        cap_trim_lens,
        bs,
        BLOCK=BLOCK,
    )
    return MixedAcceptSelectResult(
        correct_len=correct_len, bonus=bonus, cap_trim_lens=cap_trim_lens
    )


class AcceptGreedy:
    @classmethod
    def execute(
        cls, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        candidates: torch.Tensor,
        target_logits: torch.Tensor,
        verify_num_draft_tokens: int,
        cutoff_verify_lens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return accept_greedy(
            candidates=candidates,
            target_logits=target_logits,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_verify_lens=cutoff_verify_lens,
        )

    @classmethod
    def triton(
        cls,
        *,
        candidates: torch.Tensor,
        target_logits: torch.Tensor,
        verify_num_draft_tokens: int,
        cutoff_verify_lens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return accept_greedy_triton(
            candidates=candidates,
            target_logits=target_logits,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_verify_lens=cutoff_verify_lens,
        )


def accept_greedy(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    verify_num_draft_tokens: int,
    cutoff_verify_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bs = candidates.shape[0]
    target_predict = torch.argmax(target_logits, dim=-1).view(
        bs, verify_num_draft_tokens
    )
    correct_len, bonus = compute_dflash_correct_drafts_and_bonus(
        candidates=candidates,
        target_predict=target_predict,
    )
    cap_trim_lens = torch.zeros_like(correct_len)
    if cutoff_verify_lens is not None:
        correct_len, cap_trim_lens = CapCorrectLen.execute(
            correct_len=correct_len, verify_lens=cutoff_verify_lens
        )
        row_ids = torch.arange(bs, device=target_predict.device)
        bonus = target_predict[row_ids, correct_len.to(torch.long)].to(torch.int64)
    return correct_len, bonus, cap_trim_lens


@triton.jit
def _gather_row_bonus_kernel(
    table_ptr,
    idx_ptr,
    out_ptr,
    cols,
    n,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = tl.load(idx_ptr + offs, mask=mask, other=0).to(tl.int64)
    val = tl.load(table_ptr + offs * cols + idx, mask=mask, other=0)
    tl.store(out_ptr + offs, val.to(tl.int64), mask=mask)


def gather_row_bonus_triton(*, table: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    bs, cols = table.shape
    table = table.contiguous()
    idx = idx.contiguous()
    out = torch.empty(bs, dtype=torch.int64, device=table.device)
    BLOCK = 256
    grid = (triton.cdiv(bs, BLOCK),)
    _gather_row_bonus_kernel[grid](table, idx, out, cols, bs, BLOCK=BLOCK)
    return out


def accept_greedy_triton(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    verify_num_draft_tokens: int,
    cutoff_verify_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bs = candidates.shape[0]
    target_predict = torch.argmax(target_logits, dim=-1).view(
        bs, verify_num_draft_tokens
    )
    correct_len, bonus = compute_dflash_correct_drafts_and_bonus(
        candidates=candidates,
        target_predict=target_predict,
    )
    cap_trim_lens = torch.zeros_like(correct_len)
    if cutoff_verify_lens is not None:
        correct_len, cap_trim_lens = CapCorrectLen.execute(
            correct_len=correct_len, verify_lens=cutoff_verify_lens
        )
        bonus = gather_row_bonus_triton(table=target_predict, idx=correct_len)
    return correct_len, bonus, cap_trim_lens


class FinalizeAcceptLensResult(msgspec.Struct):
    commit_lens: torch.Tensor
    new_seq_lens: torch.Tensor
    cap_trim_lens: torch.Tensor


class FinalizeAcceptLens:
    @classmethod
    def execute(cls, *args, **kwargs) -> FinalizeAcceptLensResult:
        if inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        correct_len: torch.Tensor,
        cap_trim_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
    ) -> FinalizeAcceptLensResult:
        return finalize_accept_lens(
            correct_len=correct_len,
            cap_trim_lens=cap_trim_lens,
            prefix_lens=prefix_lens,
        )

    @classmethod
    def triton(
        cls,
        *,
        correct_len: torch.Tensor,
        cap_trim_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
    ) -> FinalizeAcceptLensResult:
        return finalize_accept_lens_triton(
            correct_len=correct_len,
            cap_trim_lens=cap_trim_lens,
            prefix_lens=prefix_lens,
        )


def finalize_accept_lens(
    *,
    correct_len: torch.Tensor,
    cap_trim_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
) -> FinalizeAcceptLensResult:
    commit_lens = correct_len.to(torch.int32) + 1
    new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
    return FinalizeAcceptLensResult(
        commit_lens=commit_lens,
        new_seq_lens=new_seq_lens,
        cap_trim_lens=cap_trim_lens.to(torch.int32),
    )


@triton.jit
def _finalize_accept_lens_kernel(
    correct_len_ptr,
    cap_trim_ptr,
    prefix_lens_ptr,
    commit_lens_ptr,
    new_seq_lens_ptr,
    cap_trim_out_ptr,
    bs,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < bs
    commit = tl.load(correct_len_ptr + offs, mask=mask, other=0).to(tl.int32) + 1
    prefix = tl.load(prefix_lens_ptr + offs, mask=mask, other=0)
    trim = tl.load(cap_trim_ptr + offs, mask=mask, other=0).to(tl.int32)
    tl.store(commit_lens_ptr + offs, commit, mask=mask)
    tl.store(new_seq_lens_ptr + offs, prefix + commit, mask=mask)
    tl.store(cap_trim_out_ptr + offs, trim, mask=mask)


def finalize_accept_lens_triton(
    *,
    correct_len: torch.Tensor,
    cap_trim_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
) -> FinalizeAcceptLensResult:
    bs = correct_len.shape[0]
    device = correct_len.device

    commit_lens = torch.empty(bs, dtype=torch.int32, device=device)
    new_seq_lens = torch.empty(bs, dtype=prefix_lens.dtype, device=device)
    cap_trim_out = torch.empty(bs, dtype=torch.int32, device=device)
    BLOCK = 256
    _finalize_accept_lens_kernel[(triton.cdiv(bs, BLOCK),)](
        correct_len,
        cap_trim_lens,
        prefix_lens,
        commit_lens,
        new_seq_lens,
        cap_trim_out,
        bs,
        BLOCK=BLOCK,
    )
    return FinalizeAcceptLensResult(
        commit_lens=commit_lens,
        new_seq_lens=new_seq_lens,
        cap_trim_lens=cap_trim_out,
    )


class CapCorrectLen:
    @classmethod
    def execute(cls, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        correct_len: torch.Tensor,
        verify_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return cap_correct_len(
            correct_len=correct_len,
            verify_lens=verify_lens,
        )

    @classmethod
    def triton(
        cls,
        *,
        correct_len: torch.Tensor,
        verify_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return cap_correct_len_triton(
            correct_len=correct_len,
            verify_lens=verify_lens,
        )


def cap_correct_len(
    *,
    correct_len: torch.Tensor,
    verify_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ell_r = (verify_lens.to(device=correct_len.device) - 1).to(correct_len.dtype)
    capped = torch.minimum(correct_len, ell_r)
    cap_trim_lens = correct_len - capped
    return capped, cap_trim_lens


@triton.jit
def _cap_correct_len_kernel(
    correct_len_ptr,
    verify_lens_ptr,
    capped_ptr,
    trim_ptr,
    n,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    cl = tl.load(correct_len_ptr + offs, mask=mask, other=0).to(tl.int64)
    vl = tl.load(verify_lens_ptr + offs, mask=mask, other=0).to(tl.int64)
    ell = vl - 1
    capped = tl.minimum(cl, ell)
    trim = cl - capped
    tl.store(capped_ptr + offs, capped, mask=mask)
    tl.store(trim_ptr + offs, trim, mask=mask)


def cap_correct_len_triton(
    *,
    correct_len: torch.Tensor,
    verify_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = correct_len.device
    correct_len = correct_len.contiguous()
    verify_lens = verify_lens.to(device=device).contiguous()
    n = correct_len.shape[0]
    capped = torch.empty_like(correct_len)
    trim = torch.empty_like(correct_len)
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _cap_correct_len_kernel[grid](
        correct_len, verify_lens, capped, trim, n, BLOCK=BLOCK
    )
    return capped, trim
