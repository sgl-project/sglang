from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.layers.utils.hash import murmur_hash32
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import (
    _get_or_create_chain_verify_buffers,
    build_dflash_verify_target_probs,
)
from sglang.srt.speculative.dspark_components.kernels.cap_correct_len import (
    CapCorrectLen,
)
from sglang.srt.speculative.dspark_components.kernels.softmax_temp import SoftmaxTemp
from sglang.srt.speculative.reject_sampling import chain_speculative_sampling_triton

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_ACCEPT_SAMPLING.get()


def _hash_uniform(
    *,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    stream: int,
) -> torch.Tensor:
    stream_ids = torch.full((1,), int(stream), dtype=torch.int64, device=seeds.device)
    hashed = murmur_hash32(
        seeds.to(torch.int64).contiguous(),
        positions.to(torch.int64).contiguous(),
        stream_ids,
    ).view(-1)
    return hashed.to(torch.float32) / float(torch.iinfo(torch.uint32).max)


def _chain_uniform_samples(
    *,
    sampling_info,
    positions_2d: Optional[torch.Tensor],
    bs: int,
    gamma: int,
    verify_num_draft_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    sampling_seed = getattr(sampling_info, "sampling_seed", None)
    if sampling_seed is None:
        return (
            torch.rand((bs, gamma), dtype=torch.float32, device=device),
            torch.rand(
                (bs, verify_num_draft_tokens), dtype=torch.float32, device=device
            ),
        )
    if positions_2d is None:
        raise RuntimeError(
            "DSpark seeded sampling needs positions_2d; refusing to fall back "
            "to unseeded random samples."
        )

    if positions_2d.shape[0] < bs or positions_2d.shape[1] < verify_num_draft_tokens:
        raise RuntimeError(
            "DSpark seeded sampling needs positions_2d shaped at least "
            f"({bs}, {verify_num_draft_tokens}), got {tuple(positions_2d.shape)}."
        )

    seeds_2d = sampling_seed.view(bs, 1).expand(bs, verify_num_draft_tokens)
    final_positions = positions_2d[:bs, :verify_num_draft_tokens].contiguous()
    final_uniform = _hash_uniform(
        seeds=seeds_2d.reshape(-1),
        positions=final_positions.reshape(-1),
        stream=1,
    ).view(bs, verify_num_draft_tokens)

    accept_uniform = _hash_uniform(
        seeds=seeds_2d[:, :gamma].reshape(-1),
        positions=final_positions[:, :gamma].reshape(-1),
        stream=0,
    ).view(bs, gamma)
    return accept_uniform, final_uniform


def _sample_cdf_token(probs: torch.Tensor, coin: torch.Tensor) -> int:
    cdf = torch.cumsum(probs.float(), dim=-1)
    total = cdf[-1].clamp_min(torch.finfo(cdf.dtype).tiny)
    target = coin.to(cdf.dtype) * total
    token = torch.searchsorted(cdf, target, right=True)
    return int(torch.clamp(token, max=probs.numel() - 1).item())


def _reference_chain_accept(
    *,
    candidates: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_final: torch.Tensor,
    gamma: int,
    cutoff_verify_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slow reference for debug-checking DSpark non-greedy chain accept.

    This mirrors `reject_sampling.speculative_sampling_classic_kernel` and is
    intentionally Python/Torch based. It is only used under debug assertions and
    in unit tests.
    """

    bs = candidates.shape[0]
    device = candidates.device
    expected_correct = torch.empty((bs,), dtype=torch.int32, device=device)
    expected_bonus = torch.empty((bs,), dtype=torch.int64, device=device)
    expected_cap_trim = torch.zeros((bs,), dtype=torch.int32, device=device)

    for row in range(bs):
        cur_prob_row = 0
        raw_correct = 0
        next_tokens_by_accept_len: list[int] = []
        all_drafts_accepted = True

        for step in range(1, gamma + 1):
            draft_token = int(candidates[row, step].item())
            p = target_probs[row, cur_prob_row, draft_token]
            q = draft_probs[row, cur_prob_row, draft_token]
            coin = uniform_samples[row, step - 1]
            if bool((coin * q < p).item()):
                next_tokens_by_accept_len.append(draft_token)
                raw_correct += 1
                cur_prob_row = step
            else:
                all_drafts_accepted = False
                break

        if all_drafts_accepted:
            final_probs = target_probs[row, cur_prob_row]
        else:
            final_probs = torch.clamp_min(
                target_probs[row, cur_prob_row] - draft_probs[row, cur_prob_row],
                0.0,
            )
        final_token = _sample_cdf_token(
            final_probs, uniform_samples_final[row, cur_prob_row]
        )
        next_tokens_by_accept_len.append(final_token)

        capped_correct = raw_correct
        if cutoff_verify_lens is not None:
            capped_correct = min(
                raw_correct,
                max(int(cutoff_verify_lens[row].item()) - 1, 0),
            )
            expected_cap_trim[row] = raw_correct - capped_correct

        expected_correct[row] = capped_correct
        expected_bonus[row] = next_tokens_by_accept_len[capped_correct]

    return expected_correct, expected_bonus, expected_cap_trim


def _assert_accept_sampling_reference(
    *,
    candidates: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_final: torch.Tensor,
    gamma: int,
    cutoff_verify_lens: Optional[torch.Tensor],
    correct_len: torch.Tensor,
    bonus: torch.Tensor,
    cap_trim_lens: torch.Tensor,
) -> None:
    expected_correct, expected_bonus, expected_cap_trim = _reference_chain_accept(
        candidates=candidates,
        target_probs=target_probs,
        draft_probs=draft_probs,
        uniform_samples=uniform_samples,
        uniform_samples_final=uniform_samples_final,
        gamma=gamma,
        cutoff_verify_lens=cutoff_verify_lens,
    )
    checks = {
        "correct_len": (correct_len, expected_correct),
        "bonus": (bonus, expected_bonus),
        "cap_trim_lens": (cap_trim_lens, expected_cap_trim),
    }
    failures = []
    for name, (actual, expected) in checks.items():
        expected = expected.to(dtype=actual.dtype)
        if not torch.equal(actual, expected):
            failures.append(
                f"{name}: actual={actual.detach().cpu().tolist()} "
                f"expected={expected.detach().cpu().tolist()}"
            )
    if failures:
        raise AssertionError(
            "DSpark non-greedy accept sampling invariant failed: "
            + "; ".join(failures)
        )


class AcceptSampling:
    @classmethod
    def execute(
        cls, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

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
        positions_2d: Optional[torch.Tensor] = None,
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
            positions_2d=positions_2d,
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
        positions_2d: Optional[torch.Tensor] = None,
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
            positions_2d=positions_2d,
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
    positions_2d: Optional[torch.Tensor],
    cutoff_verify_lens: Optional[torch.Tensor],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
]:
    bs = candidates.shape[0]
    device = candidates.device
    if (
        not sampling_info.need_top_k_sampling
        and not sampling_info.need_top_p_sampling
        and not sampling_info.need_min_p_sampling
    ):
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
    uniform_samples, uniform_samples_final = _chain_uniform_samples(
        sampling_info=sampling_info,
        positions_2d=positions_2d,
        bs=bs,
        gamma=gamma,
        verify_num_draft_tokens=verify_num_draft_tokens,
        device=device,
    )
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
    debug_tensors = None
    if envs.SGLANG_DSPARK_VERIFY_TRACE_ASSERT.get():
        debug_tensors = (
            target_probs,
            uniform_samples,
            uniform_samples_final,
            cutoff_verify_lens,
        )
    return correct_len, cap_trim_lens, accept_index, predicts, debug_tensors


def accept_sampling(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    draft_probs: torch.Tensor,
    sampling_info,
    draft_input: DFlashDraftInputV2,
    gamma: int,
    verify_num_draft_tokens: int,
    positions_2d: Optional[torch.Tensor] = None,
    cutoff_verify_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bs = candidates.shape[0]
    device = candidates.device
    correct_len, cap_trim_lens, accept_index, predicts, debug_tensors = (
        _accept_sampling_core(
            candidates=candidates,
            target_logits=target_logits,
            draft_probs=draft_probs,
            sampling_info=sampling_info,
            draft_input=draft_input,
            gamma=gamma,
            verify_num_draft_tokens=verify_num_draft_tokens,
            positions_2d=positions_2d,
            cutoff_verify_lens=cutoff_verify_lens,
        )
    )
    row_ids = torch.arange(bs, dtype=torch.long, device=device)
    accept_pos = accept_index[row_ids, correct_len.to(torch.long)].to(torch.long)
    bonus = predicts[accept_pos].to(torch.int64)
    if debug_tensors is not None:
        target_probs, uniform_samples, uniform_samples_final, debug_cutoff_lens = (
            debug_tensors
        )
        _assert_accept_sampling_reference(
            candidates=candidates,
            target_probs=target_probs,
            draft_probs=draft_probs,
            uniform_samples=uniform_samples,
            uniform_samples_final=uniform_samples_final,
            gamma=gamma,
            cutoff_verify_lens=debug_cutoff_lens,
            correct_len=correct_len,
            bonus=bonus,
            cap_trim_lens=cap_trim_lens,
        )
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
    positions_2d: Optional[torch.Tensor] = None,
    cutoff_verify_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    correct_len, cap_trim_lens, accept_index, predicts, debug_tensors = (
        _accept_sampling_core(
            candidates=candidates,
            target_logits=target_logits,
            draft_probs=draft_probs,
            sampling_info=sampling_info,
            draft_input=draft_input,
            gamma=gamma,
            verify_num_draft_tokens=verify_num_draft_tokens,
            positions_2d=positions_2d,
            cutoff_verify_lens=cutoff_verify_lens,
        )
    )
    bonus = gather_two_level_bonus_triton(
        accept_index=accept_index, predicts=predicts, correct_len=correct_len
    )
    if debug_tensors is not None:
        target_probs, uniform_samples, uniform_samples_final, debug_cutoff_lens = (
            debug_tensors
        )
        _assert_accept_sampling_reference(
            candidates=candidates,
            target_probs=target_probs,
            draft_probs=draft_probs,
            uniform_samples=uniform_samples,
            uniform_samples_final=uniform_samples_final,
            gamma=gamma,
            cutoff_verify_lens=debug_cutoff_lens,
            correct_len=correct_len,
            bonus=bonus,
            cap_trim_lens=cap_trim_lens,
        )
    return correct_len, bonus, cap_trim_lens
