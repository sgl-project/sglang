"""DSpark static greedy verify: the accept and its bookkeeping in one launch
(SGLANG_DSPARK_FUSED_ACCEPT_TAIL), one program per request row.

Per row, from the target's argmax ids: the leading draft matches and the bonus
(AcceptGreedy; ``bonus_parts`` replaces the bonus with the in-graph simulated
bonus's per-split maxima reduced here),
the simulated correct length override (build_accept_outs), commit / new seq lens and
a zero cap trim (FinalizeAcceptLens), the output tokens (BuildOutTokens), and the
gated commit lengths the static epilogue's draft-KV commit reads. It can also run
what run_static issues after the accept: the FutureMap relay scatter (``relay``), the
engram history commit and the deferred draft KV staging (``commit_tail``). Every
output equals the separate ops' bit for bit. With a simulated length and
``bonus_parts`` the argmax ids are not read.
"""

from __future__ import annotations

from typing import Optional

import msgspec
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.speculative.dspark.dispatch import inputs_on_cuda
from sglang.kernels.ops.speculative.dspark.simulated_bonus import (
    simulated_bonus_select_partials,
)


class CommitStage(msgspec.Struct, frozen=True):
    """stage_commit's operands: the verify rows' write locations and positions, and
    the deferred draft KV store's buffers they go to (with per-request state slots
    when it keeps them)."""

    cache_loc: torch.Tensor
    positions: torch.Tensor
    cache_loc_dst: torch.Tensor
    positions_dst: torch.Tensor
    commit_lens_dst: torch.Tensor
    state_slot_dst: Optional[torch.Tensor]


class CommitTail(msgspec.Struct, frozen=True):
    """The commits run_static issues after the accept, run in the accept's launch:
    the engram history (``history``) and the deferred draft KV staging (``stage``)
    of the requests in ``req_slots``."""

    req_slots: torch.Tensor
    history: Optional[torch.Tensor]
    stage: Optional[CommitStage]


@triton.jit
def _static_accept_tail_kernel(
    candidates_ptr,
    target_ptr,
    seq_lens_ptr,
    verify_lens_ptr,
    gate_ptr,
    sim_ptr,
    part_val_ptr,
    part_idx_ptr,
    correct_out_ptr,
    bonus_out_ptr,
    cap_trim_out_ptr,
    commit_out_ptr,
    new_seq_out_ptr,
    out_tokens_ptr,
    gated_out_ptr,
    rows_ptr,
    relay_seq_ptr,
    relay_tok_ptr,
    history_ptr,
    loc_src_ptr,
    loc_dst_ptr,
    pos_src_ptr,
    pos_dst_ptr,
    slot_dst_ptr,
    lens_dst_ptr,
    bs,
    max_bs,
    cand_stride,
    out_tokens_stride,
    vocab,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
    NSPLIT: tl.constexpr,
    HAS_SIM: tl.constexpr,
    HAS_PARTS: tl.constexpr,
    NEED_TARGET: tl.constexpr,
    HAS_GATED: tl.constexpr,
    HAS_RELAY: tl.constexpr,
    HISTORY_STRIDE: tl.constexpr,
    HISTORY_WIDTH: tl.constexpr,
    HISTORY_BLOCK: tl.constexpr,
    HAS_HISTORY: tl.constexpr,
    HAS_STAGE: tl.constexpr,
    HAS_SLOT: tl.constexpr,
    LENS_BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_T)
    in_t = offs < T
    gamma = T - 1
    in_draft = offs < gamma
    drafts = tl.load(
        candidates_ptr + row * cand_stride + 1 + offs, mask=in_draft, other=-1
    )
    if NEED_TARGET:
        targets = tl.load(target_ptr + row * T + offs, mask=in_draft, other=-2)
        eq = (drafts == targets) & in_draft
        num_correct = tl.min(tl.where(eq, BLOCK_T, offs), 0)
    if HAS_SIM:
        correct = tl.load(sim_ptr).to(tl.int32)
    else:
        correct = num_correct.to(tl.int32)
    if HAS_PARTS:
        # _simulated_bonus_row_select_kernel's reduction.
        s = tl.arange(0, NSPLIT)
        val = tl.load(part_val_ptr + row * NSPLIT + s)
        idx = tl.load(part_idx_ptr + row * NSPLIT + s)
        top = tl.max(val, axis=0)
        bonus = tl.min(tl.where(val == top, idx, vocab), axis=0).to(tl.int64)
    else:
        bonus = tl.load(target_ptr + row * T + correct).to(tl.int64)
    commit = correct + 1
    new_seq = tl.load(seq_lens_ptr + row) + commit
    tl.store(correct_out_ptr + row, correct)
    tl.store(bonus_out_ptr + row, bonus)
    tl.store(cap_trim_out_ptr + row, 0)
    tl.store(commit_out_ptr + row, commit)
    tl.store(new_seq_out_ptr + row, new_seq)
    token = tl.where(
        offs == correct, bonus, tl.where(offs < correct, drafts.to(tl.int64), 0)
    )
    tl.store(out_tokens_ptr + row * out_tokens_stride + offs, token, mask=in_t)
    if HAS_GATED:
        verify_len = tl.load(verify_lens_ptr + row)
        gate = tl.load(gate_ptr)
        gated = tl.minimum(commit, verify_len.to(tl.int32)) * gate
        tl.store(gated_out_ptr + row, gated)
    if HAS_RELAY:
        slot = tl.load(rows_ptr + row).to(tl.int64)
        tl.store(relay_seq_ptr + slot, new_seq)
        tl.store(relay_tok_ptr + slot, bonus)
    if HAS_HISTORY:
        # engram_commit_history: padded rows (verify_len 0) and gate 0 skip.
        live = (tl.load(verify_lens_ptr + row) != 0) & (tl.load(gate_ptr) != 0)
        req = tl.load(rows_ptr + row).to(tl.int64)
        col = tl.arange(0, HISTORY_BLOCK)
        source_col = commit + col
        valid = (col < HISTORY_WIDTH) & live
        old = tl.load(
            history_ptr + req * HISTORY_STRIDE + source_col,
            mask=valid & (source_col < HISTORY_WIDTH),
            other=0,
        )
        new = tl.load(
            candidates_ptr + row * cand_stride + source_col - HISTORY_WIDTH,
            mask=valid & (source_col >= HISTORY_WIDTH),
            other=0,
        )
        values = tl.where(source_col < HISTORY_WIDTH, old, new)
        tl.debug_barrier()
        tl.store(history_ptr + req * HISTORY_STRIDE + col, values, mask=valid)
    if HAS_STAGE:
        # stage_commit: this request's rows, and zeros past the staged requests.
        r = row * T + offs
        tl.store(loc_dst_ptr + r, tl.load(loc_src_ptr + r, mask=in_t), mask=in_t)
        tl.store(pos_dst_ptr + r, tl.load(pos_src_ptr + r, mask=in_t), mask=in_t)
        if HAS_SLOT:
            req_slot = tl.load(rows_ptr + row).to(tl.int64)
            tl.store(
                slot_dst_ptr + r, tl.zeros([BLOCK_T], tl.int64) + req_slot, mask=in_t
            )
        tl.store(lens_dst_ptr + row, gated)
        if row == 0:
            q = tl.arange(0, LENS_BLOCK)
            pad = (q >= bs) & (q < max_bs)
            tl.store(lens_dst_ptr + q, tl.zeros([LENS_BLOCK], tl.int32), mask=pad)


def static_accept_tail(
    *,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
    seq_lens: torch.Tensor,
    verify_lens: torch.Tensor,
    gate: torch.Tensor,
    correct_len_out: torch.Tensor,
    bonus_out: torch.Tensor,
    cap_trim_lens_out: torch.Tensor,
    commit_lens_out: torch.Tensor,
    new_seq_lens_out: torch.Tensor,
    out_tokens_out: torch.Tensor,
    simulated_correct_len: Optional[torch.Tensor] = None,
    bonus_parts: Optional[tuple[torch.Tensor, torch.Tensor, int]] = None,
    gated_commit_lens_out: Optional[torch.Tensor] = None,
    relay: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    commit_tail: Optional[CommitTail] = None,
) -> None:
    """``candidates`` [bs, T] (anchor + drafts), ``target_predict`` [bs * T] argmax ids,
    ``bonus_parts`` (part_val [bs, NSPLIT], part_idx, vocab); outputs go to rows [:bs].
    ``relay`` (req_pool_indices, new_seq_lens table, output_tokens table) scatters the
    row results by slot; ``commit_tail`` also runs the engram history commit and the
    deferred-commit staging (it needs ``gated_commit_lens_out``)."""
    fn = (
        _static_accept_tail_triton
        if inputs_on_cuda(candidates)
        else static_accept_tail_torch
    )
    fn(
        candidates=candidates,
        target_predict=target_predict,
        seq_lens=seq_lens,
        verify_lens=verify_lens,
        gate=gate,
        correct_len_out=correct_len_out,
        bonus_out=bonus_out,
        cap_trim_lens_out=cap_trim_lens_out,
        commit_lens_out=commit_lens_out,
        new_seq_lens_out=new_seq_lens_out,
        out_tokens_out=out_tokens_out,
        simulated_correct_len=simulated_correct_len,
        bonus_parts=bonus_parts,
        gated_commit_lens_out=gated_commit_lens_out,
        relay=relay,
        commit_tail=commit_tail,
    )


def _static_accept_tail_triton(**kw) -> None:
    candidates = kw["candidates"]
    bs, T = candidates.shape
    assert candidates.stride(1) == 1 and kw["out_tokens_out"].stride(1) == 1
    dummy = kw["seq_lens"]
    sim, parts, gated = (
        kw["simulated_correct_len"],
        kw["bonus_parts"],
        kw["gated_commit_lens_out"],
    )
    relay, commit = kw["relay"], kw["commit_tail"]
    req_slots = relay[0] if relay is not None else None
    history = commit.history if commit is not None else None
    stage = commit.stage if commit is not None else None
    if commit is not None:
        assert req_slots is None or req_slots.data_ptr() == commit.req_slots.data_ptr()
        req_slots = commit.req_slots
    if history is not None:
        assert history.stride(1) == 1 and history.shape[1] > 0
    if stage is not None:
        assert gated is not None, "the staging stores the gated commit lengths"
        assert stage.cache_loc.is_contiguous() and stage.positions.is_contiguous()
    max_bs = stage.commit_lens_dst.shape[0] if stage is not None else bs
    _static_accept_tail_kernel[(bs,)](
        candidates,
        kw["target_predict"],
        kw["seq_lens"],
        kw["verify_lens"],
        kw["gate"],
        sim if sim is not None else dummy,
        parts[0] if parts is not None else dummy,
        parts[1] if parts is not None else dummy,
        kw["correct_len_out"],
        kw["bonus_out"],
        kw["cap_trim_lens_out"],
        kw["commit_lens_out"],
        kw["new_seq_lens_out"],
        kw["out_tokens_out"],
        gated if gated is not None else dummy,
        req_slots if req_slots is not None else dummy,
        relay[1] if relay is not None else dummy,
        relay[2] if relay is not None else dummy,
        history if history is not None else dummy,
        stage.cache_loc if stage is not None else dummy,
        stage.cache_loc_dst if stage is not None else dummy,
        stage.positions if stage is not None else dummy,
        stage.positions_dst if stage is not None else dummy,
        (
            stage.state_slot_dst
            if stage is not None and stage.state_slot_dst is not None
            else dummy
        ),
        stage.commit_lens_dst if stage is not None else dummy,
        bs,
        max_bs,
        candidates.stride(0),
        kw["out_tokens_out"].stride(0),
        parts[2] if parts is not None else 0,
        T=T,
        BLOCK_T=triton.next_power_of_2(T),
        NSPLIT=parts[0].shape[1] if parts is not None else 1,
        HAS_SIM=sim is not None,
        HAS_PARTS=parts is not None,
        # The argmax ids only decide a length or bonus nothing else supplies.
        NEED_TARGET=sim is None or parts is None,
        HAS_GATED=gated is not None,
        HAS_RELAY=relay is not None,
        HISTORY_STRIDE=history.stride(0) if history is not None else 0,
        HISTORY_WIDTH=history.shape[1] if history is not None else 0,
        HISTORY_BLOCK=(
            triton.next_power_of_2(history.shape[1]) if history is not None else 1
        ),
        HAS_HISTORY=history is not None,
        HAS_STAGE=stage is not None,
        HAS_SLOT=stage is not None and stage.state_slot_dst is not None,
        LENS_BLOCK=triton.next_power_of_2(max_bs),
        num_warps=4,
    )


def static_accept_tail_torch(
    *,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
    seq_lens: torch.Tensor,
    verify_lens: torch.Tensor,
    gate: torch.Tensor,
    correct_len_out: torch.Tensor,
    bonus_out: torch.Tensor,
    cap_trim_lens_out: torch.Tensor,
    commit_lens_out: torch.Tensor,
    new_seq_lens_out: torch.Tensor,
    out_tokens_out: torch.Tensor,
    simulated_correct_len: Optional[torch.Tensor] = None,
    bonus_parts: Optional[tuple[torch.Tensor, torch.Tensor, int]] = None,
    gated_commit_lens_out: Optional[torch.Tensor] = None,
    relay: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    commit_tail: Optional[CommitTail] = None,
) -> None:
    """Reference: the same outputs from per-row torch ops (CPU or GPU)."""
    bs, T = candidates.shape
    gamma = T - 1
    target = target_predict.view(bs, T)
    matches = (candidates[:, 1:] == target[:, :gamma]).to(torch.int32)
    num_correct = matches.cumprod(dim=1).sum(dim=1).to(torch.int32)
    rows = torch.arange(bs, device=candidates.device)
    if simulated_correct_len is not None:
        correct = simulated_correct_len.repeat(bs).to(torch.int32)
    else:
        correct = num_correct
    if bonus_parts is not None:
        part_val, part_idx, vocab = bonus_parts
        bonus = simulated_bonus_select_partials(
            part_val=part_val[:bs], part_idx=part_idx[:bs], vocab=vocab
        )
    else:
        bonus = target[rows, correct.long()].to(torch.int64)
    commit = correct + 1
    new_seq = seq_lens[:bs] + commit.to(seq_lens.dtype)
    correct_len_out[:bs].copy_(correct)
    bonus_out[:bs].copy_(bonus)
    cap_trim_lens_out[:bs].zero_()
    commit_lens_out[:bs].copy_(commit)
    new_seq_lens_out[:bs].copy_(new_seq)
    tokens = torch.zeros((bs, T), dtype=torch.int64, device=candidates.device)
    keep = torch.arange(gamma, device=candidates.device)[None, :] < correct[:, None]
    tokens[:, :gamma] = torch.where(keep, candidates[:, 1:].to(torch.int64), 0)
    tokens.scatter_(1, correct.long()[:, None], bonus[:, None])
    out_tokens_out[:bs].copy_(tokens)
    if gated_commit_lens_out is not None:
        gated = torch.minimum(commit, verify_lens[:bs].to(torch.int32)) * gate
        gated_commit_lens_out[:bs].copy_(gated)
    if relay is not None:
        slots = relay[0][:bs].to(torch.int64)
        relay[1][slots] = new_seq.to(relay[1].dtype)
        relay[2][slots] = bonus.to(relay[2].dtype)
    if commit_tail is not None:
        assert commit_tail.stage is None or gated_commit_lens_out is not None
        _commit_tail_torch(
            commit_tail,
            candidates=candidates,
            commit=commit,
            verify_lens=verify_lens,
            gate=gate,
            gated=(
                gated_commit_lens_out[:bs]
                if gated_commit_lens_out is not None
                else None
            ),
            bs=bs,
            T=T,
        )


def _commit_tail_torch(tail, *, candidates, commit, verify_lens, gate, gated, bs, T):
    """engram_commit_history and stage_commit, row by row."""
    live = (verify_lens[:bs] != 0) & (gate.view(-1)[0] != 0)
    slots = tail.req_slots[:bs].to(torch.int64)
    history = tail.history
    if history is not None:
        width = history.shape[1]
        for b in range(bs):
            if not bool(live[b]):
                continue
            c = int(commit[b])
            row = torch.cat([history[slots[b]], candidates[b].to(history.dtype)])
            history[slots[b]] = row[c : c + width]
    stage = tail.stage
    if stage is not None:
        n = bs * T
        stage.cache_loc_dst[:n].copy_(stage.cache_loc[:n])
        stage.positions_dst[:n].copy_(stage.positions[:n])
        if stage.state_slot_dst is not None:
            stage.state_slot_dst[:n].copy_(slots.repeat_interleave(T))
        stage.commit_lens_dst[:bs].copy_(gated)
        stage.commit_lens_dst[bs:].zero_()
