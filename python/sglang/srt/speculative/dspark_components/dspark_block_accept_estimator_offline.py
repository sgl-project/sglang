from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

import msgspec
import torch

from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

logger = logging.getLogger(__name__)

_GATHER_ROW_CHUNK = 128
_STATE_SWEEP_INTERVAL = 1024
_STATE_EXPIRE_STEPS = 4096
_FLUSH_EVERY_STEPS = 16


class _PendingBlock(msgspec.Struct):
    forward_ct: int
    anchor_pos: int
    window: int
    trimmed_tokens: List[int]
    next_offset: int


class _RequestState(msgspec.Struct):
    expected_seq_len: int = -1
    last_seen_ct: int = 0
    pending: List[_PendingBlock] = []


class _GatherPlan(msgspec.Struct):
    row_indices: List[int] = []
    token_indices: List[int] = []


class _StepInputs(msgspec.Struct):
    forward_ct: int
    rids: List[str]
    bs: int
    drafts: List[List[int]]
    correct_lens: List[int]
    cap_trims: List[int]
    bonus_tokens: List[int]
    greedy_rows: List[bool]
    truncated_rows: List[bool]
    seq_lens: List[int]
    verify_lens: List[int]


class _RowPlan(msgspec.Struct):
    record: dict
    q_lp_slot: int
    num_trimmed: int
    pending: List[Tuple[_PendingBlock, int, int]]


class OfflineBlockAcceptEstimateRecorder:
    def __init__(self, *, path: str, gamma: int) -> None:
        self._gamma = gamma
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("w")
        self._states: dict[str, _RequestState] = {}
        self._steps_since_flush = 0
        self._observed_step_ct = 0
        self._discontinuity_drop_ct = 0
        self._skipped_step_ct = 0
        self._warned_skip_reasons: set[str] = set()

        logger.info(
            "DSPARK offline block accept estimate recorder enabled: path=%s gamma=%d",
            path,
            gamma,
        )

    def observe_verify_step(
        self,
        *,
        forward_ct: int,
        rids: List[str],
        draft_tokens: torch.Tensor,
        corrected_logits: Optional[torch.Tensor],
        draft_temperatures: torch.Tensor,
        greedy_mask: torch.Tensor,
        target_logits: torch.Tensor,
        target_temperatures: torch.Tensor,
        truncated_sampling_mask: Optional[torch.Tensor],
        logits_adjustments_are_noop: bool,
        correct_len: torch.Tensor,
        cap_trim_lens: torch.Tensor,
        bonus: torch.Tensor,
        prefix_lens: torch.Tensor,
        layout: Optional[RaggedVerifyLayout],
    ) -> None:
        skip_reason = self._skip_reason(
            logits_adjustments_are_noop=logits_adjustments_are_noop,
            corrected_logits=corrected_logits,
        )
        if skip_reason is not None:
            self._skip_step(reason=skip_reason)
            return

        gamma = self._gamma
        rows_per_request = gamma + 1
        bs = len(rids)
        assert draft_tokens.shape == (bs, gamma)
        assert corrected_logits.shape[0] == bs and corrected_logits.shape[1] == gamma
        assert target_logits.shape[0] == bs * rows_per_request

        inputs = self._read_inputs(
            forward_ct=forward_ct,
            rids=rids,
            draft_tokens=draft_tokens,
            correct_len=correct_len,
            cap_trim_lens=cap_trim_lens,
            bonus=bonus,
            greedy_mask=greedy_mask,
            truncated_sampling_mask=truncated_sampling_mask,
            prefix_lens=prefix_lens,
            layout=layout,
        )

        target_plan = _GatherPlan()
        draft_plan = _GatherPlan()
        row_plans = self._plan_rows(
            inputs, target_plan=target_plan, draft_plan=draft_plan
        )

        draft_temps_full = (
            draft_temperatures.reshape(bs).to(torch.float32).repeat_interleave(gamma)
        )
        target_temps_full = (
            target_temperatures.reshape(bs)
            .to(torch.float32)
            .repeat_interleave(rows_per_request)
        )
        draft_logprobs = self._gather(
            logits=corrected_logits.reshape(bs * gamma, -1),
            temps=draft_temps_full,
            plan=draft_plan,
        )
        target_logprobs = self._gather(
            logits=target_logits, temps=target_temps_full, plan=target_plan
        )

        self._write_rows(
            row_plans=row_plans,
            draft_logprobs=draft_logprobs,
            target_logprobs=target_logprobs,
        )
        self._finish_step(forward_ct=forward_ct)

    def flush(self) -> None:
        self._file.flush()
        self._steps_since_flush = 0

    def estimate_log_suffix(self) -> Optional[str]:
        return None

    def note_request_finished(self, *, rid: str, natural_stop: bool) -> None:
        state = self._states.pop(rid, None)
        if state is None or not state.pending:
            return
        if natural_stop:
            marker = {
                "rid": rid,
                "eos_end": [block.forward_ct for block in state.pending],
            }
            self._file.write(json.dumps(marker) + "\n")

    def _read_inputs(
        self,
        *,
        forward_ct: int,
        rids: List[str],
        draft_tokens: torch.Tensor,
        correct_len: torch.Tensor,
        cap_trim_lens: torch.Tensor,
        bonus: torch.Tensor,
        greedy_mask: torch.Tensor,
        truncated_sampling_mask: Optional[torch.Tensor],
        prefix_lens: torch.Tensor,
        layout: Optional[RaggedVerifyLayout],
    ) -> _StepInputs:
        bs = len(rids)
        rows_per_request = self._gamma + 1
        if truncated_sampling_mask is not None:
            truncated_rows = truncated_sampling_mask.tolist()
        else:
            truncated_rows = [False] * bs
        if layout is not None:
            verify_lens = layout.verify_lens.tolist()
        else:
            verify_lens = [rows_per_request] * bs
        return _StepInputs(
            forward_ct=forward_ct,
            rids=list(rids),
            bs=bs,
            drafts=draft_tokens.tolist(),
            correct_lens=correct_len.tolist(),
            cap_trims=cap_trim_lens.tolist(),
            bonus_tokens=bonus.tolist(),
            greedy_rows=greedy_mask.tolist(),
            truncated_rows=truncated_rows,
            seq_lens=prefix_lens.tolist(),
            verify_lens=verify_lens,
        )

    def _plan_rows(
        self, inputs: _StepInputs, *, target_plan: _GatherPlan, draft_plan: _GatherPlan
    ) -> List[_RowPlan]:
        gamma = self._gamma
        row_plans: List[_RowPlan] = []
        for b in range(inputs.bs):
            rid = inputs.rids[b]
            state = self._states.setdefault(rid, _RequestState())
            state.last_seen_ct = inputs.forward_ct

            cl = int(inputs.correct_lens[b])
            window = int(inputs.verify_lens[b]) - 1
            seq_len = int(inputs.seq_lens[b])
            assert 0 <= cl <= window <= gamma

            self._drop_pending_on_discontinuity(state, seq_len=seq_len)
            state.expected_seq_len = seq_len + cl + 1

            if inputs.greedy_rows[b] or inputs.truncated_rows[b]:
                if inputs.truncated_rows[b] and not inputs.greedy_rows[b]:
                    self._warn_once(
                        reason="requests with top-k/top-p/min-p sampling are "
                        "excluded per-row; the estimator only supports "
                        "pure-temperature sampling (processed target distribution "
                        "would differ from plain softmax(logits/T))"
                    )
                state.pending = []
                continue

            realized = inputs.drafts[b][:cl] + [int(inputs.bonus_tokens[b])]
            record: dict[str, Any] = {
                "rid": rid,
                "fct": inputs.forward_ct,
                "w": window,
                "cl": cl,
                "ct": int(inputs.cap_trims[b]),
            }
            q_lp_slot = -1
            num_trimmed = 0
            if cl == window and window < gamma:
                q_lp_slot, num_trimmed = self._open_block(
                    state,
                    record,
                    inputs=inputs,
                    b=b,
                    window=window,
                    seq_len=seq_len,
                    draft_plan=draft_plan,
                )
            pending = self._plan_pending(
                state,
                inputs=inputs,
                b=b,
                cl=cl,
                seq_len=seq_len,
                realized=realized,
                target_plan=target_plan,
            )
            row_plans.append(
                _RowPlan(
                    record=record,
                    q_lp_slot=q_lp_slot,
                    num_trimmed=num_trimmed,
                    pending=pending,
                )
            )
        return row_plans

    def _open_block(
        self,
        state: _RequestState,
        record: dict[str, Any],
        *,
        inputs: _StepInputs,
        b: int,
        window: int,
        seq_len: int,
        draft_plan: _GatherPlan,
    ) -> Tuple[int, int]:
        gamma = self._gamma
        trimmed_tokens = inputs.drafts[b][window:gamma]
        state.pending.append(
            _PendingBlock(
                forward_ct=inputs.forward_ct,
                anchor_pos=seq_len - 1,
                window=window,
                trimmed_tokens=trimmed_tokens,
                next_offset=window + 1,
            )
        )
        q_lp_slot = len(draft_plan.row_indices)
        for offset in range(window + 1, gamma + 1):
            draft_plan.row_indices.append(b * gamma + offset - 1)
            draft_plan.token_indices.append(inputs.drafts[b][offset - 1])
        record["trimmed_tokens"] = trimmed_tokens
        return q_lp_slot, gamma - window

    def _plan_pending(
        self,
        state: _RequestState,
        *,
        inputs: _StepInputs,
        b: int,
        cl: int,
        seq_len: int,
        realized: List[int],
        target_plan: _GatherPlan,
    ) -> List[Tuple[_PendingBlock, int, int]]:
        gamma = self._gamma
        rows_per_request = gamma + 1
        pending_gathers: List[Tuple[_PendingBlock, int, int]] = []
        kept_pending: List[_PendingBlock] = []
        for block in state.pending:
            diverged = False
            while block.next_offset <= gamma:
                row = block.anchor_pos + block.next_offset - seq_len
                assert row >= 0
                if row > cl:
                    break
                token = block.trimmed_tokens[block.next_offset - block.window - 1]
                target_plan.row_indices.append(b * rows_per_request + row)
                target_plan.token_indices.append(token)
                pending_gathers.append((block, block.next_offset, realized[row]))
                block.next_offset += 1
                if realized[row] != token:
                    diverged = True
                    break
            if not diverged and block.next_offset <= gamma:
                kept_pending.append(block)
        state.pending = kept_pending
        return pending_gathers

    def _write_rows(
        self,
        *,
        row_plans: List[_RowPlan],
        draft_logprobs: List[float],
        target_logprobs: List[float],
    ) -> None:
        target_cursor = 0
        for plan in row_plans:
            record = plan.record
            if plan.q_lp_slot >= 0:
                record["q_lp"] = draft_logprobs[
                    plan.q_lp_slot : plan.q_lp_slot + plan.num_trimmed
                ]
            if plan.pending:
                record["pg"] = [
                    [
                        block.forward_ct,
                        offset,
                        target_logprobs[target_cursor + i],
                        block.trimmed_tokens[offset - block.window - 1],
                        realized_token,
                    ]
                    for i, (block, offset, realized_token) in enumerate(plan.pending)
                ]
                target_cursor += len(plan.pending)
            self._file.write(json.dumps(record) + "\n")

    def _gather(
        self, *, logits: torch.Tensor, temps: torch.Tensor, plan: _GatherPlan
    ) -> List[float]:
        if not plan.row_indices:
            return []
        device = logits.device
        row_indices = torch.tensor(plan.row_indices, dtype=torch.long, device=device)
        token_indices = torch.tensor(
            plan.token_indices, dtype=torch.long, device=device
        )
        per_row_temps = temps.to(device=device)[row_indices].clamp_min(1e-5)
        results: List[torch.Tensor] = []
        for start in range(0, row_indices.shape[0], _GATHER_ROW_CHUNK):
            end = start + _GATHER_ROW_CHUNK
            rows = logits[row_indices[start:end]].to(torch.float32)
            rows = rows / per_row_temps[start:end, None]
            log_norm = torch.logsumexp(rows, dim=-1)
            token_logits = rows.gather(
                dim=1, index=token_indices[start:end, None]
            ).squeeze(1)
            results.append(token_logits - log_norm)
        return torch.cat(results).cpu().tolist()

    def _drop_pending_on_discontinuity(
        self, state: _RequestState, *, seq_len: int
    ) -> None:
        if state.expected_seq_len < 0 or seq_len == state.expected_seq_len:
            return
        if state.pending:
            self._discontinuity_drop_ct += len(state.pending)
            state.pending = []

    def _finish_step(self, *, forward_ct: int) -> None:
        self._observed_step_ct += 1
        self._steps_since_flush += 1
        if self._steps_since_flush >= _FLUSH_EVERY_STEPS:
            self._file.flush()
            self._steps_since_flush = 0
        if self._observed_step_ct % _STATE_SWEEP_INTERVAL == 0:
            self._sweep_states(forward_ct=forward_ct)

    def _sweep_states(self, *, forward_ct: int) -> None:
        expired = [
            rid
            for rid, state in self._states.items()
            if forward_ct - state.last_seen_ct > _STATE_EXPIRE_STEPS
        ]
        for rid in expired:
            del self._states[rid]

    def _skip_reason(
        self,
        *,
        logits_adjustments_are_noop: bool,
        corrected_logits: Optional[torch.Tensor],
    ) -> Optional[str]:
        if not logits_adjustments_are_noop:
            return (
                "non-noop logits adjustments (penalizer/logit_bias/grammar) "
                "in batch; cross-step conditioning of the gathered target "
                "probabilities would be state-dependent"
            )
        if corrected_logits is None:
            return "corrected_logits unavailable (folded draft path)"
        return None

    def _skip_step(self, *, reason: str) -> None:
        self._skipped_step_ct += 1
        self._warn_once(
            reason=f"skipping step: {reason} (pending blocks of affected requests "
            "are dropped by the seq-len continuity check)"
        )

    def _warn_once(self, *, reason: str) -> None:
        if reason not in self._warned_skip_reasons:
            self._warned_skip_reasons.add(reason)
            logger.warning(
                "DSPARK block accept estimate recorder: %s (warned once)", reason
            )
