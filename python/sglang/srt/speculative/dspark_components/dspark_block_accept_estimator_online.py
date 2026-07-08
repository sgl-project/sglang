from __future__ import annotations

import json
import logging
import math
from collections import deque
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import msgspec
import torch

from sglang.srt.kv_canary.runner.future_tensor import DelayedDeviceHostHandler
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

logger = logging.getLogger(__name__)

_GATHER_ROW_CHUNK = 512
_STATE_SWEEP_INTERVAL = 1024
_STATE_EXPIRE_STEPS = 4096
_FLUSH_EVERY_STEPS = 16
_PENDING_BUCKET_MIN = 16
_DEFAULT_ONLINE_WINDOW_STEPS = 256


def _pending_bucket(count: int) -> int:
    if count == 0:
        return 0
    bucket = _PENDING_BUCKET_MIN
    while bucket < count:
        bucket *= 2
    return bucket


class _CeilingSnapshot(msgspec.Struct):
    window_lo: float
    window_hi: float
    window_blocks: int
    window_horizon: int
    cumulative_lo: float
    cumulative_hi: float
    cumulative_blocks: int


class _OnlineCeiling:
    def __init__(self, *, log_interval: int, window_steps: int) -> None:
        self._log_interval = log_interval
        self._window_steps = window_steps
        self._steps: deque[Tuple[int, float, float, int]] = deque()
        self._win_lo = 0.0
        self._win_hi = 0.0
        self._win_count = 0
        self._cum_lo = 0.0
        self._cum_hi = 0.0
        self._cum_count = 0
        self._max_forward_ct = 0

    def add(self, *, forward_ct: int, lo: float, hi: float) -> None:
        self._max_forward_ct = max(self._max_forward_ct, forward_ct)
        if self._steps and self._steps[-1][0] == forward_ct:
            fct, slo, shi, c = self._steps[-1]
            self._steps[-1] = (fct, slo + lo, shi + hi, c + 1)
        else:
            self._steps.append((forward_ct, lo, hi, 1))
        self._win_lo += lo
        self._win_hi += hi
        self._win_count += 1
        self._cum_lo += lo
        self._cum_hi += hi
        self._cum_count += 1
        self._evict(forward_ct=self._max_forward_ct)

    def _evict(self, *, forward_ct: int) -> None:
        cutoff = forward_ct - self._window_steps
        while self._steps and self._steps[0][0] <= cutoff:
            _, slo, shi, c = self._steps.popleft()
            self._win_lo -= slo
            self._win_hi -= shi
            self._win_count -= c

    def estimate(self) -> Optional[_CeilingSnapshot]:
        if self._cum_count == 0:
            return None
        return _CeilingSnapshot(
            window_lo=self._win_lo / self._win_count,
            window_hi=self._win_hi / self._win_count,
            window_blocks=self._win_count,
            window_horizon=min(self._window_steps, self._max_forward_ct),
            cumulative_lo=self._cum_lo / self._cum_count,
            cumulative_hi=self._cum_hi / self._cum_count,
            cumulative_blocks=self._cum_count,
        )

    def maybe_log(self, *, forward_ct: int) -> None:
        if self._log_interval <= 0 or forward_ct % self._log_interval != 0:
            return
        snap = self.estimate()
        if snap is None:
            return
        logger.info(
            "DSpark uncapped-acc-len estimate (forward_ct=%d): "
            "last %d passes ~%.3f [%.3f, %.3f] w=%.3f (%d blocks) | "
            "cumulative ~%.3f [%.3f, %.3f] w=%.3f (%d blocks)",
            forward_ct,
            snap.window_horizon,
            0.5 * (snap.window_lo + snap.window_hi),
            snap.window_lo,
            snap.window_hi,
            snap.window_hi - snap.window_lo,
            snap.window_blocks,
            0.5 * (snap.cumulative_lo + snap.cumulative_hi),
            snap.cumulative_lo,
            snap.cumulative_hi,
            snap.cumulative_hi - snap.cumulative_lo,
            snap.cumulative_blocks,
        )


class _PendingBlock(msgspec.Struct):
    forward_ct: int
    anchor_pos: int
    window: int
    trimmed_tokens: List[int]
    next_offset: int
    q_lps: List[float] = []
    est_prod: float = 1.0
    est_lo_extra: float = 0.0


class _RequestState(msgspec.Struct):
    expected_seq_len: int = -1
    last_seen_ct: int = 0
    pending: List[_PendingBlock] = []


class _PendingPlan(msgspec.Struct):
    rows: List[int]
    tokens: List[int]
    slot_lookup: dict[tuple[int, int, int], int]


class _SettleBatch(msgspec.Struct):
    forward_ct: int
    rids: List[str]
    row_meta: List[List[int]]
    drafts: List[List[int]]
    q_all: List[List[float]]
    target_diag: List[List[float]]
    pending_logprobs: List[float]
    slot_lookup: dict[tuple[int, int, int], int]

    @classmethod
    def from_bundle(cls, bundle: dict[str, Any]) -> _SettleBatch:
        return cls(
            forward_ct=bundle["forward_ct"],
            rids=bundle["rids"],
            row_meta=bundle["row_meta"].tolist(),
            drafts=bundle["draft_tokens"].tolist(),
            q_all=bundle["q_all"].tolist(),
            target_diag=bundle["target_diag_logprobs"].tolist(),
            pending_logprobs=bundle["pending_logprobs"].tolist(),
            slot_lookup=bundle["pending_slot_lookup"],
        )


class OnlineBlockAcceptEstimateRecorder:
    def __init__(
        self,
        *,
        path: str,
        gamma: int,
        device: Union[str, torch.device],
        online_log_interval: int = 0,
        online_window_steps: int = 0,
    ) -> None:
        self._gamma = gamma
        self._last_forward_ct = 0
        if path:
            self._path: Optional[Path] = Path(path)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self._path.open("w")
        else:
            self._path = None
            self._file = None
        self._device = torch.device(device)
        self._states: dict[str, _RequestState] = {}
        self._steps_since_flush = 0
        self._observed_step_ct = 0
        self._discontinuity_drop_ct = 0
        self._skipped_step_ct = 0
        self._warned_skip_reasons: set[str] = set()
        self._finish_intents: dict[str, bool] = {}

        self._online = _OnlineCeiling(
            log_interval=online_log_interval,
            window_steps=(
                online_window_steps
                if online_window_steps > 0
                else (
                    online_log_interval
                    if online_log_interval > 0
                    else _DEFAULT_ONLINE_WINDOW_STEPS
                )
            ),
        )

        self._retained_h2d: List[torch.Tensor] = []
        self._delayed: Optional[DelayedDeviceHostHandler] = None
        if self._device.type == "cuda":
            self._delayed = DelayedDeviceHostHandler(
                d2h_stream=torch.cuda.Stream(device=self._device)
            )

        logger.info(
            "DSPARK block accept estimate recorder enabled: path=%s gamma=%d "
            "async=%s online_log_interval=%d",
            path,
            gamma,
            self._delayed is not None,
            online_log_interval,
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
        if (
            self._delayed is not None
            and torch.cuda.is_available()
            and torch.cuda.is_current_stream_capturing()
        ):
            return

        skip_reason = self._skip_reason(
            logits_adjustments_are_noop=logits_adjustments_are_noop,
            corrected_logits=corrected_logits,
        )
        if skip_reason is not None:
            self._skip_step(reason=skip_reason)

        def compute_on_device() -> Optional[dict[str, Any]]:
            if skip_reason is not None:
                return None
            return self._build_device_bundle(
                forward_ct=forward_ct,
                rids=rids,
                draft_tokens=draft_tokens,
                corrected_logits=corrected_logits,
                draft_temperatures=draft_temperatures,
                greedy_mask=greedy_mask,
                target_logits=target_logits,
                target_temperatures=target_temperatures,
                truncated_sampling_mask=truncated_sampling_mask,
                correct_len=correct_len,
                cap_trim_lens=cap_trim_lens,
                bonus=bonus,
                prefix_lens=prefix_lens,
                layout=layout,
            )

        if self._delayed is not None:
            self._delayed.step(
                compute_on_device=compute_on_device,
                postprocess_on_host=self._settle_and_write,
            )
        else:
            bundle = compute_on_device()
            if bundle is not None:
                self._settle_and_write(bundle)

    def flush(self) -> None:
        if self._delayed is not None:
            self._delayed.step(
                compute_on_device=lambda: None,
                postprocess_on_host=self._settle_and_write,
            )
            self._apply_all_finish_intents()
        if self._file is not None:
            self._file.flush()
        self._steps_since_flush = 0

    def note_request_finished(self, *, rid: str, natural_stop: bool) -> None:
        if self._delayed is None:
            self._finalize_request(
                rid=rid, natural_stop=natural_stop, forward_ct=self._last_forward_ct
            )
        else:
            self._finish_intents[rid] = natural_stop

    def _apply_all_finish_intents(self) -> None:
        for rid in list(self._finish_intents):
            self._finalize_request(
                rid=rid,
                natural_stop=self._finish_intents.pop(rid),
                forward_ct=self._last_forward_ct,
            )

    def _finalize_request(
        self, *, rid: str, natural_stop: bool, forward_ct: int
    ) -> None:
        state = self._states.pop(rid, None)
        if state is None:
            return
        for block in state.pending:
            if natural_stop:
                self._finalize_eos_online(block, forward_ct=forward_ct)
            else:
                self._finalize_at_end_online(block, forward_ct=forward_ct)
        if natural_stop and state.pending:
            self._write_eos_marker(rid=rid, blocks=state.pending)

    def _finalize_eos_online(self, block: _PendingBlock, *, forward_ct: int) -> None:
        lo = block.window + 1.0 + block.est_lo_extra
        self._online.add(forward_ct=forward_ct, lo=lo, hi=lo)

    def _write_eos_marker(self, *, rid: str, blocks: List[_PendingBlock]) -> None:
        if self._file is None:
            return
        marker = {"rid": rid, "eos_end": [block.forward_ct for block in blocks]}
        self._file.write(json.dumps(marker) + "\n")

    def online_estimate(self) -> Optional[_CeilingSnapshot]:
        return self._online.estimate()

    def estimate_log_suffix(self) -> Optional[str]:
        snap = self.online_estimate()
        if snap is None:
            return None
        mid = 0.5 * (snap.cumulative_lo + snap.cumulative_hi)
        return (
            f"est uncap acc len: {mid:.2f} "
            f"[{snap.cumulative_lo:.2f}, {snap.cumulative_hi:.2f}]"
        )

    def drain_pending_online(self) -> None:
        for state in self._states.values():
            for block in state.pending:
                self._finalize_at_end_online(block, forward_ct=self._last_forward_ct)
            state.pending = []

    def _finalize_walk_online(
        self, block: _PendingBlock, *, diverged: bool, forward_ct: int
    ) -> None:
        base = block.window + 1.0
        lo = base + block.est_lo_extra
        if diverged:
            offset = block.next_offset - 1
            tail = (
                block.est_prod * (self._gamma - offset) if offset < self._gamma else 0.0
            )
        else:
            tail = 0.0
        self._online.add(forward_ct=forward_ct, lo=lo, hi=lo + tail)

    def _finalize_at_end_online(self, block: _PendingBlock, *, forward_ct: int) -> None:
        base = block.window + 1.0
        lo = base + block.est_lo_extra
        tail = block.est_prod * (self._gamma - block.next_offset + 1)
        self._online.add(forward_ct=forward_ct, lo=lo, hi=lo + tail)

    def _build_device_bundle(
        self,
        *,
        forward_ct: int,
        rids: List[str],
        draft_tokens: torch.Tensor,
        corrected_logits: torch.Tensor,
        draft_temperatures: torch.Tensor,
        greedy_mask: torch.Tensor,
        target_logits: torch.Tensor,
        target_temperatures: torch.Tensor,
        truncated_sampling_mask: Optional[torch.Tensor],
        correct_len: torch.Tensor,
        cap_trim_lens: torch.Tensor,
        bonus: torch.Tensor,
        prefix_lens: torch.Tensor,
        layout: Optional[RaggedVerifyLayout],
    ) -> dict[str, Any]:
        gamma = self._gamma
        rows_per_request = gamma + 1
        bs = len(rids)
        device = target_logits.device
        assert draft_tokens.shape == (bs, gamma)
        assert corrected_logits.shape[0] == bs and corrected_logits.shape[1] == gamma
        assert target_logits.shape[0] == bs * rows_per_request

        if truncated_sampling_mask is not None:
            truncated_mask = truncated_sampling_mask
        else:
            truncated_mask = torch.zeros(bs, dtype=torch.bool, device=device)
        if layout is not None:
            verify_lens = layout.verify_lens
        else:
            verify_lens = torch.full(
                (bs,), rows_per_request, dtype=torch.int32, device=device
            )

        draft_temps_full = (
            draft_temperatures.reshape(bs).to(torch.float32).repeat_interleave(gamma)
        )
        target_temps_full = (
            target_temperatures.reshape(bs)
            .to(torch.float32)
            .repeat_interleave(rows_per_request)
        )
        draft_flat = draft_tokens.reshape(-1)

        q_all = self._gather_logprobs(
            logits=corrected_logits.reshape(bs * gamma, -1),
            row_indices=torch.arange(bs * gamma, device=device),
            token_indices=draft_flat,
            temps=draft_temps_full,
        ).reshape(bs, gamma)
        target_diag = self._gather_logprobs(
            logits=target_logits,
            row_indices=self._diag_rows(bs=bs, rows_per_request=rows_per_request),
            token_indices=draft_flat,
            temps=target_temps_full,
        ).reshape(bs, gamma)

        self._retained_h2d = []
        plan = self._plan_pending(bs=bs, rows_per_request=rows_per_request, rids=rids)
        pending_logprobs = self._gather_pending(
            plan=plan,
            target_logits=target_logits,
            target_temps_full=target_temps_full,
            device=device,
        )

        return {
            "forward_ct": int(forward_ct),
            "rids": list(rids),
            "row_meta": self._pack_row_meta(
                correct_len=correct_len,
                cap_trim_lens=cap_trim_lens,
                bonus=bonus,
                prefix_lens=prefix_lens,
                greedy_mask=greedy_mask,
                truncated_mask=truncated_mask,
                verify_lens=verify_lens,
            ),
            "draft_tokens": draft_tokens,
            "q_all": q_all,
            "target_diag_logprobs": target_diag,
            "pending_logprobs": pending_logprobs,
            "pending_slot_lookup": plan.slot_lookup,
        }

    def _diag_rows(self, *, bs: int, rows_per_request: int) -> torch.Tensor:
        device = self._device
        return (
            (torch.arange(bs, device=device) * rows_per_request)[:, None]
            + torch.arange(self._gamma, device=device)[None, :]
        ).reshape(-1)

    def _plan_pending(
        self, *, bs: int, rows_per_request: int, rids: List[str]
    ) -> _PendingPlan:
        gamma = self._gamma
        rows: List[int] = []
        tokens: List[int] = []
        slot_lookup: dict[tuple[int, int, int], int] = {}
        for b in range(bs):
            state = self._states.get(rids[b])
            if state is None or not state.pending or state.expected_seq_len < 0:
                continue
            expected_seq_len = state.expected_seq_len
            for block_idx, block in enumerate(state.pending):
                offset = block.next_offset
                while offset <= gamma:
                    row = block.anchor_pos + offset - expected_seq_len
                    if row < 0 or row >= rows_per_request:
                        break
                    slot_lookup[(b, block_idx, offset)] = len(rows)
                    rows.append(b * rows_per_request + row)
                    tokens.append(block.trimmed_tokens[offset - block.window - 1])
                    offset += 1
        return _PendingPlan(rows=rows, tokens=tokens, slot_lookup=slot_lookup)

    def _gather_pending(
        self,
        *,
        plan: _PendingPlan,
        target_logits: torch.Tensor,
        target_temps_full: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        bucket = _pending_bucket(len(plan.rows))
        rows = plan.rows + [0] * (bucket - len(plan.rows))
        tokens = plan.tokens + [0] * (bucket - len(plan.tokens))
        return self._gather_logprobs(
            logits=target_logits,
            row_indices=self._host_to_device_async(rows, device=device),
            token_indices=self._host_to_device_async(tokens, device=device),
            temps=target_temps_full,
        )

    def _pack_row_meta(
        self,
        *,
        correct_len: torch.Tensor,
        cap_trim_lens: torch.Tensor,
        bonus: torch.Tensor,
        prefix_lens: torch.Tensor,
        greedy_mask: torch.Tensor,
        truncated_mask: torch.Tensor,
        verify_lens: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            [
                correct_len.to(torch.int64),
                cap_trim_lens.to(torch.int64),
                bonus.to(torch.int64),
                prefix_lens.to(torch.int64),
                greedy_mask.to(torch.int64),
                truncated_mask.to(torch.int64),
                verify_lens.to(torch.int64),
            ],
            dim=1,
        )

    def _settle_and_write(self, bundle: dict[str, Any]) -> None:
        batch = _SettleBatch.from_bundle(bundle)
        self._last_forward_ct = batch.forward_ct
        for b in range(len(batch.rids)):
            self._settle_row(b=b, batch=batch)
        self._finish_step(forward_ct=batch.forward_ct)
        self._apply_all_finish_intents()

    def _settle_row(self, *, b: int, batch: _SettleBatch) -> None:
        forward_ct = batch.forward_ct
        rid = batch.rids[b]
        state = self._states.setdefault(rid, _RequestState())
        state.last_seen_ct = forward_ct

        cl, cap_trim, bonus_token, seq_len, is_greedy, is_truncated, verify_len = (
            batch.row_meta[b]
        )
        window = verify_len - 1
        assert 0 <= cl <= window <= self._gamma

        self._drop_pending_on_discontinuity(
            state, seq_len=seq_len, forward_ct=forward_ct
        )
        state.expected_seq_len = seq_len + cl + 1

        if is_greedy or is_truncated:
            if is_truncated and not is_greedy:
                self._warn_once(
                    reason="requests with top-k/top-p/min-p sampling are "
                    "excluded per-row; the estimator only supports "
                    "pure-temperature sampling (processed target distribution "
                    "would differ from plain softmax(logits/T))"
                )
            state.pending = []
            return

        record: dict[str, Any] = {
            "rid": rid,
            "fct": forward_ct,
            "w": window,
            "cl": cl,
            "ct": cap_trim,
        }
        num_old_pending = len(state.pending)
        if cl == window and window < self._gamma:
            self._open_block(
                state,
                record,
                drafts_row=batch.drafts[b],
                q_all_row=batch.q_all[b],
                window=window,
                seq_len=seq_len,
                forward_ct=forward_ct,
            )
        else:
            self._online.add(forward_ct=forward_ct, lo=cl + 1.0, hi=cl + 1.0)

        pending_gathers = self._settle_pending(
            b=b,
            batch=batch,
            state=state,
            realized=batch.drafts[b][:cl] + [bonus_token],
            cl=cl,
            seq_len=seq_len,
            num_old_pending=num_old_pending,
        )
        if pending_gathers:
            record["pg"] = pending_gathers
        if self._file is not None:
            self._file.write(json.dumps(record) + "\n")

    def _open_block(
        self,
        state: _RequestState,
        record: dict[str, Any],
        *,
        drafts_row: List[int],
        q_all_row: List[float],
        window: int,
        seq_len: int,
        forward_ct: int,
    ) -> None:
        trimmed_tokens = drafts_row[window : self._gamma]
        q_lps = q_all_row[window : self._gamma]
        state.pending.append(
            _PendingBlock(
                forward_ct=forward_ct,
                anchor_pos=seq_len - 1,
                window=window,
                trimmed_tokens=trimmed_tokens,
                next_offset=window + 1,
                q_lps=q_lps,
            )
        )
        record["trimmed_tokens"] = trimmed_tokens
        record["q_lp"] = q_lps

    def _settle_pending(
        self,
        *,
        b: int,
        batch: _SettleBatch,
        state: _RequestState,
        realized: List[int],
        cl: int,
        seq_len: int,
        num_old_pending: int,
    ) -> List[list]:
        gamma = self._gamma
        pending_gathers: List[list] = []
        kept_pending: List[_PendingBlock] = []
        for block_idx, block in enumerate(state.pending):
            diverged = False
            while block.next_offset <= gamma:
                row = block.anchor_pos + block.next_offset - seq_len
                assert row >= 0
                if row > cl:
                    break
                token = block.trimmed_tokens[block.next_offset - block.window - 1]
                if block_idx < num_old_pending:
                    p_lp = batch.pending_logprobs[
                        batch.slot_lookup[(b, block_idx, block.next_offset)]
                    ]
                else:
                    p_lp = batch.target_diag[b][row]
                pending_gathers.append(
                    [block.forward_ct, block.next_offset, p_lp, token, realized[row]]
                )
                self._accumulate_online(block, p_lp=p_lp)
                block.next_offset += 1
                if realized[row] != token:
                    diverged = True
                    break
            if not diverged and block.next_offset <= gamma:
                kept_pending.append(block)
            else:
                self._finalize_walk_online(
                    block, diverged=diverged, forward_ct=batch.forward_ct
                )
        state.pending = kept_pending
        return pending_gathers

    def _accumulate_online(self, block: _PendingBlock, *, p_lp: float) -> None:
        a = min(1.0, math.exp(p_lp - block.q_lps[block.next_offset - block.window - 1]))
        block.est_prod *= a
        block.est_lo_extra += block.est_prod

    def _drop_pending_on_discontinuity(
        self, state: _RequestState, *, seq_len: int, forward_ct: int
    ) -> None:
        if state.expected_seq_len < 0 or seq_len == state.expected_seq_len:
            return
        if not state.pending:
            return
        self._discontinuity_drop_ct += len(state.pending)
        for block in state.pending:
            self._finalize_at_end_online(block, forward_ct=forward_ct)
        state.pending = []

    def _finish_step(self, *, forward_ct: int) -> None:
        self._observed_step_ct += 1
        if self._file is not None:
            self._steps_since_flush += 1
            if self._steps_since_flush >= _FLUSH_EVERY_STEPS:
                self._file.flush()
                self._steps_since_flush = 0
        if self._observed_step_ct % _STATE_SWEEP_INTERVAL == 0:
            self._sweep_states(forward_ct=forward_ct)
        self._online.maybe_log(forward_ct=forward_ct)

    def _host_to_device_async(
        self, values: List[int], *, device: torch.device
    ) -> torch.Tensor:
        host = torch.tensor(values, dtype=torch.long, pin_memory=device.type == "cuda")
        self._retained_h2d.append(host)
        return host.to(device=device, non_blocking=True)

    def _gather_logprobs(
        self,
        *,
        logits: torch.Tensor,
        row_indices: torch.Tensor,
        token_indices: torch.Tensor,
        temps: torch.Tensor,
    ) -> torch.Tensor:
        if row_indices.numel() == 0:
            return torch.zeros(0, dtype=torch.float32, device=logits.device)
        per_row_temps = temps[row_indices].clamp_min(1e-5)
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
        return torch.cat(results)

    def _sweep_states(self, *, forward_ct: int) -> None:
        expired = [
            rid
            for rid, state in self._states.items()
            if forward_ct - state.last_seen_ct > _STATE_EXPIRE_STEPS
        ]
        for rid in expired:
            for block in self._states[rid].pending:
                self._finalize_at_end_online(block, forward_ct=forward_ct)
            del self._states[rid]
            self._finish_intents.pop(rid, None)

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
