"""ASD acceptance adapter for the fixed greedy DSpark verify seam.

The adapter is deliberately the only place that knows both the engine-neutral
ASD rule and DSpark's candidate/logit/request-slot layout.  Its hot interface
returns the same ``(correct_len, bonus, cap_trim_lens)`` tuple as the native
greedy verifier, so the existing finalization, KV commit, request-token, next
draft input, and metrics paths all consume one authoritative accepted length.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Mapping, Optional

import msgspec
import torch

try:
    from asd.reproduce.dspark.asd_config import DSparkASDConfig
    from asd.reproduce.dspark.torch_rule import choose_prefix_batch
except ImportError:
    # Optional research dependency, not published to PyPI; install from the
    # ASD source repository only when enabling ASD acceptance. A missing
    # install keeps DSpark fully native (ASD_ENABLED=0, no trace).
    DSparkASDConfig = None
    choose_prefix_batch = None


ASD_MODE_DISABLED = "disabled"
ASD_MODE_CALIBRATION = "calibration"
ASD_MODE_ENABLED = "enabled"
_ASD_MODES = {
    ASD_MODE_DISABLED,
    ASD_MODE_CALIBRATION,
    ASD_MODE_ENABLED,
}

_ENABLED_ENV = "ASD_ENABLED"
_CALIBRATION_TRACE_ENV = "SGLANG_DSPARK_ASD_CALIBRATION_TRACE"
_LEGACY_MODE_ENV = "SGLANG_DSPARK_ASD_MODE"
_CONFIG_JSON_ENV = "SGLANG_DSPARK_ASD_CONFIG_JSON"
_CONFIG_PATH_ENV = "SGLANG_DSPARK_ASD_CONFIG_PATH"
_TRACE_CAPACITY_ENV = "SGLANG_DSPARK_ASD_TRACE_CAPACITY"
_DEFAULT_TRACE_CAPACITY = 65536


class DSparkASDSettings(msgspec.Struct, frozen=True):
    """Process-start settings for one of the three experiment behaviours."""

    mode: str = ASD_MODE_DISABLED
    config: Optional[DSparkASDConfig] = None
    trace_capacity: int = _DEFAULT_TRACE_CAPACITY

    def __post_init__(self) -> None:
        if self.mode not in _ASD_MODES:
            raise ValueError(
                f"internal ASD mode must be one of {sorted(_ASD_MODES)}, "
                f"got {self.mode!r}; experiment arms use {_ENABLED_ENV}=0|1"
            )
        if self.mode == ASD_MODE_ENABLED and self.config is None:
            raise ValueError("ASD enabled mode requires a decode configuration")
        if self.mode != ASD_MODE_ENABLED and self.config is not None:
            raise ValueError(
                "an ASD decode configuration is valid only in enabled mode"
            )
        if (
            isinstance(self.trace_capacity, bool)
            or not isinstance(self.trace_capacity, int)
            or self.trace_capacity <= 0
        ):
            raise ValueError("trace_capacity must be a positive integer")

    @classmethod
    def from_environ(
        cls, environ: Optional[Mapping[str, str]] = None
    ) -> "DSparkASDSettings":
        """Resolve immutable settings without touching config in native mode."""

        env = os.environ if environ is None else environ
        if _LEGACY_MODE_ENV in env:
            raise ValueError(
                f"{_LEGACY_MODE_ENV} is not an experiment arm switch; use "
                f"{_ENABLED_ENV}=0|1 and {_CALIBRATION_TRACE_ENV}=0|1"
            )

        enabled_raw = env.get(_ENABLED_ENV, "0")
        trace_raw = env.get(_CALIBRATION_TRACE_ENV, "0")
        if enabled_raw not in {"0", "1"}:
            raise ValueError(f"{_ENABLED_ENV} must be exactly 0 or 1")
        if trace_raw not in {"0", "1"}:
            raise ValueError(f"{_CALIBRATION_TRACE_ENV} must be exactly 0 or 1")
        enabled = enabled_raw == "1"
        trace = trace_raw == "1"
        if enabled and trace:
            raise ValueError(
                f"{_ENABLED_ENV}=1 conflicts with " f"{_CALIBRATION_TRACE_ENV}=1"
            )

        inline = env.get(_CONFIG_JSON_ENV)
        path = env.get(_CONFIG_PATH_ENV)
        if not enabled and (inline or path):
            raise ValueError("ASD config is invalid when ASD_ENABLED=0")
        if trace:
            raw_capacity = env.get(_TRACE_CAPACITY_ENV, str(_DEFAULT_TRACE_CAPACITY))
            try:
                capacity = int(raw_capacity)
            except ValueError as error:
                raise ValueError(f"{_TRACE_CAPACITY_ENV} must be an integer") from error
            return cls(mode=ASD_MODE_CALIBRATION, trace_capacity=capacity)
        if not enabled:
            return cls(mode=ASD_MODE_DISABLED)

        if bool(inline) == bool(path):
            raise ValueError(
                "ASD enabled mode requires exactly one of "
                f"{_CONFIG_JSON_ENV} or {_CONFIG_PATH_ENV}"
            )

        if DSparkASDConfig is None:
            raise RuntimeError(
                f"{_ENABLED_ENV}=1 requires the optional ASD package; "
                "install it from the ASD source repository or unset "
                f"{_ENABLED_ENV} to run DSpark with strict verification"
            )

        if path:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        else:
            payload = json.loads(inline)
        if not isinstance(payload, Mapping):
            raise ValueError("ASD config JSON must be an object")
        return cls(
            mode=ASD_MODE_ENABLED,
            config=DSparkASDConfig.from_mapping(payload),
        )

    @property
    def active(self) -> bool:
        return self.mode != ASD_MODE_DISABLED

    @property
    def fingerprint(self) -> Optional[str]:
        return None if self.config is None else self.config.fingerprint()


class DSparkASDAdapter:
    """Deep adapter at the DSpark native-vs-ASD acceptance seam.

    Request identities are bound at prefill/finish lifecycle points.  The
    decode interface uses only request-pool device indices, compact score
    tensors, and device state.  Snapshot conversion is intentionally separate
    and may synchronize only when an arm/calibration run is being sealed.
    """

    _PROPOSALS = 0
    _DRAFTS_VERIFIABLE = 1
    _STRICT_ACCEPTED = 2
    _ASD_ACCEPTED = 3
    _RELAXED_MISMATCHES = 4
    _CAP_TRIMS = 5
    _BUDGET_EXHAUSTIONS = 6
    _REQUESTS_INITIALIZED = 7
    _REQUESTS_FINISHED = 8
    _REQUESTS_NON_NATURAL = 9
    _SLOT_REUSE_RESETS = 10
    _INT_COUNTER_COUNT = 11

    _REGRET_CHARGED = 0
    _COMPLETED_REMAINING_BUDGET = 1
    _FLOAT_COUNTER_COUNT = 2

    _TRACE_VALID = 0
    _TRACE_PROPOSAL = 1
    _TRACE_SLOT = 2
    _TRACE_REQUEST_SERIAL = 3
    _TRACE_FORWARD = 4
    _TRACE_BARRIER = 5
    _TRACE_INT_FIELD_COUNT = 6

    _TRACE_REGRET = 0
    _TRACE_VALUE = 1
    _TRACE_RATIO = 2
    _TRACE_FLOAT_FIELD_COUNT = 3

    def __init__(
        self,
        *,
        settings: DSparkASDSettings,
        gamma: int,
        verify_num_draft_tokens: int,
        device: torch.device | str,
    ) -> None:
        self.settings = settings
        self.gamma = int(gamma)
        self.verify_num_draft_tokens = int(verify_num_draft_tokens)
        self.device = torch.device(device)
        if self.gamma <= 0:
            raise ValueError("DSpark gamma must be positive")
        if self.verify_num_draft_tokens != self.gamma + 1:
            raise ValueError(
                "DSpark ASD requires verify_num_draft_tokens == gamma + 1, "
                f"got {self.verify_num_draft_tokens} and {self.gamma}"
            )
        if settings.config is not None:
            settings.config.validate_block_size(self.gamma)

        self._remaining_budget: Optional[torch.Tensor] = None
        self._active_slots: Optional[torch.Tensor] = None
        self._request_serial_by_slot: Optional[torch.Tensor] = None
        self._int_counters: Optional[torch.Tensor] = None
        self._float_counters: Optional[torch.Tensor] = None
        self._accepted_by_position: Optional[torch.Tensor] = None
        self._trace_cursor: Optional[torch.Tensor] = None
        self._trace_dropped: Optional[torch.Tensor] = None
        self._trace_int: Optional[torch.Tensor] = None
        self._trace_float: Optional[torch.Tensor] = None

        self._rid_to_slot: dict[str, int] = {}
        self._slot_to_rid: dict[int, str] = {}
        self._serial_to_rid: dict[int, str] = {}
        self._next_request_serial = 1

    @property
    def active(self) -> bool:
        return self.settings.active

    def identity(self) -> dict:
        return {
            "mode": self.settings.mode,
            "experiment_switches": {
                "ASD_ENABLED": (1 if self.settings.mode == ASD_MODE_ENABLED else 0),
                "SGLANG_DSPARK_ASD_CALIBRATION_TRACE": (
                    1 if self.settings.mode == ASD_MODE_CALIBRATION else 0
                ),
            },
            "config": (
                None
                if self.settings.config is None
                else self.settings.config.to_mapping()
            ),
            "config_fingerprint": self.settings.fingerprint,
            "gamma": self.gamma,
            "verify_num_draft_tokens": self.verify_num_draft_tokens,
            "candidate_alignment": {
                "anchor_index": 0,
                "draft_indices": [1, self.verify_num_draft_tokens],
                "target_draft_logit_indices": [0, self.gamma],
            },
            "score_seam": (
                "native full-vocab next_token_logits after the existing "
                "LogitsProcessor TP all-gather; ASD adds no TP collective"
            ),
        }

    def require_supported_runtime(
        self,
        *,
        disable_cuda_graph: bool,
        disable_overlap_schedule: bool,
        ragged_verify_mode: str,
        simulate_accept_length: float,
    ) -> None:
        """Fail loudly instead of silently bypassing an active ASD mode."""

        if not self.active:
            return
        unsupported = []
        if not disable_cuda_graph:
            unsupported.append("CUDA graph")
        if not disable_overlap_schedule:
            unsupported.append("overlap schedule")
        if ragged_verify_mode != "static":
            unsupported.append(
                f"ragged verify mode {ragged_verify_mode!r} (requires 'static')"
            )
        if simulate_accept_length > 0:
            unsupported.append("simulated acceptance length")
        if unsupported:
            raise ValueError(
                "DSpark ASD calibration/enabled mode supports only the frozen "
                "eager static greedy experiment path; disable " + ", ".join(unsupported)
            )

    def require_unfolded_accept(self) -> None:
        if self.active:
            raise RuntimeError(
                "active DSpark ASD mode cannot use folded native acceptance"
            )

    def _allocate_state(self, pool_capacity: int) -> None:
        if self._remaining_budget is not None:
            if self._remaining_budget.shape[0] != pool_capacity:
                raise RuntimeError(
                    "request-pool capacity changed after ASD state allocation"
                )
            return
        if pool_capacity <= 1:
            raise ValueError("request-pool capacity must include at least one slot")

        self._remaining_budget = torch.zeros(
            (pool_capacity,), dtype=torch.float64, device=self.device
        )
        self._active_slots = torch.zeros(
            (pool_capacity,), dtype=torch.bool, device=self.device
        )
        self._request_serial_by_slot = torch.full(
            (pool_capacity,), -1, dtype=torch.int64, device=self.device
        )
        self._int_counters = torch.zeros(
            (self._INT_COUNTER_COUNT,), dtype=torch.int64, device=self.device
        )
        self._float_counters = torch.zeros(
            (self._FLOAT_COUNTER_COUNT,), dtype=torch.float64, device=self.device
        )
        self._accepted_by_position = torch.zeros(
            (self.gamma,), dtype=torch.int64, device=self.device
        )
        if self.settings.mode == ASD_MODE_CALIBRATION:
            capacity = self.settings.trace_capacity
            self._trace_cursor = torch.zeros((), dtype=torch.int64, device=self.device)
            self._trace_dropped = torch.zeros((), dtype=torch.int64, device=self.device)
            self._trace_int = torch.zeros(
                (capacity + 1, self._TRACE_INT_FIELD_COUNT),
                dtype=torch.int64,
                device=self.device,
            )
            self._trace_float = torch.zeros(
                (capacity + 1, self._TRACE_FLOAT_FIELD_COUNT),
                dtype=torch.float64,
                device=self.device,
            )

    @staticmethod
    def _pool_capacity(batch) -> int:
        pool = batch.req_to_token_pool
        if hasattr(pool, "_alloc_size"):
            return int(pool._alloc_size)
        return int(pool.size) + 1

    @staticmethod
    def _cpu_slots(batch) -> list[int]:
        mirror = batch.req_pool_indices_cpu
        if mirror is not None:
            return [int(value) for value in mirror]
        slots = [req.req_pool_idx for req in batch.reqs]
        if any(slot is None for slot in slots):
            raise RuntimeError("ASD prefill lifecycle saw an unallocated request")
        return [int(slot) for slot in slots]

    def bind_batch(self, batch) -> None:
        """Bind stable request identities at the low-frequency prefill seam."""

        if not self.active or not batch.reqs:
            return
        self._allocate_state(self._pool_capacity(batch))
        slots = self._cpu_slots(batch)
        if len(slots) != len(batch.reqs):
            raise RuntimeError("request and request-pool slot counts differ")

        for req, slot in zip(batch.reqs, slots):
            rid = str(req.rid)
            old_slot = self._rid_to_slot.get(rid)
            if old_slot is not None and old_slot != slot:
                self._clear_slot(old_slot)
                self._slot_to_rid.pop(old_slot, None)
                self._rid_to_slot.pop(rid, None)

            old_rid = self._slot_to_rid.get(slot)
            if old_rid == rid:
                continue
            if old_rid is not None:
                self._rid_to_slot.pop(old_rid, None)
                self._clear_slot(slot)
                self._int_counters[self._SLOT_REUSE_RESETS].add_(1)

            serial = self._next_request_serial
            self._next_request_serial += 1
            self._rid_to_slot[rid] = slot
            self._slot_to_rid[slot] = rid
            self._serial_to_rid[serial] = rid
            slot_tensor = torch.tensor([slot], dtype=torch.int64, device=self.device)
            self._active_slots.index_fill_(0, slot_tensor, True)
            self._request_serial_by_slot.index_fill_(0, slot_tensor, serial)
            initial_budget = (
                0.0
                if self.settings.config is None
                else self.settings.config.risk_budget
            )
            self._remaining_budget.index_fill_(0, slot_tensor, initial_budget)
            self._int_counters[self._REQUESTS_INITIALIZED].add_(1)

    def _clear_slot(self, slot: int) -> None:
        if self._remaining_budget is None:
            return
        slot_tensor = torch.tensor([slot], dtype=torch.int64, device=self.device)
        self._remaining_budget.index_fill_(0, slot_tensor, 0.0)
        self._active_slots.index_fill_(0, slot_tensor, False)
        self._request_serial_by_slot.index_fill_(0, slot_tensor, -1)

    def note_request_finished(self, *, rid: str, natural_stop: bool) -> None:
        """Clear request-owned device state on finish, cancellation, or abort."""

        if not self.active:
            return
        slot = self._rid_to_slot.pop(rid, None)
        if slot is None:
            return
        self._slot_to_rid.pop(slot, None)
        slot_tensor = torch.tensor([slot], dtype=torch.int64, device=self.device)
        if self.settings.mode == ASD_MODE_ENABLED:
            self._float_counters[self._COMPLETED_REMAINING_BUDGET].add_(
                self._remaining_budget.index_select(0, slot_tensor).sum()
            )
        self._int_counters[self._REQUESTS_FINISHED].add_(1)
        if not natural_stop:
            self._int_counters[self._REQUESTS_NON_NATURAL].add_(1)
        self._clear_slot(slot)

    def _assert_bound(self, req_pool_indices: torch.Tensor) -> None:
        if self._active_slots is None:
            raise RuntimeError(
                "active ASD mode reached decode before prefill lifecycle binding"
            )
        active = self._active_slots.index_select(0, req_pool_indices.to(torch.int64))
        torch._assert_async(
            active.all(),
            "DSpark ASD decode saw an inactive/reused request-pool slot",
        )

    def _compact_scores(
        self,
        *,
        candidates: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if candidates.ndim != 2:
            raise ValueError("DSpark ASD candidates must be [batch, verify_width]")
        bs, width = candidates.shape
        if width != self.verify_num_draft_tokens:
            raise ValueError(
                f"expected candidate width {self.verify_num_draft_tokens}, got {width}"
            )
        if target_logits.ndim != 2:
            raise ValueError("DSpark target logits must be [batch*verify_width, vocab]")
        if target_logits.shape[0] != bs * width:
            raise ValueError(
                "target logits row count does not match DSpark candidate layout"
            )
        logits = target_logits.view(bs, width, -1)
        top_logits, top_token_ids = logits.max(dim=-1)
        draft_token_ids = candidates[:, 1:]
        draft_logits = logits[:, : self.gamma].gather(
            dim=-1, index=draft_token_ids.unsqueeze(-1)
        )
        return top_logits, top_token_ids, draft_logits.squeeze(-1)

    def _strict_uncapped(
        self, *, draft_token_ids: torch.Tensor, top_token_ids: torch.Tensor
    ) -> torch.Tensor:
        return (
            (draft_token_ids == top_token_ids[:, : self.gamma])
            .to(torch.int32)
            .cumprod(dim=1)
            .sum(dim=1)
        )

    def _max_verifiable_drafts(
        self,
        *,
        bs: int,
        device: torch.device,
        cutoff_verify_lens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if cutoff_verify_lens is None:
            return torch.full((bs,), self.gamma, dtype=torch.int32, device=device)
        return (cutoff_verify_lens.to(device=device, dtype=torch.int32) - 1).clamp_(
            min=0, max=self.gamma
        )

    def _record_calibration(
        self,
        *,
        req_pool_indices: torch.Tensor,
        forward_ct: int,
        strict_uncapped: torch.Tensor,
        max_verifiable: torch.Tensor,
        top_logits: torch.Tensor,
        draft_logits: torch.Tensor,
    ) -> None:
        bs = strict_uncapped.shape[0]
        regrets = (
            top_logits[:, : self.gamma].to(torch.float64)
            - draft_logits.to(torch.float64)
        ).clamp_min_(0.0)
        values = torch.arange(
            self.gamma,
            0,
            -1,
            dtype=torch.float64,
            device=regrets.device,
        ) / float(self.gamma)
        barrier = strict_uncapped.to(torch.int64)
        safe_barrier = barrier.clamp(max=self.gamma - 1)
        row = torch.arange(bs, dtype=torch.int64, device=regrets.device)
        barrier_regret = regrets[row, safe_barrier]
        barrier_value = values[safe_barrier]
        valid = (
            (barrier < self.gamma)
            & (barrier < max_verifiable.to(torch.int64))
            & (barrier_regret > 0.0)
        )

        ordinal = self._trace_cursor + torch.arange(
            bs, dtype=torch.int64, device=regrets.device
        )
        in_capacity = ordinal < self.settings.trace_capacity
        # Keep overflow writes on a device-only sentinel row.  Clamping to the
        # last real row would let later proposals corrupt the final retained
        # calibration record, while a host-side capacity branch would
        # synchronize the decode hot path.
        safe_ordinal = ordinal.clamp(max=self.settings.trace_capacity)
        serial = self._request_serial_by_slot.index_select(
            0, req_pool_indices.to(torch.int64)
        )
        self._trace_int[safe_ordinal, self._TRACE_VALID] = (valid & in_capacity).to(
            torch.int64
        )
        self._trace_int[safe_ordinal, self._TRACE_PROPOSAL] = ordinal
        self._trace_int[safe_ordinal, self._TRACE_SLOT] = req_pool_indices
        self._trace_int[safe_ordinal, self._TRACE_REQUEST_SERIAL] = serial
        self._trace_int[safe_ordinal, self._TRACE_FORWARD] = int(forward_ct)
        self._trace_int[safe_ordinal, self._TRACE_BARRIER] = barrier
        self._trace_float[safe_ordinal, self._TRACE_REGRET] = barrier_regret
        self._trace_float[safe_ordinal, self._TRACE_VALUE] = barrier_value
        self._trace_float[safe_ordinal, self._TRACE_RATIO] = (
            barrier_regret / barrier_value
        )
        self._trace_cursor.add_(bs)
        self._trace_dropped.add_((~in_capacity).to(torch.int64).sum())

    def accept_or_native(
        self,
        *,
        candidates: torch.Tensor,
        target_logits: Optional[torch.Tensor],
        cutoff_verify_lens: Optional[torch.Tensor],
        req_pool_indices: torch.Tensor,
        forward_ct: int,
        all_greedy: bool,
        native_accept: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Choose one acceptance implementation and return its authoritative tuple."""

        if not self.active:
            return native_accept()
        if not all_greedy:
            raise ValueError(
                "DSpark ASD calibration/enabled mode supports greedy verify only"
            )
        if target_logits is None:
            raise RuntimeError("DSpark ASD requires target logits")
        self._assert_bound(req_pool_indices)

        top_logits, top_token_ids, draft_logits = self._compact_scores(
            candidates=candidates,
            target_logits=target_logits,
        )
        draft_token_ids = candidates[:, 1:]
        strict_uncapped = self._strict_uncapped(
            draft_token_ids=draft_token_ids,
            top_token_ids=top_token_ids,
        )
        max_verifiable = self._max_verifiable_drafts(
            bs=candidates.shape[0],
            device=candidates.device,
            cutoff_verify_lens=cutoff_verify_lens,
        )

        if self.settings.mode == ASD_MODE_CALIBRATION:
            native_result = native_accept()
            self._record_calibration(
                req_pool_indices=req_pool_indices,
                forward_ct=forward_ct,
                strict_uncapped=strict_uncapped,
                max_verifiable=max_verifiable,
                top_logits=top_logits,
                draft_logits=draft_logits,
            )
            return native_result

        remaining_before = self._remaining_budget.index_select(
            0, req_pool_indices.to(torch.int64)
        )
        decision = choose_prefix_batch(
            draft_token_ids=draft_token_ids,
            top_logits=top_logits[:, : self.gamma],
            top_token_ids=top_token_ids[:, : self.gamma],
            draft_logits=draft_logits,
            remaining_budget=remaining_before,
            config=self.settings.config,
        )
        uncapped = decision.accepted.to(torch.int32)
        accepted = torch.minimum(uncapped, max_verifiable)
        cap_trim_lens = uncapped - accepted

        positions = torch.arange(
            self.gamma, dtype=torch.int32, device=candidates.device
        ).unsqueeze(0)
        committed = positions < accepted.unsqueeze(1)
        committed_mismatches = committed & decision.mismatched
        charged = torch.where(
            committed_mismatches,
            decision.regrets,
            torch.zeros_like(decision.regrets),
        ).sum(dim=1)
        relaxed = committed_mismatches.to(torch.int64).sum(dim=1)
        remaining_after = (remaining_before.to(torch.float64) - charged).clamp_min(0.0)
        self._remaining_budget.index_copy_(
            0, req_pool_indices.to(torch.int64), remaining_after
        )

        row = torch.arange(
            candidates.shape[0], dtype=torch.int64, device=candidates.device
        )
        bonus = top_token_ids[row, accepted.to(torch.int64)].to(torch.int64)

        strict_capped = torch.minimum(
            strict_uncapped, max_verifiable.to(strict_uncapped.dtype)
        )
        self._int_counters[self._PROPOSALS].add_(candidates.shape[0])
        self._int_counters[self._DRAFTS_VERIFIABLE].add_(
            max_verifiable.to(torch.int64).sum()
        )
        self._int_counters[self._STRICT_ACCEPTED].add_(
            strict_capped.to(torch.int64).sum()
        )
        self._int_counters[self._ASD_ACCEPTED].add_(accepted.to(torch.int64).sum())
        self._int_counters[self._RELAXED_MISMATCHES].add_(relaxed.sum())
        self._int_counters[self._CAP_TRIMS].add_(cap_trim_lens.to(torch.int64).sum())
        self._int_counters[self._BUDGET_EXHAUSTIONS].add_(
            ((remaining_before > 0.0) & (remaining_after <= 0.0) & (charged > 0.0))
            .to(torch.int64)
            .sum()
        )
        self._float_counters[self._REGRET_CHARGED].add_(charged.sum())
        self._accepted_by_position.add_(committed.to(torch.int64).sum(dim=0))
        return accepted, bonus, cap_trim_lens

    @staticmethod
    def _cpu_scalar(tensor: torch.Tensor) -> int | float:
        return tensor.detach().cpu().item()

    def snapshot(self) -> dict:
        """Bulk-copy counters/traces after an arm; never called per proposal."""

        out = {
            **self.identity(),
            "counter_schema_version": 1,
            "proposals": 0,
            "draft_tokens_verifiable": 0,
            "strict_accepted_draft_tokens": 0,
            "asd_accepted_draft_tokens": 0,
            "asd_accepted_draft_tokens_by_position": [0] * self.gamma,
            "relaxed_mismatches": 0,
            "regret_charged": 0.0,
            "cap_trim_lens": 0,
            "budget_exhaustion_events": 0,
            "remaining_budget_total": 0.0,
            "requests_initialized": 0,
            "requests_finished": 0,
            "requests_non_natural": 0,
            "slot_reuse_resets": 0,
            "active_request_states": len(self._rid_to_slot),
            "state_leaks": len(self._rid_to_slot),
            "remaining_budget_by_request": [],
            "strict_rejection_trace": [],
            "trace_rows_seen": 0,
            "trace_rows_dropped": 0,
        }
        if self._int_counters is None:
            return out

        ints = self._int_counters.detach().cpu().tolist()
        floats = self._float_counters.detach().cpu().tolist()
        out.update(
            proposals=int(ints[self._PROPOSALS]),
            draft_tokens_verifiable=int(ints[self._DRAFTS_VERIFIABLE]),
            strict_accepted_draft_tokens=int(ints[self._STRICT_ACCEPTED]),
            asd_accepted_draft_tokens=int(ints[self._ASD_ACCEPTED]),
            asd_accepted_draft_tokens_by_position=[
                int(value)
                for value in self._accepted_by_position.detach().cpu().tolist()
            ],
            relaxed_mismatches=int(ints[self._RELAXED_MISMATCHES]),
            regret_charged=float(floats[self._REGRET_CHARGED]),
            cap_trim_lens=int(ints[self._CAP_TRIMS]),
            budget_exhaustion_events=int(ints[self._BUDGET_EXHAUSTIONS]),
            requests_initialized=int(ints[self._REQUESTS_INITIALIZED]),
            requests_finished=int(ints[self._REQUESTS_FINISHED]),
            requests_non_natural=int(ints[self._REQUESTS_NON_NATURAL]),
            slot_reuse_resets=int(ints[self._SLOT_REUSE_RESETS]),
        )

        if self.settings.mode == ASD_MODE_ENABLED:
            active_items = sorted(self._rid_to_slot.items())
            if active_items:
                slots = torch.tensor(
                    [slot for _, slot in active_items],
                    dtype=torch.int64,
                    device=self.device,
                )
                budgets = (
                    self._remaining_budget.index_select(0, slots)
                    .detach()
                    .cpu()
                    .tolist()
                )
                out["remaining_budget_by_request"] = [
                    {"rid": rid, "slot": slot, "remaining_budget": float(budget)}
                    for (rid, slot), budget in zip(active_items, budgets)
                ]
                active_remaining = float(sum(budgets))
            else:
                active_remaining = 0.0
            out["remaining_budget_total"] = (
                float(floats[self._COMPLETED_REMAINING_BUDGET]) + active_remaining
            )

        if self.settings.mode == ASD_MODE_CALIBRATION:
            seen = int(self._cpu_scalar(self._trace_cursor))
            dropped = int(self._cpu_scalar(self._trace_dropped))
            stored = min(seen, self.settings.trace_capacity)
            trace_int = self._trace_int[:stored].detach().cpu().tolist()
            trace_float = self._trace_float[:stored].detach().cpu().tolist()
            records = []
            for ints_row, floats_row in zip(trace_int, trace_float):
                if not ints_row[self._TRACE_VALID]:
                    continue
                serial = int(ints_row[self._TRACE_REQUEST_SERIAL])
                records.append(
                    {
                        "proposal_ordinal": int(ints_row[self._TRACE_PROPOSAL]),
                        "request_serial": serial,
                        "rid": self._serial_to_rid.get(serial),
                        "request_pool_slot": int(ints_row[self._TRACE_SLOT]),
                        "forward_ct": int(ints_row[self._TRACE_FORWARD]),
                        "barrier_position": int(ints_row[self._TRACE_BARRIER]),
                        "regret": float(floats_row[self._TRACE_REGRET]),
                        "value": float(floats_row[self._TRACE_VALUE]),
                        "regret_per_value": float(floats_row[self._TRACE_RATIO]),
                    }
                )
            out["strict_rejection_trace"] = records
            out["trace_rows_seen"] = seen
            out["trace_rows_dropped"] = dropped
        return out

    def clear_metrics(self) -> None:
        """Reset arm-level evidence without resetting live request budgets."""

        if self._int_counters is None:
            return
        self._int_counters.zero_()
        self._float_counters.zero_()
        self._accepted_by_position.zero_()
        if self._trace_cursor is not None:
            self._trace_cursor.zero_()
            self._trace_dropped.zero_()
            self._trace_int.zero_()
            self._trace_float.zero_()
