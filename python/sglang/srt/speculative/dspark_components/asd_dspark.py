"""ASD acceptance adapter for the fixed greedy DSpark verify seam.

ASD (approximate speculative decoding) relaxes strict greedy verification by
admitting draft tokens whose target-logit gap ("regret") stays within a
bounded per-request budget; a zero budget recovers strict verification.

The adapter is deliberately the only place that knows both the engine-neutral
ASD rule and DSpark's candidate/logit/request-slot layout.  Its hot interface
returns the same ``(correct_len, bonus, cap_trim_lens)`` tuple as the native
greedy verifier, so the existing finalization, KV commit, request-token, and
next-draft input paths all consume one authoritative accepted length.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping, Optional

import msgspec
import torch

try:
    from asd.reproduce.dspark.asd_config import DSparkASDConfig
    from asd.reproduce.dspark.torch_rule import choose_prefix_batch
except ImportError:
    # Optional research dependency, not published to PyPI; install from the
    # ASD source repository only when using ASD acceptance. A missing install
    # keeps DSpark fully native (no --speculative-dspark-asd-config-path).
    DSparkASDConfig = None
    choose_prefix_batch = None


class DSparkASDSettings(msgspec.Struct, frozen=True):
    """Immutable acceptance settings resolved once at worker startup.

    ``config`` is annotated ``object`` on purpose: the concrete type comes
    from an optional dependency, and msgspec skips validation for ``object``
    fields, so unit tests may inject duck-typed stand-in configs to exercise
    the acceptance-accounting path without the package.
    """

    config: object = None

    @classmethod
    def from_server_args(cls, server_args) -> "DSparkASDSettings":
        """Resolve from --speculative-dspark-asd-config-path (None = strict)."""

        path = getattr(server_args, "speculative_dspark_asd_config_path", None)
        if path is None:
            return cls()
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("ASD config JSON must be an object")
        if DSparkASDConfig is None:
            raise RuntimeError(
                "--speculative-dspark-asd-config-path requires the optional "
                "ASD package; install it from the ASD source repository or "
                "unset the flag to run DSpark with strict greedy verification"
            )
        return cls(config=DSparkASDConfig.from_mapping(payload))

    @property
    def active(self) -> bool:
        return self.config is not None

    @property
    def fingerprint(self) -> Optional[str]:
        return None if self.config is None else self.config.fingerprint()


class DSparkASDAdapter:
    """Deep adapter at the DSpark native-vs-ASD acceptance seam.

    Request identities are bound at the prefill lifecycle point.  The decode
    interface uses only request-pool device indices, compact score tensors,
    and device state, so the relaxed-acceptance hot path never synchronizes.
    """

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
        if settings.config is not None:
            # ASD-only invariants; a native (strict) DSpark start must not be
            # coupled to ASD assumptions.
            if self.gamma <= 0:
                raise ValueError("DSpark ASD gamma must be positive")
            if self.verify_num_draft_tokens != self.gamma + 1:
                raise ValueError(
                    "DSpark ASD requires verify_num_draft_tokens == gamma + 1, "
                    f"got {self.verify_num_draft_tokens} and {self.gamma}"
                )
            settings.config.validate_block_size(self.gamma)

        self._remaining_budget: Optional[torch.Tensor] = None
        self._active_slots: Optional[torch.Tensor] = None

        self._rid_to_slot: dict[str, int] = {}
        self._slot_to_rid: dict[int, str] = {}

    @property
    def active(self) -> bool:
        return self.settings.active

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
                "DSpark ASD acceptance supports only the frozen eager static "
                "greedy experiment path; disable " + ", ".join(unsupported)
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

            self._rid_to_slot[rid] = slot
            self._slot_to_rid[slot] = rid
            slot_tensor = torch.tensor([slot], dtype=torch.int64, device=self.device)
            self._active_slots.index_fill_(0, slot_tensor, True)
            initial_budget = (
                0.0
                if self.settings.config is None
                else self.settings.config.risk_budget
            )
            self._remaining_budget.index_fill_(0, slot_tensor, initial_budget)

    def _clear_slot(self, slot: int) -> None:
        if self._remaining_budget is None:
            return
        slot_tensor = torch.tensor([slot], dtype=torch.int64, device=self.device)
        self._remaining_budget.index_fill_(0, slot_tensor, 0.0)
        self._active_slots.index_fill_(0, slot_tensor, False)

    def note_request_finished(self, *, rid: str) -> None:
        """Clear request-owned device state on finish, cancellation, or abort."""

        if not self.active:
            return
        slot = self._rid_to_slot.pop(rid, None)
        if slot is None:
            return
        self._slot_to_rid.pop(slot, None)
        self._clear_slot(slot)

    def _assert_bound(self, req_pool_indices: torch.Tensor) -> None:
        # Device-side backstop only: in eager mode torch._assert_async does
        # NOT raise synchronously here -- a tripped assert surfaces as a CUDA
        # error at the next host synchronization point (e.g. the decode output
        # copy), possibly several steps later.  The primary guards are the
        # Python-side slot binding in bind_batch and the state clearing in
        # note_request_finished; do not rely on this check for correctness.
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

    def accept_or_native(
        self,
        *,
        candidates: torch.Tensor,
        target_logits: Optional[torch.Tensor],
        cutoff_verify_lens: Optional[torch.Tensor],
        req_pool_indices: torch.Tensor,
        all_greedy: bool,
        native_accept: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Choose one acceptance implementation and return its authoritative tuple."""

        if not self.active:
            return native_accept()
        if not all_greedy:
            raise ValueError("DSpark ASD acceptance supports greedy verify only")
        if target_logits is None:
            raise RuntimeError("DSpark ASD requires target logits")
        self._assert_bound(req_pool_indices)

        top_logits, top_token_ids, draft_logits = self._compact_scores(
            candidates=candidates,
            target_logits=target_logits,
        )
        draft_token_ids = candidates[:, 1:]
        max_verifiable = self._max_verifiable_drafts(
            bs=candidates.shape[0],
            device=candidates.device,
            cutoff_verify_lens=cutoff_verify_lens,
        )

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
        remaining_after = (remaining_before.to(torch.float64) - charged).clamp_min(0.0)
        self._remaining_budget.index_copy_(
            0, req_pool_indices.to(torch.int64), remaining_after
        )

        row = torch.arange(
            candidates.shape[0], dtype=torch.int64, device=candidates.device
        )
        bonus = top_token_ids[row, accepted.to(torch.int64)].to(torch.int64)
        return accepted, bonus, cap_trim_lens
