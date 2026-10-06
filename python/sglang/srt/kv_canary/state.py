from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.kernels.ops.kv_canary.consts import VIOLATION_FIELDS
from sglang.srt.kv_canary.config import CanaryConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class ViolationLog:
    """Global violation sink shared across all canary launches.

    One instance per canary runner — every launch (head / tail / sweep, K / V half, FULL / SWA group) writes
    into the same ring. The kernel_kind field stamped into each violation row identifies which launch fired
    (kernel_kind is a static IntEnum tag — :class:`CanaryLaunchTag` in
    ``sglang.kernels.ops.kv_canary.verify`` — with a unique value per (head|tail|sweep, K|V, FULL|SWA) tuple).

    Ring capacity is sized generously (≥ 1024) so overflow is a non-concern in practice — violations are
    cold-path and the host raises at the first one anyway (or just logs it in mode="log"). atomicAdd
    contention on a single counter is also negligible since violation events are rare.

    Derived state (host computes on read; not stored):
        is_errored = violation_write_index[0] > 0
        first_violation = violation_ring[0]   (valid iff is_errored)
        ring_valid_count = min(violation_write_index[0], ring_capacity)

    The ring is fill-once: writes beyond ring_capacity are dropped but the counter still increments. Whoever
    wins atomicAdd for idx == 0 permanently occupies row 0.

    Fields:
        violation_ring: Append-only violation sink, shape [ring_capacity, VIOLATION_FIELDS], int64. Row 0 is
            the first violation; rows 1..min(write_index, capacity) follow in atomic order. Fill-once.
        violation_write_index: Monotonic violation counter, shape [1], int32. Incremented on every violation
            regardless of ring capacity.
    """

    violation_ring: torch.Tensor
    violation_write_index: torch.Tensor

    @classmethod
    def allocate(cls, *, ring_capacity: int, device: torch.device) -> ViolationLog:
        if ring_capacity <= 0:
            raise ValueError(
                f"kv-canary: ViolationLog ring_capacity must be positive, got {ring_capacity}"
            )
        return cls(
            violation_ring=torch.zeros(
                ring_capacity, VIOLATION_FIELDS, dtype=torch.int64, device=device
            ),
            violation_write_index=torch.zeros(1, dtype=torch.int32, device=device),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryDeviceState:
    """Device-side state owned by one CanaryManager instance.

    One instance per ModelRunner. Held on the same device as the KV pool. All tensors are allocated up
    front (sizes fixed by CanaryConfig + cuda-graph capture capacity) and reused across forward steps —
    no per-step allocation.

    Fields:
        violation_log: The single ViolationLog shared by every launch (head / tail / sweep × K / V ×
            FULL / SWA). All kernels atomicAdd into violation_log.violation_write_index and stamp their
            CanaryLaunchTag into each violation row.
        kernel_run_counters: Per-CanaryLaunchTag int64 counter array, shape [num_tags], device. The
            kernel itself does NOT index this array; runner takes a 1-element view at tag's slot (via
            CanaryEndpoint.kernel_run_counter_view) and hands a shape [1] tensor to the kernel,
            which atomicAdds 1 regardless of whether the plan had any active entry. Health watchdog
            reads this array to confirm "canary path actually ran".
        slot_run_counters: Per-CanaryLaunchTag int64 counter array, shape [num_tags], device. Same
            view-handed-to-kernel pattern as kernel_run_counters; each launch adds its active entry
            count to its slot. Used for periodic stats ("protected N tokens").
        enable_chain_position_assert: int32 [1] device flag gating the write kernel's chain-step
            write_position assert. allocate() defaults to 1; CanaryManager zeros it during
            __init__ for the warmup window and mark_init_finished() flips it back to 1.
        req_to_verify_expected_tokens: Optional int32 device tensor shape
            ``[req_to_token_alloc_size, max_context_len]``. Mirrors ReqToTokenPool layout;
            ``pool[req_idx, p]`` = source-of-truth token at logical position ``p`` for the
            req in slot ``req_idx``. Allocated only when
            ``CanaryConfig.enable_verify_token_assert`` is True. The plan-side entries
            kernel gathers from this pool (via ``kv_token_id_vs_position_offset`` per buffer
            group) into ``VerifyPlan.verify_expected_tokens``; the verify kernel then
            compares against each canary slot's stored token.
    """

    violation_log: ViolationLog
    kernel_run_counters: torch.Tensor
    slot_run_counters: torch.Tensor
    enable_chain_position_assert: torch.Tensor
    req_to_verify_expected_tokens: Optional[torch.Tensor]

    @classmethod
    def allocate(
        cls,
        *,
        config: CanaryConfig,
        device: torch.device,
        num_tags: int,
        req_to_token_alloc_size: Optional[int] = None,
        max_context_len: Optional[int] = None,
    ) -> CanaryDeviceState:
        if num_tags <= 0:
            raise ValueError(
                f"kv-canary: CanaryDeviceState num_tags must be positive, got {num_tags}"
            )
        violation_log = ViolationLog.allocate(
            ring_capacity=config.ring_capacity, device=device
        )
        kernel_run_counters = torch.zeros(num_tags, dtype=torch.int64, device=device)
        slot_run_counters = torch.zeros(num_tags, dtype=torch.int64, device=device)
        enable_chain_position_assert = torch.ones(1, dtype=torch.int32, device=device)
        if config.enable_verify_token_assert:
            if req_to_token_alloc_size is None or max_context_len is None:
                raise ValueError(
                    "kv-canary: CanaryDeviceState.allocate requires req_to_token_alloc_size "
                    "and max_context_len when CanaryConfig.enable_verify_token_assert is on"
                )
            req_to_verify_expected_tokens = torch.empty(
                (req_to_token_alloc_size, max_context_len),
                dtype=torch.int32,
                device=device,
            )
        else:
            req_to_verify_expected_tokens = None
        return cls(
            violation_log=violation_log,
            kernel_run_counters=kernel_run_counters,
            slot_run_counters=slot_run_counters,
            enable_chain_position_assert=enable_chain_position_assert,
            req_to_verify_expected_tokens=req_to_verify_expected_tokens,
        )
