from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.jit_kernel.kv_canary.verify import VIOLATION_FIELDS
from sglang.srt.kv_canary.config import CanaryConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class ViolationLog:
    """Global violation sink shared across all canary launches.

    One instance per canary runner — every launch (head / tail / sweep, K / V half, FULL / SWA group) writes
    into the same ring. The kernel_kind field stamped into each violation row identifies which launch fired
    (kernel_kind is a unique int per (head|tail|sweep, K|V, FULL|SWA) tuple; assigned by the runner).

    Ring capacity is sized generously (≥ 1024) so overflow is a non-concern in practice — violations are
    cold-path and the host raises at the first one anyway. atomicAdd contention on a single counter is also
    negligible since violation events are rare.

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
    def allocate(cls, *, ring_capacity: int, device: torch.device) -> "ViolationLog":
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

    def clear(self) -> None:
        """Reset to all-zero (forgets all past violations)."""
        self.violation_ring.zero_()
        self.violation_write_index.zero_()


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryDeviceState:
    """Device-side state owned by one CanaryRunner instance.

    One instance per ModelRunner. Held on the same device as the KV pool. All tensors are allocated up
    front (sizes fixed by CanaryConfig + cuda-graph capture capacity) and reused across forward steps —
    no per-step allocation.

    Fields:
        violation_log: The single ViolationLog shared by every launch (head / tail / sweep × K / V ×
            FULL / SWA). All kernels atomicAdd into violation_log.violation_write_index and stamp their
            CanaryLaunchTag into each violation row.
        kernel_run_counters: Per-CanaryLaunchTag int64 counter array, shape [num_tags], device. The
            kernel itself does NOT index this array; runner takes a 1-element view at tag's slot (via
            CanaryEndpoint.kernel_run_counter_view, §2.4) and hands a shape [1] tensor to the kernel,
            which atomicAdds 1 regardless of whether the plan had any active entry. Health watchdog
            reads this array to confirm "canary path actually ran".
        slot_run_counters: Per-CanaryLaunchTag int64 counter array, shape [num_tags], device. Same
            view-handed-to-kernel pattern as kernel_run_counters; each launch adds its active entry
            count to its slot. Used for periodic stats ("protected N tokens").
        allreduce_buf: Device byte buffer used for cross-rank allreduce of is_errored, shape [1], uint8.
            Only allocated when CanaryConfig.allreduce_violation_signal is True.
    """

    violation_log: ViolationLog
    kernel_run_counters: torch.Tensor
    slot_run_counters: torch.Tensor
    allreduce_buf: Optional[torch.Tensor]

    @classmethod
    def allocate(
        cls,
        *,
        config: CanaryConfig,
        device: torch.device,
        num_tags: int,
    ) -> "CanaryDeviceState":
        if num_tags <= 0:
            raise ValueError(
                f"kv-canary: CanaryDeviceState num_tags must be positive, got {num_tags}"
            )
        violation_log = ViolationLog.allocate(
            ring_capacity=config.ring_capacity, device=device
        )
        kernel_run_counters = torch.zeros(num_tags, dtype=torch.int64, device=device)
        slot_run_counters = torch.zeros(num_tags, dtype=torch.int64, device=device)
        allreduce_buf: Optional[torch.Tensor] = None
        if config.allreduce_violation_signal:
            allreduce_buf = torch.zeros(1, dtype=torch.uint8, device=device)
        return cls(
            violation_log=violation_log,
            kernel_run_counters=kernel_run_counters,
            slot_run_counters=slot_run_counters,
            allreduce_buf=allreduce_buf,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryHostState:
    """Pinned-host state owned by one CanaryRunner instance.

    All tensors are pinned-host (pin_memory=True) for async D2H transfers. One instance per
    ModelRunner, allocated at install time alongside CanaryDeviceState.

    Fields:
        violation_signal_host: Pinned-host uint8 scalar, shape [1]. async D2H snapshot of
            (violation_log.violation_write_index > 0). Snapshotted on a side stream at end_of_step N
            and consumed at start of step N+1 after pump_event.synchronize(); this 1-step delay is
            intentional — bug is detected one step later but forward path stays sync-free.
        allreduce_signal_host: Pinned-host uint8 scalar, shape [1]. async D2H mirror of allreduce_buf
            after the cross-rank reduction. Same 1-step delay pattern as violation_signal_host.
            Only allocated when CanaryConfig.allreduce_violation_signal is True.
        kernel_run_counters_host: Pinned-host int64 mirror of kernel_run_counters, shape [num_tags].
            Health watchdog stages a D2H copy here on the alt stream and reads on the next call.
        slot_run_counters_sum_host: Pinned-host int64 scalar, shape [1]. Stat path stages
            ``slot_run_counters.sum()`` here on the alt stream and reads on the next call.
        violation_write_index_host: Pinned-host int32 mirror of violation_log.violation_write_index,
            shape [1]. Stat path stages a D2H copy here on the alt stream and reads on the next call.
    """

    violation_signal_host: torch.Tensor
    allreduce_signal_host: Optional[torch.Tensor]
    kernel_run_counters_host: torch.Tensor
    slot_run_counters_sum_host: torch.Tensor
    violation_write_index_host: torch.Tensor

    @classmethod
    def allocate(
        cls,
        *,
        config: CanaryConfig,
        num_tags: int,
    ) -> "CanaryHostState":
        if num_tags <= 0:
            raise ValueError(
                f"kv-canary: CanaryHostState num_tags must be positive, got {num_tags}"
            )
        violation_signal_host = torch.zeros(1, dtype=torch.uint8, pin_memory=True)
        allreduce_signal_host: Optional[torch.Tensor] = None
        if config.allreduce_violation_signal:
            allreduce_signal_host = torch.zeros(1, dtype=torch.uint8, pin_memory=True)
        kernel_run_counters_host = torch.zeros(
            num_tags, dtype=torch.int64, pin_memory=True
        )
        slot_run_counters_sum_host = torch.zeros(1, dtype=torch.int64, pin_memory=True)
        violation_write_index_host = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        return cls(
            violation_signal_host=violation_signal_host,
            allreduce_signal_host=allreduce_signal_host,
            kernel_run_counters_host=kernel_run_counters_host,
            slot_run_counters_sum_host=slot_run_counters_sum_host,
            violation_write_index_host=violation_write_index_host,
        )
