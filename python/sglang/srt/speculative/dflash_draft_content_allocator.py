"""Allocator for the device-only DFlash radix content subrange."""

from __future__ import annotations

import torch


class DFlashDraftContentAllocator:
    """Own exactly one fixed, contiguous subrange of draft KV row indices."""

    def __init__(self, *, start: int, size: int, device: str | torch.device):
        if start < 0 or size < 0:
            raise ValueError(
                f"invalid DFlash content range: start={start}, size={size}"
            )
        self.start = int(start)
        self.size = int(size)
        self.end = self.start + self.size
        self.device = torch.device(device)
        self.clear()

    def clear(self) -> None:
        self._free_rows = torch.arange(
            self.start, self.end, dtype=torch.int64, device="cpu"
        )
        self._is_free = torch.ones(self.size, dtype=torch.bool, device="cpu")
        # Hold returned tensors strongly so their Python ids remain unique while
        # the one-shot unstaged allocation leases are outstanding.
        self._leases: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def available_size(self) -> int:
        return int(self._free_rows.numel())

    def alloc(self, need_size: int) -> torch.Tensor | None:
        need_size = int(need_size)
        if need_size < 0:
            raise ValueError(f"allocation size must be nonnegative, got {need_size}")
        if need_size > self.available_size():
            return None
        rows = self._free_rows[:need_size]
        self._free_rows = self._free_rows[need_size:]
        if rows.numel():
            offsets = rows - self.start
            if not bool(torch.all(self._is_free[offsets])):
                raise RuntimeError("DFlash content allocator free-list corruption")
            self._is_free[offsets] = False
        device_rows = rows.to(self.device)
        if device_rows.numel():
            self._leases[id(device_rows)] = (device_rows, rows.clone())
        return device_rows

    def _take_lease(self, rows: torch.Tensor) -> torch.Tensor:
        record = self._leases.get(id(rows))
        if record is None or record[0] is not rows:
            raise RuntimeError(
                "DFlash content rows are not a current one-shot allocation lease"
            )
        del self._leases[id(rows)]
        cpu_rows = record[1]
        if bool(torch.any(self._is_free[cpu_rows - self.start])):
            raise RuntimeError("DFlash content allocation lease refers to a free row")
        return cpu_rows

    def claim_lease(self, rows: torch.Tensor) -> None:
        """Consume exactly one fresh allocation lease for publication."""
        self._take_lease(rows)

    def free_lease(self, rows: torch.Tensor) -> None:
        """Return an allocation that failed before publication staging."""
        self._free_cpu_rows(self._take_lease(rows))

    def free(self, rows: torch.Tensor) -> None:
        record = self._leases.pop(id(rows), None)
        if record is not None and record[0] is rows:
            cpu_rows = record[1]
        else:
            cpu_rows = rows.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        if cpu_rows.numel() == 0:
            return
        # Arbitrary test/debug frees may overlap a larger lease. Production
        # unstaged cleanup uses free_lease and never invalidates a partial lease.
        for lease_id, (_, lease_rows) in list(self._leases.items()):
            if bool(torch.isin(lease_rows, cpu_rows).any()):
                del self._leases[lease_id]
        self._free_cpu_rows(cpu_rows)

    def _free_cpu_rows(self, rows: torch.Tensor) -> None:
        if torch.unique(rows).numel() != rows.numel():
            raise RuntimeError("DFlash content allocator received duplicate rows")
        if bool(torch.any(rows < self.start)) or bool(torch.any(rows >= self.end)):
            raise RuntimeError(
                "DFlash content row is outside allocator range: "
                f"valid=[{self.start},{self.end}), rows={rows.tolist()}"
            )
        offsets = rows - self.start
        if bool(torch.any(self._is_free[offsets])):
            raise RuntimeError(
                f"DFlash content row was freed twice: rows={rows.tolist()}"
            )
        self._is_free[offsets] = True
        self._free_rows = torch.cat((self._free_rows, rows))

    def assert_allocated(self, rows: torch.Tensor) -> None:
        if rows.is_cuda:
            # Tree publication can only consume a one-shot allocator lease, and
            # the match plan pins those rows against eviction. Keep the hot match
            # path sync-free while still rejecting a corrupted physical range.
            valid = (rows >= self.start) & (rows < self.end)
            torch._assert_async(
                torch.all(valid),
                "DFlash content source is outside the content subrange",
            )
            return
        rows = rows.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        if rows.numel() == 0:
            return
        if bool(torch.any(rows < self.start)) or bool(torch.any(rows >= self.end)):
            raise RuntimeError("DFlash content source is outside the content subrange")
        if bool(torch.any(self._is_free[rows - self.start])):
            raise RuntimeError("DFlash content source refers to an unallocated row")
