"""Device-resident main KV buffers and row reuse for fused sparse attention."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_budget import (
        BudgetPlan,
    )


class ResidentKVPool:
    """Own the C buffers and the state that is invalidated on row reuse."""

    def __init__(self, budget: BudgetPlan, device: str | torch.device) -> None:
        self.device = device
        self.request_rows = budget.request_rows
        self.resident_rows = budget.resident_rows
        self.device_k_buffer: list[torch.Tensor]
        self.device_v_buffer: list[torch.Tensor]
        self.device_slot_map: list[torch.Tensor]
        self.short_resident_end: torch.Tensor
        self.last_prefix: torch.Tensor
        self.request_to_resident: torch.Tensor | None = None
        for name, spec in budget.resident_buffers.items():
            try:
                if name in ("device_k_buffer", "device_v_buffer", "device_slot_map"):
                    fill_value = -1 if name == "device_slot_map" else None
                    buffer = [
                        spec.allocate(device, fill_value) for _ in range(spec.count)
                    ]
                else:
                    fill_value = 0 if name == "request_to_resident" else -1
                    buffer = spec.allocate(device, fill_value)
                setattr(self, name, buffer)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to allocate sparse KV buffer "
                    f"{name}: req_capacity={budget.request_rows}, "
                    f"max_context_len={budget.context_len}, sparse_topk={budget.top_k}. "
                    "The sparse KV request capacity may be too large; set a smaller "
                    "--max-running-requests for sparse KV offload."
                ) from exc

        if self.request_to_resident is not None:
            self.clear()

    def reset_rows(
        self, req_ids: list[int], resident_ids: list[int] | None = None
    ) -> None:
        if not req_ids:
            return

        req_indices = torch.tensor(req_ids, dtype=torch.long).to(
            self.device, non_blocking=True
        )
        resident_indices = (
            req_indices
            if resident_ids is None
            else torch.tensor(resident_ids, dtype=torch.long).to(
                self.device, non_blocking=True
            )
        )
        self.short_resident_end.index_fill_(1, req_indices, -1)
        for slot_map in self.device_slot_map:
            slot_map.index_fill_(0, resident_indices, -1)
        self.last_prefix.index_fill_(0, resident_indices, -1)

    def clear(self) -> None:
        self._request_to_resident_cpu = [0] * self.request_rows
        self._free_resident_rows = list(range(self.resident_rows - 1, 0, -1))

    def available(self) -> int:
        return len(self._free_resident_rows)

    def release(self, req_id: int) -> None:
        resident_id = self._request_to_resident_cpu[req_id]
        if resident_id:
            self._request_to_resident_cpu[req_id] = 0
            self._free_resident_rows.append(resident_id)

    def invalidate_request(self, req_id: int) -> None:
        """Invalidate one request's C row after an external payload restore."""
        if not 0 < req_id < self.request_rows:
            return
        if self.request_to_resident is None:
            resident_id = req_id
            if resident_id >= self.resident_rows:
                return
        else:
            resident_id = self._request_to_resident_cpu[req_id]
            if not resident_id:
                return
        self.reset_rows([req_id], [resident_id])

    def admit(self, req_ids: list[int]) -> None:
        """Reset and bind new C rows on the forward consumer stream."""
        device_mapping = self.request_to_resident
        assert device_mapping is not None
        new_req_ids = [
            req_id
            for req_id in dict.fromkeys(req_ids)
            if not self._request_to_resident_cpu[req_id]
        ]
        if not new_req_ids:
            return
        count = len(new_req_ids)
        if count > len(self._free_resident_rows):
            raise RuntimeError(
                "PD decode admitted more requests than resident KV rows."
            )

        # Keep the CPU ownership unchanged if reset or publication raises.
        resident_ids = self._free_resident_rows[-count:][::-1]
        mapping = self._request_to_resident_cpu.copy()
        for req_id, resident_id in zip(new_req_ids, resident_ids):
            mapping[req_id] = resident_id
        self.reset_rows(new_req_ids, resident_ids)
        # Publish releases and new bindings in-place before graph replay.
        device_mapping.copy_(
            torch.tensor(mapping, dtype=device_mapping.dtype, device="cpu"),
            non_blocking=True,
        )
        self._request_to_resident_cpu = mapping
        del self._free_resident_rows[-count:]
