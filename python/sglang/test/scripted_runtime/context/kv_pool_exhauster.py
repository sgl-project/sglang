from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch

    from sglang.srt.managers.scheduler import Scheduler


class ScriptedKvPoolExhauster:

    def __init__(self, scheduler: "Scheduler") -> None:
        self.scheduler = scheduler
        self._held: List["torch.Tensor"] = []

    def exhaust(self, *, leave_pages: int) -> None:
        allocator = self.scheduler.token_to_kv_pool_allocator

        leave_tokens = leave_pages * self.scheduler.page_size
        need = allocator.available_size() - leave_tokens
        if need <= 0:
            return

        held = allocator.alloc(need)
        assert (
            held is not None
        ), f"exhaust_kv: allocator could not grab {need} tokens to create pressure"
        self._held.append(held)

    def release(self) -> None:
        for held in self._held:
            self.scheduler.token_to_kv_pool_allocator.free(held)
        self._held.clear()
