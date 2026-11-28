# SPDX-License-Identifier: Apache-2.0
"""Build CPU adapter state off the scheduler, then commit on every TP rank."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

import torch
import torch.distributed as dist


class LoRAPreparationQueue:
    def __init__(self, prepare):
        self.prepare = prepare
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="lora-prepare"
        )
        self.pending = deque()
        self.cancelled = set()

    def __bool__(self):
        return bool(self.pending)

    def submit(self, request):
        future = self.executor.submit(
            copy_context().run, self.prepare, request.to_ref()
        )
        self.pending.append((request, future))

    def cancel(self, lora_id):
        for request, future in self.pending:
            if request.lora_id == lora_id:
                self.cancelled.add(lora_id)
                future.cancel()

    def poll(self, validate, cpu_group):
        if not self.pending:
            return None
        request, future = self.pending[0]
        ready = torch.tensor([future.done()], dtype=torch.int32, device="cpu")
        distributed = dist.is_initialized()
        if distributed:
            dist.all_reduce(ready, op=dist.ReduceOp.MIN, group=cpu_group)
        if not ready.item():
            return None
        prepared, error = None, None
        try:
            if request.lora_id in self.cancelled:
                raise RuntimeError("Adapter unloaded during CPU preparation")
            prepared = future.result()
            validate(request, prepared)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = [error]
        if distributed:
            errors = [None] * dist.get_world_size(cpu_group)
            dist.all_gather_object(errors, error, group=cpu_group)
        self.pending.popleft()
        self.cancelled.discard(request.lora_id)
        error = "; ".join(e for e in errors if e is not None) if any(errors) else None
        return request, prepared, error

    def close(self, wait=True):
        self.executor.shutdown(wait=wait, cancel_futures=True)
