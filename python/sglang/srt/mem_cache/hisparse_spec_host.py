# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Opt-in eager host commit adapter; scheduler ownership hooks are still required.

Calls and request teardown must be serialized by the coordinator. The owner
resolver must return (request_object, generation). Teardown must drain/cancel
live transactions BEFORE freeing committed host pages, including reused tails.
Allocated page padding is addressable but never KV-valid until committed.
"""

from dataclasses import dataclass

import torch

from sglang.srt.mem_cache.hisparse_spec_lifecycle import (
    CommitPlan,
    CompletionFence,
    HostReservation,
)


@dataclass
class _Pending:
    owner: object
    old_len: int
    allocated: int
    snapshot: object
    mapping: object
    reservation: HostReservation
    plan: CommitPlan | None = None
    fence: CompletionFence | None = None
    published: bool = False
    quarantined: bool = False
    transfer_tensors: tuple = ()


class HiSparseSpecHostBackend:
    def __init__(self, coordinator, device_module, resolve_owner):
        if coordinator.mem_pool_host.page_size != 64:
            raise ValueError("speculative host commits require page64")
        self.coordinator = coordinator
        self.device_module = device_module
        self.resolve_owner = resolve_owner
        self._bindings = {}
        self._pending = {}
        self._latest = {}
        self.valid_lengths = {}

    def bind(self, key, owner, committed_length):
        """Register a transaction after admission; root KV starts at this length."""
        current, generation = self.resolve_owner(key.request_slot)
        if current is not owner or generation != key.request_generation:
            raise ValueError("stale request ownership")
        if key.request_slot in self._bindings:
            raise ValueError("request already has a bound host transaction")
        stamp = (key.request_generation, key.iteration_id)
        if stamp <= self._latest.get(key.request_slot, (-1, -1)):
            raise ValueError("stale host transaction")
        if type(committed_length) is not int or committed_length < 0:
            raise ValueError("invalid committed host length")
        prior = self.valid_lengths.get(key.request_slot)
        if prior is not None and prior[0] == key.request_generation:
            if prior[1] != committed_length:
                raise ValueError("committed host length changed")
        self._bindings[key.request_slot] = (key, owner, committed_length)
        self._latest[key.request_slot] = stamp
        self.valid_lengths[key.request_slot] = (
            key.request_generation,
            committed_length,
        )

    def _binding(self, key):
        binding = self._bindings.get(key.request_slot)
        current, generation = self.resolve_owner(key.request_slot)
        if binding is None or binding[0] != key:
            raise ValueError("unbound host transaction")
        if current is not binding[1] or generation != key.request_generation:
            raise ValueError("stale request ownership")
        return binding

    def allocate(self, key, positions):
        _, owner, old_len = self._binding(key)
        if key in self._pending:
            raise ValueError("host reservation already allocated")
        if not positions or positions != tuple(
            range(old_len, old_len + len(positions))
        ):
            raise ValueError("commit must be a contiguous accepted prefix")
        c = self.coordinator
        slot = key.request_slot
        allocated = int(c.req_to_host_pool_allocated_len[slot])
        end = ((positions[-1] + 64) // 64) * 64
        if allocated % 64 or old_len > allocated or end > c.req_to_host_pool.shape[1]:
            raise ValueError("invalid host allocation bounds")
        snapshot = c.req_to_host_pool[slot].clone()
        if (snapshot[:allocated] < 0).any().item():
            raise ValueError("missing allocated host page")
        mapping = snapshot.unsqueeze(0).clone()
        # Use the mixin page API directly so a failed private mapping write
        # still leaves every newly allocated row available for rollback.
        new_rows = None
        needed = max(0, end - allocated)
        if needed:
            new_rows = c.mem_pool_host.alloc_page(needed // 64)
            if new_rows is None:
                return None
        try:
            if new_rows is not None:
                mapping[0, allocated:end] = new_rows.to(mapping.device)
            host = tuple(mapping[0, list(positions)].tolist())
            rows = () if new_rows is None else tuple(new_rows.tolist())
            reservation = HostReservation(key, positions, host, rows)
            self._pending[key] = _Pending(
                owner, old_len, allocated, snapshot, mapping[0], reservation
            )
            return reservation
        except Exception:
            if new_rows is not None:
                c.mem_pool_host.free(new_rows)
            raise

    def copy(self, plan):
        self._binding(plan.key)
        pending = self._pending[plan.key]
        if pending.plan is not None or pending.quarantined:
            raise ValueError("host copy already submitted")
        if (
            plan.positions != pending.reservation.positions
            or plan.host_ids != pending.reservation.host_ids
        ):
            raise ValueError("copy does not match reservation")
        if len(plan.device_ids) != len(plan.host_ids):
            raise ValueError("copy source count mismatch")
        c, dm = self.coordinator, self.device_module
        stream = c.decode_backup_stream
        host = torch.tensor(
            plan.host_ids, dtype=torch.int64, device=c.req_to_host_pool.device
        )
        device = torch.tensor(
            plan.device_ids, dtype=torch.int64, device=c.req_to_host_pool.device
        )
        pending.transfer_tensors = (host, device)
        fence = CompletionFence(dm.Event())
        schedule_stream = dm.current_stream()
        try:
            with dm.stream(stream):
                stream.wait_stream(schedule_stream)
                if c.decode_producer_stream is not None:
                    stream.wait_stream(c.decode_producer_stream)
                c.mem_pool_host.backup_from_device_all_layer(
                    c.mem_pool_device, host, device, io_backend="kernel"
                )
                for tensor in (host, device):
                    if tensor.is_cuda:
                        tensor.record_stream(stream)
                fence.record(stream)
        except Exception:
            # The all-layer routine can fail after submitting some layers.
            # A failed drain quarantines ownership: release must refuse to free.
            pending.quarantined = True
            stream.synchronize()
            pending.quarantined = False
            raise
        pending.plan, pending.fence = plan, fence
        return fence

    def publish(self, plan, reservation):
        self._binding(plan.key)
        pending = self._pending[plan.key]
        if reservation is not pending.reservation or plan != pending.plan:
            raise ValueError("publication does not match submitted copy")
        if pending.published:
            return
        if pending.fence is None or not pending.fence.query():
            raise ValueError("host copy is incomplete")
        c, slot = self.coordinator, plan.key.request_slot
        if int(
            c.req_to_host_pool_allocated_len[slot]
        ) != pending.allocated or not torch.equal(
            c.req_to_host_pool[slot], pending.snapshot
        ):
            raise ValueError("request host mapping changed during commit")
        end = max(pending.allocated, ((plan.positions[-1] + 64) // 64) * 64)
        try:
            c.req_to_host_pool[slot].copy_(pending.mapping)
            c.req_to_host_pool_allocated_len[slot] = end
        except Exception:
            c.req_to_host_pool[slot].copy_(pending.snapshot)
            c.req_to_host_pool_allocated_len[slot] = pending.allocated
            raise
        self.valid_lengths[slot] = (plan.key.request_generation, plan.positions[-1] + 1)
        pending.published = True

    def release(self, reservation):
        pending = self._pending.get(reservation.key)
        if pending is None:
            return
        if reservation is not pending.reservation:
            raise ValueError("foreign host reservation")
        if pending.quarantined or (
            pending.fence is not None and not pending.fence.query()
        ):
            raise ValueError("cannot free host pages before copy completion")
        if not pending.published and reservation.new_page_rows:
            rows = torch.tensor(reservation.new_page_rows, dtype=torch.int64)
            self.coordinator.mem_pool_host.free(rows)
        del self._pending[reservation.key]
        self._bindings.pop(reservation.key.request_slot, None)

    def finish(self, key):
        """After lifecycle retirement, forget published bookkeeping or empty bind."""
        pending = self._pending.get(key)
        if pending is not None:
            if not pending.published:
                raise ValueError("unpublished reservation must be cancelled first")
            self.release(pending.reservation)
        elif self._bindings.get(key.request_slot, (None,))[0] == key:
            del self._bindings[key.request_slot]
