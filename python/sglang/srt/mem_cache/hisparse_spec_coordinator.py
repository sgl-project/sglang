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
"""Opt-in eager HiSparse verifier ownership and union staging.

Scheduler calls prepare on every iteration (including retained logical pages),
registers each reader before submission, records its fence after the last read,
then commits accepted input KV and retires after draft extension. All methods
are serialized on the scheduling thread. Tables are retained until retirement.
"""

from dataclasses import dataclass, field

import torch

from sglang.srt.mem_cache.hisparse_spec_host import HiSparseSpecHostBackend
from sglang.srt.mem_cache.hisparse_spec_lifecycle import (
    CompletionFence,
    SpeculativeKVLifecycle,
)
from sglang.srt.mem_cache.hisparse_spec_state import SpecTxnKey, plan_union


@dataclass
class _Owner:
    req: object
    generation: int
    iteration: int = 0
    key: object = None
    arena: object = None
    old_len: int = 0
    tables: dict = field(default_factory=dict)
    readers: list = field(default_factory=list)
    failed: bool = False
    verified: bool = False


class HiSparseSpecCoordinator:
    def __init__(self, coordinator, device_module):
        c = coordinator
        if (
            c.is_dsv4_hisparse
            or c.is_m3_hisparse
            or c.compress_ratio != 1
            or c.mem_pool_device.page_size != 64
        ):
            raise ValueError("eager speculative HiSparse requires page64 MLA")
        self.c, self.dm = c, device_module
        self.owners = {}
        self.generations = {}
        self.backend = HiSparseSpecHostBackend(c, device_module, self.resolve_owner)
        self.lifecycle = SpeculativeKVLifecycle(
            c.token_to_kv_pool_allocator, self.backend
        )

    def resolve_owner(self, slot):
        owner = self.owners.get(slot)
        return (None, -1) if owner is None else (owner.req, owner.generation)

    def admit(self, req):
        slot = req.kv.req_pool_idx
        owner = self.owners.get(slot)
        if owner is not None:
            if owner.req is not req:
                raise ValueError("request slot reused before teardown")
            return owner
        generation = self.generations.get(slot, -1) + 1
        self.generations[slot] = generation
        owner = self.owners[slot] = _Owner(req, generation)
        return owner

    def _get(self, req, key=None):
        owner = self.owners.get(req.kv.req_pool_idx)
        if owner is None or owner.req is not req:
            raise ValueError("stale request owner")
        if key is not None and owner.key != key:
            raise ValueError("stale speculative callback")
        return owner

    def prepare(self, req, old_kv_len, logical_write_ids, reserved_rows):
        if req.hisparse_staging:
            raise ValueError("request staging has not completed")
        owner = self.admit(req)
        if owner.key is not None:
            raise ValueError("previous verifier readers still own residency")
        self.c._prepare_speculative_stream()
        owner.iteration += 1
        key = SpecTxnKey(req.kv.req_pool_idx, owner.generation, owner.iteration)
        self.backend.bind(key, req, old_kv_len)
        try:
            # Accepted KV is durable in the host pool, but a short request may
            # still own only its initial page of hot slots. Grow before any
            # new arena/readers; the stream boundary above drained prior users.
            c, slot = self.c, key.request_slot
            old_cap = int(c.req_device_buffer_size[slot])
            needed = min(old_kv_len, c.device_buffer_size)
            if old_cap < needed:
                lengths_cpu = torch.tensor([needed], dtype=torch.int64)
                slots_cpu = torch.tensor([slot], dtype=torch.int64)
                c._grow_device_buffers(
                    lengths_cpu.to(c.device),
                    slots_cpu.to(c.device),
                    lengths_cpu,
                    slots_cpu,
                )
                new_cap = min(int(c.req_device_buffer_size[slot]), c.device_buffer_size)
                # Initial allocation labels even unallocated padding with an
                # arange. New physical slots contain no committed KV yet: force
                # host misses on every layer, retaining existing hot mappings.
                c.req_device_buffer_tokens[:, slot, old_cap:new_cap] = -1
            arena = self.lifecycle.begin(
                key, old_kv_len, logical_write_ids, reserved_rows
            )
        except Exception:
            self.backend.finish(key)
            raise
        owner.key, owner.arena, owner.old_len = key, arena, old_kv_len
        return key

    def add_reader(self, req, key):
        owner = self._get(req, key)
        fence = CompletionFence(self.dm.Event())
        self.lifecycle.add_reader(key, fence)
        owner.readers.append(fence)
        return fence

    def stage_layer(self, req, key, layer_id, rows):
        """Stage all real rows for ONE request/layer on the current stream.

        Rows carry request-relative positions and -1 padding. The caller must
        register/record attention and draft reader fences. No shared-index
        prefetch or decode scratch is used: each layer retains its own table.
        """
        owner = self._get(req, key)
        if owner.failed or owner.verified:
            raise ValueError("staging requires an unverified healthy transaction")
        if layer_id in owner.tables:
            raise ValueError("layer already staged for this transaction")
        c, slot = self.c, key.request_slot
        cap = min(int(c.req_device_buffer_size[slot]), c.device_buffer_size)
        tokens = c.req_device_buffer_tokens[layer_id, slot, :cap].tolist()
        # alloc_device_buffer labels its padding; padding is not valid KV.
        tokens = [p if 0 <= p < owner.old_len else -1 for p in tokens]
        slots = c.req_device_buffer_token_locs[layer_id, slot, :cap].tolist()
        order = [i for i in c.lru_slots[layer_id, slot].tolist() if i < cap]
        host = c.req_to_host_pool[slot, : owner.old_len].tolist()
        plan = plan_union(
            key=key,
            layer_id=layer_id,
            hot_key=key,
            hot_layer_id=layer_id,
            old_kv_len=owner.old_len,
            rows=rows,
            host_slots={p: h for p, h in enumerate(host) if h >= 0},
            hot_tokens=tokens,
            hot_slots=slots,
            victim_order=order,
            arena=owner.arena,
        )
        table = torch.tensor(plan.row_device_tables, dtype=torch.int32, device=c.device)
        # Retain submission fence and tensors before any copy can be enqueued.
        fence = self.add_reader(req, key)
        src = torch.tensor([plan.miss_src], dtype=torch.int64, device=c.device)
        dst = torch.tensor([plan.miss_dst], dtype=torch.int32, device=c.device)
        count = torch.tensor([len(plan.miss_src)], dtype=torch.int32, device=c.device)
        real = torch.ones(1, dtype=torch.int32, device=c.device)
        owner.tables[layer_id] = (table, plan, src, dst, count, real)
        try:
            if plan.miss_src:
                c._copy_speculative_union(layer_id, src, dst, count, real)
            c.req_device_buffer_tokens[layer_id, slot, :cap] = torch.tensor(
                plan.pending_hot_tokens, dtype=torch.int32, device=c.device
            )
        except Exception:
            owner.failed = True
            # A partial transfer may have overwritten old victim contents.
            # Invalidate the snapshot so cancellation/retry cannot claim hits.
            c.req_device_buffer_tokens[layer_id, slot, :cap] = -1
            raise
        finally:
            # Covers partial copy submission too; teardown retains ownership.
            fence.record(self.dm.current_stream())
        return table

    def verified(self, req, key):
        owner = self._get(req, key)
        if owner.failed:
            raise ValueError("failed staging cannot be verified")
        self.lifecycle.verified(key)
        owner.verified = True

    def commit(self, req, key, accept_len):
        self._get(req, key)
        try:
            return self.lifecycle.commit(key, accept_len)
        except Exception:
            # Allocation/submission failures can cancel synchronously inside
            # lifecycle.commit. Keep the external owner registry consistent.
            if not self.lifecycle.is_active(key):
                self._finish(self._get(req, key), key)
            raise

    def retire(self, req, key, cancel=False):
        owner = self._get(req, key)
        done = self.lifecycle.cancel(key) if cancel else self.lifecycle.release(key)
        if done:
            self._finish(owner, key)
        return done

    def _finish(self, owner, key):
        self.backend.finish(key)
        owner.key = owner.arena = None
        owner.failed = owner.verified = False
        owner.tables.clear()
        owner.readers.clear()

    def teardown(self, req):
        owner = self._get(req)
        if owner.key is not None:
            # Active-only conservative drain, including consumer streams whose
            # CompletionFences may have been recorded outside the coordinator.
            self.dm.synchronize()
            if not self.retire(req, owner.key, cancel=True):
                raise RuntimeError("speculative readers are not recorded/completed")
        self.backend.valid_lengths.pop(req.kv.req_pool_idx, None)
        del self.owners[req.kv.req_pool_idx]

    def destroy(self):
        for owner in list(self.owners.values()):
            self.teardown(owner.req)
