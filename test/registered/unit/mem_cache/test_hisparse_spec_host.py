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
import contextlib
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.hisparse_spec_host import HiSparseSpecHostBackend
from sglang.srt.mem_cache.hisparse_spec_lifecycle import CommitPlan
from sglang.srt.mem_cache.hisparse_spec_state import SpecTxnKey
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class Event:
    done = False

    def record(self, stream):
        self.recorded = stream

    def query(self):
        return self.done


class Stream:
    drained = 0

    def wait_stream(self, other):
        pass

    def synchronize(self):
        self.drained += 1


class Pool:
    page_size = 64

    def __init__(self):
        self.freed = []
        self.copies = []
        self.fail = False

    def alloc_page(self, count):
        return torch.arange(512, 512 + count * 64)

    def free(self, rows):
        self.freed.append(tuple(rows.tolist()))

    def backup_from_device_all_layer(self, pool, host, device, io_backend):
        self.copies.append((host.tolist(), device.tolist()))
        if self.fail:
            raise RuntimeError("partially submitted")


class HostTests(unittest.TestCase):
    def setUp(self):
        self.owner = object()
        self.generation = 1
        self.pool = Pool()
        self.stream = Stream()
        self.event = Event()
        self.c = SimpleNamespace(
            mem_pool_host=self.pool,
            mem_pool_device=object(),
            req_to_host_pool=torch.full((1, 256), -1, dtype=torch.int64),
            req_to_host_pool_allocated_len=torch.tensor([64]),
            decode_backup_stream=self.stream,
            decode_producer_stream=Stream(),
        )
        self.c.req_to_host_pool[0, :64] = torch.arange(128, 192)
        dm = SimpleNamespace(
            Event=lambda: self.event,
            stream=lambda _: contextlib.nullcontext(),
            current_stream=lambda: Stream(),
        )
        self.b = HiSparseSpecHostBackend(
            self.c, dm, lambda _: (self.owner, self.generation)
        )
        self.key = SpecTxnKey(0, 1, 1)
        self.b.bind(self.key, self.owner, 63)

    def reserve(self):
        r = self.b.allocate(self.key, (63, 64))
        p = CommitPlan(self.key, (63, 64), (900, 901), (70, 71), r.host_ids)
        return r, p

    def test_private_pages_publish_validity(self):
        r, p = self.reserve()
        self.assertEqual(r.host_ids, (191, 512))
        self.assertEqual(len(r.new_page_rows), 64)
        self.assertEqual(int(self.c.req_to_host_pool[0, 64]), -1)
        fence = self.b.copy(p)
        self.assertFalse(fence.query())
        with self.assertRaises(ValueError):
            self.b.publish(p, r)
        self.event.done = True
        self.b.publish(p, r)
        self.assertEqual(self.b.valid_lengths[0], (1, 65))
        self.assertEqual(int(self.c.req_to_host_pool_allocated_len[0]), 128)
        self.assertEqual(int(self.c.req_to_host_pool[0, 127]), 575)
        self.assertEqual(self.pool.copies, [([191, 512], [70, 71])])
        self.b.finish(self.key)
        self.assertEqual(self.pool.freed, [])

    def test_cancel_and_stale_owner_only_free_new_pages(self):
        r, p = self.reserve()
        self.b.copy(p)
        with self.assertRaises(ValueError):
            self.b.release(r)
        self.event.done = True
        self.owner = object()
        self.generation = 2
        with self.assertRaises(ValueError):
            self.b.publish(p, r)
        self.b.release(r)
        self.b.release(r)
        self.assertEqual(self.pool.freed, [tuple(range(512, 576))])
        self.assertEqual(int(self.c.req_to_host_pool[0, 63]), 191)
        self.assertEqual(int(self.c.req_to_host_pool[0, 64]), -1)

    def test_partial_copy_drained(self):
        r, p = self.reserve()
        self.pool.fail = True
        with self.assertRaises(RuntimeError):
            self.b.copy(p)
        self.assertEqual(self.stream.drained, 1)
        self.b.release(r)
        self.assertEqual(len(self.pool.freed[0]), 64)

    def test_failed_drain_quarantines_pages_and_indices(self):
        r, p = self.reserve()
        self.pool.fail = True

        def fail_drain():
            raise RuntimeError("device failure")

        self.stream.synchronize = fail_drain
        with self.assertRaises(RuntimeError):
            self.b.copy(p)
        with self.assertRaises(ValueError):
            self.b.release(r)
        self.assertEqual(len(self.b._pending[self.key].transfer_tensors), 2)
        self.assertEqual(self.pool.freed, [])

    def test_changed_mapping_cannot_publish(self):
        r, p = self.reserve()
        self.b.copy(p)
        self.event.done = True
        self.c.req_to_host_pool[0, 0] = 1000
        with self.assertRaises(ValueError):
            self.b.publish(p, r)
        self.b.release(r)
        self.assertEqual(self.b.valid_lengths[0], (1, 63))

    def test_reused_tail_has_no_new_owned_pages(self):
        r = self.b.allocate(self.key, (63,))
        self.assertEqual(r.new_page_rows, ())
        self.b.release(r)
        self.assertEqual(self.pool.freed, [])


if __name__ == "__main__":
    unittest.main()
