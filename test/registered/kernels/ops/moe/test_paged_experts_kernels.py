import threading
import unittest

import torch

from sglang.kernels.ops.moe.paged_experts import (
    paged_experts_decide,
    paged_experts_gather,
    paged_experts_host_device_pointer,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

E, K = 64, 16


class _State:
    """Residency state on the GPU, slots starting with experts 0..K-1."""

    def __init__(self):
        i32 = dict(dtype=torch.int32, device="cuda")
        self.step = torch.zeros(1, **i32)
        self.slot_expert = torch.arange(K, **i32)
        self.expert_slot = torch.full((E,), -1, **i32)
        self.expert_slot[:K] = self.slot_expert
        self.slot_lastuse = torch.zeros(K, **i32)
        self.src = torch.zeros(K, **i32)
        self.dst = torch.zeros(K, **i32)
        self.count = torch.zeros(1, **i32)

    def decide(self, topk, weights=None):
        """Run the decide kernel; return the runner inputs it wrote (slots, weights)."""
        if weights is None:
            weights = torch.ones(topk.shape, device="cuda")
        run_ids, run_w = torch.empty_like(topk), torch.empty_like(weights)
        paged_experts_decide(
            topk,
            weights,
            self.step,
            self.slot_expert,
            self.expert_slot,
            self.slot_lastuse,
            self.src,
            self.dst,
            self.count,
            run_ids,
            run_w,
        )
        return run_ids, run_w


class _Reference:
    """The decision the kernel must make, in Python."""

    def __init__(self):
        self.step = 0
        self.slot_expert = list(range(K))
        self.slot_lastuse = [0] * K

    def decide(self, topk):
        self.step += 1
        for e in topk:
            if e in self.slot_expert:
                self.slot_lastuse[self.slot_expert.index(e)] = self.step
        plan = []
        for e in topk:
            if e < 0 or e in self.slot_expert:
                continue
            free = [s for s in range(K) if self.slot_lastuse[s] != self.step]
            victim = min(free, key=lambda s: (self.slot_lastuse[s], s))
            self.slot_expert[victim] = e
            self.slot_lastuse[victim] = self.step
            plan.append((e, victim))
        return plan


def _random_topk(generator, num_entries):
    topk = torch.randint(0, E, (num_entries,), generator=generator, dtype=torch.int32)
    topk[torch.rand(num_entries, generator=generator) < 0.1] = -1  # padding
    return topk


@unittest.skipUnless(torch.cuda.is_available(), "paged experts kernels need CUDA")
class TestPagedExpertsKernels(CustomTestCase):
    def assert_state(self, state, ref, plan):
        n = int(state.count.item())
        got_plan = list(zip(state.src[:n].tolist(), state.dst[:n].tolist()))
        self.assertEqual(got_plan, plan)
        self.assertEqual(state.slot_expert.tolist(), ref.slot_expert)
        self.assertEqual(state.slot_lastuse.tolist(), ref.slot_lastuse)
        expert_slot = [-1] * E
        for s, e in enumerate(ref.slot_expert):
            expert_slot[e] = s
        self.assertEqual(state.expert_slot.tolist(), expert_slot)

    def test_decide_matches_reference(self):
        """The LRU decision and plan, and the runner inputs: each entry's slot and weight,
        slot 0 and weight 0 for padding."""
        generator = torch.Generator().manual_seed(0)
        state, ref = _State(), _Reference()
        for _ in range(200):
            topk = _random_topk(generator, int(torch.randint(1, K + 1, (1,))))
            weights = torch.rand(topk.shape, generator=generator)
            plan = ref.decide(topk.tolist())
            run_ids, run_w = state.decide(topk.cuda(), weights.cuda())
            self.assert_state(state, ref, plan)
            routed = topk >= 0
            slots = [ref.slot_expert.index(e) if e >= 0 else 0 for e in topk.tolist()]
            self.assertEqual(run_ids.tolist(), slots)
            self.assertTrue(torch.equal(run_w.cpu(), weights.masked_fill(~routed, 0)))

    def test_decide_pads_ids_outside_the_experts(self):
        topk = torch.tensor([3, E, -1, 3, E + 5], dtype=torch.int32, device="cuda")
        weights = torch.arange(1, 6, dtype=torch.float32, device="cuda")
        state = _State()
        run_ids, run_w = state.decide(topk, weights)
        self.assertEqual(run_ids.tolist(), [3, 0, 0, 3, 0])
        self.assertEqual(run_w.tolist(), [1.0, 0.0, 0.0, 4.0, 0.0])
        self.assertEqual(state.count.item(), 0)

    def test_decide_rejects_more_entries_than_slots(self):
        with self.assertRaises(Exception):
            _State().decide(torch.zeros(K + 1, dtype=torch.int32, device="cuda"))

    def test_gather_copies_the_plan_for_every_tensor(self):
        hosts = [
            torch.randn(E, 8, 32, dtype=torch.bfloat16).pin_memory(),
            torch.randint(0, 100, (E, 12), dtype=torch.int32).pin_memory(),
            torch.randn(E, dtype=torch.float32).pin_memory(),  # 4-byte rows
            torch.randn(E, 3, dtype=torch.float32).pin_memory(),  # 12-byte rows
            torch.randn(E, dtype=torch.bfloat16).pin_memory(),  # 2-byte rows
            torch.randn(E, 3, dtype=torch.bfloat16).pin_memory(),  # 6-byte rows
        ]
        gpus = [
            torch.zeros(K, *h.shape[1:], dtype=h.dtype, device="cuda") for h in hosts
        ]
        args = _gather_args(hosts, gpus)
        src = torch.tensor([5, 63, 17] + [0] * (K - 3), dtype=torch.int32).cuda()
        dst = torch.tensor([2, 0, 15] + [0] * (K - 3), dtype=torch.int32).cuda()
        count = torch.tensor([3], dtype=torch.int32, device="cuda")

        paged_experts_gather(*args, src, dst, count)

        for host, gpu in zip(hosts, gpus):
            for e, s in ((5, 2), (63, 0), (17, 15)):
                self.assertTrue(torch.equal(gpu[s].cpu(), host[e]))
            self.assertFalse(gpu[1].any())  # untouched

    def test_store_pins_its_exact_size_and_the_gather_reads_it(self):
        """The host store pins exactly its size (torch's pinned allocator rounds up to a power
        of two), and the gather kernel reads it through the GPU."""
        from sglang.srt.layers.moe.paged_experts.store import HostExpertStore

        gpu = torch.zeros(K, 3, 1000, dtype=torch.bfloat16, device="cuda")
        host = HostExpertStore({"w": ((3, 1000), torch.bfloat16)}, E).host["w"]
        host.copy_(torch.randn(E, 3, 1000, dtype=torch.bfloat16))
        self.assertTrue(host.is_pinned())
        self.assertEqual(host.untyped_storage().nbytes(), E * 3 * 1000 * 2)
        src = torch.tensor([5, 63] + [0] * (K - 2), dtype=torch.int32).cuda()
        dst = torch.tensor([2, 0] + [0] * (K - 2), dtype=torch.int32).cuda()
        count = torch.tensor([2], dtype=torch.int32, device="cuda")

        paged_experts_gather(*_gather_args([host], [gpu]), src, dst, count)

        self.assertTrue(torch.equal(gpu[2].cpu(), host[5]))
        self.assertTrue(torch.equal(gpu[0].cpu(), host[63]))

    def test_store_is_on_the_host_under_a_default_cuda_device(self):
        """Models are constructed under ``torch.device("cuda")``; a store created there (by
        a format that pages its loaded rows as they are) is still pinned host memory."""
        from sglang.srt.layers.moe.paged_experts.store import HostExpertStore

        with torch.device("cuda"):
            host = HostExpertStore({"w": ((3, 1000), torch.bfloat16)}, E).host["w"]
        self.assertEqual(host.device.type, "cpu")
        self.assertTrue(host.is_pinned())

    def test_store_unregisters_its_pages_once_the_last_tensor_is_freed(self):
        """The pinned pages stay registered while any tensor on them lives (here a slice that
        outlives the store and its tensor), and are unregistered once the last one is freed."""
        import gc

        from sglang.srt.layers.moe.paged_experts.store import HostExpertStore

        gpu = torch.zeros(K, 3, 1000, dtype=torch.bfloat16, device="cuda")
        store = HostExpertStore({"w": ((3, 1000), torch.bfloat16)}, E)
        ptr = store.host["w"].data_ptr()
        expert = store.host["w"][5]
        gpu[0].copy_(expert, non_blocking=True)  # a read still queued when it is freed
        del store
        gc.collect()
        self.assertTrue(expert.is_pinned())

        del expert
        gc.collect()
        # Unregistering again must fail: the pages are no longer registered. In a thread of its
        # own, as CUDA keeps the error as the thread's last error for a later call to report.
        rc = []
        unregister = lambda: rc.append(int(torch.cuda.cudart().cudaHostUnregister(ptr)))
        thread = threading.Thread(target=unregister)
        thread.start()
        thread.join()
        self.assertNotEqual(rc[0], 0)

    def test_expert_cache_reads_into_the_pinned_store_byte_for_byte(self):
        """The expert cache reads with direct I/O straight into the pinned store's rows of
        whole pages and through its bounce buffer for the others, byte for byte. Negative
        control: a byte changed in the file shows in that expert's row, and only there."""
        import mmap
        import os
        import tempfile
        from unittest.mock import patch

        from sglang.srt.layers.moe.paged_experts import expert_cache
        from sglang.srt.layers.moe.paged_experts.store import (
            HostExpertStore,
            same_bytes,
        )

        rows = {"w13": ((2, 2048), torch.bfloat16), "scales": ((3, 5), torch.float32)}
        source = {
            "w13": torch.randn(E, 2, 2048).bfloat16(),
            "scales": torch.randn(E, 3, 5),
        }
        opened = []
        open_direct = expert_cache._open_direct

        def recorded(*args):
            fd, direct, result = open_direct(*args)
            opened.append(direct)
            return fd, direct, result

        expert_cache._write_failed = False
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(expert_cache, "_FREE_MARGIN_BYTES", 0),
            patch.object(expert_cache, "_open_direct", recorded),
        ):
            entry = expert_cache.ExpertCache(
                os.path.join(folder, "layer-0.experts"), {"key": "k"}
            )
            entry.write(source)
            store = HostExpertStore(rows, E)
            self.assertTrue(entry.read(store.host))
            host = store.host["w13"]
            self.assertTrue(host.is_pinned())
            self.assertEqual(host[1].data_ptr() % mmap.PAGESIZE, 0)  # read straight
            for name in rows:
                self.assertTrue(same_bytes(store.host[name], source[name]), name)

            layout = expert_cache.ObjectLayout.of_host(source)
            with open(entry.path, "r+b") as f:
                f.seek(expert_cache._HEADER_BYTES + 3 * layout.object_bytes)
                f.write(b"\x7f\x7f")
            again = HostExpertStore(rows, E)
            self.assertTrue(entry.read(again.host))
            others = [e for e in range(E) if e != 3]
            self.assertFalse(same_bytes(again.host["w13"][3], source["w13"][3]))
            self.assertTrue(
                same_bytes(again.host["w13"][others], source["w13"][others])
            )
        if not all(opened):
            self.skipTest(
                f"{tempfile.gettempdir()} refuses O_DIRECT: read through the page cache"
            )

    def test_graph_replays_decide_on_new_routing(self):
        """A captured decide + gather plans and pages each replay's routing, not the routing
        seen at capture."""
        host = torch.randn(E, 64, dtype=torch.float32).pin_memory()
        gpu = torch.zeros(K, 64, device="cuda")
        gpu.copy_(host[:K])
        args = _gather_args([host], [gpu])
        state, ref = _State(), _Reference()
        topk = torch.zeros(K // 2, dtype=torch.int32, device="cuda")

        def step():
            state.decide(topk)
            paged_experts_gather(*args, state.src, state.dst, state.count)

        topk.copy_(torch.arange(K // 2, dtype=torch.int32))
        ref.decide(topk.tolist())
        step()  # warm up outside the capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        ref.decide(topk.tolist())  # capture does not run the kernels, ...
        graph.replay()  # ... so this replay is the second decide of the same routing

        generator = torch.Generator().manual_seed(1)
        for _ in range(20):
            routing = _random_topk(generator, K // 2)
            topk.copy_(routing)
            plan = ref.decide(routing.tolist())
            graph.replay()
            self.assert_state(state, ref, plan)
            for s, e in enumerate(ref.slot_expert):
                self.assertTrue(torch.equal(gpu[s].cpu(), host[e]))


class TestPooledSlotTables(CustomTestCase):
    def test_a_pooled_table_returns_its_memory_once_dropped(self):
        """``auto`` keeps the K = top_k slot tables in a pool of their own so that freeing them
        returns their segments. Negative control: while the pool lives, they stay reserved."""
        from sglang.srt.layers.moe.paged_experts.method import pooled_empty

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        before = torch.cuda.memory_reserved()
        table, pool = pooled_empty(64 << 20, torch.device("cuda"))
        self.assertGreaterEqual(torch.cuda.memory_reserved() - before, 64 << 20)
        del table
        torch.cuda.empty_cache()
        self.assertGreaterEqual(torch.cuda.memory_reserved() - before, 64 << 20)
        del pool
        torch.cuda.empty_cache()
        self.assertLess(torch.cuda.memory_reserved() - before, 64 << 20)

    def test_released_slot_tables_return_their_memory(self):
        """A layer that auto has sized drops its K = top_k tables (``release_slots``) and their
        memory goes back. Negative control: a view of a table kept elsewhere holds it."""
        from types import SimpleNamespace

        from sglang.srt.layers.moe.paged_experts.method import (
            PagedExpertsMoEMethod,
            _slot_table_bytes,
            pooled_empty,
        )

        host = torch.randn(E, 2, 1 << 17)  # 1 MiB per expert
        nbytes = K * host[0].numel() * host.element_size()
        for keep_a_view in (False, True):
            with self.subTest(keep_a_view=keep_a_view):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                before = torch.cuda.memory_reserved()
                method = PagedExpertsMoEMethod(None, None, E, K, auto_size=True)
                method.store = SimpleNamespace(host={"w13_weight": host})
                layer = torch.nn.Module()
                layer.w13_weight = torch.nn.Parameter(
                    torch.empty(0, *host.shape[1:], device="cuda"), requires_grad=False
                )
                slab, method._slot_pool = pooled_empty(
                    _slot_table_bytes(method, K), torch.device("cuda")
                )
                method._place(layer, K, slab, 0)
                self.assertTrue(torch.equal(layer.w13_weight.cpu(), host[:K]))
                del slab
                kept = layer.w13_weight.data[0] if keep_a_view else None
                method.release_slots(layer)
                torch.cuda.empty_cache()
                held = torch.cuda.memory_reserved() - before
                self.assertEqual(held >= nbytes, keep_a_view)
                del kept


def _gather_args(hosts, gpus):
    return [
        torch.tensor(v, dtype=torch.int64, device="cuda")
        for v in (
            [paged_experts_host_device_pointer(h) for h in hosts],
            [g.data_ptr() for g in gpus],
            [h[0].numel() * h.element_size() for h in hosts],
        )
    ]


if __name__ == "__main__":
    unittest.main()
