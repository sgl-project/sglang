import contextlib
import os
import tempfile
import threading
import unittest
import unittest.mock
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.moe.paged_experts import (
    paged_experts_decide,
    paged_experts_gather,
    paged_experts_host_device_pointer,
    paged_experts_precopy_plan,
    paged_experts_rows_to_slots,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-small")

E, K = 64, 16


class _State:
    """Residency state on the GPU, slots starting with experts 0..K-1."""

    def __init__(self, num_experts=E, num_slots=K):
        self.num_experts, self.num_slots = num_experts, num_slots
        i32 = dict(dtype=torch.int32, device="cuda")
        self.step = torch.zeros(1, **i32)
        self.slot_expert = torch.arange(num_slots, **i32)
        self.expert_slot = torch.full((num_experts,), -1, **i32)
        self.expert_slot[:num_slots] = self.slot_expert
        self.slot_lastuse = torch.zeros(num_slots, **i32)
        self.src = torch.zeros(num_slots, **i32)
        self.dst = torch.zeros(num_slots, **i32)
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

    def __init__(self, num_slots=K):
        self.num_slots = num_slots
        self.step = 0
        self.slot_expert = list(range(num_slots))
        self.slot_lastuse = [0] * num_slots

    def decide(self, topk):
        self.step += 1
        for e in topk:
            if e in self.slot_expert:
                self.slot_lastuse[self.slot_expert.index(e)] = self.step
        plan = []
        for e in topk:
            if e < 0 or e in self.slot_expert:
                continue
            free = [
                s for s in range(self.num_slots) if self.slot_lastuse[s] != self.step
            ]
            victim = min(free, key=lambda s: (self.slot_lastuse[s], s))
            self.slot_expert[victim] = e
            self.slot_lastuse[victim] = self.step
            plan.append((e, victim))
        return plan


def _random_topk(generator, num_entries, num_experts=E):
    topk = torch.randint(
        0, num_experts, (num_entries,), generator=generator, dtype=torch.int32
    )
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
        expert_slot = [-1] * state.num_experts
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

    def test_decide_at_model_size(self):
        """E=512/K=400/T=80, the shapes a decode step runs at: the block-wide staging and the
        ballot compaction still decide the reference plan, page-in included."""
        generator = torch.Generator().manual_seed(2)
        state, ref = _State(512, 400), _Reference(400)
        for _ in range(3):
            topk = _random_topk(generator, 80, 512)  # most experts cold, a few padding
            weights = torch.rand(80, generator=generator)
            plan = ref.decide(topk.tolist())
            _, run_w = state.decide(topk.cuda(), weights.cuda())
            self.assert_state(state, ref, plan)
            self.assertTrue(
                torch.equal(run_w.cpu(), weights.masked_fill(topk < 0, 0.0))
            )

    def test_sum_top_k_falls_back_past_the_grid_y_limit(self):
        """``moe_topk_sum`` grids over (..., M) and gridDim.y stops at 65,535, so a prefill chunk
        that large must keep the general reduce instead of failing to launch."""
        from sglang.srt.layers.moe.paged_experts.runners import MarlinContract

        entries = torch.randn(65536, 1, 2560, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(65536, 2560, device="cuda", dtype=torch.bfloat16)
        MarlinContract()._sum_top_k(entries, out, 1.3)
        self.assertTrue(
            torch.equal(out, (entries.float().sum(1) * 1.3).to(torch.bfloat16))
        )

    def test_gather_with_an_empty_plan_touches_nothing(self):
        """Most decode steps page nothing; count = 0 must leave every slot as it was."""
        hosts = [torch.randn(E, 8, dtype=torch.bfloat16).pin_memory()]
        gpu = torch.randn(K, 8, dtype=torch.bfloat16, device="cuda")
        before = gpu.clone()
        src = torch.full((K,), 7, dtype=torch.int32, device="cuda")
        dst = torch.full((K,), 3, dtype=torch.int32, device="cuda")
        count = torch.zeros(1, dtype=torch.int32, device="cuda")
        paged_experts_gather(*_gather_args(hosts, [gpu]), src, dst, count)
        self.assertTrue(torch.equal(gpu, before))

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
        self.assertFalse(host.any())
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

        def unregister():
            rc.append(int(torch.cuda.cudart().cudaHostUnregister(ptr)))

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

    def test_rows_to_slots_copies_one_way(self):
        """Rows of 16-, 4- and 2-byte multiples: the slot gets the row's expert, the row keeps
        it; negative rows are skipped."""
        hosts = [
            torch.randn(E, 8, 32, dtype=torch.bfloat16).pin_memory(),
            torch.randn(E, 3, dtype=torch.float32).pin_memory(),  # 12-byte rows
            torch.randn(E, 3, dtype=torch.bfloat16).pin_memory(),  # 6-byte rows
        ]
        gpus = [
            torch.randn(K, *h.shape[1:], dtype=h.dtype, device="cuda") for h in hosts
        ]
        rows = torch.tensor([5, -1, 17] + [0] * (K - 3), dtype=torch.int32).cuda()
        slots = torch.tensor([2, 3, 15] + [0] * (K - 3), dtype=torch.int32).cuda()
        count = torch.tensor([3], dtype=torch.int32, device="cuda")

        before = [(h.clone(), g.cpu()) for h, g in zip(hosts, gpus)]
        paged_experts_rows_to_slots(*_gather_args(hosts, gpus), rows, slots, count)
        torch.cuda.synchronize()
        for host, gpu, (host0, gpu0) in zip(hosts, gpus, before):
            for r, s in ((5, 2), (17, 15)):
                self.assertTrue(torch.equal(gpu[s].cpu(), host0[r]))
                self.assertTrue(torch.equal(host[r], host0[r]))
            self.assertTrue(torch.equal(gpu[3].cpu(), gpu0[3]))  # skipped

    def test_captured_step_waits_for_a_disk_read_or_replays_after_it(self):
        """A replayed step copies an expert in a host row into its slot and flags an expert
        only on disk, publishes its row and waits: the replay hook reads the expert into a host
        row and answers, and the step copies it in (no replay). A step that gives up waiting
        (timeout 0) runs on with the slot's old expert; the hook then copies it in after the
        step and asks for another replay. Both experts keep their rows (inclusive rows), the
        ones the slots held are dropped."""
        for wait in (True, False):
            with self.subTest(wait=wait):
                self._step_with_a_disk_read(wait)

    def _step_with_a_disk_read(self, wait: bool):
        from sglang.srt.layers.moe.paged_experts import disk_tier
        from sglang.srt.layers.moe.paged_experts.expert_cache import ExpertCache
        from sglang.srt.model_executor.runner_backend_utils import replay_hooks

        experts = torch.randn(E, 8, 32, dtype=torch.bfloat16)
        layer = SimpleNamespace(
            w=torch.nn.Parameter(experts[:K].clone().cuda(), requires_grad=False)
        )
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(disk_tier, "_MODEL", disk_tier.ModelContext()),
            patch.object(replay_hooks, "_HOOKS", []),
            patch(
                "sglang.srt.runtime_context.process_model_config",
                return_value=SimpleNamespace(num_hidden_layers=1),
            ),
            patch.object(disk_tier, "WAIT_TIMEOUT_MS", 2000 if wait else 0),
        ):
            cache = ExpertCache(os.path.join(folder, "layer-0.experts"), {"key": "k"})
            cache.write({"w": experts})
            store = disk_tier.TieredExpertStore(
                {"w": ((8, 32), torch.bfloat16)},
                E,
                K,  # host rows hold experts K..2K-1, slots 0..K-1; R < E
                cache,
            )
            store.bind(layer, device_planned=True)

            i32 = dict(dtype=torch.int32, device="cuda")
            src, dst = torch.zeros(K, **i32), torch.zeros(K, **i32)
            count = torch.zeros(1, **i32)
            store.gather(src, dst, count)  # warm up before capture
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                store.gather(src, dst, count)

            src[:2] = torch.tensor([K + 3, 40], **i32)  # in a host row, only on disk
            dst[:2] = torch.tensor([0, 1], **i32)
            count.fill_(2)
            graph.replay()
            if wait:  # the hook answers while the step waits at the layer
                self.assertFalse(replay_hooks.needs_replay())
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(layer.w[0].cpu(), experts[K + 3]))
                self.assertTrue(torch.equal(layer.w[1].cpu(), experts[40]))
            else:
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(layer.w[0].cpu(), experts[K + 3]))
                self.assertTrue(torch.equal(layer.w[1].cpu(), experts[1]))  # on disk
                self.assertTrue(replay_hooks.needs_replay())  # read 40, copied it in
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(layer.w[1].cpu(), experts[40]))
                count.zero_()  # the replayed step finds both resident
                graph.replay()
                self.assertFalse(replay_hooks.needs_replay())

            store.read_maps(store.state.cpu())
            rows = {e: r for r, e in enumerate(store.row_expert) if e >= 0}
            for e, r in rows.items():  # every row holds what the maps say
                self.assertTrue(torch.equal(store.host["w"][r], experts[e]))
            # the moved experts keep their rows, the slots' old ones (0, 1) are dropped
            self.assertTrue({K + 3, 40} <= set(rows))
            self.assertFalse({0, 1} & set(rows))

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


@unittest.skipUnless(torch.cuda.is_available(), "paged experts kernels need CUDA")
class TestPublish(CustomTestCase):
    """The map update kernel writes the layer's state row to pinned host memory and then the
    new clock as its flag, captured and replayed."""

    def test_the_row_then_the_flag(self):
        from sglang.kernels.ops.moe.paged_experts import paged_experts_tiered_plan

        k, e, r = 4, 16, 6
        i32 = dict(dtype=torch.int32, device="cuda")
        state = torch.zeros(3 * k + e + 2 * r + 1, **i32)
        doorbell, expert_row, row_expert, row_stamp, clock = state.split(
            [3 * k, e, r, r, 1]
        )
        expert_row.fill_(-1)
        for row in range(r):
            row_expert[row] = 4 + row
            expert_row[4 + row] = row
        src = torch.tensor([5, 13, 7, 0], **i32)  # 13 is on disk
        dst = torch.tensor([0, 1, 2, 0], **i32)
        count = torch.tensor([3], **i32)
        rows_out = torch.full((k,), -1, **i32)
        host = torch.zeros(2, state.numel(), dtype=torch.int32, pin_memory=True)
        flags = torch.zeros(2, dtype=torch.int32, pin_memory=True)
        row_at = paged_experts_host_device_pointer(host) + state.numel() * 4
        flag_at = paged_experts_host_device_pointer(flags) + 4
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            paged_experts_tiered_plan(
                src,
                dst,
                count,
                expert_row,
                row_expert,
                row_stamp,
                clock,
                doorbell.view(3, k),
                rows_out,
                row_at,
                flag_at,
                state.numel(),
            )
        for step in (1, 2, 3):
            host.zero_()
            flags.zero_()
            graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(int(flags[1]), step)  # the new clock
            self.assertEqual(int(flags[0]), 0)
            self.assertEqual(host[1].tolist(), state.cpu().tolist())
            self.assertEqual(host[0].abs().sum().item(), 0)
            self.assertEqual(int(host[1, -1]), step)
            self.assertEqual(host[1, :k].tolist(), [0, 1, 0, 0])  # 13 flagged
        with self.assertRaises(Exception):  # a row shorter than the doorbell
            paged_experts_tiered_plan(
                src,
                dst,
                count,
                expert_row,
                row_expert,
                row_stamp,
                clock,
                doorbell.view(3, k),
                rows_out,
                row_at,
                flag_at,
                3 * k - 1,
            )

    def test_every_warp_stamps_the_published_clock(self):
        """With more than 32 planned loads the update kernel runs several warps: every host-row
        hit must carry the clock the step publishes, over consecutive miss-free replays (the
        host protects exactly the rows stamped with the published clock)."""
        from sglang.kernels.ops.moe.paged_experts import paged_experts_tiered_plan

        i32 = dict(dtype=torch.int32, device="cuda")
        for k in (64, 1024):
            with self.subTest(k=k):
                e = r = k
                state = torch.zeros(3 * k + e + 2 * r + 1, **i32)
                doorbell, expert_row, row_expert, row_stamp, clock = state.split(
                    [3 * k, e, r, r, 1]
                )
                perm = torch.randperm(e, generator=torch.Generator().manual_seed(k))
                row_expert.copy_(perm.to(torch.int32))
                expert_row[perm.cuda()] = torch.arange(r, **i32)
                src = torch.arange(k, **i32)  # every expert planned, all in host rows
                dst = torch.arange(k, **i32)
                count = torch.tensor([k], **i32)
                rows_out = torch.full((k,), -1, **i32)
                host = torch.zeros(state.numel(), dtype=torch.int32, pin_memory=True)
                flags = torch.zeros(1, dtype=torch.int32, pin_memory=True)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    paged_experts_tiered_plan(
                        src,
                        dst,
                        count,
                        expert_row,
                        row_expert,
                        row_stamp,
                        clock,
                        doorbell.view(3, k),
                        rows_out,
                        paged_experts_host_device_pointer(host),
                        paged_experts_host_device_pointer(flags),
                        state.numel(),
                    )
                for step in range(1, 201):
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assertEqual(int(flags[0]), step)
                    self.assertEqual(int(clock), step)
                    stamps = row_stamp.cpu()
                    self.assertTrue(
                        bool((stamps == step).all()),
                        f"step {step}: stamps {sorted(set(stamps.tolist()))}",
                    )
                    self.assertTrue(
                        bool((host[3 * k + e + r : 3 * k + e + 2 * r] == step).all())
                    )
                    self.assertEqual(int(doorbell[:k].sum()), 0)  # miss-free


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


def _precopy_ref(ref, logits, top_k, margin, max_copies, tally):
    """The pre-copy plan the kernel must make, on a ``_Reference`` (``tally``: last, last_stamp,
    counts, updated in place). ``logits`` float, distinct."""
    import math

    expert_slot = {e: s for s, e in enumerate(ref.slot_expert)}
    for e in tally["last"]:
        s = expert_slot.get(e)
        tally["counts"][
            1 if s is not None and ref.slot_lastuse[s] > tally["stamp"] else 2
        ] += 1
    tally["last"] = []
    order = sorted(range(len(logits)), key=lambda i: (-logits[i], i))[: top_k + 1]
    v = [logits[i] for i in order]
    denom = sum(math.exp(x - v[0]) for x in logits)
    m = [(math.exp(a - v[0]) - math.exp(v[top_k] - v[0])) / denom for a in v[:top_k]]
    predicted = set(order[:top_k])
    plan, skipped = [], 0
    for j in range(top_k):
        if len(plan) >= min(max_copies, K - top_k):
            break
        e = order[j]
        if not m[j] > margin or e in expert_slot:
            skipped += 1
            continue
        picked = {s for _, s in plan}
        free = [
            s
            for s in range(K)
            if s not in picked and ref.slot_expert[s] not in predicted
        ]
        if not free:
            break
        victim = min(free, key=lambda s: (ref.slot_lastuse[s], s))
        plan.append((e, victim))
        del expert_slot[ref.slot_expert[victim]]
        ref.slot_expert[victim] = e
        expert_slot[e] = victim
        ref.slot_lastuse[victim] = ref.step
        tally["last"].append(e)
    tally["stamp"] = ref.step
    tally["counts"][0] += len(plan)
    tally["counts"][3] += skipped
    return plan, m


@unittest.skipUnless(torch.cuda.is_available(), "paged experts kernels need CUDA")
class TestPrecopyPlan(CustomTestCase):
    """``paged_experts_precopy_plan`` against a Python reference: the plan, the residency it moves
    (slots stamped with the step as it is), the counters, then the decide that follows; captured
    with the 32-block gather and replayed over chained random states."""

    def _run(self, margin, max_copies, steps=40):
        top_k = 4
        host = torch.randn(E, 64, dtype=torch.float32).pin_memory()
        gpu = torch.zeros(K, 64, device="cuda")
        gpu.copy_(host[:K])
        args = _gather_args([host], [gpu])
        state, ref = _State(), _Reference()
        i32 = dict(dtype=torch.int32, device="cuda")
        logits = torch.zeros(E, dtype=torch.bfloat16, device="cuda")
        src, dst, count = (
            torch.zeros(K, **i32),
            torch.zeros(K, **i32),
            torch.zeros(1, **i32),
        )
        last, stamp, counts = (
            torch.full((K,), -1, **i32),
            torch.zeros(1, **i32),
            torch.zeros(4, **i32),
        )
        topk = torch.zeros(top_k, dtype=torch.int32, device="cuda")
        tally = {"last": [], "stamp": 0, "counts": [0, 0, 0, 0]}

        def step():
            paged_experts_precopy_plan(
                logits,
                top_k,
                margin,
                max_copies,
                state.step,
                state.slot_expert,
                state.expert_slot,
                state.slot_lastuse,
                src,
                dst,
                count,
                last,
                stamp,
                counts,
            )
            paged_experts_gather(*args, src, dst, count, blocks=32)
            state.decide(topk)
            paged_experts_gather(*args, state.src, state.dst, state.count)

        topk.copy_(torch.arange(top_k, dtype=torch.int32))
        step()  # warm up outside the capture
        torch.cuda.synchronize()
        # Restart from the initial state after the warm-up.
        state, ref = _State(), _Reference()
        gpu.copy_(host[:K])
        last.fill_(-1), stamp.zero_(), counts.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        generator = torch.Generator().manual_seed(2)
        for t in range(steps):
            # Distinct bf16 logits; half the steps route the predicted top-k, else random.
            vals = (torch.randperm(4 * E, generator=generator)[:E].float() / 16 - 8).to(
                torch.bfloat16
            )
            logits.copy_(vals)
            lv = vals.float().tolist()
            predicted = sorted(range(E), key=lambda i: (-lv[i], i))[:top_k]
            routing = (
                predicted
                if t % 2 == 0
                else torch.randperm(E, generator=generator)[:top_k].tolist()
            )
            topk.copy_(torch.tensor(routing, dtype=torch.int32))
            plan, _ = _precopy_ref(ref, lv, top_k, margin, max_copies, tally)
            ref.decide(routing)
            graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(
                list(zip(src[: len(plan)].tolist(), dst[: len(plan)].tolist())), plan
            )
            self.assertEqual(state.slot_expert.tolist(), ref.slot_expert)
            self.assertEqual(state.slot_lastuse.tolist(), ref.slot_lastuse)
            self.assertEqual(counts.tolist(), tally["counts"])
            for s, e in enumerate(ref.slot_expert):
                self.assertTrue(torch.equal(gpu[s].cpu(), host[e]), (t, s, e))
        return tally["counts"]

    def test_margin_zero_two_copies(self):
        planned, used, wasted, _ = self._run(0.0, 2)
        self.assertGreater(used, 0)
        self.assertGreater(wasted, 0)
        self.assertGreaterEqual(
            planned, used + wasted
        )  # the last plan is not tallied yet

    def test_a_margin_skips_and_four_copies(self):
        planned, used, _, skipped = self._run(0.02, 4)
        self.assertGreater(planned, 0)
        self.assertGreater(skipped, 0)

    def test_reference_catches_a_wrong_stamp(self):
        """Negative control of the stamp check: a reference that stamps the pre-copied slot with
        the next step (as the decide would) disagrees with the kernel."""
        orig = _precopy_ref

        def next_step_stamp(ref, *a, **k):
            out = orig(ref, *a, **k)
            for e in a[4]["last"]:
                ref.slot_lastuse[ref.slot_expert.index(e)] = ref.step + 1
            return out

        with unittest.mock.patch(f"{__name__}._precopy_ref", next_step_stamp):
            with self.assertRaises(AssertionError):
                self._run(0.0, 2, steps=12)

    def test_rejects_too_many_experts(self):
        i32 = dict(dtype=torch.int32, device="cuda")
        state = _State()
        big = torch.zeros(1025, dtype=torch.bfloat16, device="cuda")
        with self.assertRaises(Exception):
            paged_experts_precopy_plan(
                big,
                4,
                0.0,
                2,
                state.step,
                state.slot_expert,
                torch.full((1025,), -1, **i32),
                state.slot_lastuse,
                state.src,
                state.dst,
                state.count,
                torch.full((K,), -1, **i32),
                torch.zeros(1, **i32),
                torch.zeros(4, **i32),
            )


@unittest.skipUnless(torch.cuda.is_available(), "paged experts kernels need CUDA")
class TestPrecopyInstall(CustomTestCase):
    """``precopy.install`` links each paged MoE layer to the next one whose router the plan can
    predict (plain softmax top-k) and whose store copies with the host store's own ``gather``; a
    layer otherwise gets no pre-copy, logged, and nothing raises (pre-copy is on wherever it
    applies). Control: all softmax, every pair linked. Each layer's routing width comes from its
    own TopK config: the model config need not have ``num_experts_per_tok`` (HunYuan sets
    ``moe_topk``, possibly per layer) and has none here."""

    def _model(self, scoring, stores=None, top_ks=None, routers=None):
        from types import SimpleNamespace

        from sglang.srt.layers.moe.paged_experts.executor import DeviceExecutor
        from sglang.srt.layers.moe.paged_experts.method import PagedExpertsMoEMethod
        from sglang.srt.layers.moe.paged_experts.residency import DeviceResidency
        from sglang.srt.layers.moe.paged_experts.store import HostExpertStore, rows_like

        layers = []
        for i, func in enumerate(scoring):
            gpu = torch.zeros(K, 64, device="cuda")
            holder = SimpleNamespace(w=gpu)
            cls = (stores or {}).get(i, HostExpertStore)
            store = cls(rows_like(holder, ["w"]), E)
            store.bind_device_gather(holder)
            method = object.__new__(PagedExpertsMoEMethod)
            method.num_resident = K
            method.device_executor = DeviceExecutor(
                base_method=None,
                store=store,
                residency=DeviceResidency(num_experts=E, num_slots=K, device="cuda"),
                contract=None,
                routed_scaling_factor=1.0,
            )
            config = SimpleNamespace(
                top_k=(top_ks or {}).get(i, 4),
                scoring_func=func,
                use_grouped_topk=False,
                custom_routing_function=None,
                correction_bias=None,
            )
            mlp = SimpleNamespace(
                experts=SimpleNamespace(quant_method=method),
                gate=SimpleNamespace(
                    weight=torch.zeros(
                        (routers or {}).get(i, E),
                        64,
                        dtype=torch.bfloat16,
                        device="cuda",
                    )
                ),
                topk=SimpleNamespace(topk_config=config),
            )
            layers.append(SimpleNamespace(mlp=mlp))
        return SimpleNamespace(
            model=SimpleNamespace(layers=layers),
            config=SimpleNamespace(),
        ), [l.mlp.experts.quant_method for l in layers]

    def test_links_softmax_layers_and_skips_the_others(self):
        from sglang.srt.layers.moe.paged_experts import precopy

        for scoring, linked in (
            (["softmax", "softmax", "softmax"], [True, True, False]),
            (["softmax", "sigmoid", "softmax"], [False, True, False]),
        ):
            model, methods = self._model(scoring)
            with (
                self.subTest(scoring=scoring),
                self.assertLogs(precopy.logger, "INFO") as logs,
                patch.object(precopy, "copy_blocks", return_value=7) as grid,
            ):
                n = precopy.install(model)
            self.assertEqual(n, sum(linked))
            self.assertEqual(
                [m.device_executor.precopy_next is not None for m in methods], linked
            )
            # Each link's grid comes from this GPU's SM count (``copy_blocks``).
            sms = torch.cuda.get_device_properties(0).multi_processor_count
            self.assertEqual(grid.call_args_list, [((sms,),)] * n)
            links = [m.device_executor.precopy_next for m in methods]
            self.assertEqual([p.blocks for p in links if p is not None], [7] * n)
            skipped = [r for r in logs.output if "no pre-copy into" in r]
            self.assertEqual(
                len(skipped), 0 if all(f == "softmax" for f in scoring) else 1
            )

    def test_each_layer_routes_with_its_own_width(self):
        """The target layer's own top_k drives its plan; a width the plan kernel cannot take
        (above 32) or a router whose width is not the expert count (a redundant expert gives
        the slot map one entry more than the router has logits) gets no pre-copy instead of
        failing in capture. Control: the first pair, top-2 on a full-width router, is linked."""
        from sglang.srt.layers.moe.paged_experts import precopy

        model, methods = self._model(
            ["softmax"] * 4, top_ks={1: 2, 3: 33}, routers={2: E - 1}
        )
        with self.assertLogs(precopy.logger, "INFO") as logs:
            n = precopy.install(model)
        self.assertEqual(n, 1)
        links = [m.device_executor.precopy_next for m in methods]
        self.assertEqual([p is not None for p in links], [True, False, False, False])
        self.assertEqual(links[0].top_k, 2)
        skipped = "".join(r for r in logs.output if "no pre-copy into" in r)
        self.assertIn("redundant experts", skipped)
        self.assertIn("top-33 routing", skipped)

    def test_skips_a_store_that_replaces_gather(self):
        """A store with its own ``gather`` (one that plans more than the copy, as a tiered store
        does) is not copied into ahead; a subclass that keeps the host store's ``gather`` is
        (control: the check follows the copy, not the class)."""
        from sglang.srt.layers.moe.paged_experts import precopy
        from sglang.srt.layers.moe.paged_experts.store import HostExpertStore

        class PlansItsOwnCopies(HostExpertStore):
            def gather(self, src, dst, count):  # no ``blocks``: a different contract
                raise AssertionError("pre-copy must not call this gather")

        class SameCopy(HostExpertStore):
            pass

        for cls, linked in (
            (PlansItsOwnCopies, [False, True, False]),
            (SameCopy, [True, True, False]),
        ):
            model, methods = self._model(["softmax"] * 3, stores={1: cls})
            with (
                self.subTest(store=cls.__name__),
                self.assertLogs(precopy.logger, "INFO") as logs,
            ):
                n = precopy.install(model)
            self.assertEqual(n, sum(linked))
            self.assertEqual(
                [m.device_executor.precopy_next is not None for m in methods], linked
            )
            skipped = [
                r for r in logs.output if "a store that plans its own copies" in r
            ]
            self.assertEqual(len(skipped), 0 if cls is SameCopy else 1)


class _SlotReader:
    """Runner stand-in: each routed entry's slot content, [T, top_k, D]."""

    def apply(self, layer, dispatch_output):
        from types import SimpleNamespace

        ids = dispatch_output.topk_output.topk_ids.long()
        return SimpleNamespace(hidden_states=layer.w.data[ids].clone())


class _Identity:
    @staticmethod
    def finish(merged, hidden_states, routed_scaling_factor):
        return merged


@unittest.skipUnless(torch.cuda.is_available(), "paged experts kernels need CUDA")
class TestPrecopyCaptured(CustomTestCase):
    """Byte oracle of pre-copy in a captured multi-layer decode step, captured as the server
    captures (``freeze_gc``): after each layer's gather every runner input holds its routed
    expert, and after every step each slot holds what the maps say. The side stream spins
    between its plan and its copy, so the join is what keeps the copy ahead of the next layer."""

    LAYERS, D, T, TOP_K = 3, 256, 1, 4

    def _chain(self, side_sleep, max_copies=2, margin=0.0, dtype=torch.bfloat16):
        from types import SimpleNamespace

        from sglang.srt.layers.moe.paged_experts.executor import DeviceExecutor
        from sglang.srt.layers.moe.paged_experts.precopy import Precopy
        from sglang.srt.layers.moe.paged_experts.residency import DeviceResidency
        from sglang.srt.layers.moe.paged_experts.store import HostExpertStore, rows_like

        torch.manual_seed(3)
        side = torch.cuda.Stream()
        weight = torch.eye(E, dtype=dtype, device="cuda")  # logits = the input
        layers = []
        for _ in range(self.LAYERS):
            gpu = torch.zeros(K, self.D, device="cuda")
            layer = SimpleNamespace(w=gpu)
            store = HostExpertStore(rows_like(layer, ["w"]), E)
            store.host["w"].copy_(torch.randn(E, self.D))
            gpu.copy_(store.host["w"][:K])
            store.bind_device_gather(layer)
            r = DeviceResidency(num_experts=E, num_slots=K, device="cuda")
            ex = DeviceExecutor(
                base_method=_SlotReader(),
                store=store,
                residency=r,
                contract=_Identity(),
                routed_scaling_factor=1.0,
            )
            layers.append(SimpleNamespace(layer=layer, store=store, r=r, ex=ex))
        links = []
        for cur, nxt in zip(layers, layers[1:]):
            pre = Precopy(
                SimpleNamespace(device_executor=nxt.ex),
                weight,
                self.TOP_K,
                side,
                margin,
                max_copies,
                32,
            )
            pre.side_sleep = side_sleep
            cur.ex.precopy_next = pre
            links.append(pre)
        return layers, links

    def _capture_and_run(
        self, layers, steps=16, defer_join=False, skip_copy=False, dtype=torch.bfloat16
    ):
        from sglang.srt.layers.moe.paged_experts import executor as executor_mod
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardDispatchOutput,
        )
        from sglang.srt.layers.moe.topk import StandardTopKOutput
        from sglang.srt.model_executor.runner.base_cuda_graph_runner import freeze_gc

        i32 = dict(dtype=torch.int32, device="cuda")
        ids = [torch.zeros(self.T, self.TOP_K, **i32) for _ in layers]
        xs = [torch.zeros(self.T, E, dtype=dtype, device="cuda") for _ in layers]
        w = torch.ones(self.T, self.TOP_K, device="cuda")
        outs = [torch.zeros(self.T, self.TOP_K, self.D, device="cuda") for _ in layers]
        if skip_copy:
            for l in layers[:-1]:
                l.ex.precopy_next.store = SimpleNamespace(gather=lambda **kw: None)

        def forward():
            for l, i, x, o in zip(layers, ids, xs, outs):
                d = StandardDispatchOutput(
                    hidden_states=x,
                    hidden_states_scale=None,
                    topk_output=StandardTopKOutput(
                        topk_weights=w, topk_ids=i, router_logits=None
                    ),
                )
                o.copy_(l.ex.run(l.layer, d).hidden_states)

        deferred = []

        def late_join(r):
            if r.precopy_join is not None:
                deferred.append(r.precopy_join)
                r.precopy_join = None

        for i, l in enumerate(ids):
            l.copy_(torch.arange(self.TOP_K, dtype=torch.int32) + i)
        forward()  # warm up outside the capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        patcher = (
            unittest.mock.patch.object(executor_mod, "join", late_join)
            if defer_join
            else contextlib.nullcontext()
        )
        with patcher, freeze_gc(False), torch.cuda.graph(graph):
            forward()
            for ev in deferred:
                torch.cuda.current_stream().wait_event(ev)
        generator = torch.Generator().manual_seed(4)
        bad = 0
        for t in range(steps):
            routing = [
                torch.randperm(E, generator=generator)[: self.TOP_K] for _ in layers
            ]
            for li, (i, x) in enumerate(zip(ids, xs)):
                i.copy_(routing[li].view(1, -1).int())
                # Layer li's input predicts layer li+1: right on even steps, wrong on odd ones.
                logits = torch.full((E,), -4.0)
                if li + 1 < len(layers):
                    target = (
                        routing[li + 1]
                        if t % 2 == 0
                        else torch.randperm(E, generator=generator)[: self.TOP_K]
                    )
                    logits[target] = 4.0 - 0.25 * torch.arange(self.TOP_K)
                x.copy_(logits.view(1, -1).to(dtype))
            graph.replay()
            torch.cuda.synchronize()
            for l, r_ids, o in zip(layers, routing, outs):
                want = l.store.host["w"][r_ids.long()].view(1, self.TOP_K, self.D)
                bad += not torch.equal(o.cpu(), want)
                for s, e in enumerate(l.r.slot_expert.tolist()):
                    bad += not torch.equal(l.layer.w[s].cpu(), l.store.host["w"][e])
        return bad

    def test_every_runner_input_and_slot_holds_its_expert(self):
        layers, links = self._chain(side_sleep=int(2e6))
        self.assertEqual(self._capture_and_run(layers), 0)
        planned, used, wasted, _ = (
            torch.stack([p.counts for p in links]).sum(0).tolist()
        )
        self.assertGreater(used, 0)
        self.assertGreater(wasted, 0)
        self.assertGreaterEqual(planned, used + wasted)
        self.assertEqual(
            len(links[0].keep), 1
        )  # the captured input is kept for the graph

    def test_an_fp16_router_is_converted_for_the_plan(self):
        """GPTQ models run in fp16: the router GEMV writes fp16 and the plan reads its bf16 copy."""
        layers, links = self._chain(side_sleep=int(2e6), dtype=torch.float16)
        self.assertEqual(self._capture_and_run(layers, dtype=torch.float16), 0)
        self.assertEqual(links[0].router_out.dtype, torch.float16)
        _, used, _, _ = torch.stack([p.counts for p in links]).sum(0).tolist()
        self.assertGreater(used, 0)

    def test_join_deferred_to_the_end_is_caught(self):
        layers, _ = self._chain(side_sleep=int(2e6))
        self.assertGreater(self._capture_and_run(layers, defer_join=True), 0)

    def test_maps_moved_without_the_copy_is_caught(self):
        layers, _ = self._chain(side_sleep=0)
        self.assertGreater(self._capture_and_run(layers, skip_copy=True), 0)


def _gather_args(hosts, gpus):
    return [
        torch.tensor(v, dtype=torch.int64, device="cuda")
        for v in (
            [paged_experts_host_device_pointer(h) for h in hosts],
            [g.data_ptr() for g in gpus],
            [h[0].numel() * h.element_size() for h in hosts],
        )
    ]


class TestCompactedTritonCalls(CustomTestCase):
    """The triton runner's no-combine output of a step, computed as the paged triton waves
    compute it: each wave's entries as top-k 1 rows in calls of a few rows, inside the config
    lookup of the step's full call (tokens x top-k)."""

    @classmethod
    def setUpClass(cls):
        # fused_experts reads the server args and allocates under the TP group.
        from sglang.srt.distributed.parallel_state import (
            init_distributed_environment,
            initialize_model_parallel,
            model_parallel_is_initialized,
        )
        from sglang.srt.server_args import (
            ServerArgs,
            set_global_server_args_for_scheduler,
        )
        from sglang.test.test_utils import publish_build_topology

        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        for key, value in (
            ("MASTER_ADDR", "127.0.0.1"),
            ("MASTER_PORT", "29653"),
            ("RANK", "0"),
            ("WORLD_SIZE", "1"),
            ("LOCAL_RANK", "0"),
        ):
            os.environ.setdefault(key, value)
        if not torch.distributed.is_initialized():
            init_distributed_environment(
                world_size=1, rank=0, local_rank=0, backend="gloo"
            )
        if not model_parallel_is_initialized():
            publish_build_topology(tp_size=1, ep_size=1, pp_size=1)
            initialize_model_parallel(backend="gloo")

    @staticmethod
    def _fp8_blocks(w):
        """Block-wise FP8 [128, 128] weights and their float32 scales, as checkpoints keep them."""
        e, n, k = w.shape
        blocks = w.view(e, n // 128, 128, k // 128, 128)
        scale = blocks.abs().amax(dim=(2, 4)).clamp(min=1e-6) / 448.0
        q = (blocks / scale[:, :, None, :, None]).to(torch.float8_e4m3fn)
        return q.view(e, n, k), scale.float()

    def _run(self, x, w1, w2, ids, weights, quant):
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.moe_runner.triton_utils import fused_experts
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        e = w1.shape[0]
        config = MoeRunnerConfig(
            num_experts=e,
            num_local_experts=e,
            top_k=ids.shape[1],
            hidden_size=x.shape[1],
            intermediate_size_per_partition=w2.shape[2],
            params_dtype=x.dtype,
            activation="silu",
            inplace=False,
            no_combine=True,
            no_combine_keep_router_weight=True,
        )
        topk = StandardTopKOutput(
            topk_weights=weights, topk_ids=ids, router_logits=None
        )
        return fused_experts(x, w1, w2, topk, config, **quant)

    def test_compacted_calls_equal_the_full_call_bitwise(self):
        """bf16 and block-wise FP8 experts, 300 tokens x top-4 over 8 experts: the entries of
        experts 0-3 (one wave), run in calls of at most 6 rows inside
        ``override_config_lookup(300, 4)``, give the full call's outputs bit for bit. Control:
        without the lookup the 6-row calls would look up another config than the full call in
        at least one case (so the lookup, not luck, keeps the numerics; block-wise FP8's
        default config does not depend on the rows on every GPU, bf16's does)."""
        from sglang.srt.layers.moe.moe_runner.triton_utils import (
            override_config_lookup,
            try_get_optimal_moe_config,
        )

        torch.manual_seed(0)
        e, top_k, tokens, hidden, inter = 8, 4, 300, 256, 256
        x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
        w1 = torch.randn(e, 2 * inter, hidden, device="cuda", dtype=torch.bfloat16) / 16
        w2 = torch.randn(e, hidden, inter, device="cuda", dtype=torch.bfloat16) / 16
        ids = torch.stack([torch.randperm(e)[:top_k] for _ in range(tokens)])
        ids = ids.to(device="cuda", dtype=torch.int32)
        weights = torch.rand(tokens, top_k, device="cuda", dtype=torch.float32)
        q1, s1 = self._fp8_blocks(w1)
        q2, s2 = self._fp8_blocks(w2)
        cases = {
            "bf16": (w1, w2, {}, None),
            "fp8-block": (
                q1,
                q2,
                dict(
                    use_fp8_w8a8=True, w1_scale=s1, w2_scale=s2, block_shape=[128, 128]
                ),
                "fp8_w8a8",
            ),
        }
        # Experts 0-3: one wave of 4 slots.
        wave = (ids < 4).flatten().nonzero().squeeze(1)
        lookup_matters = []
        for name, (a, b, quant, dtype) in cases.items():
            with self.subTest(experts=name):
                full = self._run(x, a, b, ids, weights, quant)
                got = torch.zeros_like(full).view(-1, hidden)
                with override_config_lookup(tokens, top_k):
                    for part in wave.split(6):
                        rows = part // top_k
                        out = self._run(
                            x[rows],
                            a,
                            b,
                            ids.flatten()[part].unsqueeze(1),
                            weights.flatten()[part].unsqueeze(1),
                            quant,
                        )
                        got[part] = out.view(-1, hidden)
                want = full.view(-1, hidden)[wave]
                mismatch = (got[wave].view(torch.int16) != want.view(torch.int16)).sum()
                self.assertEqual(int(mismatch), 0)

                block = quant.get("block_shape")
                shapes = (a.shape, (b.shape[0], b.shape[1], b.shape[2]))
                full_config = try_get_optimal_moe_config(
                    *shapes, top_k, dtype, tokens, block_shape=block
                )
                row_config = try_get_optimal_moe_config(
                    *shapes, 1, dtype, 6, block_shape=block
                )
                lookup_matters.append(full_config != row_config)
        self.assertTrue(any(lookup_matters))


if __name__ == "__main__":
    unittest.main()
