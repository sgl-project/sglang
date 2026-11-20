import os
import tempfile
import threading
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.moe.paged_experts import (
    paged_experts_decide,
    paged_experts_gather,
    paged_experts_host_device_pointer,
    paged_experts_rows_to_slots,
)
from sglang.srt.layers.moe.paged_experts.store import HostExpertStore
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=75, stage="base-b-kernel-unit", runner_config="1-gpu-small")

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

    def decide(self, topk, weights=None, host_ids=False):
        """Run the decide kernel; return the runner inputs it wrote (slots, weights).
        ``host_ids``: the fused host copy's ids (K + e for an entry paged in)."""
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
            host_ids=host_ids,
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


class _Layer:
    """Every expert in a pinned host store, K GPU slots holding experts 0..K-1 at the start."""

    def __init__(self, tensors):  # name -> [E, ...] tensor on the GPU
        rows = {n: (tuple(t.shape[1:]), t.dtype) for n, t in tensors.items()}
        self.store = HostExpertStore(rows, E)
        for n, t in tensors.items():
            self.store.host[n].copy_(t.cpu())
        self.initial = {n: t[:K].clone() for n, t in tensors.items()}
        self.slots = {n: t.clone() for n, t in self.initial.items()}

    def reset(self):
        for n, t in self.initial.items():
            self.slots[n].copy_(t)

    def gather(self, state):
        """The copy kernel that runs before the MoE kernel without the host copy: bring the plan's experts into their slots."""
        names = list(self.slots)
        args = _gather_args(
            [self.store.host[n] for n in names], [self.slots[n] for n in names]
        )
        paged_experts_gather(*args, state.src, state.dst, state.count)

    def assert_slots_hold(self, case, state):
        """Every slot holds, byte for byte, the expert the residency state says it holds."""
        for n, gpu in self.slots.items():
            for s, e in enumerate(state.slot_expert.tolist()):
                case.assertTrue(
                    torch.equal(gpu[s].cpu(), self.store.host[n][e]), f"{n} slot {s}"
                )


# Routings for 4 tokens x top-2 (8 entries, K = 16 slots, experts 0..15 resident at the start).
ROUTINGS = {
    "none missing": [[0, 1], [2, 3], [4, 5], [6, 7]],
    "all missing": [[16, 17], [18, 19], [20, 21], [22, 23]],
    "mixed": [[0, 40], [41, 3], [5, 42], [63, 7]],
    "one missing expert in every token": [[50, 1], [50, 2], [50, 3], [50, 4]],
    "padding": [[0, 40], [-1, 3], [41, -1], [-1, -1]],
}


class TestFusedHostCopyDecide(CustomTestCase):
    def test_host_ids_mark_only_the_paged_in_entries(self):
        """With host_ids the decide kernel makes the same decision (plan, slot tables) and
        writes the same run ids, except that an entry whose expert is paged in gets K + e."""
        weights = torch.rand(8, device="cuda")
        for name, routing in ROUTINGS.items():
            with self.subTest(routing=name):
                topk = torch.tensor(routing, dtype=torch.int32, device="cuda").flatten()
                copy_first, fused = _State(), _State()
                ids_s, w_s = copy_first.decide(topk, weights, host_ids=False)
                ids_f, w_f = fused.decide(topk, weights, host_ids=True)
                for a in (
                    "src",
                    "dst",
                    "count",
                    "slot_expert",
                    "expert_slot",
                    "slot_lastuse",
                ):
                    self.assertTrue(
                        torch.equal(getattr(copy_first, a), getattr(fused, a)), a
                    )
                self.assertTrue(torch.equal(w_s, w_f))
                n = int(copy_first.count.item())
                paged_in = set(copy_first.src[:n].tolist())
                want = [
                    K + e if e in paged_in else s
                    for e, s in zip(topk.tolist(), ids_s.tolist())
                ]
                self.assertEqual(ids_f.tolist(), want)


class TestFusedHostCopyTriton(CustomTestCase):
    """The Triton MoE kernel reading missing experts from host memory: same output as
    copy-then-compute, bit for bit, and every slot filled."""

    setUpClass = classmethod(TestCompactedTritonCalls.setUpClass.__func__)
    _run = TestCompactedTritonCalls._run

    def setUp(self):
        torch.manual_seed(0)
        hidden, inter = 256, 128
        self.layer = _Layer(
            {
                "w13": torch.randn(
                    E, 2 * inter, hidden, device="cuda", dtype=torch.bfloat16
                )
                / 16,
                "w2": torch.randn(E, hidden, inter, device="cuda", dtype=torch.bfloat16)
                / 16,
            }
        )
        self.x = torch.randn(4, hidden, device="cuda", dtype=torch.bfloat16)

    def _paged_expert_info(self, state):
        # What PagedExpertsMoEMethod._fuse_host_copy puts on the layer's runner config.
        w13, w2 = (
            dict(
                host_table=self.layer.store.host[n],
                expert_slot=state.expert_slot,
                missing_expert_count=state.count,
            )
            for n in ("w13", "w2")
        )
        paged_expert_info = torch.empty(0, device="cuda")
        paged_expert_info.info = (w13, w2)
        return {"paged_expert_info": paged_expert_info}

    def _copy_then_compute(self, topk, weights):
        state = _State()
        ids, w = state.decide(topk.flatten(), weights.flatten(), host_ids=False)
        self.layer.reset()
        self.layer.gather(state)
        out = self._run(
            self.x,
            self.layer.slots["w13"],
            self.layer.slots["w2"],
            ids.view(topk.shape),
            w.view(topk.shape),
            {},
        )
        return out, state

    def _fused(self, topk, weights):
        state = _State()
        ids, w = state.decide(topk.flatten(), weights.flatten(), host_ids=True)
        self.layer.reset()
        out = self._run(
            self.x,
            self.layer.slots["w13"],
            self.layer.slots["w2"],
            ids.view(topk.shape),
            w.view(topk.shape),
            self._paged_expert_info(state),
        )
        return out, state

    def test_fused_equals_copy_then_compute_bitwise(self):
        for name, routing in ROUTINGS.items():
            with self.subTest(routing=name):
                topk = torch.tensor(routing, dtype=torch.int32, device="cuda")
                weights = torch.rand(topk.shape, device="cuda")
                want, _ = self._copy_then_compute(topk, weights)
                got, state = self._fused(topk, weights)
                self.assertTrue(torch.equal(got, want))
                self.layer.assert_slots_hold(self, state)

    def test_many_tokens_on_one_paged_in_expert(self):
        """32 tokens share expert 50 (missing), so several thread blocks read it from the
        host and store it into the same slot; the other entry is a resident expert. More
        entries than the decide kernel takes (K), so the residency is set by hand: expert 50
        replaces expert 15 in slot 15."""
        state = _State()
        state.slot_expert[15] = 50
        state.expert_slot[15], state.expert_slot[50] = -1, 15
        state.src[0], state.dst[0], state.count[0] = 50, 15, 1
        resident = torch.tensor(
            [t % 8 for t in range(32)], dtype=torch.int32, device="cuda"
        )
        slot_ids = torch.stack([torch.full_like(resident, 15), resident], dim=1)
        fused_ids = torch.stack([torch.full_like(resident, K + 50), resident], dim=1)
        weights = torch.rand(32, 2, device="cuda")
        x = torch.randn(32, self.x.shape[1], device="cuda", dtype=torch.bfloat16)
        w13, w2 = self.layer.slots["w13"], self.layer.slots["w2"]

        self.layer.reset()
        self.layer.gather(state)
        want = self._run(x, w13, w2, slot_ids, weights, {})
        self.layer.reset()
        got = self._run(x, w13, w2, fused_ids, weights, self._paged_expert_info(state))
        self.assertTrue(torch.equal(got, want))
        self.layer.assert_slots_hold(self, state)

    def test_graph_replays_on_new_routing(self):
        """decide + fused MoE captured once, replayed on new routings: each replay matches
        copy-then-compute and leaves every slot holding its expert."""
        state = _State()
        paged_expert_info = self._paged_expert_info(state)
        topk = torch.zeros(4, 2, dtype=torch.int32, device="cuda")
        weights = torch.ones(4, 2, device="cuda")

        def step():
            ids, w = state.decide(topk.flatten(), weights.flatten(), host_ids=True)
            return self._run(
                self.x,
                self.layer.slots["w13"],
                self.layer.slots["w2"],
                ids.view(4, 2),
                w.view(4, 2),
                paged_expert_info,
            )

        step()  # warm up (compiles) outside the capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = step()
        generator = torch.Generator().manual_seed(1)
        for _ in range(10):
            routing = torch.stack(
                [torch.randperm(E, generator=generator)[:2] for _ in range(4)]
            )
            topk.copy_(routing.to(torch.int32))
            weights.copy_(torch.rand(4, 2, generator=generator))
            graph.replay()
            self.layer.assert_slots_hold(self, state)
            # Every routed expert is now resident: the MoE kernel on slot ids, without the host copy,
            # must agree.
            slot_ids = state.expert_slot[topk.long()]
            want = self._run(
                self.x,
                self.layer.slots["w13"],
                self.layer.slots["w2"],
                slot_ids,
                weights,
                {},
            )
            self.assertTrue(torch.equal(out, want))


def _torch_moe(x, w1, w2, ids, weights):
    """The MoE layer in PyTorch on dequantized weights (w1 [E, 2I, H] gate then up, w2
    [E, H, I]), as test_marlin_moe.py's torch_experts computes it."""
    from sglang.srt.layers.activation import SiluAndMul

    out = torch.zeros(x.shape[0], w2.shape[1], dtype=torch.float32, device=x.device)
    for t in range(x.shape[0]):
        for e, w in zip(ids[t].tolist(), weights[t].tolist()):
            if e < 0:
                continue
            h = SiluAndMul()((x[t : t + 1] @ w1[e].T))
            out[t] += w * (h @ w2[e].T)[0].float()
    return out.to(x.dtype)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0),
    "Marlin needs SM80+",
)
class TestFusedHostCopyMarlin(CustomTestCase):
    """NVFP4 experts through Marlin with the host copy: the output matches the PyTorch MoE on
    the dequantized weights, and the slots hold their experts byte for byte. Same settings as test_marlin_moe.py: inputs and weights / 10, atol 5e-2,
    rtol 0."""

    def setUp(self):
        from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            prepare_moe_nvfp4_layer_for_marlin,
        )
        from sglang.test.test_marlin_utils import make_nvfp4_weight_and_ref

        torch.manual_seed(0)
        H, I, dtype = 256, 128, torch.bfloat16

        def experts(n, k):
            # One NVFP4 weight per expert, scaled by 1/10 like test_marlin_moe.py's weights.
            parts = [make_nvfp4_weight_and_ref(n, k, dtype) for _ in range(E)]
            return (
                torch.stack([p[0] for p in parts]),
                torch.stack([p[1] for p in parts]),
                torch.stack([p[2].float() / 10 for p in parts]),
                torch.stack([p[3] / 10 for p in parts]),
            )

        q13, s13, g13, self.ref13 = experts(2 * I, H)
        q2, s2, g2, self.ref2 = experts(H, I)
        ns = types.SimpleNamespace(
            w13_weight=q13,
            w2_weight=q2,
            w13_weight_scale=s13,
            w2_weight_scale=s2,
            w13_weight_scale_2=g13,
            w2_weight_scale_2=g2,
            moe_runner_config=types.SimpleNamespace(is_gated=True),
            intermediate_size_per_partition=I,
            params_dtype=dtype,
        )
        prepare_moe_nvfp4_layer_for_marlin(ns, group_size=16)
        names = (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
            "w13_weight_scale_2",
            "w2_weight_scale_2",
        )
        self.layer = _Layer({n: getattr(ns, n).data for n in names})
        self.x = torch.randn(4, H, device="cuda", dtype=dtype) / 10
        self.workspace = lambda: marlin_make_workspace(self.x.device, 4)

    def _paged_expert_info(self, state):
        # What PagedExpertsMoEMethod._fuse_host_copy puts on the layer's runner config.
        ptr = lambda n: paged_experts_host_device_pointer(self.layer.store.host[n])
        w13, w2 = (
            dict(
                host_weight_address=ptr(w),
                host_scales_address=ptr(f"{w}_scale"),
                host_global_scale_address=ptr(f"{w}_scale_2"),
                expert_slot=state.expert_slot,
                missing_expert_count=state.count,
            )
            for w in ("w13_weight", "w2_weight")
        )
        paged_expert_info = torch.empty(0, device="cuda")
        paged_expert_info.info = (w13, w2)
        return paged_expert_info

    def _run(self, ids, weights, paged_expert_info):
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )

        s = self.layer.slots
        return fused_marlin_moe(
            self.x,
            s["w13_weight"],
            s["w2_weight"],
            s["w13_weight_scale"],
            s["w2_weight_scale"],
            torch.empty(self.x.shape[0], 1, device="cuda"),
            weights,
            ids,
            global_num_experts=K,
            w1_global_scale=s["w13_weight_scale_2"],
            w2_global_scale=s["w2_weight_scale_2"],
            workspace=self.workspace(),
            num_bits=4,
            is_k_full=True,
            activation="silu",
            is_gated=True,
            paged_expert_info=paged_expert_info,
        )

    def test_both_paths_match_the_torch_moe(self):
        for name, routing in ROUTINGS.items():
            with self.subTest(routing=name):
                topk = torch.tensor(routing, dtype=torch.int32, device="cuda")
                weights = torch.rand(topk.shape, device="cuda")
                ref = _torch_moe(self.x, self.ref13, self.ref2, topk, weights)

                fused = _State()
                ids, w = fused.decide(topk.flatten(), weights.flatten(), host_ids=True)
                self.layer.reset()
                fused_out = self._run(
                    ids.view(topk.shape),
                    w.view(topk.shape),
                    self._paged_expert_info(fused),
                )

                torch.testing.assert_close(fused_out, ref, atol=5e-2, rtol=0)
                self.layer.assert_slots_hold(self, fused)


if __name__ == "__main__":
    unittest.main()
