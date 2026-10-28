import unittest

import torch

from sglang.kernels.ops.moe.paged_experts import (
    paged_experts_decide,
    paged_experts_gather,
    paged_experts_host_device_pointer,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

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

    def decide(self, topk):
        paged_experts_decide(
            topk,
            self.step,
            self.slot_expert,
            self.expert_slot,
            self.slot_lastuse,
            self.src,
            self.dst,
            self.count,
        )


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
        generator = torch.Generator().manual_seed(0)
        state, ref = _State(), _Reference()
        for _ in range(200):
            topk = _random_topk(generator, int(torch.randint(1, K + 1, (1,))))
            plan = ref.decide(topk.tolist())
            state.decide(topk.cuda())
            self.assert_state(state, ref, plan)

    def test_decide_rejects_more_entries_than_slots(self):
        with self.assertRaises(Exception):
            _State().decide(torch.zeros(K + 1, dtype=torch.int32, device="cuda"))

    def test_gather_copies_the_plan_for_every_tensor(self):
        hosts = [
            torch.randn(E, 8, 32, dtype=torch.bfloat16).pin_memory(),
            torch.randint(0, 100, (E, 12), dtype=torch.int32).pin_memory(),
            torch.randn(E, dtype=torch.float32).pin_memory(),  # 4-byte rows
            torch.randn(E, 3, dtype=torch.float32).pin_memory(),  # 12-byte rows
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


def _gather_args(hosts, gpus):
    return [
        torch.tensor(v, dtype=torch.int64, device="cuda")
        for v in (
            [paged_experts_host_device_pointer(h) for h in hosts],
            [g.data_ptr() for g in gpus],
            [h[0].numel() * h.element_size() // 4 for h in hosts],
        )
    ]


if __name__ == "__main__":
    unittest.main()
