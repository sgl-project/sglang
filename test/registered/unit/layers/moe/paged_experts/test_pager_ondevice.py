"""GPU unit test for the captured (on-device) residency path in
srt/layers/moe/paged_experts/pager.py.

Drives ``decide_and_page_ondevice`` + ``remap_mask_ondevice`` — the capture-safe keep-warm path that runs
inside the decode CUDA graph — on a synthetic pinned K-slot layer, and asserts it pages the right experts
into the right slots and remaps routing exactly (losslessness). The eager ``decide_keep_warm`` path is
covered by ``test_pager.py``; this closes the gap that the on-device path (the default when CUDA graphs
are on) had no unit coverage. No server/model launch.
"""

import unittest

import torch

from sglang.srt.layers.moe.paged_experts.pager import ExpertPager
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


def _make_pinned_pager(E: int, K: int, dev: str) -> ExpertPager:
    layer = torch.nn.Module()
    # On-device gather is float4 -> per-expert blocks must be 16-byte aligned (fp32 numel % 4 == 0).
    layer.w13_weight = torch.nn.Parameter(
        torch.zeros(K, 4, 4, device=dev), requires_grad=False
    )  # 64 B/expert
    layer.w2_weight = torch.nn.Parameter(
        torch.zeros(K, 8, device=dev), requires_grad=False
    )  # 32 B/expert
    pager = ExpertPager(layer, E, K, dev, pin_host=True)
    for name in pager.host:
        for e in range(E):
            pager.host[name][e].fill_(
                float(e + 1)
            )  # expert e's data == e+1 (identity check)
    return pager


class TestPagerOnDevice(CustomTestCase):
    ACTIVE = [
        1,
        5,
        6,
    ]  # expert 1 is a resident hit; 5 and 6 are misses (distinct=3 <= K=4)

    def _seed_and_decide(self, E=8, K=4, dev="cuda") -> ExpertPager:
        pager = _make_pinned_pager(E, K, dev)
        pager.setup_ondevice()
        # On-device seed: gather experts 0..K-1 into slots 0..K-1 (matches setup_ondevice's initial map).
        ar = torch.arange(K, dtype=torch.int32, device=dev)
        pager._src_d[:K] = ar
        pager._dst_d[:K] = ar
        pager._n_out_d[0] = K
        pager._gather_planned_ondevice()
        torch.cuda.synchronize()
        topk = torch.tensor([self.ACTIVE], dtype=torch.int64, device=dev)
        pager.decide_and_page_ondevice(topk)
        torch.cuda.synchronize()
        return pager

    def test_decide_ondevice_pages_active_experts(self):
        K = 4
        pager = self._seed_and_decide(K=K)
        l2g = pager.logical_to_gpu_index_cuda

        # every active expert is resident, and its slot holds ITS data (e+1) in every paged tensor.
        for e in self.ACTIVE:
            slot = int(l2g[e])
            self.assertGreaterEqual(
                slot, 0, f"expert {e} not resident after on-device decide"
            )
            for name in pager.gpu:
                first = float(pager.gpu[name].data[slot].flatten()[0])
                self.assertAlmostEqual(
                    first, e + 1.0, msg=f"{name} slot {slot} != expert {e}"
                )

        # residency stays exactly K experts, and the slot->expert map agrees with the expert->slot map.
        resident = [int(pager._slot_expert_d[s]) for s in range(K)]
        self.assertEqual(len(set(resident)), K, "duplicate/empty slots after decide")
        for slot, e in enumerate(resident):
            self.assertEqual(
                int(l2g[e]), slot, "slot_expert and logical_to_gpu_index disagree"
            )
        for e in self.ACTIVE:
            self.assertIn(e, resident)

    def test_decide_ondevice_matches_eager_resident_set(self):
        # Same routing through the tested eager path must page in the same set of experts (lossless;
        # the captured path is not allowed to page a different residency than the eager reference).
        E, K, dev = 8, 4, "cuda"
        eager = _make_pinned_pager(E, K, dev)
        ar = torch.arange(K, dtype=torch.int64, device=dev)
        eager.page_in(ar, ar)
        torch.cuda.synchronize()
        src, dst = eager.decide_keep_warm(
            torch.tensor([self.ACTIVE], dtype=torch.int64, device=dev)
        )
        eager.page_in(src, dst)
        torch.cuda.synchronize()

        od = self._seed_and_decide(E=E, K=K, dev=dev)
        eager_set = {int(s) for s in eager.slot_expert}
        od_set = {int(od._slot_expert_d[s]) for s in range(K)}
        self.assertEqual(od_set, eager_set)

    def test_remap_mask_ondevice(self):
        pager = self._seed_and_decide()
        l2g = pager.logical_to_gpu_index_cuda
        topk = torch.tensor([self.ACTIVE], dtype=torch.int64, device="cuda")
        tw = torch.ones(1, len(self.ACTIVE), dtype=torch.float32, device="cuda")

        res = pager.remap_mask_ondevice(topk, tw)
        self.assertIsNotNone(
            res, "fused remap_mask should support fp32 contiguous weights"
        )
        safe_ids, masked_tw = res
        self.assertEqual(tuple(safe_ids.shape), tuple(topk.shape))
        # all active experts are resident here -> safe_ids == their slots, weights unmasked.
        for j, e in enumerate(self.ACTIVE):
            self.assertEqual(int(safe_ids[0, j]), int(l2g[e]))
            self.assertAlmostEqual(float(masked_tw[0, j]), 1.0)


if __name__ == "__main__":
    unittest.main()
