"""K3SortPlan: the model-owned inputs of the K3 sort kernels inside aiter fused_moe."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.moe import k3_moe_epi
from sglang.kernels.ops.moe import moe_sorting_small as mss
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _route_state(topk_ids, qx):
    # duck-typed k3_route_sort_mid.RouteState: the fields the plan reads
    return SimpleNamespace(
        topk_ids=topk_ids,
        qx=qx,
        counts=torch.ones(8, dtype=torch.int32),
        consumed=False,
    )


def _fake_preact(act, w_down, out):
    out.copy_(act @ w_down.t())


class TestK3SortPlanScope(CustomTestCase):
    def test_scope_activates_and_restores_plan(self):
        self.assertIsNone(mss._k3_sort_plan.get())
        with mss.k3_sort_plan() as outer:
            self.assertIs(mss._k3_sort_plan.get(), outer)
            inner_plan = mss.K3SortPlan()
            with mss.k3_sort_plan(inner_plan) as inner:
                self.assertIs(inner, inner_plan)
                self.assertIs(mss._k3_sort_plan.get(), inner_plan)
            self.assertIs(mss._k3_sort_plan.get(), outer)
        self.assertIsNone(mss._k3_sort_plan.get())

    def test_scope_restores_plan_on_error(self):
        with self.assertRaises(RuntimeError):
            with mss.k3_sort_plan():
                raise RuntimeError
        self.assertIsNone(mss._k3_sort_plan.get())


class TestK3SortPlanRouteState(CustomTestCase):
    def setUp(self):
        self.topk_ids = torch.zeros(4, 16, dtype=torch.int32)
        self.qx = torch.zeros(4, 64, dtype=torch.uint8)
        self.st = _route_state(self.topk_ids, self.qx)
        self.plan = mss.K3SortPlan(route_state=self.st)

    def test_taken_only_for_its_own_tensors(self):
        self.assertIs(self.plan.take_route_state(self.topk_ids, self.qx), self.st)
        other_ids = torch.zeros_like(self.topk_ids)
        self.assertIsNone(self.plan.take_route_state(other_ids, self.qx))
        self.assertIsNone(self.plan.take_route_state(self.topk_ids, None))
        self.assertIsNone(self.plan.take_route_state(self.topk_ids, self.qx[:2]))
        strided = self.qx.view(4, 2, 32)[:, 0]
        self.assertIsNone(self.plan.take_route_state(self.topk_ids, strided))

    def test_consumed_state_is_not_taken(self):
        self.st.consumed = True
        self.assertIsNone(self.plan.take_route_state(self.topk_ids, self.qx))

    def test_exit_zeroes_counts_only_when_sort_did_not_run(self):
        with mss.k3_sort_plan(self.plan):
            pass
        self.assertEqual(int(self.st.counts.abs().sum()), 0)

        st = _route_state(self.topk_ids, self.qx)
        with mss.k3_sort_plan(mss.K3SortPlan(route_state=st)):
            st.consumed = True
        self.assertTrue(torch.equal(st.counts, torch.ones(8, dtype=torch.int32)))


class TestK3SortPlanSharedGemm(CustomTestCase):
    def setUp(self):
        self.topk_ids = torch.zeros(4, 16, dtype=torch.int32)
        self.act = torch.randn(4, 32)
        self.w_down = torch.randn(8, 32)
        self.out = torch.zeros(4, 8)
        self.gemm = (self.topk_ids, self.act, self.w_down, self.out)

    def test_taken_once_for_its_top_k(self):
        plan = mss.K3SortPlan(shared_gemm=self.gemm)
        self.assertIsNone(plan.take_shared_gemm(torch.zeros_like(self.topk_ids)))
        taken = plan.take_shared_gemm(self.topk_ids)
        self.assertEqual(len(taken), 3)
        for got, want in zip(taken, self.gemm[1:]):
            self.assertIs(got, want)
        self.assertIsNone(plan.take_shared_gemm(self.topk_ids))

    def test_exit_runs_untaken_gemm(self):
        with mock.patch.object(k3_moe_epi, "shared_gemm_preact", _fake_preact):
            with mss.k3_sort_plan(mss.K3SortPlan(shared_gemm=self.gemm)) as plan:
                pass
        self.assertIsNone(plan.shared_gemm)
        torch.testing.assert_close(self.out, self.act @ self.w_down.t())

    def test_exit_skips_taken_gemm(self):
        with mock.patch.object(k3_moe_epi, "shared_gemm_preact", _fake_preact):
            with mss.k3_sort_plan(mss.K3SortPlan(shared_gemm=self.gemm)) as plan:
                plan.take_shared_gemm(self.topk_ids)
        self.assertEqual(int(self.out.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
