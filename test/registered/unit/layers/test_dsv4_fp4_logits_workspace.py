"""CPU contract tests for the DSV4 HIP FP4 logits workspace."""

import unittest
from unittest import mock

import torch

from sglang.srt.layers.attention.dsv4.fp4_logits_workspace import (
    FP4LogitsWorkspace,
    fp4_logits_width_for_context,
    fp4_logits_width_from_page_table,
    limit_plan_to_available_memory,
    plan_fp4_logits_workspace,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFP4LogitsWorkspacePlan(unittest.TestCase):
    def test_context_width_uses_c4_page_geometry(self):
        self.assertEqual(fp4_logits_width_from_page_table(1), 256)
        self.assertEqual(fp4_logits_width_from_page_table(5), 512)
        self.assertEqual(fp4_logits_width_for_context(131072, 256), 32768)

    def test_small_workload_does_not_allocate_the_ceiling(self):
        plan = plan_fp4_logits_workspace(
            max_seq_len=32768,
            max_query_rows=64,
            runtime_headroom_bytes=8 << 30,
            free_memory_fraction=0.2,
            max_workspace_bytes=2 << 30,
        )
        self.assertEqual(plan.capacity_bytes, 64 * 32768 * 4)
        self.assertEqual(plan.rows_at_max_width, 64)
        self.assertEqual(plan.limiting_reason, "workload")

    def test_runtime_headroom_and_user_ceiling_are_hard_bounds(self):
        by_headroom = plan_fp4_logits_workspace(
            max_seq_len=32768,
            max_query_rows=4096,
            runtime_headroom_bytes=512 << 20,
            free_memory_fraction=0.25,
            max_workspace_bytes=2 << 30,
        )
        self.assertLessEqual(by_headroom.capacity_bytes, 128 << 20)
        self.assertEqual(by_headroom.limiting_reason, "runtime_headroom")

        by_ceiling = plan_fp4_logits_workspace(
            max_seq_len=32768,
            max_query_rows=4096,
            runtime_headroom_bytes=8 << 30,
            free_memory_fraction=0.5,
            max_workspace_bytes=64 << 20,
        )
        self.assertEqual(by_ceiling.capacity_bytes, 64 << 20)
        self.assertEqual(by_ceiling.limiting_reason, "user_ceiling")

    def test_budget_must_hold_one_maximum_width_row(self):
        with self.assertRaisesRegex(ValueError, "cannot hold one row"):
            plan_fp4_logits_workspace(
                max_seq_len=1 << 20,
                max_query_rows=8,
                runtime_headroom_bytes=1 << 20,
                free_memory_fraction=0.25,
                max_workspace_bytes=None,
            )

    def test_live_free_memory_only_shrinks_the_plan(self):
        plan = plan_fp4_logits_workspace(
            max_seq_len=1024,
            max_query_rows=1024,
            runtime_headroom_bytes=64 << 20,
            free_memory_fraction=1.0,
            max_workspace_bytes=4 << 20,
        )
        shrunk = limit_plan_to_available_memory(plan, 2 << 20, safety_fraction=0.5)
        self.assertEqual(shrunk.capacity_bytes, 1 << 20)
        self.assertEqual(shrunk.limiting_reason, "live_free_memory")

    def test_explicitly_reserved_one_row_survives_live_check(self):
        plan = plan_fp4_logits_workspace(
            max_seq_len=1024,
            max_query_rows=1,
            runtime_headroom_bytes=4096,
            free_memory_fraction=1.0,
            max_workspace_bytes=None,
        )
        checked = limit_plan_to_available_memory(
            plan, plan.capacity_bytes, safety_fraction=1.0
        )
        self.assertIs(checked, plan)


class TestFP4LogitsWorkspace(unittest.TestCase):
    def setUp(self):
        self.plan = plan_fp4_logits_workspace(
            max_seq_len=16,
            max_query_rows=8,
            runtime_headroom_bytes=4096,
            free_memory_fraction=1.0,
            max_workspace_bytes=None,
        )
        self.workspace = FP4LogitsWorkspace(plan=self.plan, device=torch.device("cpu"))

    def tearDown(self):
        self.workspace.close()

    def test_rows_per_chunk_is_strict_and_capped(self):
        self.assertEqual(self.workspace.rows_per_chunk(16), 8)
        self.assertEqual(self.workspace.rows_per_chunk(8, max_rows=4), 4)
        with self.assertRaisesRegex(RuntimeError, "exceeds the workspace plan"):
            self.workspace.rows_per_chunk(self.plan.capacity_elems + 1)

    def test_sequential_leases_reuse_the_same_storage(self):
        with self.workspace.acquire(2, 16) as first:
            first.fill_(3)
            first_ptr = first.data_ptr()
        with self.workspace.acquire(4, 8) as second:
            self.assertEqual(second.data_ptr(), first_ptr)
            self.assertTrue(torch.equal(second, torch.full_like(second, 3)))

    def test_oversized_lease_is_rejected_without_allocating(self):
        ptr = self.workspace.data_ptr
        with self.assertRaisesRegex(RuntimeError, "exceeds the planned capacity"):
            self.workspace.acquire(9, 16)
        self.assertEqual(self.workspace.data_ptr, ptr)

    def test_close_is_idempotent_and_rejects_future_use(self):
        self.workspace.close()
        self.workspace.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.workspace.acquire(1, 16)

    def test_capture_domain_is_rejected(self):
        self.workspace.device = torch.device("cuda")
        with mock.patch("torch.cuda.is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "graph capture"):
                self.workspace.acquire(1, 16)
        self.workspace.device = torch.device("cpu")

    def test_cross_stream_reuse_waits_for_consumer_event(self):
        class FakeStream:
            def __init__(self):
                self.waited = []

            def wait_event(self, event):
                self.waited.append(event)

        class FakeEvent:
            def __init__(self, *args, **kwargs):
                self.recorded_on = []
                self.synchronized = False

            def record(self, stream):
                self.recorded_on.append(stream)

            def synchronize(self):
                self.synchronized = True

        stream_a = FakeStream()
        stream_b = FakeStream()
        self.workspace.device = torch.device("cuda")
        self.workspace._reuse_event = FakeEvent()
        with mock.patch("torch.cuda.is_current_stream_capturing", return_value=False):
            with self.workspace.acquire(1, 16, stream=stream_a):
                pass
            event = self.workspace._reuse_event
            with self.workspace.acquire(1, 16, stream=stream_b):
                pass
            with self.workspace.acquire(1, 16, stream=stream_b):
                pass
            self.assertEqual(event.recorded_on, [stream_a])
            self.assertEqual(stream_b.waited, [event])
        self.workspace.device = torch.device("cpu")


if __name__ == "__main__":
    unittest.main()
