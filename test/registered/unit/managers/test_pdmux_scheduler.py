import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import sglang.srt.distributed.parallel_state as parallel_state
from sglang.srt.distributed.parallel_state import (
    is_pdmux_enabled,
    is_pdmux_prefill_enabled,
    set_pdmux_status,
)
from sglang.srt.multiplex.multiplexing_mixin import SchedulerMultiplexMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Batch:
    def __init__(self, empty):
        self._empty = empty

    def is_empty(self):
        return self._empty


class TestPDMuxScheduler(unittest.TestCase):
    def tearDown(self):
        set_pdmux_status(False)

    def _make_scheduler(
        self,
        *,
        decode_empty,
        split_index=0,
        extend_num_tokens=128000,
        token_budget=65536,
    ):
        return SimpleNamespace(
            model_config=SimpleNamespace(num_hidden_layers=61),
            pdmux_config=SimpleNamespace(split_forward_token_budget=token_budget),
            running_batch=_Batch(decode_empty),
            split_prefill_batch=SimpleNamespace(
                split_index=split_index,
                extend_num_tokens=extend_num_tokens,
            ),
        )

    def test_prefill_runs_remaining_layers_without_decode_work(self):
        scheduler = self._make_scheduler(decode_empty=True, split_index=7)

        count = SchedulerMultiplexMixin._get_split_forward_count(scheduler)

        self.assertEqual(count, 54)

    def test_prefill_uses_token_budget_with_decode_work(self):
        scheduler = self._make_scheduler(decode_empty=False)

        count = SchedulerMultiplexMixin._get_split_forward_count(scheduler)

        self.assertEqual(count, 1)

    def test_prefill_count_is_clamped_to_remaining_layers(self):
        scheduler = self._make_scheduler(
            decode_empty=False,
            split_index=59,
            extend_num_tokens=8192,
            token_budget=65536,
        )

        count = SchedulerMultiplexMixin._get_split_forward_count(scheduler)

        self.assertEqual(count, 2)

    def test_dsv4_prefill_admission_uses_planner_hard_limit(self):
        scheduler = SimpleNamespace(
            enable_pdmux=True,
            pdmux_max_prefill_plan_tokens=(1 << 16) - 1,
            page_size=16,
        )

        budget, enforce = SchedulerMultiplexMixin._get_prefill_admission_config(
            scheduler, 131072
        )

        self.assertEqual(budget, 65520)
        self.assertTrue(enforce)

    def test_non_dsv4_prefill_admission_preserves_soft_budget(self):
        scheduler = SimpleNamespace(
            enable_pdmux=True,
            pdmux_max_prefill_plan_tokens=None,
            page_size=16,
        )

        budget, enforce = SchedulerMultiplexMixin._get_prefill_admission_config(
            scheduler, 131072
        )

        self.assertEqual(budget, 131072)
        self.assertFalse(enforce)

    def test_dsv4_request_length_stays_within_planner_limit(self):
        scheduler = SimpleNamespace(
            enable_pdmux=True,
            pdmux_max_prefill_plan_tokens=(1 << 16) - 1,
            max_prefill_tokens=131072,
            page_size=16,
        )

        max_input_len = SchedulerMultiplexMixin._get_max_req_input_len(
            scheduler, 1048576
        )

        self.assertEqual(max_input_len, 65521)

    def test_dsv4_request_limit_matches_smaller_prefill_budget(self):
        scheduler = SimpleNamespace(
            enable_pdmux=True,
            pdmux_max_prefill_plan_tokens=(1 << 16) - 1,
            max_prefill_tokens=32767,
            page_size=16,
        )

        budget, enforce = SchedulerMultiplexMixin._get_prefill_admission_config(
            scheduler, scheduler.max_prefill_tokens
        )
        max_input_len = SchedulerMultiplexMixin._get_max_req_input_len(
            scheduler, 1048576
        )

        self.assertEqual(budget, 32752)
        self.assertTrue(enforce)
        self.assertEqual(max_input_len, budget + 1)

    def test_pdmux_initialization_uses_parallel_state_gpu_id(self):
        config = object()
        scheduler = SimpleNamespace(
            server_args=SimpleNamespace(pdmux_config_path="pdmux.yaml"),
            ps=SimpleNamespace(gpu_id=3),
        )

        with (
            patch(
                "sglang.srt.multiplex.multiplexing_mixin.load_pdmux_config",
                return_value=config,
            ),
            patch(
                "sglang.srt.multiplex.multiplexing_mixin.initialize_stream_groups"
            ) as initialize_stream_groups,
            patch(
                "sglang.srt.multiplex.multiplexing_mixin.get_stream_groups",
                return_value=[object(), object(), object()],
            ),
            patch(
                "sglang.srt.multiplex.multiplexing_mixin.get_sm_counts",
                return_value=[(1, 0), (1, 1), (0, 1)],
            ),
        ):
            SchedulerMultiplexMixin.init_pdmux(scheduler)

        initialize_stream_groups.assert_called_once_with(3, config)
        self.assertEqual(scheduler.real_sm_group_num, 3)

    def test_pdmux_prefill_status_is_observable(self):
        self.assertFalse(is_pdmux_prefill_enabled())

        set_pdmux_status(True)
        self.assertTrue(is_pdmux_prefill_enabled())

        set_pdmux_status(False)
        self.assertFalse(is_pdmux_prefill_enabled())

    def test_pdmux_process_status_does_not_follow_prefill_phase(self):
        with patch.object(parallel_state, "_PDMUX_PREFILL_TP_GROUP", object()):
            set_pdmux_status(False)

            self.assertTrue(is_pdmux_enabled())
            self.assertFalse(is_pdmux_prefill_enabled())

    def test_finished_prefill_merge_publishes_decode_dependency(self):
        operations = []
        split_batch = Mock()
        running_batch = Mock()
        running_batch.is_empty.return_value = False
        running_batch.merge_batch.side_effect = lambda batch: operations.append(
            ("merge", batch)
        )
        prefill_stream = Mock()
        merge_done = object()
        prefill_stream.record_event.side_effect = lambda: (
            operations.append(("record", None)) or merge_done
        )
        decode_stream = Mock()
        decode_stream.wait_event.side_effect = lambda event: operations.append(
            ("wait", event)
        )
        scheduler = SimpleNamespace(
            running_batch=running_batch,
            split_prefill_batch=split_batch,
            process_batch_result=Mock(),
        )
        prefill_result = object()

        SchedulerMultiplexMixin._merge_finished_prefill_batch(
            scheduler,
            prefill_result,
            prefill_stream,
            decode_stream,
        )

        scheduler.process_batch_result.assert_called_once_with(
            split_batch, prefill_result
        )
        self.assertEqual(
            operations,
            [("merge", split_batch), ("record", None), ("wait", merge_done)],
        )
        self.assertIsNone(scheduler.split_prefill_batch)


if __name__ == "__main__":
    unittest.main()
