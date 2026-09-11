import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import sglang.srt.distributed.parallel_state as parallel_state
from sglang.srt.distributed.parallel_state import (
    is_pdmux_enabled,
    is_pdmux_prefill_enabled,
    set_pdmux_status,
)
from sglang.srt.managers.utils import validate_input_length
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

    def test_hybrid_prefill_limit_is_optional_and_delegated(self):
        from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend

        runner = SimpleNamespace(
            kv_cache_dtype="auto",
            token_to_kv_pool=None,
            req_to_token_pool=None,
            kv_index_translator=None,
            model_config=SimpleNamespace(context_len=131072),
        )
        for limit in (None, 65535):
            with self.subTest(limit=limit):
                prefill = SimpleNamespace(needs_cpu_seq_lens=False)
                if limit is not None:
                    prefill.max_prefill_plan_tokens = limit
                with patch(
                    "sglang.srt.layers.attention.hybrid_attn_backend.get_spec",
                    return_value=SimpleNamespace(speculative_attention_mode="decode"),
                ):
                    backend = HybridAttnBackend(
                        runner, prefill, SimpleNamespace(needs_cpu_seq_lens=False)
                    )
                self.assertEqual(backend.max_prefill_plan_tokens, limit)

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

        self.assertEqual(max_input_len, 65520)

        req = SimpleNamespace(origin_input_ids=list(range(65536)))
        self.assertIsNone(validate_input_length(req, max_input_len, True))
        self.assertEqual(len(req.origin_input_ids), 65520)
        budget, _ = SchedulerMultiplexMixin._get_prefill_admission_config(
            scheduler, scheduler.max_prefill_tokens
        )
        paged_tokens = (
            (len(req.origin_input_ids) + scheduler.page_size - 1)
            // scheduler.page_size
            * scheduler.page_size
        )
        self.assertLessEqual(paged_tokens, budget)

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
        self.assertEqual(max_input_len, budget)

    def test_retracted_history_over_planner_limit_is_aborted_before_queueing(self):
        from sglang.srt.disaggregation.utils import DisaggregationMode
        from sglang.srt.managers.scheduler import Scheduler

        for limit, output_len, should_abort in (
            (65520, 600, True),
            (65520, 520, False),
            (None, 600, False),
        ):
            with self.subTest(limit=limit, output_len=output_len):
                req = MagicMock(
                    rid="retracted",
                    is_retracted=True,
                    origin_input_ids=list(range(65000)),
                    output_ids=list(range(output_len)),
                )
                scheduler = SimpleNamespace(
                    enable_pdmux=True,
                    max_prefill_tokens=131072,
                    disaggregation_mode=DisaggregationMode.NULL,
                    waiting_queue=[],
                    processed_tokens_counter=0,
                    _set_or_validate_priority=Mock(return_value=True),
                    _get_pdmux_prefill_token_limit=Mock(return_value=limit),
                    _release_aborted_request=Mock(),
                    _abort_on_queued_limit=Mock(return_value=False),
                    _prefetch_kvcache=Mock(),
                    beam_coordinator=Mock(),
                    ipc_channels=Mock(),
                )
                with patch(
                    "sglang.srt.managers.scheduler._make_abort_req"
                ) as make_abort:
                    Scheduler._add_request_to_queue(scheduler, req, is_retracted=True)
                if should_abort:
                    self.assertEqual(scheduler.waiting_queue, [])
                    scheduler._prefetch_kvcache.assert_not_called()
                    scheduler._release_aborted_request.assert_called_once_with(req.rid)
                    scheduler.beam_coordinator.retire_group.assert_called_once_with(req)
                    reason = make_abort.call_args.kwargs["finished_reason"]
                    self.assertEqual(reason["status_code"], 503)
                    self.assertIn("65600", reason["message"])
                    scheduler.ipc_channels.send_to_tokenizer.send_output.assert_called_once_with(
                        make_abort.return_value, req
                    )
                else:
                    self.assertEqual(scheduler.waiting_queue, [req])
                    make_abort.assert_not_called()

    def test_pdmux_initialization_uses_parallel_state_gpu_id(self):
        config = object()
        scheduler = SimpleNamespace(
            ps=SimpleNamespace(gpu_id=3),
        )

        with (
            patch(
                "sglang.srt.multiplex.multiplexing_mixin.load_pdmux_config",
                return_value=config,
            ) as load_pdmux_config,
            patch(
                "sglang.srt.multiplex.multiplexing_mixin.get_disagg",
                return_value=SimpleNamespace(pdmux_config_path="pdmux.yaml"),
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

        load_pdmux_config.assert_called_once_with("pdmux.yaml")
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

        merged_batch = SchedulerMultiplexMixin._merge_finished_prefill_batch(
            scheduler,
            prefill_result,
            prefill_stream,
            decode_stream,
            running_batch,
        )

        scheduler.process_batch_result.assert_called_once_with(
            split_batch, prefill_result
        )
        self.assertEqual(
            operations,
            [("merge", split_batch), ("record", None), ("wait", merge_done)],
        )
        self.assertIs(merged_batch, running_batch)
        self.assertIs(scheduler.running_batch, running_batch)
        self.assertIsNone(scheduler.split_prefill_batch)


if __name__ == "__main__":
    unittest.main()
