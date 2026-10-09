import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.multiplex.multiplexing_mixin import SchedulerMultiplexMixin
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Batch:
    def __init__(self, empty):
        self._empty = empty

    def is_empty(self):
        return self._empty


class _ChunkedReq:
    """Hashable stand-in for Req (the merge path stores it in a set)."""

    def __init__(self, *, extend_end, prefix_len):
        self.extend_range = SimpleNamespace(end=extend_end)
        self.prefix_indices = [0] * prefix_len


def _make_chunked_req(*, extend_end, prefix_len):
    return _ChunkedReq(extend_end=extend_end, prefix_len=prefix_len)


class TestPDMuxScheduler(unittest.TestCase):
    @staticmethod
    def _bind_merge(scheduler):
        scheduler._merge_completed_prefill_batch = lambda **kwargs: (
            SchedulerMultiplexMixin._merge_completed_prefill_batch(scheduler, **kwargs)
        )
        return scheduler

    def _make_scheduler(
        self,
        *,
        decode_empty,
        split_index=0,
        extend_num_tokens=128000,
        scheduler_global_num_tokens=None,
        token_budget=65536,
        max_split_forward_layers=0,
    ):
        return SimpleNamespace(
            model_config=SimpleNamespace(num_hidden_layers=61),
            pdmux_config=SimpleNamespace(
                split_forward_token_budget=token_budget,
                max_split_forward_layers=max_split_forward_layers,
            ),
            running_batch=_Batch(decode_empty),
            split_prefill_batch=SimpleNamespace(
                split_index=split_index,
                extend_num_tokens=extend_num_tokens,
                scheduler_global_num_tokens=scheduler_global_num_tokens,
            ),
        )

    def test_layer_cap_respects_token_budget_and_remaining_layers(self):
        decode_batch = SimpleNamespace(scheduler_global_num_tokens=[1])
        cases = (
            (2048, 0, 2, 2),
            (131072, 0, 2, 1),
            (4096, 60, 2, 1),
            (4096, 0, 0, 16),
        )
        for tokens, split_index, cap, expected in cases:
            with self.subTest(tokens=tokens, split_index=split_index, cap=cap):
                scheduler = self._make_scheduler(
                    decode_empty=False,
                    split_index=split_index,
                    extend_num_tokens=tokens,
                    max_split_forward_layers=cap,
                )
                self.assertEqual(
                    SchedulerMultiplexMixin._get_split_forward_count(
                        scheduler, decode_batch
                    ),
                    expected,
                )

    def test_layer_cap_keeps_active_and_idle_dp_ranks_aligned(self):
        global_num_tokens = [4096, 1024, 0]
        decode_batch = SimpleNamespace(scheduler_global_num_tokens=[0, 1, 0])
        for local_num_tokens in global_num_tokens:
            with self.subTest(local_num_tokens=local_num_tokens):
                scheduler = self._make_scheduler(
                    decode_empty=local_num_tokens == 0,
                    extend_num_tokens=local_num_tokens,
                    scheduler_global_num_tokens=global_num_tokens,
                    max_split_forward_layers=2,
                )
                self.assertEqual(
                    SchedulerMultiplexMixin._get_split_forward_count(
                        scheduler, decode_batch
                    ),
                    2,
                )

    def test_layer_cap_does_not_split_without_global_decode(self):
        scheduler = self._make_scheduler(
            decode_empty=True,
            split_index=7,
            extend_num_tokens=4096,
            max_split_forward_layers=2,
        )
        for decode_batch in (
            None,
            SimpleNamespace(scheduler_global_num_tokens=[0, 0]),
        ):
            with self.subTest(decode_batch=decode_batch):
                self.assertEqual(
                    SchedulerMultiplexMixin._get_split_forward_count(
                        scheduler, decode_batch
                    ),
                    54,
                )

    def test_prefill_runs_remaining_layers_without_decode_work(self):
        scheduler = self._make_scheduler(decode_empty=True, split_index=7)

        count = SchedulerMultiplexMixin._get_split_forward_count(
            scheduler, decode_batch=None
        )

        self.assertEqual(count, 54)

    def test_prefill_uses_token_budget_with_decode_work(self):
        scheduler = self._make_scheduler(decode_empty=False)

        count = SchedulerMultiplexMixin._get_split_forward_count(
            scheduler,
            decode_batch=SimpleNamespace(scheduler_global_num_tokens=[1]),
        )

        self.assertEqual(count, 1)

    def test_prefill_count_is_clamped_to_remaining_layers(self):
        scheduler = self._make_scheduler(
            decode_empty=False,
            split_index=59,
            extend_num_tokens=8192,
            token_budget=65536,
        )

        count = SchedulerMultiplexMixin._get_split_forward_count(
            scheduler,
            decode_batch=SimpleNamespace(scheduler_global_num_tokens=[1]),
        )

        self.assertEqual(count, 2)

    def test_max_rank_token_budget_runs_45_layers_for_small_prefills(self):
        scheduler = self._make_scheduler(
            decode_empty=False,
            extend_num_tokens=1024,
            scheduler_global_num_tokens=[1024] * 8,
        )
        scheduler.model_config.num_hidden_layers = 45
        decode_batch = SimpleNamespace(scheduler_global_num_tokens=[1] * 8)
        counts = []

        while scheduler.split_prefill_batch.split_index < 45:
            count = SchedulerMultiplexMixin._get_split_forward_count(
                scheduler, decode_batch
            )
            self.assertGreater(count, 0)
            counts.append(count)
            scheduler.split_prefill_batch.split_index += count

        self.assertEqual(counts, [45])

    def test_global_token_budget_matches_on_uneven_active_and_idle_dp_ranks(self):
        global_num_tokens = [4096, 1024, 0]
        decode_batch = SimpleNamespace(scheduler_global_num_tokens=[0, 1, 0])

        for local_num_tokens in global_num_tokens:
            with self.subTest(local_num_tokens=local_num_tokens):
                scheduler = self._make_scheduler(
                    decode_empty=local_num_tokens == 0,
                    extend_num_tokens=local_num_tokens,
                    scheduler_global_num_tokens=global_num_tokens,
                )

                count = SchedulerMultiplexMixin._get_split_forward_count(
                    scheduler, decode_batch
                )

                self.assertEqual(count, 16)

    def test_empty_global_prefill_tokens_do_not_divide_by_zero(self):
        for global_num_tokens in ([], [0, 0]):
            with self.subTest(global_num_tokens=global_num_tokens):
                scheduler = self._make_scheduler(
                    decode_empty=False,
                    split_index=7,
                    extend_num_tokens=0,
                    scheduler_global_num_tokens=global_num_tokens,
                )

                count = SchedulerMultiplexMixin._get_split_forward_count(
                    scheduler,
                    decode_batch=SimpleNamespace(scheduler_global_num_tokens=[1, 0]),
                )

                self.assertEqual(count, 54)

    def test_prefill_split_count_matches_on_active_and_idle_dp_ranks(self):
        active_scheduler = self._make_scheduler(
            decode_empty=False,
            extend_num_tokens=16384,
            scheduler_global_num_tokens=[16384, 0],
        )
        idle_scheduler = self._make_scheduler(
            decode_empty=True,
            extend_num_tokens=0,
            scheduler_global_num_tokens=[16384, 0],
        )

        active_count = SchedulerMultiplexMixin._get_split_forward_count(
            active_scheduler,
            decode_batch=SimpleNamespace(scheduler_global_num_tokens=[0, 3]),
        )
        idle_count = SchedulerMultiplexMixin._get_split_forward_count(
            idle_scheduler,
            decode_batch=SimpleNamespace(scheduler_global_num_tokens=[0, 3]),
        )

        self.assertEqual(active_count, 4)
        self.assertEqual(idle_count, active_count)

    def test_idle_decode_participants_do_not_force_prefill_splitting(self):
        scheduler = self._make_scheduler(
            decode_empty=True,
            split_index=7,
            extend_num_tokens=16384,
            scheduler_global_num_tokens=[16384, 0],
        )

        count = SchedulerMultiplexMixin._get_split_forward_count(
            scheduler,
            decode_batch=SimpleNamespace(scheduler_global_num_tokens=[0, 0]),
        )

        self.assertEqual(count, 54)

    @staticmethod
    @contextmanager
    def _stubbed_stream_idx():
        """Stand in for the module-level stream-index state.

        The real setter validates against `STREAM_GROUPS`, which only
        `initialize_stream_groups` fills and which needs a GPU.
        """
        state = {"idx": 0}
        with (
            patch(
                "sglang.srt.multiplex.multiplexing_mixin.set_current_stream_idx",
                lambda idx: state.update(idx=idx),
            ),
            patch(
                "sglang.srt.multiplex.multiplexing_mixin.get_current_stream_idx",
                lambda: state["idx"],
            ),
        ):
            yield

    def _make_stream_group_scheduler(self, *, manual_divisions, group_num):
        model_runner = SimpleNamespace(update_decode_attn_backend=Mock())
        scheduler = SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.NONE,
            split_prefill_batch=object(),
            pdmux_config=SimpleNamespace(
                manual_divisions=manual_divisions, decode_bs_divisor=36
            ),
            real_sm_group_num=group_num,
            tp_worker=SimpleNamespace(model_runner=model_runner),
            stream_groups=[(f"p{i}", f"d{i}") for i in range(group_num)],
        )
        scheduler._update_decode_attn_backends = lambda index: (
            SchedulerMultiplexMixin._update_decode_attn_backends(scheduler, index)
        )
        return scheduler

    def test_manual_division_below_every_threshold_uses_first_shared_group(self):
        """A decode batch under every configured threshold still needs a group.

        The selection loop only assigns a stream index on a threshold it meets,
        so a batch below all of them left the index unbound -- an
        UnboundLocalError raised from the scheduler loop. A single-division
        config (`--sm-group-num 3`) whose threshold is above 1 hits this for
        every small decode batch that overlaps a split prefill.
        """
        scheduler = self._make_stream_group_scheduler(
            manual_divisions=[[128, 0, 8]], group_num=3
        )
        decode_batch = SimpleNamespace(is_empty=lambda: False, batch_size=lambda: 1)

        with self._stubbed_stream_idx():
            stream_idx, stream_group = SchedulerMultiplexMixin.adjust_stream_groups(
                scheduler, decode_batch
            )

        self.assertEqual(stream_idx, 1)
        self.assertEqual(stream_group, ("p1", "d1"))

    def test_manual_division_picks_the_highest_met_threshold(self):
        scheduler = self._make_stream_group_scheduler(
            manual_divisions=[[32, 0, 1], [64, 0, 8]], group_num=4
        )
        decode_batch = SimpleNamespace(is_empty=lambda: False, batch_size=lambda: 12)

        with self._stubbed_stream_idx():
            stream_idx, _ = SchedulerMultiplexMixin.adjust_stream_groups(
                scheduler, decode_batch
            )

        self.assertEqual(stream_idx, 2)

    def test_idle_decode_rank_uses_active_peers_stream_group(self):
        scheduler = self._make_stream_group_scheduler(
            manual_divisions=[[104, 0, 0]], group_num=3
        )
        active = SimpleNamespace(
            is_empty=lambda: False,
            batch_size=lambda: 2,
            scheduler_global_num_tokens=[2, 0],
        )
        idle = SimpleNamespace(
            is_empty=lambda: True,
            batch_size=lambda: 0,
            scheduler_global_num_tokens=[2, 0],
        )

        with self._stubbed_stream_idx():
            active_idx, _ = SchedulerMultiplexMixin.adjust_stream_groups(
                scheduler, active
            )
            idle_idx, _ = SchedulerMultiplexMixin.adjust_stream_groups(scheduler, idle)

        self.assertEqual(active_idx, 1)
        self.assertEqual(idle_idx, 1)

        idle.scheduler_global_num_tokens = [0, 0]
        with self._stubbed_stream_idx():
            no_decode_idx, _ = SchedulerMultiplexMixin.adjust_stream_groups(
                scheduler, idle
            )
        self.assertEqual(no_decode_idx, 0)

    def test_manual_division_uses_global_decode_size(self):
        scheduler = self._make_stream_group_scheduler(
            manual_divisions=[[32, 0, 1], [64, 0, 8]], group_num=4
        )
        batches = [
            SimpleNamespace(
                is_empty=lambda size=local_size: size == 0,
                batch_size=lambda size=local_size: size,
                scheduler_global_num_tokens=[2, 12, 0],
            )
            for local_size in (2, 12, 0)
        ]

        with self._stubbed_stream_idx():
            indices = [
                SchedulerMultiplexMixin.adjust_stream_groups(scheduler, batch)[0]
                for batch in batches
            ]

        self.assertEqual(indices, [2, 2, 2])

    def test_stream_switch_uses_speculative_worker_backend_hook(self):
        scheduler = self._make_stream_group_scheduler(
            manual_divisions=[[32, 100, 1]], group_num=3
        )
        scheduler.model_worker = SimpleNamespace(
            update_pdmux_decode_attn_backend=Mock()
        )
        batch = SimpleNamespace(is_empty=lambda: False, batch_size=lambda: 2)
        with self._stubbed_stream_idx():
            index, _ = SchedulerMultiplexMixin.adjust_stream_groups(scheduler, batch)
        scheduler.model_worker.update_pdmux_decode_attn_backend.assert_called_once_with(
            index
        )
        scheduler.tp_worker.model_runner.update_decode_attn_backend.assert_not_called()

    def test_decode_only_and_globally_idle_stream_selection(self):
        scheduler = self._make_stream_group_scheduler(
            manual_divisions=[[32, 0, 1]], group_num=3
        )
        scheduler.split_prefill_batch = None
        idle_peer = SimpleNamespace(
            is_empty=lambda: True,
            batch_size=lambda: 0,
            scheduler_global_num_tokens=[2, 0],
        )
        with self._stubbed_stream_idx():
            active_idx, _ = SchedulerMultiplexMixin.adjust_stream_groups(
                scheduler, idle_peer
            )
            idle_peer.scheduler_global_num_tokens = [0, 0]
            empty_idx, _ = SchedulerMultiplexMixin.adjust_stream_groups(
                scheduler, idle_peer
            )
            absent_idx, _ = SchedulerMultiplexMixin.adjust_stream_groups(
                scheduler, None
            )
        self.assertEqual((active_idx, empty_idx, absent_idx), (2, 0, 0))

    def test_split_prefill_routes_spec_worker_and_publishes_only_final_draft(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.future_map = object()
        scheduler.tp_worker = Mock()
        scheduler.model_worker = Mock()
        scheduler._relay_forward_payload = Mock()
        scheduler._copy_auxiliary_output_to_cpu = Mock()
        state = object()
        old_draft = object()
        seq_lens = torch.tensor([3, 4])
        batch = SimpleNamespace(
            split_index=0,
            split_forward_batch=state,
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            spec_info=old_draft,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.clone(),
            seq_lens_sum=7,
            req_pool_indices=torch.tensor([0, 1]),
            input_ids=torch.tensor([1]),
        )
        intermediate = GenerationBatchResult(
            logits_output=None, can_run_cuda_graph=False
        )
        final_draft = object()
        new_seq_lens = torch.tensor([4, 5])
        final = GenerationBatchResult(
            logits_output=None,
            can_run_cuda_graph=False,
            next_draft_input=final_draft,
            new_seq_lens=new_seq_lens,
        )
        scheduler.model_worker.forward_batch_split_prefill.side_effect = [
            intermediate,
            final,
        ]
        with patch("sglang.srt.managers.scheduler.resolve_forward_inputs") as resolve:
            self.assertIs(scheduler._run_pdmux_split_prefill(batch), intermediate)
            self.assertIs(batch.spec_info, old_draft)
            self.assertIs(batch.seq_lens, seq_lens)
            self.assertIs(batch.split_forward_batch, state)
            scheduler._relay_forward_payload.assert_not_called()
            batch.split_index = 1
            self.assertIs(scheduler._run_pdmux_split_prefill(batch), final)
            resolve.assert_called_once_with(batch, scheduler.future_map)
        scheduler.tp_worker.forward_batch_split_prefill.assert_not_called()
        self.assertIs(batch.spec_info, final_draft)
        self.assertIs(batch.seq_lens, new_seq_lens)
        self.assertEqual(batch.seq_lens_sum, 7)
        torch.testing.assert_close(batch.seq_lens_cpu, seq_lens)
        self.assertIsNotNone(final.new_seq_lens_cpu)
        scheduler._retire_pdmux_seq_lens(batch, final)
        self.assertEqual(batch.seq_lens_sum, 9)
        torch.testing.assert_close(batch.seq_lens_cpu, new_seq_lens)
        self.assertIsNone(final.new_seq_lens_cpu)
        self.assertIsNone(batch.input_ids)
        scheduler._relay_forward_payload.assert_called_once_with(
            batch, batch.req_pool_indices, final
        )

    def test_plain_split_prefill_uses_target_worker(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.model_worker = Mock()
        scheduler.tp_worker = Mock()
        scheduler._copy_auxiliary_output_to_cpu = Mock()
        scheduler._relay_forward_payload = Mock()
        result = GenerationBatchResult(logits_output=None, can_run_cuda_graph=False)
        scheduler.tp_worker.forward_batch_split_prefill.return_value = result
        batch = SimpleNamespace(split_index=1, spec_algorithm=SpeculativeAlgorithm.NONE)
        self.assertIs(scheduler._run_pdmux_split_prefill(batch), result)
        scheduler.model_worker.forward_batch_split_prefill.assert_not_called()

    def test_decode_seq_lens_cpu_is_published_only_after_retirement(self):
        for pdmux in (True, False):
            with self.subTest(pdmux=pdmux):
                scheduler = Scheduler.__new__(Scheduler)
                scheduler.metrics_reporter = Mock()
                scheduler.forward_ct = 0
                scheduler._sched_idled = False
                scheduler.scripted_scheduler_hook = None
                scheduler.profiler_manager = Mock()
                scheduler.forward_sleep_time = None
                scheduler.disaggregation_mode = DisaggregationMode.NULL
                scheduler.is_generation = True
                scheduler.enable_overlap = False
                scheduler.enable_pdmux = pdmux
                scheduler.split_prefill_batch = None
                scheduler.pp_group = SimpleNamespace(is_last_rank=True)
                scheduler.future_map = Mock()
                scheduler._forward_isolation = lambda *args, **kwargs: nullcontext()
                scheduler.update_cache_from_scheduler = Mock()
                scheduler._maybe_report_active_ranks = Mock()
                scheduler.device_module = SimpleNamespace(Event=Mock())
                result = GenerationBatchResult(
                    logits_output=None,
                    can_run_cuda_graph=False,
                    next_draft_input=object(),
                    new_seq_lens=torch.tensor([4, 6]),
                )
                scheduler.model_worker = Mock()
                scheduler.model_worker.forward_batch_generation.return_value = result
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.DECODE,
                    extend_num_tokens=0,
                    spec_algorithm=SpeculativeAlgorithm.DSPARK,
                    spec_info=None,
                    seq_lens=torch.tensor([3, 4]),
                    seq_lens_cpu=torch.tensor([3, 4]),
                    seq_lens_sum=7,
                    return_logprob=False,
                    return_hidden_states=False,
                )
                with (
                    patch("sglang.srt.managers.scheduler.resolve_forward_inputs"),
                    patch(
                        "sglang.srt.managers.scheduler.get_parallel",
                        return_value=SimpleNamespace(pp_size=1),
                    ),
                ):
                    # Exercise the actual run_batch body, without the metrics decorator.
                    self.assertIs(
                        Scheduler.run_batch.__wrapped__(scheduler, batch), result
                    )
                self.assertIs(batch.seq_lens, result.new_seq_lens)
                if pdmux:
                    self.assertEqual(batch.seq_lens_sum, 7)
                    torch.testing.assert_close(batch.seq_lens_cpu, torch.tensor([3, 4]))
                    scheduler._retire_pdmux_seq_lens(batch, result)
                self.assertEqual(batch.seq_lens_sum, 10)
                torch.testing.assert_close(batch.seq_lens_cpu, torch.tensor([4, 6]))

    def test_first_target_slice_passes_hidden_capture_request(self):
        worker = TpModelWorker.__new__(TpModelWorker)
        worker.set_hicache_consumer = Mock()
        worker._maybe_finalize_elastic_cuda_graph_scale = Mock()
        worker._model_runner = Mock()
        worker._model_runner.forward.return_value = SimpleNamespace(
            logits_output=None, can_run_graph=False, expert_distribution_metrics=None
        )
        batch = SimpleNamespace(
            hicache_consumer_index=7, split_index=0, split_forward_count=1
        )
        persistent = object()
        with patch(
            "sglang.srt.managers.tp_worker.ForwardBatch.init_new",
            return_value=persistent,
        ) as initialize:
            TpModelWorker.forward_batch_split_prefill(
                worker, batch, capture_hidden_mode=CaptureHiddenMode.FULL
            )
        self.assertEqual(
            initialize.call_args.kwargs["capture_hidden_mode"], CaptureHiddenMode.FULL
        )
        self.assertIs(batch.split_forward_batch, persistent)

    def test_split_prefill_forward_installs_hicache_consumer_first(self):
        """Every split-prefill segment must install the HiCache consumer index
        before running the model.

        `set_hicache_consumer` selects which layer-transfer event set the KV
        pool waits on before reading loaded-back pages, and the decode forward
        that runs between segments resets it to the decode batch's -1 --
        which disables the wait entirely. A segment that skips the install
        therefore reads host-loaded KV while the transfer stream is still
        copying: garbage indices out of the DSV4 top-k indexer and a
        device-side IndexKernel assert under load (the PDMux + HiCache
        benchmark crash of 2026-08-26). This is the only forward entry point
        besides `forward_batch_generation`, which does install it.
        """
        worker = TpModelWorker.__new__(TpModelWorker)
        calls = Mock()
        worker.set_hicache_consumer = calls.consumer
        worker._maybe_finalize_elastic_cuda_graph_scale = Mock()
        worker._model_runner = SimpleNamespace(forward=calls.forward)
        calls.forward.return_value = SimpleNamespace(
            logits_output=None, can_run_graph=False, expert_distribution_metrics=None
        )
        batch = SimpleNamespace(
            hicache_consumer_index=7,
            split_index=1,
            split_forward_batch=object(),
            split_forward_count=2,
        )
        TpModelWorker.forward_batch_split_prefill(worker, batch)
        self.assertEqual(calls.mock_calls[0].args, (7,))
        self.assertEqual(calls.mock_calls[0][0], "consumer")
        self.assertEqual(calls.mock_calls[1][0], "forward")

    def test_pdmux_initialization_uses_device_context_gpu_id(self):
        config = object()
        scheduler = SimpleNamespace()

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
                "sglang.srt.multiplex.multiplexing_mixin.get_device",
                return_value=SimpleNamespace(gpu_id=3),
            ),
            patch(
                "sglang.srt.multiplex.multiplexing_mixin.get_parallel",
                return_value=SimpleNamespace(attn_dp_enabled=False),
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
            patch("torch.cuda.Stream", return_value=Mock()),
        ):
            SchedulerMultiplexMixin.init_pdmux(scheduler)

        load_pdmux_config.assert_called_once_with("pdmux.yaml")
        initialize_stream_groups.assert_called_once_with(3, config)
        self.assertEqual(scheduler.real_sm_group_num, 3)

    def _make_merge_streams(self, operations):
        prefill_stream = Mock()
        merge_done = object()
        prefill_stream.record_event.side_effect = lambda: (
            operations.append(("record", None)) or merge_done
        )
        decode_stream = Mock()
        decode_stream.wait_event.side_effect = lambda event: operations.append(
            ("wait", event)
        )
        return prefill_stream, decode_stream, merge_done

    def test_finished_prefill_merge_publishes_decode_dependency(self):
        operations = []
        split_batch = Mock()
        split_batch.chunked_req = None
        split_batch.is_empty.return_value = False
        # The unconditional filter drops nothing here: same size before/after.
        split_batch.batch_size.side_effect = [2, 2]
        running_batch = Mock()
        running_batch.is_empty.return_value = False
        running_batch.batch_is_full = True
        running_batch.merge_batch.side_effect = lambda batch: operations.append(
            ("merge", batch)
        )
        prefill_stream, decode_stream, merge_done = self._make_merge_streams(operations)
        scheduler = SimpleNamespace(
            running_batch=running_batch,
            split_prefill_batch=split_batch,
            chunked_req=None,
            process_batch_result=Mock(),
        )
        prefill_result = object()

        merged_batch = SchedulerMultiplexMixin._merge_finished_prefill_batch(
            self._bind_merge(scheduler),
            prefill_result,
            prefill_stream,
            decode_stream,
            running_batch,
        )

        scheduler.process_batch_result.assert_called_once_with(
            split_batch, prefill_result
        )
        split_batch.filter_batch.assert_called_once_with(chunked_req_to_exclude=[])
        self.assertTrue(running_batch.batch_is_full)
        self.assertEqual(
            operations,
            [("merge", split_batch), ("record", None), ("wait", merge_done)],
        )
        self.assertIs(merged_batch, running_batch)
        self.assertIs(scheduler.running_batch, running_batch)
        self.assertIsNone(scheduler.split_prefill_batch)

    def test_finished_prefill_releases_persistent_forward_batch(self):
        split_forward_batch = object()
        split_batch = Mock()
        split_batch.chunked_req = None
        split_batch.split_forward_batch = split_forward_batch
        split_batch.split_index = 61
        split_batch.split_forward_count = 4
        split_batch.split_prefill_finished = True
        split_batch.batch_size.side_effect = [1, 1]
        split_batch.is_empty.return_value = False
        running_batch = Mock()
        running_batch.is_empty.return_value = True
        running_batch.batch_is_full = True
        prefill_stream, decode_stream, merge_done = self._make_merge_streams([])
        scheduler = SimpleNamespace(
            running_batch=running_batch,
            split_prefill_batch=split_batch,
            chunked_req=None,
            process_batch_result=Mock(),
        )

        returned = SchedulerMultiplexMixin._merge_finished_prefill_batch(
            self._bind_merge(scheduler),
            prefill_result=object(),
            prefill_stream=prefill_stream,
            decode_stream=decode_stream,
            running_batch=running_batch,
        )

        self.assertIs(returned, split_batch)
        self.assertIsNone(split_batch.split_forward_batch)
        self.assertEqual(split_batch.split_index, 0)
        self.assertEqual(split_batch.split_forward_count, 1)
        self.assertFalse(split_batch.split_prefill_finished)
        self.assertIsNone(scheduler.split_prefill_batch)
        decode_stream.wait_event.assert_called_once_with(merge_done)

    def test_merge_excludes_and_stashes_unfinished_chunked_request(self):
        """A request that only finished a middle chunk must be stashed and
        kept out of the decode batch; merging it would start decoding with a
        partial prefill."""
        operations = []
        chunked_req = _make_chunked_req(extend_end=32, prefix_len=16)
        split_batch = Mock()
        split_batch.chunked_req = chunked_req
        split_batch.split_prefill_finished = True
        split_batch.batch_size.side_effect = [2, 1]
        split_batch.is_empty.return_value = False
        split_batch.filter_batch.side_effect = lambda **kwargs: operations.append(
            ("filter", kwargs)
        )
        running_batch = Mock()
        running_batch.is_empty.return_value = False
        running_batch.batch_is_full = True
        running_batch.merge_batch.side_effect = lambda batch: operations.append(
            ("merge", batch)
        )
        prefill_stream, decode_stream, merge_done = self._make_merge_streams(operations)
        scheduler = SimpleNamespace(
            running_batch=running_batch,
            split_prefill_batch=split_batch,
            chunked_req=chunked_req,
            process_batch_result=Mock(),
            stash_chunked_request=Mock(
                side_effect=lambda req: operations.append(("stash", req))
            ),
        )

        merged_batch = SchedulerMultiplexMixin._merge_finished_prefill_batch(
            self._bind_merge(scheduler),
            object(),
            prefill_stream,
            decode_stream,
            running_batch,
        )

        scheduler.stash_chunked_request.assert_called_once_with(chunked_req)
        (filter_op,) = [op for op in operations if op[0] == "filter"]
        self.assertEqual(filter_op[1]["chunked_req_to_exclude"], [chunked_req])
        self.assertFalse(running_batch.batch_is_full)
        self.assertEqual(
            [op[0] for op in operations],
            ["stash", "filter", "merge", "record", "wait"],
        )
        self.assertIs(merged_batch, running_batch)
        self.assertIsNone(scheduler.split_prefill_batch)

    def test_merge_of_pure_middle_chunk_keeps_decode_batch(self):
        """A batch holding only a middle chunk merges nothing into decode, but
        the dependency event must still be published: the stash frees
        deduplicated KV pages on the prefill stream that decode may
        reallocate right after."""
        operations = []
        chunked_req = _make_chunked_req(extend_end=32, prefix_len=16)
        split_batch = Mock()
        split_batch.chunked_req = chunked_req
        split_batch.split_prefill_finished = True
        split_batch.batch_size.side_effect = [1, 0]
        split_batch.is_empty.return_value = True
        running_batch = Mock()
        running_batch.batch_is_full = True
        prefill_stream, decode_stream, merge_done = self._make_merge_streams(operations)
        scheduler = SimpleNamespace(
            running_batch=running_batch,
            split_prefill_batch=split_batch,
            chunked_req=chunked_req,
            process_batch_result=Mock(),
            stash_chunked_request=Mock(),
        )

        merged_batch = SchedulerMultiplexMixin._merge_finished_prefill_batch(
            self._bind_merge(scheduler),
            object(),
            prefill_stream,
            decode_stream,
            running_batch,
        )

        running_batch.merge_batch.assert_not_called()
        self.assertIs(merged_batch, running_batch)
        self.assertIs(scheduler.running_batch, running_batch)
        self.assertFalse(running_batch.batch_is_full)
        self.assertEqual(
            [op[0] for op in operations],
            ["record", "wait"],
        )

    def test_merge_skips_stash_for_parked_chunk(self):
        """A parked chunk (no new KV beyond the cached prefix) must be
        excluded from the merge without being stashed — stashing it would be
        a no-op insert that still pays radix-cache work."""
        chunked_req = _make_chunked_req(extend_end=16, prefix_len=16)
        split_batch = Mock()
        split_batch.chunked_req = chunked_req
        split_batch.batch_size.side_effect = [1, 0]
        split_batch.is_empty.return_value = True
        running_batch = Mock()
        prefill_stream, decode_stream, _ = self._make_merge_streams([])
        scheduler = SimpleNamespace(
            running_batch=running_batch,
            split_prefill_batch=split_batch,
            chunked_req=chunked_req,
            process_batch_result=Mock(),
            stash_chunked_request=Mock(),
        )

        SchedulerMultiplexMixin._merge_finished_prefill_batch(
            self._bind_merge(scheduler),
            object(),
            prefill_stream,
            decode_stream,
            running_batch,
        )

        scheduler.stash_chunked_request.assert_not_called()
        split_batch.filter_batch.assert_called_once_with(
            chunked_req_to_exclude=[chunked_req]
        )

    def test_update_split_prefill_batch_processes_pending_chunked_abort(self):
        """PDMux never calls get_next_batch_to_run, so the mixin must drain
        pending chunked aborts itself; without this an aborted chunked request
        leaks its KV forever."""
        running_batch = _Batch(empty=True)
        scheduler = SimpleNamespace(
            split_prefill_batch=None,
            process_pending_chunked_abort=Mock(),
            _process_hicache_events=Mock(),
            get_new_batch_prefill=Mock(
                return_value=SimpleNamespace(
                    batch_to_run=None, running_batch=running_batch
                )
            ),
            dp_attn_adapter=SimpleNamespace(
                maybe_prepare_mlp_sync_batch=Mock(return_value=None)
            ),
        )

        created, returned = SchedulerMultiplexMixin.update_split_prefill_batch(
            scheduler, 1, running_batch
        )

        scheduler.process_pending_chunked_abort.assert_called_once_with()
        scheduler._process_hicache_events.assert_called_once_with()
        scheduler.dp_attn_adapter.maybe_prepare_mlp_sync_batch.assert_called_once_with(
            None
        )
        self.assertFalse(created)
        self.assertIs(returned, running_batch)

    def test_update_split_prefill_batch_accepts_peer_dp_idle_batch(self):
        running_batch = _Batch(empty=True)
        idle_mode = SimpleNamespace(is_idle=lambda: True)
        idle_batch = _Batch(empty=True)
        idle_batch.forward_mode = idle_mode
        adapter = SimpleNamespace(
            maybe_prepare_mlp_sync_batch=Mock(return_value=idle_batch)
        )
        scheduler = SimpleNamespace(
            split_prefill_batch=None,
            process_pending_chunked_abort=Mock(),
            _process_hicache_events=Mock(),
            get_new_batch_prefill=Mock(
                return_value=SimpleNamespace(
                    batch_to_run=None, running_batch=running_batch
                )
            ),
            dp_attn_adapter=adapter,
        )

        created, returned = SchedulerMultiplexMixin.update_split_prefill_batch(
            scheduler, 1, running_batch
        )

        self.assertTrue(created)
        self.assertIs(returned, running_batch)
        self.assertIs(scheduler.split_prefill_batch, idle_batch)
        self.assertIs(idle_batch.forward_mode, idle_mode)
        self.assertEqual(idle_batch.split_index, 0)
        self.assertFalse(idle_batch.split_prefill_finished)
        self.assertEqual(idle_batch.split_forward_count, 1)
        self.assertIsNone(idle_batch.split_forward_batch)
        adapter.maybe_prepare_mlp_sync_batch.assert_called_once_with(None)

    def test_update_split_prefill_batch_defers_abort_while_chunk_in_flight(self):
        """Tearing down a chunked request while its split forward is running
        is unsafe; the abort must wait for the between-chunks safe point."""
        running_batch = _Batch(empty=True)
        scheduler = SimpleNamespace(
            split_prefill_batch=Mock(),
            process_pending_chunked_abort=Mock(),
            _process_hicache_events=Mock(),
        )

        created, returned = SchedulerMultiplexMixin.update_split_prefill_batch(
            scheduler, 1, running_batch
        )

        scheduler.process_pending_chunked_abort.assert_not_called()
        scheduler._process_hicache_events.assert_not_called()
        self.assertFalse(created)
        self.assertIs(returned, running_batch)

    def test_dsv4_prefill_admission_uses_planner_hard_limit(self):
        scheduler = SimpleNamespace(
            enable_pdmux=True,
            pdmux_max_prefill_plan_tokens=(1 << 16) - 1,
            page_size=16,
            chunked_prefill_size=None,
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
            chunked_prefill_size=None,
        )

        budget, enforce = SchedulerMultiplexMixin._get_prefill_admission_config(
            scheduler, 131072
        )

        self.assertEqual(budget, 131072)
        self.assertFalse(enforce)

    def test_chunked_prefill_admission_preserves_soft_budget(self):
        """Chunked prefill enforces the planner limit per chunk, so the hard
        admission clamp must deactivate — keeping it would re-reject the long
        requests chunking exists to serve."""
        scheduler = SimpleNamespace(
            enable_pdmux=True,
            pdmux_max_prefill_plan_tokens=(1 << 16) - 1,
            page_size=16,
            chunked_prefill_size=16384,
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
            chunked_prefill_size=None,
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
            chunked_prefill_size=None,
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

    def test_init_rejects_chunked_prefill_size_over_plan_limit(self):
        """65520 is the page-aligned uint16 compressor-plan cap; a larger
        chunk budget would overflow a single prefill plan at runtime."""
        scheduler = SimpleNamespace(
            enable_pdmux=True,
            page_size=16,
            chunked_prefill_size=65536,
            max_prefill_tokens=131072,
            max_req_input_len=1048576,
        )
        attn_backend = SimpleNamespace(max_prefill_plan_tokens=(1 << 16) - 1)

        with self.assertRaisesRegex(ValueError, "65520"):
            SchedulerMultiplexMixin.init_pdmux_prefill_plan_limit(
                scheduler, attn_backend=attn_backend
            )

    def test_init_accepts_chunked_prefill_size_at_plan_limit(self):
        """With a valid chunk budget the per-request length clamp must stay
        off: chunking is what serves requests beyond the planner limit."""
        scheduler = SimpleNamespace(
            enable_pdmux=True,
            page_size=16,
            chunked_prefill_size=65520,
            max_prefill_tokens=131072,
            max_req_input_len=1048576,
        )
        attn_backend = SimpleNamespace(max_prefill_plan_tokens=(1 << 16) - 1)

        SchedulerMultiplexMixin.init_pdmux_prefill_plan_limit(
            scheduler, attn_backend=attn_backend
        )

        self.assertEqual(scheduler.pdmux_max_prefill_plan_tokens, (1 << 16) - 1)
        self.assertEqual(scheduler.max_req_input_len, 1048576)

    def test_init_tightens_request_length_without_chunked_prefill(self):
        """Without chunking PDMux cannot split an oversized request, so init
        must clamp request validation to the planner limit."""
        scheduler = SimpleNamespace(
            enable_pdmux=True,
            page_size=16,
            chunked_prefill_size=None,
            max_prefill_tokens=131072,
            max_req_input_len=1048576,
        )
        attn_backend = SimpleNamespace(max_prefill_plan_tokens=(1 << 16) - 1)

        SchedulerMultiplexMixin.init_pdmux_prefill_plan_limit(
            scheduler, attn_backend=attn_backend
        )

        self.assertEqual(scheduler.max_req_input_len, 65521)


if __name__ == "__main__":
    unittest.main()
