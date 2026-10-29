import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers import overlap_utils
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.allocation_sizing import (
    get_alloc_len_per_decode,
    get_alloc_reserve_per_decode,
    get_req_to_token_extra_context_len,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.platforms.rocm import RocmSRTPlatform
from sglang.srt.runtime_context import get_context, get_parallel, get_resources
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative import eagle_utils, prepared_next_draft
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.srt.speculative.prepared_next_draft import (
    NextDraftInputStager,
    PreparedNextDraftInputs,
    PreparedNextDraftProducer,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _req(slot: int, *, allocated: int, committed: int = 100) -> Req:
    sampling_params = SamplingParams(max_new_tokens=16, temperature=0)
    sampling_params.normalize(None)
    req = Req(
        rid=f"r{slot}",
        origin_input_text="",
        origin_input_ids=[1, 2, 3],
        sampling_params=sampling_params,
    )
    req.kv.req_pool_idx = slot
    req.kv.kv_committed_len = committed
    req.kv.kv_allocated_len = allocated
    return req


def _batch(slots, allocated, *, mode=ForwardMode.DECODE):
    return SimpleNamespace(
        reqs=[
            _req(slot, allocated=alloc)
            for slot, alloc in zip(slots, allocated, strict=True)
        ],
        forward_mode=mode,
        multimodal_inputs=[None] * len(slots),
    )


def _pool(size: int = 32):
    return SimpleNamespace(req_generation=torch.ones(size, dtype=torch.int64))


def _stager(pool=None, *, graph_sizes=(2, 4), with_mrope=False):
    return NextDraftInputStager(
        req_to_token_pool=_pool() if pool is None else pool,
        window_size=7,
        graph_sizes=graph_sizes,
        with_mrope=with_mrope,
    )


def _prepared(num_requests: int = 2) -> PreparedNextDraftInputs:
    return PreparedNextDraftInputs(
        num_requests=num_requests,
        runner_request_pool_indices=torch.zeros(num_requests, dtype=torch.int64),
        runner_sequence_lengths=torch.zeros(num_requests, dtype=torch.int64),
        positions=torch.zeros(num_requests, dtype=torch.int64),
        mrope_positions=None,
        draft_out_cache_loc=torch.zeros(num_requests * 6, dtype=torch.int64),
        write_window=torch.zeros(num_requests * 7, dtype=torch.int64),
    )


def _decode_prepare_batch(reqs, *, penalized=False):
    return SimpleNamespace(
        reqs=reqs,
        maybe_evict_swa=lambda: None,
        sampling_info=SimpleNamespace(
            penalizer_orchestrator=SimpleNamespace(is_required=penalized)
        ),
        cumulate_penalty_output_tokens=lambda: None,
        token_to_kv_pool_allocator=SimpleNamespace(page_size=1),
        tree_cache=SimpleNamespace(supports_swa=lambda: False),
        req_to_token_pool=SimpleNamespace(),
        req_pool_indices=torch.tensor([req.kv.req_pool_idx for req in reqs]),
        seq_lens_cpu=torch.tensor([req.kv.kv_committed_len for req in reqs]),
        device="cpu",
        spec_decode_schedule_stream_idle=False,
    )


class TestPreparedNextDraft(CustomTestCase):
    def test_enable_policy_honors_explicit_child_override(self):
        spec = SimpleNamespace(speculative_algorithm="EAGLE", speculative_eagle_topk=1)
        schedule = SimpleNamespace(disable_overlap_schedule=False)
        parallel = SimpleNamespace(moe_ep_size=1)
        for master, child, expected in (
            (False, True, True),
            (True, False, False),
        ):
            prepared_next_draft.prepared_next_draft_enabled.cache_clear()
            with (
                self.subTest(master=master, child=child),
                envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(master),
                envs.SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH.override(child),
                patch.object(prepared_next_draft, "get_spec", return_value=spec),
                patch.object(
                    prepared_next_draft, "get_schedule", return_value=schedule
                ),
                patch.object(
                    prepared_next_draft, "get_parallel", return_value=parallel
                ),
                patch.object(
                    prepared_next_draft.current_platform,
                    "is_cuda",
                    return_value=True,
                ),
            ):
                self.assertIs(
                    prepared_next_draft.prepared_next_draft_enabled(), expected
                )
        prepared_next_draft.prepared_next_draft_enabled.cache_clear()

    def test_enable_policy_rejects_non_cuda_platforms(self):
        spec = SimpleNamespace(speculative_algorithm="EAGLE", speculative_eagle_topk=1)
        schedule = SimpleNamespace(disable_overlap_schedule=False)
        parallel = SimpleNamespace(moe_ep_size=1)
        prepared_next_draft.prepared_next_draft_enabled.cache_clear()
        with (
            envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True),
            patch.object(prepared_next_draft, "get_spec", return_value=spec),
            patch.object(prepared_next_draft, "get_schedule", return_value=schedule),
            patch.object(prepared_next_draft, "get_parallel", return_value=parallel),
            patch.object(prepared_next_draft, "current_platform", RocmSRTPlatform()),
        ):
            self.assertFalse(prepared_next_draft.prepared_next_draft_enabled())
        prepared_next_draft.prepared_next_draft_enabled.cache_clear()

    def test_rocm_keeps_stock_two_window_reserve(self):
        with (
            get_context().override_server_args(
                speculative_algorithm="EAGLE",
                speculative_num_steps=6,
                speculative_eagle_topk=1,
                speculative_num_draft_tokens=7,
                page_size=1,
            ),
            get_parallel().override(attn_dcp_size=1),
            envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True),
            patch.object(prepared_next_draft, "current_platform", RocmSRTPlatform()),
        ):
            prepared_next_draft.prepared_next_draft_enabled.cache_clear()
            window = get_alloc_len_per_decode()
            self.assertFalse(prepared_next_draft.prepared_next_draft_enabled())
            self.assertEqual(get_alloc_reserve_per_decode(), 2 * window)
        prepared_next_draft.prepared_next_draft_enabled.cache_clear()

    def test_inputs_survive_the_length_read_and_are_consumed_once(self):
        """The scheduler may read prepared lengths before the worker consumes the
        matching buffers; reordered, padded, or under-reserved batches must decline."""
        pool = _pool()
        stager = _stager(pool)
        inputs = _prepared()
        seq_lens = torch.tensor([104, 105])
        batch = _batch([5, 9], [121, 121])

        stager.note_decode(batch, prepared=inputs, new_seq_lens=seq_lens)
        self.assertIs(stager.take_scheduler_sequence_lengths(batch), seq_lens)
        # Freshness is resolved exactly once; scheduler rereads and the worker
        # consume the same lease, rather than independently rebinding it.
        pool.req_generation[5] += 1
        self.assertIs(stager.take_scheduler_sequence_lengths(batch), seq_lens)
        self.assertIs(stager.take_inputs(batch), inputs)
        self.assertIsNone(stager.take_inputs(batch))

        for previous, current in (
            (batch, _batch([9, 5], [121, 121])),
            (_batch([5, 9, 3], [121] * 3), _batch([5, 9, 3], [121] * 3)),
            (_batch([5, 9], [121, 120]), batch),
        ):
            with self.subTest(slots=[req.kv.req_pool_idx for req in current.reqs]):
                stager.note_decode(previous, prepared=inputs, new_seq_lens=seq_lens)
                self.assertIsNone(stager.take_scheduler_sequence_lengths(current))
                self.assertIsNone(stager.take_inputs(current))

    def test_generation_reuse_and_sequence_alias_fail_closed(self):
        """The same row after free/realloc is an ABA miss, and the scheduler's
        accepted lengths may never alias the draft runner's static input."""
        pool = _pool()
        stager = _stager(pool)
        batch = _batch([5, 9], [121, 121])
        prepared = _prepared()
        scheduler_sequence_lengths = torch.tensor([104, 105])

        stager.note_decode(
            batch,
            prepared=prepared,
            new_seq_lens=scheduler_sequence_lengths,
        )
        pool.req_generation[9] += 1
        self.assertIsNone(stager.take_scheduler_sequence_lengths(batch))
        self.assertIsNone(stager.take_inputs(batch))

        stager.note_decode(
            batch,
            prepared=prepared,
            new_seq_lens=prepared.runner_sequence_lengths,
        )
        self.assertIsNone(stager.take_scheduler_sequence_lengths(batch))
        self.assertIsNone(stager.take_inputs(batch))

        stager.note_decode(
            batch,
            prepared=prepared,
            new_seq_lens=scheduler_sequence_lengths,
        )
        batch.reqs[0].kv.kv_allocated_len -= 1
        self.assertIsNone(stager.take_scheduler_sequence_lengths(batch))
        self.assertIsNone(stager.take_inputs(batch))

    def test_only_decode_can_take_prepared_lengths(self):
        """A prefill reusing the same request slots cannot consume decode lengths."""
        worker = EAGLEWorkerV2.__new__(EAGLEWorkerV2)
        worker._next_draft_stager = _stager(graph_sizes=(2,))
        seq_lens = torch.tensor([104, 105])
        decode = _batch([5, 9], [121, 121])
        worker._next_draft_stager.note_decode(
            decode, prepared=_prepared(), new_seq_lens=seq_lens
        )
        self.assertIs(worker.take_prepared_seq_lens(decode), seq_lens)
        decode.forward_mode = ForwardMode.EXTEND
        self.assertIsNone(worker.take_prepared_seq_lens(decode))

    def test_adaptive_states_keep_prepared_next_disabled(self):
        """Adaptive transitions replace runner buffers, so their rounds stay stock."""
        worker = EAGLEWorkerV2.__new__(EAGLEWorkerV2)
        worker._draft_worker = object()
        worker.adaptive_controller = object()
        worker._next_draft_stager = None

        with get_resources().override(prepared_next_draft_reserve_enabled=True):
            worker._init_prepared_next_draft_inputs()
            self.assertFalse(
                prepared_next_draft.prepared_next_draft_reserve_enabled()
            )

        self.assertIsNone(worker._next_draft_stager)

    def test_three_window_reserve_tracks_constructed_stager(self):
        """Only a constructed, PP-local stager may reserve the third KV window."""
        draft_model_runner = SimpleNamespace(
            runner_ids=SimpleNamespace(is_translating=False),
            model_config=SimpleNamespace(model_is_mrope=False),
        )
        draft_graph_runner = SimpleNamespace(
            require_mlp_tp_gather=False,
            capture_bs=(2,),
            seq_len_fill_value=1,
            buffers=SimpleNamespace(),
        )
        worker = EAGLEWorkerV2.__new__(EAGLEWorkerV2)
        worker._draft_worker = SimpleNamespace(
            cuda_graph_runner=draft_graph_runner,
            draft_runner=draft_model_runner,
        )
        worker.adaptive_controller = None
        worker._post_verify_graphs = Mock()
        worker.speculative_num_steps = 3
        worker.speculative_num_draft_tokens = 4
        worker.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.zeros((4, 80), dtype=torch.int32)
        )

        with (
            get_context().override_server_args(
                speculative_algorithm="EAGLE",
                speculative_num_steps=3,
                speculative_eagle_topk=1,
                speculative_num_draft_tokens=4,
                page_size=1,
            ),
            get_parallel().override(attn_dcp_size=1, pp_size=1),
            envs.SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH.override(True),
            get_resources().override(prepared_next_draft_reserve_enabled=None),
            patch.object(
                prepared_next_draft, "prepared_next_draft_enabled", return_value=True
            ),
        ):
            worker._init_prepared_next_draft_inputs()
            self.assertIsNotNone(worker._next_draft_stager)
            self.assertTrue(
                prepared_next_draft.prepared_next_draft_reserve_enabled()
            )
            self.assertEqual(get_alloc_reserve_per_decode(), 3 * 4)

            worker._next_draft_stager = None
            with get_parallel().override(pp_size=2):
                worker._init_prepared_next_draft_inputs()
            self.assertIsNone(worker._next_draft_stager)
            self.assertFalse(
                prepared_next_draft.prepared_next_draft_reserve_enabled()
            )
            self.assertEqual(get_alloc_reserve_per_decode(), 2 * 4)

    def test_graph_and_eager_use_one_mapping_and_eager_reuses_a_ring(self):
        """Captured and eager misses bind the same six runner outputs; eager
        rounds reuse two write windows without overwriting the preceding one."""
        runner_buffers = SimpleNamespace(
            req_pool_indices=torch.zeros(4, dtype=torch.int64),
            seq_lens=torch.zeros(4, dtype=torch.int64),
            positions=torch.zeros(4, dtype=torch.int64),
            mrope_positions=torch.zeros((3, 4), dtype=torch.int64),
            out_cache_loc=torch.zeros(24, dtype=torch.int64),
        )
        token_map = torch.zeros((16, 128), dtype=torch.int32)
        producer = PreparedNextDraftProducer(
            num_steps=6,
            window_size=7,
            seq_len_fill_value=1,
            with_mrope=False,
            req_to_token=token_map,
            runner_buffers=runner_buffers,
            capture_sizes=(2,),
        )
        request_indices = torch.tensor([5, 9])
        scheduler_sequence_lengths = torch.tensor([104, 105])
        graph_inputs = producer.allocate(num_requests=2, device="cpu")

        def publish(prepared, **_):
            return prepared

        with patch(
            "sglang.srt.speculative.prepared_next_draft.prepare_next_draft_inputs",
            side_effect=publish,
        ) as prepare:
            graph = producer.prepare(
                graph_inputs,
                req_pool_indices=request_indices,
                scheduler_sequence_lengths=scheduler_sequence_lengths,
            )
            eager0 = producer.prepare_eager(
                req_pool_indices=request_indices,
                scheduler_sequence_lengths=scheduler_sequence_lengths,
            )
            eager1 = producer.prepare_eager(
                req_pool_indices=request_indices,
                scheduler_sequence_lengths=scheduler_sequence_lengths,
            )
            eager2 = producer.prepare_eager(
                req_pool_indices=request_indices,
                scheduler_sequence_lengths=scheduler_sequence_lengths,
            )

        self.assertEqual(prepare.call_count, 4)
        for call in prepare.call_args_list:
            self.assertIs(call.kwargs["req_pool_indices"], request_indices)
            self.assertIs(call.kwargs["new_seq_lens"], scheduler_sequence_lengths)
            self.assertIs(call.kwargs["req_to_token"], token_map)
            self.assertEqual(call.kwargs["num_steps"], 6)
            self.assertEqual(call.kwargs["window_size"], 7)
        for prepared in (graph, eager0, eager1, eager2):
            self.assertEqual(
                prepared.runner_request_pool_indices.data_ptr(),
                runner_buffers.req_pool_indices.data_ptr(),
            )
            self.assertEqual(
                prepared.runner_sequence_lengths.data_ptr(),
                runner_buffers.seq_lens.data_ptr(),
            )
            self.assertNotEqual(
                prepared.runner_sequence_lengths.data_ptr(),
                scheduler_sequence_lengths.data_ptr(),
            )
        self.assertNotEqual(
            eager0.write_window.data_ptr(), eager1.write_window.data_ptr()
        )
        self.assertEqual(eager0.write_window.data_ptr(), eager2.write_window.data_ptr())
        self.assertIsNone(producer.allocate(num_requests=3, device="cpu"))

    def test_eager_producer_rebinds_request_rows_after_a_draft_graph_miss(self):
        """A prior eager draft leaves the graph runner's request-row buffer stale.
        The eager post-verify producer must refresh it before publishing a lease."""
        runner_buffers = SimpleNamespace(
            req_pool_indices=torch.tensor([1, 2], dtype=torch.int64),
            seq_lens=torch.zeros(2, dtype=torch.int64),
            positions=torch.zeros(2, dtype=torch.int64),
            mrope_positions=torch.zeros((3, 2), dtype=torch.int64),
            out_cache_loc=torch.zeros(12, dtype=torch.int64),
        )
        producer = PreparedNextDraftProducer(
            num_steps=6,
            window_size=7,
            seq_len_fill_value=1,
            with_mrope=False,
            req_to_token=torch.zeros((16, 128), dtype=torch.int32),
            runner_buffers=runner_buffers,
            capture_sizes=(2,),
        )
        request_indices = torch.tensor([5, 9], dtype=torch.int64)
        scheduler_sequence_lengths = torch.tensor([104, 105], dtype=torch.int64)

        def fake_prepare_draft_window(req_pool_indices, _seq_lens, _table, **kwargs):
            kwargs["req_pool_indices_out"].copy_(req_pool_indices)

        with (
            patch(
                "sglang.srt.speculative.prepared_next_draft."
                "can_prepare_next_draft_inputs",
                return_value=True,
            ),
            patch(
                "sglang.kernels.ops.speculative.cache_locs.prepare_draft_window",
                side_effect=fake_prepare_draft_window,
            ) as kernel,
        ):
            prepared = producer.prepare_eager(
                req_pool_indices=request_indices,
                scheduler_sequence_lengths=scheduler_sequence_lengths,
            )

        kernel.assert_called_once()
        self.assertIsNotNone(prepared)
        self.assertEqual(
            prepared.runner_request_pool_indices.data_ptr(),
            runner_buffers.req_pool_indices.data_ptr(),
        )
        self.assertEqual(runner_buffers.req_pool_indices.tolist(), [5, 9])

        stager = _stager(graph_sizes=(2,))
        batch = _batch([5, 9], [121, 121])
        stager.note_decode(
            batch,
            prepared=prepared,
            new_seq_lens=scheduler_sequence_lengths,
        )
        self.assertIs(stager.take_inputs(batch), prepared)

    def test_unsupported_producer_declines_before_the_kernel(self):
        runner_buffers = SimpleNamespace(
            req_pool_indices=torch.zeros(2, dtype=torch.int64),
            seq_lens=torch.zeros(2, dtype=torch.int64),
            positions=torch.zeros(2, dtype=torch.int64),
            mrope_positions=torch.zeros((3, 2), dtype=torch.int64),
            out_cache_loc=torch.zeros(12, dtype=torch.int64),
        )
        producer = PreparedNextDraftProducer(
            num_steps=6,
            window_size=7,
            seq_len_fill_value=1,
            with_mrope=False,
            req_to_token=torch.zeros((16, 128), dtype=torch.int32),
            runner_buffers=runner_buffers,
            capture_sizes=(2,),
        )
        prepared = producer.allocate(num_requests=2, device="cpu")
        with patch(
            "sglang.kernels.ops.speculative.cache_locs.prepare_draft_window"
        ) as kernel:
            self.assertIsNone(
                producer.prepare(
                    prepared,
                    req_pool_indices=torch.tensor([5, 9]),
                    scheduler_sequence_lengths=torch.tensor([104, 105]),
                )
            )
        kernel.assert_not_called()

    def test_fast_path_reserves_three_decode_windows(self):
        """Prepared inputs remain valid through the two overlap buffers plus the
        subsequent draft window; every other configuration retains two windows."""
        with (
            get_context().override_server_args(
                speculative_algorithm="EAGLE",
                speculative_num_steps=6,
                speculative_eagle_topk=1,
                speculative_num_draft_tokens=7,
                page_size=1,
            ),
            get_parallel().override(attn_dcp_size=1),
            envs.SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH.override(True),
            get_resources().override(prepared_next_draft_reserve_enabled=True),
        ):
            window = get_alloc_len_per_decode()
            self.assertEqual(get_alloc_reserve_per_decode(), 3 * window)
        with (
            get_context().override_server_args(
                speculative_algorithm="EAGLE",
                speculative_num_steps=6,
                speculative_eagle_topk=1,
                speculative_num_draft_tokens=7,
                page_size=1,
            ),
            get_parallel().override(attn_dcp_size=1),
            envs.SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH.override(False),
            get_resources().override(prepared_next_draft_reserve_enabled=False),
        ):
            self.assertEqual(get_alloc_reserve_per_decode(), 2 * window)

    def test_three_window_page_one_row_does_not_overwrite_next_request(self):
        """The third W=4 window must fit at the page-size-one context edge."""
        context_len = 64
        sentinel = -1
        with (
            get_context().override_server_args(
                speculative_algorithm="EAGLE",
                speculative_num_steps=3,
                speculative_eagle_topk=1,
                speculative_num_draft_tokens=4,
                page_size=1,
            ),
            get_parallel().override(attn_dcp_size=1),
            envs.SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH.override(True),
            get_resources().override(prepared_next_draft_reserve_enabled=None),
            patch(
                "sglang.srt.speculative.prepared_next_draft."
                "prepared_next_draft_enabled",
                return_value=True,
            ),
        ):
            self.assertIsNone(
                get_resources().prepared_next_draft_reserve_enabled
            )
            self.assertTrue(
                prepared_next_draft.prepared_next_draft_reserve_configured()
            )
            reserve = get_alloc_reserve_per_decode()
            extra = get_req_to_token_extra_context_len()
            row_width = context_len + extra

        committed = context_len - 3
        table = torch.full((2, row_width), sentinel, dtype=torch.int64)
        table.view(-1)[committed : committed + reserve] = torch.arange(reserve)

        self.assertEqual(reserve, 3 * 4)
        self.assertEqual(extra, reserve)
        self.assertEqual(table[1, 0].item(), sentinel)

    def test_single_step_configuration_keeps_two_window_row(self):
        """A single draft step has no draft graph, so it cannot arm 3W sizing."""
        with (
            get_context().override_server_args(
                speculative_algorithm="EAGLE",
                speculative_num_steps=1,
                speculative_eagle_topk=1,
                speculative_num_draft_tokens=4,
                page_size=1,
            ),
            get_parallel().override(attn_dcp_size=1),
            envs.SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH.override(True),
            get_resources().override(prepared_next_draft_reserve_enabled=None),
            patch.object(
                prepared_next_draft, "prepared_next_draft_enabled", return_value=True
            ),
        ):
            self.assertFalse(
                prepared_next_draft.prepared_next_draft_reserve_configured()
            )
            self.assertEqual(get_alloc_reserve_per_decode(), 2 * 4)
            self.assertEqual(get_req_to_token_extra_context_len(), 4 + 4)

    def test_future_map_uses_prepared_lengths_without_waiting_for_the_relay(self):
        """An unchanged batch must not wait for or gather the relay that the
        preceding post-verify graph has already materialized."""
        pool = SimpleNamespace(req_to_token=torch.zeros((8, 16), dtype=torch.int32))
        with (
            patch.object(overlap_utils, "_is_cuda", False),
            patch.object(
                overlap_utils, "spec_relay_bypass_enabled", return_value=False
            ),
            patch(
                "sglang.srt.speculative.prepared_next_draft."
                "prepared_next_draft_enabled",
                return_value=True,
            ),
        ):
            future_map = FutureMap(
                torch.device("cpu"), SpeculativeAlgorithm.EAGLE, pool
            )
        future_map.publish_ready = Mock()
        future_indices = torch.tensor([3, 1])
        batch = SimpleNamespace(
            spec_info=SimpleNamespace(future_indices=future_indices),
            req_pool_indices_cpu=future_indices,
        )
        prepared = torch.tensor([30, 10])

        future_map.resolve_seq_lens_cpu(batch, prepared_seq_lens=prepared)

        self.assertIs(batch.seq_lens, prepared)
        self.assertEqual(batch.seq_lens_cpu.tolist(), [30, 10])
        future_map.publish_ready.wait.assert_not_called()

    def test_schedule_stream_join_is_skipped_only_after_idle_preparation(self):
        """Prepared lengths alone do not waive a schedule-stream join when KV
        allocation or penalty bookkeeping launched work on that stream."""
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.forward_stream = "forward"
        scheduler.schedule_stream = "schedule"
        prepared = torch.tensor([10])
        cases = (
            (prepared, True, False),
            (prepared, False, True),
            (None, True, True),
        )
        for seq_lens, idle, joins in cases:
            batch = SimpleNamespace(spec_decode_schedule_stream_idle=idle)
            with (
                self.subTest(prepared=seq_lens is not None, idle=idle),
                patch.object(
                    scheduler_module.spec_decode_resources, "wait_stream"
                ) as wait,
            ):
                scheduler._join_schedule_stream(batch, seq_lens)
                self.assertEqual(wait.call_count, int(joins))
                if seq_lens is not None:
                    self.assertFalse(batch.spec_decode_schedule_stream_idle)

    def test_decode_without_new_pages_updates_only_host_watermarks(self):
        """A fully reserved round avoids allocation tensors and marks the schedule
        stream idle only when penalty and cache bookkeeping also launch no work."""
        reserve = 21
        rounds = (
            (121, False, True),
            (121, True, False),
            (120, False, False),
        )
        for allocated, penalized, idle in rounds:
            req = _req(5, allocated=allocated)
            batch = _decode_prepare_batch([req], penalized=penalized)
            with (
                self.subTest(allocated=allocated, penalized=penalized),
                patch.object(
                    eagle_utils,
                    "get_alloc_reserve_per_decode",
                    return_value=reserve,
                ),
                patch.object(
                    eagle_utils,
                    "prepared_next_draft_enabled",
                    return_value=True,
                ),
                patch.object(
                    eagle_utils,
                    "get_exec",
                    return_value=SimpleNamespace(
                        mamba=SimpleNamespace(enable_mamba_extra_buffer_lazy=False)
                    ),
                ),
                patch.object(
                    eagle_utils, "update_spec_decode_allocated_lens"
                ) as update,
                patch.object(eagle_utils, "alloc_for_spec_decode") as allocate,
            ):
                eagle_utils.eagle_prepare_for_decode(batch)
                self.assertEqual(batch.spec_decode_schedule_stream_idle, idle)
                if allocated == 121:
                    update.assert_called_once_with([req], [121])
                    allocate.assert_not_called()
                else:
                    update.assert_not_called()
                    allocate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
