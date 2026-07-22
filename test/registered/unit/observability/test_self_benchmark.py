from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import json
import os
import types
import unittest
from collections import deque
from tempfile import TemporaryDirectory
from unittest import mock

import sglang.srt.managers.scheduler_components.self_benchmark_decode as self_benchmark_decode_module
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import compute_extend_logprob_start_len
from sglang.srt.managers.scheduler_components.self_benchmark import (
    SELF_BENCHMARK_DUMMY_TOKEN_ID,
    BenchmarkPhase,
    BenchmarkPoint,
    BenchmarkPointResult,
    SelfBenchmark,
)
from sglang.srt.observability.forward_pass_metrics import (
    ForwardPassMetrics,
    ScheduledRequestMetrics,
)
from sglang.test.test_utils import CustomTestCase


class _FakeForwardMode:
    def __init__(self, *, is_decode: bool = False, is_extend: bool = False):
        self._is_decode = is_decode
        self._is_extend = is_extend

    def is_prebuilt(self):
        return False

    def is_decode(self):
        return self._is_decode

    def is_extend(self):
        return self._is_extend


def _make_scheduler(output_path: str):
    return types.SimpleNamespace(
        instance_id="test-run",
        server_args=types.SimpleNamespace(
            model_path="test-model",
            served_model_name=None,
            benchmark_mode="agg",
            benchmark_prefill_granularity=2,
            benchmark_prefill_kv_read_granularity=1,
            benchmark_decode_length_granularity=2,
            benchmark_decode_batch_granularity=2,
            benchmark_warmup_iterations=0,
            benchmark_output_path=output_path,
            chunked_prefill_size=None,
            cuda_graph_config=types.SimpleNamespace(
                decode=types.SimpleNamespace(max_bs=4)
            ),
            node_rank=0,
            nnodes=1,
        ),
        enable_fpm=True,
        max_req_input_len=16,
        max_total_num_tokens=64,
        max_running_requests=4,
        max_req_len=32,
        max_prefill_tokens=64,
        page_size=1,
        disaggregation_mode=DisaggregationMode.NULL,
        ps=types.SimpleNamespace(
            dp_rank=0,
            dp_size=1,
            tp_rank=0,
            tp_size=1,
            attn_tp_rank=0,
            attn_tp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
        ),
        result_queue=deque(),
        waiting_queue=[],
        running_batch=None,
        chunked_req=None,
        tree_cache=types.SimpleNamespace(disable=True),
    )


class _FakeReq:
    def __init__(self, finished: bool = False):
        self._finished = finished

    def finished(self):
        return self._finished


def _prepare_decode_scheduler(scheduler):
    scheduler.is_generation = True
    scheduler.model_config = types.SimpleNamespace(
        is_encoder_decoder=False,
        hf_eos_token_id=None,
        vocab_size=32000,
    )
    scheduler.spec_algorithm = types.SimpleNamespace(is_none=lambda: True)
    scheduler.server_args.allow_auto_truncate = False
    scheduler.server_args.disaggregation_bootstrap_port = None
    scheduler.tokenizer = None
    scheduler.running_batch = types.SimpleNamespace(is_empty=lambda: True)
    scheduler.init_req_max_new_tokens = lambda req: None
    return scheduler


class TestSelfBenchmark(CustomTestCase):
    def setUp(self):
        super().setUp()
        self._tmpdir = TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.output_path = os.path.join(self._tmpdir.name, "benchmark.json")

    def _scheduler(self):
        return _make_scheduler(self.output_path)

    @staticmethod
    def _read_output(benchmark):
        with open(benchmark._output_path) as f:
            return json.load(f)

    @staticmethod
    def _start_point(benchmark, point, *, active_reqs=()):
        benchmark.phase = BenchmarkPhase.SWEEP
        benchmark._grid = [point]
        benchmark._grid_index = 0
        benchmark._current = BenchmarkPointResult(point=point)
        benchmark._active_reqs = list(active_reqs)

    @staticmethod
    def _fpm(**scheduled_fields):
        return ForwardPassMetrics(
            scheduled_requests=ScheduledRequestMetrics(**scheduled_fields)
        )

    def test_prefill_point_collects_fpm_and_writes_output(self):
        benchmark = SelfBenchmark(self._scheduler())
        running_output = self._read_output(benchmark)
        self.assertEqual(running_output["scope"], "local_diagnostics")
        self.assertEqual(running_output["status"], "running")
        self.assertFalse(running_output["valid"])
        self.assertEqual(running_output["run_id"], "test-run")

        point = BenchmarkPoint(point_type="prefill", isl=10)
        self._start_point(benchmark, point)

        fpm = ForwardPassMetrics(
            worker_id="worker-1",
            scheduled_requests=ScheduledRequestMetrics(
                num_prefill_requests=1,
                sum_prefill_tokens=10,
                sum_prefill_kv_tokens=10,
            ),
        )
        batch = types.SimpleNamespace(
            forward_mode=_FakeForwardMode(is_extend=True),
        )

        benchmark.observe_forward_pass(batch, fpm)
        benchmark.maybe_schedule_next()

        self.assertFalse(benchmark.active)
        output = self._read_output(benchmark)
        self.assertEqual(output["scope"], "local_diagnostics")
        self.assertEqual(output["status"], "complete")
        self.assertTrue(output["valid"])
        self.assertEqual(
            output["coverage"],
            {"expected_points": 1, "completed_points": 1, "skipped_points": 0},
        )
        self.assertEqual(output["skipped_points"], [])
        self.assertEqual(output["run_id"], "test-run")
        self.assertEqual(output["identity"]["model_path"], "test-model")
        self.assertEqual(output["identity"]["disaggregation_mode"], "null")
        self.assertEqual(output["config"]["mode"], "agg")
        self.assertEqual(output["results"][0]["point"]["point_type"], "prefill")
        self.assertEqual(
            output["results"][0]["fpms"][0]["scheduled_requests"][
                "num_prefill_requests"
            ],
            1,
        )

    def test_finish_notifies_scheduler_after_writing_output(self):
        scheduler = self._scheduler()
        calls = []
        scheduler.on_self_benchmark_finished = lambda: calls.append("done")
        benchmark = SelfBenchmark(scheduler)
        benchmark.phase = BenchmarkPhase.SWEEP
        benchmark._grid_index = len(benchmark._grid)

        benchmark.maybe_schedule_next()

        self.assertEqual(calls, ["done"])
        self.assertFalse(benchmark.active)
        self.assertIn("results", self._read_output(benchmark))

    def test_output_invalidation_replaces_stale_result_for_worker_path(self):
        # dp_rank 0 writes the caller-assigned base path; a stale prior-run
        # result there must be invalidated at init.
        with open(self.output_path, "w") as f:
            json.dump({"valid": True, "run_id": "old-run"}, f)

        scheduler = self._scheduler()
        scheduler.instance_id = "new-run"
        benchmark = SelfBenchmark(scheduler)

        self.assertEqual(benchmark._output_path, self.output_path)
        output = self._read_output(benchmark)
        self.assertFalse(output["valid"])
        self.assertEqual(output["status"], "running")
        self.assertEqual(output["run_id"], "new-run")

    def test_output_path_follows_dp_rank_contract(self):
        # dp_rank 0 -> caller-assigned base path, unchanged.
        dp0 = SelfBenchmark(self._scheduler())
        self.assertEqual(dp0._output_path, self.output_path)

        # dp_rank N -> the "_dpN" sibling the consumer addresses.
        dp1_scheduler = self._scheduler()
        dp1_scheduler.ps.dp_rank = 1
        dp1 = SelfBenchmark(dp1_scheduler)
        self.assertEqual(
            dp1._output_path,
            os.path.join(self._tmpdir.name, "benchmark_dp1.json"),
        )

    def test_role_recorded_in_contents_not_path(self):
        # Role/run/rank identity lives in the file contents, not the filename;
        # co-located workers are kept distinct by caller-assigned base paths.
        prefill_scheduler = self._scheduler()
        prefill_scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        prefill_benchmark = SelfBenchmark(prefill_scheduler)

        self.assertEqual(prefill_benchmark._output_path, self.output_path)
        self.assertEqual(prefill_benchmark._identity["disaggregation_mode"], "prefill")

    def test_finish_waits_for_inflight_scheduler_state(self):
        cases = [
            ("result_queue", lambda s: s.result_queue.append(object())),
            ("waiting_queue", lambda s: s.waiting_queue.append(object())),
            ("chunked_req", lambda s: setattr(s, "chunked_req", object())),
            (
                "running_batch",
                lambda s: setattr(
                    s,
                    "running_batch",
                    types.SimpleNamespace(is_empty=lambda: False),
                ),
            ),
            (
                "disagg_prefill_bootstrap_queue",
                lambda s: setattr(
                    s,
                    "disagg_prefill_bootstrap_queue",
                    types.SimpleNamespace(queue=[object()]),
                ),
            ),
            (
                "disagg_prefill_inflight_queue",
                lambda s: setattr(s, "disagg_prefill_inflight_queue", [object()]),
            ),
            (
                "disagg_decode_prealloc_queue",
                lambda s: setattr(
                    s,
                    "disagg_decode_prealloc_queue",
                    types.SimpleNamespace(queue=[object()]),
                ),
            ),
            (
                "disagg_decode_transfer_queue",
                lambda s: setattr(
                    s,
                    "disagg_decode_transfer_queue",
                    types.SimpleNamespace(queue=[object()]),
                ),
            ),
            (
                "grammar_queue",
                lambda s: setattr(
                    s,
                    "grammar_manager",
                    types.SimpleNamespace(grammar_queue=[object()]),
                ),
            ),
            (
                "disagg_decode_retracted_queue",
                lambda s: setattr(
                    s,
                    "disagg_decode_prealloc_queue",
                    types.SimpleNamespace(
                        queue=[], retracted_queue=[object()], pending_reqs=[]
                    ),
                ),
            ),
            (
                "disagg_decode_pending_reqs",
                lambda s: setattr(
                    s,
                    "disagg_decode_prealloc_queue",
                    types.SimpleNamespace(
                        queue=[], retracted_queue=[], pending_reqs=[object()]
                    ),
                ),
            ),
        ]

        for name, mark_inflight in cases:
            with self.subTest(name=name):
                scheduler = self._scheduler()
                calls = []
                scheduler.on_self_benchmark_finished = lambda: calls.append("done")
                mark_inflight(scheduler)
                benchmark = SelfBenchmark(scheduler)
                benchmark.phase = BenchmarkPhase.SWEEP
                benchmark._grid = []
                benchmark._grid_index = 0
                benchmark._write_results = False

                benchmark.maybe_schedule_next()

                self.assertEqual(calls, [])
                self.assertTrue(benchmark.active)

    def test_empty_optional_queues_are_not_inflight(self):
        empty_queues = (
            ("disagg_prefill_inflight_queue", []),
            ("grammar_manager", types.SimpleNamespace(grammar_queue=[])),
            (
                "disagg_decode_prealloc_queue",
                types.SimpleNamespace(queue=[], retracted_queue=[], pending_reqs=[]),
            ),
        )
        for name, value in empty_queues:
            with self.subTest(name=name):
                scheduler = self._scheduler()
                setattr(scheduler, name, value)
                self.assertFalse(SelfBenchmark(scheduler)._has_inflight_work())

    def test_decode_point_ignores_setup_prefill_until_decode_pass(self):
        benchmark = SelfBenchmark(self._scheduler())
        point = BenchmarkPoint(point_type="decode", context_length=8, batch_size=2)
        self._start_point(benchmark, point)

        prefill_fpm = self._fpm(num_prefill_requests=2)
        decode_fpm = self._fpm(num_decode_requests=2)

        benchmark.observe_forward_pass(
            types.SimpleNamespace(forward_mode=_FakeForwardMode(is_extend=True)),
            prefill_fpm,
        )
        self.assertIsNotNone(benchmark._current)
        self.assertEqual(len(benchmark._results), 0)

        benchmark.observe_forward_pass(
            types.SimpleNamespace(forward_mode=_FakeForwardMode(is_decode=True)),
            decode_fpm,
        )
        self.assertIsNone(benchmark._current)
        self.assertEqual(len(benchmark._results), 1)
        self.assertEqual(
            benchmark._results[0].fpms[0]["scheduled_requests"]["num_decode_requests"],
            2,
        )

    def test_prefill_point_accumulates_fpms_until_request_finishes(self):
        benchmark = SelfBenchmark(self._scheduler())
        point = BenchmarkPoint(point_type="prefill", isl=32)
        req = _FakeReq(finished=False)
        self._start_point(benchmark, point, active_reqs=[req])

        first_chunk = ForwardPassMetrics(
            scheduled_requests=ScheduledRequestMetrics(
                num_prefill_requests=1,
                sum_prefill_tokens=16,
                sum_prefill_kv_tokens=0,
            )
        )
        final_chunk = ForwardPassMetrics(
            scheduled_requests=ScheduledRequestMetrics(
                num_prefill_requests=1,
                sum_prefill_tokens=16,
                sum_prefill_kv_tokens=16,
            )
        )
        batch = types.SimpleNamespace(forward_mode=_FakeForwardMode(is_extend=True))

        benchmark.observe_forward_pass(batch, first_chunk)

        self.assertIsNotNone(benchmark._current)
        self.assertEqual(len(benchmark._results), 0)
        self.assertEqual(len(benchmark._current.fpms), 1)

        req._finished = True
        benchmark.observe_forward_pass(batch, final_chunk)

        self.assertIsNone(benchmark._current)
        self.assertEqual(len(benchmark._results), 1)
        self.assertEqual(len(benchmark._results[0].fpms), 2)
        self.assertEqual(
            benchmark._results[0].fpms[1]["scheduled_requests"][
                "sum_prefill_kv_tokens"
            ],
            16,
        )

    def test_maybe_schedule_next_advances_current_point_finished_after_observe(self):
        benchmark = SelfBenchmark(self._scheduler())
        point = BenchmarkPoint(point_type="prefill", isl=32)
        req = _FakeReq(finished=False)
        self._start_point(benchmark, point, active_reqs=[req])

        fpm = ForwardPassMetrics(
            scheduled_requests=ScheduledRequestMetrics(
                num_prefill_requests=1,
                sum_prefill_tokens=32,
                sum_prefill_kv_tokens=0,
            )
        )
        batch = types.SimpleNamespace(forward_mode=_FakeForwardMode(is_extend=True))

        benchmark.observe_forward_pass(batch, fpm)
        self.assertIsNotNone(benchmark._current)
        self.assertEqual(len(benchmark._results), 0)

        req._finished = True
        benchmark.maybe_schedule_next()

        self.assertIsNone(benchmark._current)
        self.assertEqual(benchmark._grid_index, 1)
        self.assertEqual(len(benchmark._results), 1)
        self.assertEqual(len(benchmark._results[0].fpms), 1)

    def test_decode_grid_preserves_room_for_one_decode_token(self):
        scheduler = self._scheduler()
        scheduler.max_req_input_len = 16
        scheduler.max_req_len = 12
        scheduler.max_total_num_tokens = 64

        benchmark = SelfBenchmark(scheduler)

        decode_points = [p for p in benchmark._grid if p.point_type == "decode"]
        self.assertGreater(len(decode_points), 0)
        self.assertLessEqual(
            max(p.context_length for p in decode_points),
            10,
        )

        scheduler = self._scheduler()
        scheduler.max_req_input_len = 128
        scheduler.max_req_len = 128
        scheduler.max_total_num_tokens = 100
        scheduler.page_size = 16

        benchmark = SelfBenchmark(scheduler)

        decode_points = [p for p in benchmark._grid if p.point_type == "decode"]
        self.assertGreater(len(decode_points), 0)
        self.assertLessEqual(
            max(p.context_length for p in decode_points),
            80,
        )

    def test_decode_grid_is_capped_by_forward_batch_limit(self):
        scheduler = self._scheduler()
        scheduler.server_args.benchmark_mode = "decode"
        scheduler.server_args.cuda_graph_config.decode.max_bs = 7
        scheduler.max_running_requests = 4096
        scheduler.max_total_num_tokens = 8192

        benchmark = SelfBenchmark(scheduler)

        decode_points = [p for p in benchmark._grid if p.point_type == "decode"]
        self.assertEqual(max(p.batch_size for p in decode_points), 7)

    def test_synthetic_decode_models_prefill_to_decode_boundary(self):
        scheduler = _prepare_decode_scheduler(self._scheduler())
        benchmark = SelfBenchmark(scheduler)
        captured = {}
        fake_batch = object()

        def fake_build_synthetic_decode_batch(reqs, context_length):
            captured["reqs"] = reqs
            captured["context_length"] = context_length
            return fake_batch

        benchmark._decode_batch_builder.build = fake_build_synthetic_decode_batch

        injected = benchmark._inject_synthetic_decode(context_length=8, batch_size=2)

        self.assertEqual(injected, 2)
        self.assertIs(scheduler.running_batch, fake_batch)
        self.assertEqual(captured["context_length"], 8)
        self.assertEqual(benchmark._active_reqs, captured["reqs"])

        for req in captured["reqs"]:
            self.assertEqual(req.sampling_params.max_new_tokens, 2)
            self.assertEqual(list(req.output_ids), [SELF_BENCHMARK_DUMMY_TOKEN_ID])
            self.assertEqual(
                list(req.fill_ids),
                list(req.origin_input_ids) + [SELF_BENCHMARK_DUMMY_TOKEN_ID],
            )
            self.assertEqual(req.seqlen, 9)
            self.assertEqual(req.kv_committed_len, 8)
            self.assertEqual(req.kv_allocated_len, 8)
            self.assertEqual(req.already_computed, 8)
            self.assertFalse(req.finished())

    def test_synthetic_decode_skips_if_two_tokens_cannot_be_generated(self):
        scheduler = _prepare_decode_scheduler(self._scheduler())
        scheduler.init_req_max_new_tokens = lambda req: setattr(
            req.sampling_params, "max_new_tokens", 1
        )
        benchmark = SelfBenchmark(scheduler)

        def fail_build_synthetic_decode_batch(_reqs, _context_length):
            raise AssertionError("synthetic decode batch should not be built")

        benchmark._decode_batch_builder.build = fail_build_synthetic_decode_batch

        injected = benchmark._inject_synthetic_decode(context_length=8, batch_size=2)

        self.assertEqual(injected, 0)
        self.assertEqual(benchmark._active_reqs, [])

    def test_prefill_grid_is_not_capped_at_chunked_prefill_size(self):
        scheduler = self._scheduler()
        scheduler.server_args.benchmark_mode = "prefill"
        scheduler.server_args.chunked_prefill_size = 8
        scheduler.max_req_input_len = 128
        scheduler.max_total_num_tokens = 64

        benchmark = SelfBenchmark(scheduler)

        prefill_points = [p for p in benchmark._grid if p.point_type == "prefill"]
        self.assertGreater(len(prefill_points), 0)
        self.assertEqual(max(p.isl for p in prefill_points), 62)
        self.assertTrue(all(p.kv_read_tokens == 0 for p in prefill_points))

    def test_prefill_grid_is_capped_by_forward_token_budget(self):
        scheduler = self._scheduler()
        scheduler.server_args.benchmark_mode = "prefill"
        scheduler.max_req_input_len = 40960
        scheduler.max_total_num_tokens = 40960
        scheduler.max_prefill_tokens = 1024

        benchmark = SelfBenchmark(scheduler)

        prefill_points = [p for p in benchmark._grid if p.point_type == "prefill"]
        self.assertGreater(len(prefill_points), 0)
        self.assertEqual(max(p.isl for p in prefill_points), 1024)
        self.assertTrue(
            all(p.isl <= scheduler.max_prefill_tokens for p in prefill_points)
        )

    def test_prefill_kv_read_grid_crosses_with_isl(self):
        scheduler = self._scheduler()
        scheduler.server_args.benchmark_mode = "prefill"
        scheduler.server_args.benchmark_prefill_kv_read_granularity = 3

        benchmark = SelfBenchmark(scheduler)

        prefill_points = [p for p in benchmark._grid if p.point_type == "prefill"]
        self.assertEqual(
            [(p.isl, p.kv_read_tokens) for p in prefill_points],
            [
                (10, 0),
                (10, 4),
                (10, 9),
                (15, 0),
                (15, 7),
                (15, 14),
            ],
        )

    def test_prefill_kv_read_grid_aligns_to_page_size(self):
        scheduler = self._scheduler()
        scheduler.page_size = 8
        scheduler.server_args.benchmark_mode = "prefill"
        scheduler.server_args.benchmark_prefill_kv_read_granularity = 4
        scheduler.max_req_input_len = 40

        benchmark = SelfBenchmark(scheduler)

        prefill_points = [p for p in benchmark._grid if p.point_type == "prefill"]
        self.assertTrue(all(p.kv_read_tokens % 8 == 0 for p in prefill_points))
        self.assertTrue(all(p.kv_read_tokens <= p.isl - 1 for p in prefill_points))
        self.assertIn(
            BenchmarkPoint(point_type="prefill", isl=39, kv_read_tokens=32),
            prefill_points,
        )

    def test_prefill_kv_read_point_seeds_then_measures_with_same_extra_key(self):
        scheduler = self._scheduler()
        scheduler.server_args.benchmark_mode = "prefill"
        benchmark = SelfBenchmark(scheduler)
        point = BenchmarkPoint(point_type="prefill", isl=16, kv_read_tokens=8)
        calls = []

        def fake_inject_requests(**kwargs):
            calls.append(kwargs)
            return 1

        benchmark.phase = BenchmarkPhase.SWEEP
        benchmark._grid = [point]
        benchmark._grid_index = 0
        benchmark._inject_requests = fake_inject_requests
        benchmark._cached_kv_read_tokens_for_point = (
            lambda cached_point, _extra_key: cached_point.kv_read_tokens
        )

        benchmark.maybe_schedule_next()

        self.assertIsNone(benchmark._current)
        self.assertIs(benchmark._pending_seed_point, point)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["prompt_len"], 8)
        self.assertEqual(calls[0]["max_tokens"], 0)
        self.assertFalse(calls[0]["track_active"])

        benchmark.maybe_schedule_next()

        self.assertIsNotNone(benchmark._current)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["prompt_len"], 16)
        self.assertEqual(calls[1]["max_tokens"], 1)
        self.assertTrue(calls[1]["track_active"])
        self.assertEqual(calls[0]["extra_key"], calls[1]["extra_key"])
        self.assertIsNone(benchmark._pending_seed_extra_key)

    def test_prefill_kv_read_point_skips_when_seed_validation_misses(self):
        scheduler = self._scheduler()
        scheduler.server_args.benchmark_mode = "prefill"
        benchmark = SelfBenchmark(scheduler)
        point = BenchmarkPoint(point_type="prefill", isl=16, kv_read_tokens=8)
        calls = []

        def fake_inject_requests(**kwargs):
            calls.append(kwargs)
            return 1

        benchmark.phase = BenchmarkPhase.SWEEP
        benchmark._grid = [point]
        benchmark._grid_index = 0
        benchmark._inject_requests = fake_inject_requests
        benchmark._cached_kv_read_tokens_for_point = lambda _point, _extra_key: 0

        benchmark.maybe_schedule_next()
        benchmark.maybe_schedule_next()

        self.assertIsNone(benchmark._current)
        self.assertIsNone(benchmark._pending_seed_point)
        self.assertEqual(benchmark._grid_index, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(benchmark._skipped_points), 1)
        self.assertEqual(
            benchmark._skipped_points[0].reason, "seed_cache_validation_failed"
        )

    def test_disaggregated_workers_only_build_supported_grid(self):
        scheduler = self._scheduler()
        scheduler.server_args.benchmark_mode = "decode"
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL

        benchmark = SelfBenchmark(scheduler)

        self.assertEqual(benchmark._grid, [])
        self.assertEqual(benchmark._inject_warmup(), 0)

        scheduler = self._scheduler()
        scheduler.server_args.benchmark_mode = "agg"
        scheduler.disaggregation_mode = DisaggregationMode.DECODE

        benchmark = SelfBenchmark(scheduler)

        self.assertTrue(all(p.point_type == "decode" for p in benchmark._grid))

    def test_decode_batch_builder_uses_canonical_allocation_and_relay_payload(self):
        scheduler = _prepare_decode_scheduler(self._scheduler())
        captured = {}

        class _FakeFutureMap:
            def stash(self, future_indices, payload):
                captured["future_indices"] = future_indices
                captured["payload"] = payload
                assert type(payload).__name__ == "RelayPayload"
                _ = payload.bonus_tokens.to

        scheduler.future_map = _FakeFutureMap()
        scheduler.req_to_token_pool = object()
        scheduler.token_to_kv_pool_allocator = object()
        scheduler.enable_overlap = False
        scheduler.dllm_config = None
        scheduler.enable_hisparse = False
        root_node = object()
        scheduler.tree_cache.root_node = root_node

        benchmark = SelfBenchmark(scheduler)

        class _FakeBatch:
            def __init__(self, **kwargs):
                self.reqs = kwargs.get("reqs", [])
                self.tree_cache = kwargs["tree_cache"]
                self.device = "cpu"
                self.input_ids = "sentinel"

        out_cache_loc = object()
        req_pool_indices = [0]
        req_pool_indices_cpu = [0]
        alloc_for_extend = mock.Mock(
            return_value=(out_cache_loc, req_pool_indices, req_pool_indices_cpu)
        )
        with mock.patch.object(
            self_benchmark_decode_module.ScheduleBatch,
            "init_new",
            staticmethod(lambda **kwargs: _FakeBatch(**kwargs)),
        ), mock.patch.object(
            self_benchmark_decode_module.SamplingBatchInfo,
            "from_schedule_batch",
            staticmethod(lambda batch, vocab_size: object()),
        ), mock.patch.object(
            self_benchmark_decode_module, "alloc_for_extend", alloc_for_extend
        ):
            reqs = [benchmark._new_synthetic_req(prompt_len=8, max_tokens=2)]
            reqs[0].output_ids.append(SELF_BENCHMARK_DUMMY_TOKEN_ID)
            batch = benchmark._decode_batch_builder.build(reqs, context_length=8)

        alloc_for_extend.assert_called_once_with(batch)
        self.assertIs(batch.out_cache_loc, out_cache_loc)
        self.assertIs(batch.req_pool_indices, req_pool_indices)
        self.assertIs(batch.req_pool_indices_cpu, req_pool_indices_cpu)
        self.assertEqual(batch.prefix_lens, [0])
        self.assertEqual(batch.extend_lens, [8])
        self.assertEqual(batch.extend_num_tokens, 8)
        self.assertEqual(batch.seq_lens_cpu.tolist(), [8])
        self.assertEqual(batch.seq_lens.tolist(), [8])
        self.assertEqual(batch.orig_seq_lens.tolist(), [8])
        self.assertEqual(batch.seq_lens_sum, 8)
        self.assertTrue(batch.forward_mode.is_extend())
        self.assertTrue(reqs[0].synthetic_benchmark_kv_placed)
        self.assertIs(reqs[0].last_node, root_node)
        self.assertIs(reqs[0].last_host_node, root_node)
        self.assertIs(reqs[0].best_match_node, root_node)
        self.assertEqual(type(captured["payload"]).__name__, "RelayPayload")
        self.assertEqual(captured["payload"].bonus_tokens.tolist(), [0])
        self.assertIsNone(batch.input_ids)

    def test_synthetic_req_matches_dp_attention_logprob_accounting(self):
        scheduler = _prepare_decode_scheduler(self._scheduler())
        benchmark = SelfBenchmark(scheduler)
        prompt_len = 8

        req = benchmark._new_synthetic_req(prompt_len=prompt_len, max_tokens=2)
        logprob_start_len = compute_extend_logprob_start_len(
            logprob_start_len=req.logprob_start_len,
            prefix_len=0,
            extend_len=prompt_len,
            full_untruncated_fill_len=prompt_len,
        )
        num_tokens_for_logprob = max(prompt_len - logprob_start_len, 1)

        self.assertFalse(req.return_logprob)
        self.assertEqual(num_tokens_for_logprob, 1)

    def test_synthetic_decode_build_failure_cleans_up_and_reraises(self):
        for cleanup_error in (None, RuntimeError("cleanup failed")):
            with self.subTest(cleanup_error=cleanup_error):
                scheduler = _prepare_decode_scheduler(self._scheduler())
                original_batch = scheduler.running_batch
                benchmark = SelfBenchmark(scheduler)
                build_error = AttributeError("synthetic build failed")
                cleanup = mock.Mock(side_effect=cleanup_error)
                benchmark._decode_batch_builder.build = mock.Mock(
                    side_effect=build_error
                )
                benchmark._decode_batch_builder.cleanup = cleanup

                with self.assertRaises(AttributeError) as raised:
                    benchmark._inject_synthetic_decode(context_length=8, batch_size=2)

                self.assertIs(raised.exception, build_error)
                cleanup.assert_called_once()
                self.assertEqual(len(cleanup.call_args.args[0]), 2)
                self.assertIs(scheduler.running_batch, original_batch)
                self.assertEqual(benchmark._active_reqs, [])

    def test_decode_batch_builder_cleanup_handles_partial_allocation(self):
        scheduler = self._scheduler()
        scheduler.req_to_token_pool = types.SimpleNamespace(free=mock.Mock())
        placed = types.SimpleNamespace(
            synthetic_benchmark_kv_placed=True, req_pool_idx=1
        )
        slot_only = types.SimpleNamespace(req_pool_idx=2)
        unallocated = types.SimpleNamespace(req_pool_idx=None)

        with mock.patch.object(
            self_benchmark_decode_module, "release_kv_cache"
        ) as release:
            benchmark = SelfBenchmark(scheduler)
            benchmark._decode_batch_builder.cleanup([placed, slot_only, unallocated])

        release.assert_called_once_with(placed, scheduler.tree_cache, is_insert=False)
        scheduler.req_to_token_pool.free.assert_called_once_with(slot_only)

    def test_partial_sweep_records_skipped_point_and_is_invalid(self):
        benchmark = SelfBenchmark(self._scheduler())
        completed = BenchmarkPoint(point_type="prefill", isl=10)
        skipped = BenchmarkPoint(point_type="decode", context_length=1, batch_size=1)
        benchmark.phase = BenchmarkPhase.SWEEP
        benchmark._grid = [completed, skipped]
        benchmark._grid_index = 1
        benchmark._results = [
            BenchmarkPointResult(point=completed, fpms=[{"wall_time": 0.1}])
        ]
        benchmark._inject_synthetic_decode = lambda **_kwargs: 0

        benchmark.maybe_schedule_next()
        benchmark.maybe_schedule_next()

        output = self._read_output(benchmark)
        self.assertEqual(output["status"], "complete")
        self.assertFalse(output["valid"])
        self.assertEqual(
            output["coverage"],
            {"expected_points": 2, "completed_points": 1, "skipped_points": 1},
        )
        self.assertEqual(len(output["results"]), 1)
        [skipped_output] = output["skipped_points"]
        self.assertEqual(skipped_output["point"], vars(skipped))
        self.assertEqual(skipped_output["reason"], "request_injection_failed")

    def test_completed_point_without_metrics_is_skipped(self):
        benchmark = SelfBenchmark(self._scheduler())
        point = BenchmarkPoint(point_type="prefill", isl=10)
        self._start_point(benchmark, point)

        benchmark._save_current_point()

        self.assertEqual(benchmark._results, [])
        self.assertEqual(benchmark._skipped_points[0].reason, "no_forward_pass_metrics")

    def test_benchmark_forced_fpm_rank_advances_then_restores(self):
        scheduler = self._scheduler()
        # Simulate metrics_reporter._init_fpm forcing FPM on a non-FPM rank.
        scheduler.enable_fpm = True
        scheduler._fpm_is_real_rank = False
        scheduler._fpm_benchmark_forced = True

        shutdown_calls = []
        scheduler.metrics_reporter = types.SimpleNamespace(
            shutdown_benchmark_forced_fpm=lambda: shutdown_calls.append("shutdown")
        )
        finished_calls = []
        scheduler.on_self_benchmark_finished = lambda: finished_calls.append("done")

        benchmark = SelfBenchmark(scheduler)
        self.assertFalse(benchmark._write_results)
        self.assertTrue(scheduler.enable_fpm)

        point = BenchmarkPoint(point_type="prefill", isl=10)
        self._start_point(benchmark, point)
        batch = types.SimpleNamespace(forward_mode=_FakeForwardMode(is_extend=True))

        benchmark.observe_forward_pass(batch, self._fpm(num_prefill_requests=1))
        benchmark.maybe_schedule_next()

        self.assertFalse(benchmark.active)
        self.assertEqual(benchmark.phase, BenchmarkPhase.DONE)
        self.assertEqual(finished_calls, ["done"])
        self.assertEqual(shutdown_calls, ["shutdown"])
        self.assertFalse(scheduler.enable_fpm)
        self.assertFalse(
            os.path.exists(benchmark._output_path),
            "benchmark-forced rank must not write output JSON",
        )

    def test_real_fpm_rank_keeps_fpm_enabled_after_sweep(self):
        scheduler = self._scheduler()
        scheduler.enable_fpm = True
        scheduler._fpm_is_real_rank = True
        scheduler._fpm_benchmark_forced = False
        shutdown_calls = []

        def fake_shutdown():
            if getattr(scheduler, "_fpm_benchmark_forced", False):
                shutdown_calls.append("shutdown")

        scheduler.metrics_reporter = types.SimpleNamespace(
            shutdown_benchmark_forced_fpm=fake_shutdown
        )

        benchmark = SelfBenchmark(scheduler)
        self.assertTrue(benchmark._write_results)

        benchmark.phase = BenchmarkPhase.SWEEP
        benchmark._grid_index = len(benchmark._grid)
        benchmark.maybe_schedule_next()

        self.assertFalse(benchmark.active)
        self.assertEqual(shutdown_calls, [])
        self.assertTrue(scheduler.enable_fpm)


def _make_rejection_scheduler():
    """Fake scheduler for exercising SelfBenchmark.create_if_enabled.

    Defaults to a configuration that passes every compatibility check; tests
    mutate one attribute at a time to exercise each rejection.
    """
    return types.SimpleNamespace(
        server_args=types.SimpleNamespace(benchmark_mode="agg", load_format="auto"),
        ps=types.SimpleNamespace(pp_size=1, tp_size=2, dp_size=2),
        enable_pdmux=False,
        enable_overlap_mlx=False,
        dllm_config=None,
        token_to_kv_pool_allocator=object(),
        is_generation=True,
        spec_algorithm=types.SimpleNamespace(is_none=lambda: True),
        model_config=types.SimpleNamespace(
            is_encoder_decoder=False, is_multimodal=False
        ),
        enable_lora=False,
    )


class TestSelfBenchmarkFactory(CustomTestCase):
    """Runtime fast-fail rejections in SelfBenchmark.create_if_enabled.

    These cannot live in the server_args tests because the validations run in
    the scheduler process (they need is_generation / spec_algorithm /
    model_config / enable_lora / load_format resolved).
    """

    @staticmethod
    def _run(fake_scheduler):
        return SelfBenchmark.create_if_enabled(fake_scheduler)

    def test_disabled_returns_none(self):
        fake = types.SimpleNamespace(
            server_args=types.SimpleNamespace(benchmark_mode=None)
        )
        self.assertIsNone(self._run(fake))

    def test_rejects_unsupported_runtime_configurations(self):
        cases = (
            (
                "pipeline parallelism",
                lambda scheduler: setattr(scheduler.ps, "pp_size", 2),
            ),
            (
                "PD multiplexing",
                lambda scheduler: setattr(scheduler, "enable_pdmux", True),
            ),
            (
                "MLX overlap",
                lambda scheduler: setattr(scheduler, "enable_overlap_mlx", True),
            ),
            (
                "diffusion LLM",
                lambda scheduler: setattr(scheduler, "dllm_config", object()),
            ),
            (
                "only supported for generative",
                lambda scheduler: setattr(scheduler, "is_generation", False),
            ),
            (
                "speculative decoding",
                lambda scheduler: setattr(
                    scheduler.spec_algorithm, "is_none", lambda: False
                ),
            ),
            (
                "encoder-decoder",
                lambda scheduler: setattr(
                    scheduler.model_config, "is_encoder_decoder", True
                ),
            ),
            (
                "multimodal",
                lambda scheduler: setattr(
                    scheduler.model_config, "is_multimodal", True
                ),
            ),
            ("LoRA", lambda scheduler: setattr(scheduler, "enable_lora", True)),
            (
                "dummy weights",
                lambda scheduler: setattr(
                    scheduler.server_args, "load_format", "dummy"
                ),
            ),
        )

        for message, mutate in cases:
            with self.subTest(message=message):
                fake = _make_rejection_scheduler()
                mutate(fake)
                with self.assertRaisesRegex(ValueError, message):
                    self._run(fake)

    def test_rejects_deepseek_v4_npu_for_all_modes(self):
        for mode in ("prefill", "decode", "agg"):
            with self.subTest(mode=mode):
                fake = _make_rejection_scheduler()
                fake.server_args.benchmark_mode = mode
                fake.token_to_kv_pool_allocator = types.SimpleNamespace(
                    c4_attn_allocator=object()
                )
                with self.assertRaisesRegex(ValueError, "DeepSeek V4 on NPU"):
                    self._run(fake)

    def test_does_not_reject_multi_rank_tp(self):
        fake = _make_rejection_scheduler()
        with mock.patch.object(SelfBenchmark, "__init__", return_value=None) as init:
            benchmark = self._run(fake)

        init.assert_called_once_with(fake)
        self.assertIsInstance(benchmark, SelfBenchmark)


if __name__ == "__main__":
    unittest.main()
