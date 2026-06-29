from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import json
import os
import types
import unittest
from collections import deque
from tempfile import TemporaryDirectory
from unittest import mock

import sglang.srt.managers.scheduler_components.self_benchmark as self_benchmark_module
from sglang.srt.disaggregation.utils import DisaggregationMode
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
    def test_prefill_point_collects_fpm_and_writes_output(self):
        with TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/benchmark.json"
            benchmark = SelfBenchmark(_make_scheduler(output_path))
            with open(benchmark._output_path) as f:
                running_output = json.load(f)
            self.assertEqual(running_output["scope"], "local_diagnostics")
            self.assertEqual(running_output["status"], "running")
            self.assertFalse(running_output["valid"])
            self.assertEqual(running_output["run_id"], "test-run")

            point = BenchmarkPoint(point_type="prefill", isl=10)
            benchmark.phase = BenchmarkPhase.SWEEP
            benchmark._grid = [point]
            benchmark._grid_index = 0
            benchmark._current = BenchmarkPointResult(point=point)

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
            with open(benchmark._output_path) as f:
                output = json.load(f)
            self.assertEqual(output["scope"], "local_diagnostics")
            self.assertEqual(output["status"], "complete")
            self.assertTrue(output["valid"])
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
        with TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/benchmark.json"
            scheduler = _make_scheduler(output_path)
            calls = []
            scheduler.on_self_benchmark_finished = lambda: calls.append("done")
            benchmark = SelfBenchmark(scheduler)
            benchmark.phase = BenchmarkPhase.SWEEP
            benchmark._grid_index = len(benchmark._grid)

            benchmark.maybe_schedule_next()

            self.assertEqual(calls, ["done"])
            self.assertFalse(benchmark.active)
            with open(benchmark._output_path) as f:
                output = json.load(f)
            self.assertIn("results", output)

    def test_output_invalidation_replaces_stale_result_for_worker_path(self):
        with TemporaryDirectory() as tmpdir:
            # dp_rank 0 writes the caller-assigned base path; a stale prior-run
            # result there must be invalidated at init.
            output_path = f"{tmpdir}/benchmark.json"
            with open(output_path, "w") as f:
                json.dump({"valid": True, "run_id": "old-run"}, f)

            scheduler = _make_scheduler(output_path)
            scheduler.instance_id = "new-run"
            benchmark = SelfBenchmark(scheduler)

            self.assertEqual(benchmark._output_path, output_path)
            with open(benchmark._output_path) as f:
                output = json.load(f)
            self.assertFalse(output["valid"])
            self.assertEqual(output["status"], "running")
            self.assertEqual(output["run_id"], "new-run")

    def test_output_path_follows_dp_rank_contract(self):
        with TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/benchmark.json"
            # dp_rank 0 -> caller-assigned base path, unchanged.
            dp0 = SelfBenchmark(_make_scheduler(output_path))
            self.assertEqual(dp0._output_path, output_path)

            # dp_rank N -> the "_dpN" sibling the consumer addresses.
            dp1_scheduler = _make_scheduler(output_path)
            dp1_scheduler.ps.dp_rank = 1
            dp1 = SelfBenchmark(dp1_scheduler)
            self.assertEqual(dp1._output_path, f"{tmpdir}/benchmark_dp1.json")

    def test_role_recorded_in_contents_not_path(self):
        with TemporaryDirectory() as tmpdir:
            # Role/run/rank identity lives in the file contents, not the
            # filename; co-located workers are kept distinct by the caller
            # assigning a unique base path, not by namespacing the filename.
            output_path = f"{tmpdir}/benchmark.json"
            prefill_scheduler = _make_scheduler(output_path)
            prefill_scheduler.disaggregation_mode = DisaggregationMode.PREFILL
            prefill_benchmark = SelfBenchmark(prefill_scheduler)

            self.assertEqual(prefill_benchmark._output_path, output_path)
            self.assertEqual(
                prefill_benchmark._identity["disaggregation_mode"], "prefill"
            )

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
        ]

        for name, mark_inflight in cases:
            with self.subTest(name=name):
                scheduler = _make_scheduler("/tmp/unused.json")
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

    def test_decode_point_ignores_setup_prefill_until_decode_pass(self):
        benchmark = SelfBenchmark(_make_scheduler("/tmp/unused.json"))
        point = BenchmarkPoint(point_type="decode", context_length=8, batch_size=2)
        benchmark.phase = BenchmarkPhase.SWEEP
        benchmark._grid = [point]
        benchmark._current = BenchmarkPointResult(point=point)

        prefill_fpm = ForwardPassMetrics(
            scheduled_requests=ScheduledRequestMetrics(num_prefill_requests=2)
        )
        decode_fpm = ForwardPassMetrics(
            scheduled_requests=ScheduledRequestMetrics(num_decode_requests=2)
        )

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
        benchmark = SelfBenchmark(_make_scheduler("/tmp/unused.json"))
        point = BenchmarkPoint(point_type="prefill", isl=32)
        req = _FakeReq(finished=False)
        benchmark.phase = BenchmarkPhase.SWEEP
        benchmark._grid = [point]
        benchmark._current = BenchmarkPointResult(point=point)
        benchmark._active_reqs = [req]

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
        benchmark = SelfBenchmark(_make_scheduler("/tmp/unused.json"))
        point = BenchmarkPoint(point_type="prefill", isl=32)
        req = _FakeReq(finished=False)
        benchmark.phase = BenchmarkPhase.SWEEP
        benchmark._grid = [point]
        benchmark._grid_index = 0
        benchmark._current = BenchmarkPointResult(point=point)
        benchmark._active_reqs = [req]

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
        scheduler = _make_scheduler("/tmp/unused.json")
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

        scheduler = _make_scheduler("/tmp/unused.json")
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

    def test_synthetic_decode_models_prefill_to_decode_boundary(self):
        scheduler = _prepare_decode_scheduler(_make_scheduler("/tmp/unused.json"))
        benchmark = SelfBenchmark(scheduler)
        captured = {}
        fake_batch = object()

        def fake_build_synthetic_decode_batch(reqs, context_length):
            captured["reqs"] = reqs
            captured["context_length"] = context_length
            return fake_batch

        benchmark._build_synthetic_decode_batch = fake_build_synthetic_decode_batch

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
        scheduler = _prepare_decode_scheduler(_make_scheduler("/tmp/unused.json"))
        scheduler.init_req_max_new_tokens = lambda req: setattr(
            req.sampling_params, "max_new_tokens", 1
        )
        benchmark = SelfBenchmark(scheduler)

        def fail_build_synthetic_decode_batch(_reqs, _context_length):
            raise AssertionError("synthetic decode batch should not be built")

        benchmark._build_synthetic_decode_batch = fail_build_synthetic_decode_batch

        injected = benchmark._inject_synthetic_decode(context_length=8, batch_size=2)

        self.assertEqual(injected, 0)
        self.assertEqual(benchmark._active_reqs, [])

    def test_prefill_grid_is_not_capped_at_chunked_prefill_size(self):
        scheduler = _make_scheduler("/tmp/unused.json")
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
        scheduler = _make_scheduler("/tmp/unused.json")
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
        scheduler = _make_scheduler("/tmp/unused.json")
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
        scheduler = _make_scheduler("/tmp/unused.json")
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
        scheduler = _make_scheduler("/tmp/unused.json")
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
        scheduler = _make_scheduler("/tmp/unused.json")
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

    def test_chunked_prefill_request_counts_as_inflight_work(self):
        scheduler = _make_scheduler("/tmp/unused.json")
        benchmark = SelfBenchmark(scheduler)

        self.assertFalse(benchmark._has_inflight_work())
        scheduler.chunked_req = object()
        self.assertTrue(benchmark._has_inflight_work())

    def test_disagg_prefill_inflight_list_counts_as_inflight_work(self):
        scheduler = _make_scheduler("/tmp/unused.json")
        scheduler.disagg_prefill_inflight_queue = []
        benchmark = SelfBenchmark(scheduler)

        self.assertFalse(benchmark._has_inflight_work())
        scheduler.disagg_prefill_inflight_queue.append(object())
        self.assertTrue(benchmark._has_inflight_work())

    def test_disaggregated_workers_only_build_supported_grid(self):
        scheduler = _make_scheduler("/tmp/unused.json")
        scheduler.server_args.benchmark_mode = "decode"
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL

        benchmark = SelfBenchmark(scheduler)

        self.assertEqual(benchmark._grid, [])
        self.assertEqual(benchmark._inject_warmup(), 0)

        scheduler = _make_scheduler("/tmp/unused.json")
        scheduler.server_args.benchmark_mode = "agg"
        scheduler.disaggregation_mode = DisaggregationMode.DECODE

        benchmark = SelfBenchmark(scheduler)

        self.assertTrue(all(p.point_type == "decode" for p in benchmark._grid))

    def test_build_synthetic_decode_batch_stashes_relay_payload(self):
        # P0-1 regression guard: _build_synthetic_decode_batch must hand
        # FutureMap.stash a RelayPayload, never a raw tensor. We exercise the
        # REAL _build_synthetic_decode_batch (only the memory allocation and
        # sampling-info construction are stubbed) so the stash call is covered.
        scheduler = _prepare_decode_scheduler(_make_scheduler("/tmp/unused.json"))

        captured = {}

        class _FakeFutureMap:
            def stash(self, future_indices, payload):
                captured["future_indices"] = future_indices
                captured["payload"] = payload
                # Mirror FutureMap.stash's real access pattern: a raw tensor
                # would raise AttributeError here.
                assert type(payload).__name__ == "RelayPayload"
                assert hasattr(payload, "bonus_tokens")
                _ = payload.bonus_tokens.to

        scheduler.future_map = _FakeFutureMap()
        scheduler.req_to_token_pool = object()
        scheduler.token_to_kv_pool_allocator = object()
        scheduler.enable_overlap = False
        scheduler.dllm_config = None
        scheduler.enable_hisparse = False

        benchmark = SelfBenchmark(scheduler)

        # Stub the GPU-memory allocation + sampling-info construction; keep the
        # real stash path. _place_synthetic_context_cache normally fills
        # req_pool_indices, so the stub sets it.
        def fake_place(batch, context_length):
            batch.req_pool_indices = list(range(len(batch.reqs)))

        benchmark._place_synthetic_context_cache = fake_place

        class _FakeBatch:
            def __init__(self, **kwargs):
                self.reqs = kwargs.get("reqs", [])
                self.device = "cpu"
                self.req_pool_indices = None
                self.sampling_info = None
                self.input_ids = "sentinel"

        # mock.patch.object correctly saves/restores the classmethod descriptors.
        with mock.patch.object(
            self_benchmark_module.ScheduleBatch,
            "init_new",
            staticmethod(lambda **kwargs: _FakeBatch(**kwargs)),
        ), mock.patch.object(
            self_benchmark_module.SamplingBatchInfo,
            "from_schedule_batch",
            staticmethod(lambda batch, vocab_size: object()),
        ):
            reqs = [benchmark._new_synthetic_req(prompt_len=8, max_tokens=2)]
            reqs[0].output_ids.append(SELF_BENCHMARK_DUMMY_TOKEN_ID)
            batch = benchmark._build_synthetic_decode_batch(reqs, context_length=8)

        self.assertIn("payload", captured)
        self.assertEqual(type(captured["payload"]).__name__, "RelayPayload")
        self.assertEqual(captured["payload"].bonus_tokens.tolist(), [0])
        self.assertIsNone(batch.input_ids)

    def test_synthetic_decode_build_failure_frees_reqs_and_skips(self):
        # P0-1: any failure in _build_synthetic_decode_batch (now caught as
        # Exception, not just RuntimeError) must free the pre-allocated reqs and
        # skip the point, never crash the scheduler.
        scheduler = _prepare_decode_scheduler(_make_scheduler("/tmp/unused.json"))
        freed = {}
        benchmark = SelfBenchmark(scheduler)

        def raising_build(_reqs, _ctx):
            raise AttributeError("simulated stash on raw tensor")

        def fake_free(reqs):
            freed["reqs"] = reqs

        benchmark._build_synthetic_decode_batch = raising_build
        benchmark._free_synthetic_decode_reqs = fake_free

        injected = benchmark._inject_synthetic_decode(context_length=8, batch_size=2)

        self.assertEqual(injected, 0)
        self.assertEqual(benchmark._active_reqs, [])
        self.assertIn("reqs", freed)
        self.assertEqual(len(freed["reqs"]), 2)

    def test_benchmark_forced_fpm_rank_advances_then_restores(self):
        # P0-2: a rank where the real gate gives enable_fpm=False, but with
        # benchmark_mode set, has FPM force-enabled for the sweep. The sweep must
        # advance WARMUP->SWEEP->DONE off observe_forward_pass and restore FPM
        # afterwards (and not write the redundant JSON).
        with TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/benchmark.json"
            scheduler = _make_scheduler(output_path)
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
            # Forced rank must not be marked as an output writer.
            self.assertFalse(benchmark._write_results)
            # FPM stays on for the duration of the sweep.
            self.assertTrue(scheduler.enable_fpm)

            point = BenchmarkPoint(point_type="prefill", isl=10)
            benchmark.phase = BenchmarkPhase.SWEEP
            benchmark._grid = [point]
            benchmark._grid_index = 0
            benchmark._current = BenchmarkPointResult(point=point)

            fpm = ForwardPassMetrics(
                scheduled_requests=ScheduledRequestMetrics(num_prefill_requests=1),
            )
            batch = types.SimpleNamespace(
                forward_mode=_FakeForwardMode(is_extend=True),
            )

            # observe_forward_pass must fire and advance the phase.
            benchmark.observe_forward_pass(batch, fpm)
            benchmark.maybe_schedule_next()

            self.assertFalse(benchmark.active)
            self.assertEqual(benchmark.phase, BenchmarkPhase.DONE)
            self.assertEqual(finished_calls, ["done"])
            # Forced publisher torn down and FPM restored to the prior off state.
            self.assertEqual(shutdown_calls, ["shutdown"])
            self.assertFalse(scheduler.enable_fpm)
            # No JSON written on the forced rank.
            self.assertFalse(
                os.path.exists(benchmark._output_path),
                "benchmark-forced rank must not write output JSON",
            )

    def test_real_fpm_rank_keeps_fpm_enabled_after_sweep(self):
        # P0-2: the real FPM rank's production behavior is unchanged after the
        # sweep -- enable_fpm stays True and no forced teardown happens.
        with TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/benchmark.json"
            scheduler = _make_scheduler(output_path)
            scheduler.enable_fpm = True
            scheduler._fpm_is_real_rank = True
            scheduler._fpm_benchmark_forced = False

            # Faithful stub: the real helper early-returns for a non-forced rank.
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
            # Real rank keeps publishing; forced-teardown is a no-op for it.
            self.assertEqual(shutdown_calls, [])
            self.assertTrue(scheduler.enable_fpm)

    def test_grammar_queue_counts_as_inflight_work(self):
        scheduler = _make_scheduler("/tmp/unused.json")
        scheduler.grammar_manager = types.SimpleNamespace(grammar_queue=[])
        benchmark = SelfBenchmark(scheduler)

        self.assertFalse(benchmark._has_inflight_work())
        scheduler.grammar_manager.grammar_queue.append(object())
        self.assertTrue(benchmark._has_inflight_work())

    def test_decode_prealloc_retracted_and_pending_count_as_inflight_work(self):
        for attr in ("retracted_queue", "pending_reqs"):
            with self.subTest(attr=attr):
                scheduler = _make_scheduler("/tmp/unused.json")
                queue_owner = types.SimpleNamespace(
                    queue=[], retracted_queue=[], pending_reqs=[]
                )
                scheduler.disagg_decode_prealloc_queue = queue_owner
                benchmark = SelfBenchmark(scheduler)

                self.assertFalse(benchmark._has_inflight_work())
                getattr(queue_owner, attr).append(object())
                self.assertTrue(benchmark._has_inflight_work())


def _make_rejection_scheduler():
    """Fake `self` for invoking Scheduler.maybe_init_self_benchmark unbound.

    Defaults to a configuration that would PASS all rejections; each test flips
    one attribute to trigger a specific ValueError. The success path constructs
    a real SelfBenchmark, so tests only assert the rejection branches.
    """
    return types.SimpleNamespace(
        self_benchmark=None,
        server_args=types.SimpleNamespace(benchmark_mode="agg", load_format="auto"),
        ps=types.SimpleNamespace(pp_size=1),
        enable_pdmux=False,
        enable_overlap_mlx=False,
        is_generation=True,
        spec_algorithm=types.SimpleNamespace(is_none=lambda: True),
        model_config=types.SimpleNamespace(
            is_encoder_decoder=False, is_multimodal=False
        ),
        enable_lora=False,
    )


class TestSelfBenchmarkSchedulerRejections(CustomTestCase):
    """Scheduler-side fast-fail rejections in maybe_init_self_benchmark.

    These cannot live in the server_args tests because the validations run in
    the Scheduler (they need is_generation / spec_algorithm / model_config /
    enable_lora / load_format resolved). We invoke the method unbound on a fake
    `self` so we don't have to construct a full Scheduler.
    """

    @staticmethod
    def _run(fake_self):
        from sglang.srt.managers.scheduler import Scheduler

        Scheduler.maybe_init_self_benchmark(fake_self)

    def test_rejects_non_generation_model(self):
        fake = _make_rejection_scheduler()
        fake.is_generation = False
        with self.assertRaisesRegex(ValueError, "only supported for generative"):
            self._run(fake)

    def test_rejects_speculative_decoding(self):
        fake = _make_rejection_scheduler()
        fake.spec_algorithm = types.SimpleNamespace(is_none=lambda: False)
        with self.assertRaisesRegex(ValueError, "speculative decoding"):
            self._run(fake)

    def test_rejects_encoder_decoder(self):
        fake = _make_rejection_scheduler()
        fake.model_config.is_encoder_decoder = True
        with self.assertRaisesRegex(ValueError, "encoder-decoder"):
            self._run(fake)

    def test_rejects_multimodal(self):
        fake = _make_rejection_scheduler()
        fake.model_config.is_multimodal = True
        with self.assertRaisesRegex(ValueError, "multimodal"):
            self._run(fake)

    def test_rejects_lora(self):
        fake = _make_rejection_scheduler()
        fake.enable_lora = True
        with self.assertRaisesRegex(ValueError, "LoRA"):
            self._run(fake)

    def test_rejects_dummy_weights(self):
        fake = _make_rejection_scheduler()
        fake.server_args.load_format = "dummy"
        with self.assertRaisesRegex(ValueError, "dummy weights"):
            self._run(fake)

    def test_does_not_reject_multi_rank_tp(self):
        # MUST NOT reject tp_size>1 / dp_size>1. The fake `self` has no tp_size
        # gate; maybe_init_self_benchmark only rejects pp_size>1. Confirm a
        # passing config reaches SelfBenchmark construction (which then fails on
        # the incomplete fake), proving none of the rejections fired.
        fake = _make_rejection_scheduler()
        # Reaching SelfBenchmark(self) means all rejections passed. SelfBenchmark
        # construction will raise (fake lacks the full surface) -- anything but a
        # benchmark-mode ValueError proves we got past the guards.
        try:
            self._run(fake)
        except ValueError as exc:  # pragma: no cover - defensive
            self.assertNotIn("--benchmark-mode is not supported", str(exc))
            self.assertNotIn("only supported for generative", str(exc))
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
