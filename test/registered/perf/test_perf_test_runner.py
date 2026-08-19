"""CPU unit tests for the ``batch_decode_scheduler`` perf patch.

These lock down the parts that make SGLang numbers comparable with RTP-LLM
``grid_perf_test`` and the vLLM ``batch_decode_scheduler`` patch:

* the RTP-LLM trimmed-mean aggregation and the CSV/table layout,
* the CLI -> ``ServerArgs`` pins that guarantee exact batching,
* the TP/DP/EP translation between vLLM/RTP semantics and SGLang semantics,
* the invariant that the measurement path synchronizes three times per round
  and *never* per step, while the ``--per-step-timing`` path does the opposite,
* the assertions that make a split batch / a ragged fake-KV batch fail loudly
  instead of producing a plausible-looking wrong number.

Nothing here touches a GPU or loads a model: the scheduler is stubbed out and
``ServerArgs`` is captured rather than constructed.
"""

import csv
import os
import tempfile
import unittest
from unittest import mock

import torch

from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.patches.batch_decode_scheduler.perf_test_harness import (
    FAKE_KV_TOKEN_BUDGET_CAP,
    FAKE_TOKEN_ID,
    PHASE_DECODE,
    PHASE_PREFILL,
    BenchHarness,
    synthetic_input_ids,
)
from sglang.srt.patches.batch_decode_scheduler.perf_test_runner import (
    BenchResult,
    RankPayload,
    _agg,
    _prefill_token_budget,
    _trimmed_mean,
    build_server_args,
    failure_reasons,
    parse_args,
    resolve_parallelism,
    run_decode_bench,
    run_prefill_bench,
    run_step_diag,
    write_csv,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


# ----------------------------------------------------------------------
# Stubs
# ----------------------------------------------------------------------


class _StubForwardMode:
    def __init__(self, kind: str):
        self.kind = kind
        self.name = kind.upper()

    def is_idle(self) -> bool:
        return self.kind == "idle"

    def is_decode(self) -> bool:
        return self.kind == "decode"

    def is_mixed(self) -> bool:
        return self.kind == "mixed"

    def is_extend(self) -> bool:
        return self.kind == "extend"


class _StubScheduleBatch:
    def __init__(self, kind: str, batch_size: int, extend_num_tokens: int = 0):
        self.forward_mode = _StubForwardMode(kind)
        self.reqs = [f"req-{i}" for i in range(batch_size)]
        self.extend_num_tokens = extend_num_tokens

    def batch_size(self) -> int:
        return len(self.reqs)


class _StubDeviceModule:
    def __init__(self):
        self.synchronize_calls = 0

    def synchronize(self) -> None:
        self.synchronize_calls += 1


class _FakeHarness(BenchHarness):
    """A ``BenchHarness`` with only the scheduler amputated.

    ``_advance_one_step`` and the three request-submission entry points are
    stubbed; everything the bench loops actually measure with — ``_sync``, the
    three timing marks, ``run_step_no_timing`` / ``run_step_timed`` and the four
    assertions — is the production implementation, so the sync accounting these
    tests assert on is the real one.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        seq_len: int,
        served_batch_size: int = None,
        served_extend_tokens: int = None,
        decode_only: bool = False,
    ):
        self.device_module = _StubDeviceModule()
        self._batch_start = 0.0
        self._sync_count = 0
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._served_batch_size = (
            batch_size if served_batch_size is None else served_batch_size
        )
        self._served_extend_tokens = (
            batch_size * seq_len
            if served_extend_tokens is None
            else served_extend_tokens
        )
        self._decode_only = decode_only
        self._prefill_pending = False
        self.steps = []
        self.submitted = []
        self.drain_calls = 0

    def submit(self, batch_size, seq_len, *, max_new_tokens=1, ignore_eos=True):
        self.submitted.append((batch_size, seq_len, max_new_tokens))
        self._prefill_pending = not self._decode_only
        return [f"stub-{i}" for i in range(batch_size)]

    def submit_decode_only(self, batch_size, seq_len, num_decode_steps):
        # Fake-KV registration consumes the extend step itself, so the next
        # step the bench loop drives is already a decode step.
        self.submitted.append((batch_size, seq_len, num_decode_steps))
        self._prefill_pending = False
        return [f"stub-{i}" for i in range(batch_size)]

    def _advance_one_step(self):
        if self._prefill_pending:
            self._prefill_pending = False
            batch = _StubScheduleBatch(
                "extend", self._served_batch_size, self._served_extend_tokens
            )
        else:
            batch = _StubScheduleBatch("decode", self._served_batch_size)
        self.steps.append(batch)
        return batch

    def drain(self):
        self._prefill_pending = False
        self.drain_calls += 1


class _StubReq:
    def __init__(self, rid: str, prompt_len: int, output_len: int):
        self.rid = rid
        self.origin_input_ids = list(range(prompt_len))
        self.output_ids = list(range(output_len))

    @property
    def seqlen(self) -> int:
        return len(self.origin_input_ids) + len(self.output_ids)


class _StubScheduler:
    def __init__(self, running_reqs, last_batch_reqs=None):
        self.chunked_req = None
        self.waiting_queue = []
        self.running_batch = _StubScheduleBatch("decode", 0)
        self.running_batch.reqs = list(running_reqs)
        self.last_batch = None
        if last_batch_reqs is not None:
            self.last_batch = _StubScheduleBatch("extend", 0)
            self.last_batch.reqs = list(last_batch_reqs)


def _harness_with_scheduler(scheduler) -> BenchHarness:
    harness = BenchHarness.__new__(BenchHarness)
    harness.scheduler = scheduler
    return harness


def _bench_args(*extra):
    return parse_args(["--model-path", "/stub/model", *extra])


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------


class TestAggregation(unittest.TestCase):
    """RTP-LLM ``batch_perf_impl.run``: sort, drop min and max, average."""

    def test_empty(self):
        self.assertEqual(_trimmed_mean([]), 0.0)
        self.assertEqual(_agg([]), (0.0, 0.0))

    def test_fewer_than_three_samples_falls_back_to_plain_mean(self):
        self.assertEqual(_trimmed_mean([7.0]), 7.0)
        self.assertEqual(_trimmed_mean([4.0, 6.0]), 5.0)

    def test_drops_min_and_max(self):
        # 1 and 100 are dropped; mean(2, 3, 4) == 3.
        self.assertEqual(_trimmed_mean([100.0, 2.0, 1.0, 4.0, 3.0]), 3.0)

    def test_exactly_three_samples_keeps_the_median(self):
        self.assertEqual(_trimmed_mean([1.0, 5.0, 99.0]), 5.0)

    def test_agg_returns_trimmed_then_raw(self):
        trimmed, raw = _agg([1.0, 2.0, 3.0, 100.0])
        self.assertEqual(trimmed, 2.5)
        self.assertEqual(raw, 26.5)


# ----------------------------------------------------------------------
# TP / DP / EP translation
# ----------------------------------------------------------------------


class TestParallelismTranslation(unittest.TestCase):
    """CLI is vLLM/RTP semantics (TP inside one DP replica); SGLang's
    ``tp_size`` is the world size under dp-attention and per-replica otherwise."""

    def test_dp_attention_multiplies_tp_by_dp(self):
        got = resolve_parallelism(
            cli_tp_size=2, cli_dp_size=2, cli_ep_size=4, enable_dp_attention=True
        )
        self.assertEqual(got, {"tp_size": 4, "dp_size": 2, "ep_size": 4})
        # attn_tp_size is derived, not a server arg: tp_size // dp_size // cp.
        self.assertEqual(got["tp_size"] // got["dp_size"] // 1, 2)

    def test_pure_dp_attention(self):
        got = resolve_parallelism(
            cli_tp_size=1, cli_dp_size=4, cli_ep_size=4, enable_dp_attention=True
        )
        self.assertEqual(got["tp_size"], 4)
        self.assertEqual(got["tp_size"] // got["dp_size"] // 1, 1)

    def test_tp_only(self):
        got = resolve_parallelism(
            cli_tp_size=8, cli_dp_size=1, cli_ep_size=1, enable_dp_attention=False
        )
        self.assertEqual(got, {"tp_size": 8, "dp_size": 1, "ep_size": 1})

    def test_plain_dp_replicates_instead_of_widening_tp(self):
        # Without dp-attention each replica is its own TP group, so tp_size
        # passes through untouched and the launcher repeats it dp_size times.
        got = resolve_parallelism(
            cli_tp_size=2, cli_dp_size=2, cli_ep_size=1, enable_dp_attention=False
        )
        self.assertEqual(got, {"tp_size": 2, "dp_size": 2, "ep_size": 1})

    def test_zero_ep_size_normalizes_to_one(self):
        got = resolve_parallelism(
            cli_tp_size=2, cli_dp_size=1, cli_ep_size=0, enable_dp_attention=False
        )
        self.assertEqual(got["ep_size"], 1)

    def test_non_divisible_ep_size_raises(self):
        # 3 % 2 != 0: moe_ep_rank math would skew silently, so fail up front.
        with self.assertRaises(ValueError):
            resolve_parallelism(
                cli_tp_size=1, cli_dp_size=3, cli_ep_size=2, enable_dp_attention=True
            )


# ----------------------------------------------------------------------
# ServerArgs pins
# ----------------------------------------------------------------------


class TestServerArgsPinning(unittest.TestCase):
    """``build_server_args`` pins whatever exact batching and the timing
    semantics depend on. ServerArgs itself needs a real model on disk, so we
    capture the kwargs instead of constructing it."""

    def _kwargs(self, *extra, batch_sizes=(1, 4), seq_lens=(128, 512)):
        args = _bench_args(*extra)
        with mock.patch(
            "sglang.srt.patches.batch_decode_scheduler.perf_test_runner.ServerArgs"
        ) as server_args_cls:
            build_server_args(args, list(batch_sizes), list(seq_lens))
        return server_args_cls.call_args.kwargs

    def test_exact_batching_pins(self):
        kwargs = self._kwargs()
        self.assertEqual(kwargs["chunked_prefill_size"], -1)
        self.assertTrue(kwargs["disable_radix_cache"])
        self.assertTrue(kwargs["skip_tokenizer_init"])
        self.assertFalse(kwargs["enable_metrics"])
        self.assertEqual(kwargs["max_running_requests"], 4)

    def test_overlap_scheduler_is_off_by_default(self):
        self.assertTrue(self._kwargs()["disable_overlap_schedule"])
        self.assertFalse(self._kwargs("--enable-overlap")["disable_overlap_schedule"])

    def test_max_running_requests_scales_with_dp_attention(self):
        # Under dp-attention SGLang divides max_running_requests by the attn dp
        # size, so it has to be pre-multiplied to keep --batch-sizes per-rank.
        kwargs = self._kwargs("--dp-size", "2", "--enable-dp-attention")
        self.assertEqual(kwargs["max_running_requests"], 8)
        # Plain DP gives each replica its own budget: no scaling.
        self.assertEqual(self._kwargs("--dp-size", "2")["max_running_requests"], 4)

    def test_parallelism_reaches_server_args(self):
        kwargs = self._kwargs(
            "--tp-size",
            "2",
            "--dp-size",
            "2",
            "--ep-size",
            "4",
            "--enable-dp-attention",
        )
        self.assertEqual(kwargs["tp_size"], 4)
        self.assertEqual(kwargs["dp_size"], 2)
        self.assertEqual(kwargs["ep_size"], 4)
        self.assertTrue(kwargs["enable_dp_attention"])

    def test_optional_args_are_omitted_when_unset(self):
        kwargs = self._kwargs()
        for name in (
            "mem_fraction_static",
            "max_total_tokens",
            "context_length",
            "attention_backend",
        ):
            self.assertNotIn(name, kwargs)
        kwargs = self._kwargs(
            "--mem-fraction-static", "0.5", "--context-length", "4096"
        )
        self.assertEqual(kwargs["mem_fraction_static"], 0.5)
        self.assertEqual(kwargs["context_length"], 4096)

    def test_prefill_budget_swallows_the_whole_batch(self):
        kwargs = self._kwargs(batch_sizes=(1, 16), seq_lens=(128, 1024))
        self.assertEqual(kwargs["max_prefill_tokens"], 16 * 1024)

    def test_fake_kv_clamps_the_prefill_budget(self):
        # No prefill forward runs, but the budget still sizes the pinned
        # staging buffers, so --partial 1 caps it.
        budget = _prefill_token_budget(
            max_batch_size=512, max_seq_len=1024, fake_kv=True
        )
        self.assertEqual(budget, FAKE_KV_TOKEN_BUDGET_CAP)
        self.assertEqual(
            _prefill_token_budget(max_batch_size=512, max_seq_len=1024, fake_kv=False),
            512 * 1024,
        )

    def test_fake_kv_budget_never_falls_below_one_batch_of_headroom(self):
        budget = _prefill_token_budget(
            max_batch_size=8, max_seq_len=131072, fake_kv=True
        )
        self.assertEqual(budget, 131072 + 8 + 64)


# ----------------------------------------------------------------------
# Synchronization accounting
# ----------------------------------------------------------------------


class TestBenchLoopSynchronization(unittest.TestCase):
    """The headline number is a per-round wall clock, so the measurement loop
    must not synchronize between steps — that would drain the GPU and destroy
    decode step overlap. Exactly three syncs per PD round, two otherwise."""

    def test_pd_round_syncs_exactly_three_times(self):
        harness = _FakeHarness(batch_size=2, seq_len=8)
        result = run_decode_bench(
            harness,
            2,
            8,
            4,
            num_iters=3,
            num_warmup_iters=1,
            skip_prefill_forward=False,
        )
        rounds = 4  # 1 warmup + 3 measured
        # start + lap + end, regardless of how many steps ran inside.
        self.assertEqual(harness.sync_count, 3 * rounds)
        self.assertEqual(harness.device_module.synchronize_calls, 3 * rounds)
        self.assertEqual(len(harness.steps), rounds * 5)  # 1 prefill + 4 decode
        self.assertEqual(harness.drain_calls, rounds)
        self.assertEqual(result.num_rounds, 3)  # warmup round discarded
        self.assertEqual(result.mode, PHASE_DECODE)
        self.assertEqual(result.seq_len, 8)

    def test_fake_kv_round_syncs_twice(self):
        # No prefill step, so no mark_lap.
        harness = _FakeHarness(batch_size=2, seq_len=8, decode_only=True)
        run_decode_bench(
            harness, 2, 8, 4, num_iters=2, num_warmup_iters=1, skip_prefill_forward=True
        )
        self.assertEqual(harness.sync_count, 2 * 3)
        self.assertEqual(len(harness.steps), 3 * 4)

    def test_prefill_round_syncs_twice(self):
        harness = _FakeHarness(batch_size=2, seq_len=8)
        result = run_prefill_bench(harness, 2, 8, num_iters=3, num_warmup_iters=1)
        self.assertEqual(harness.sync_count, 2 * 4)
        self.assertEqual(result.mode, PHASE_PREFILL)
        self.assertEqual(result.num_rounds, 3)
        # For a single step, prefill and cost are the same measurement.
        self.assertEqual(result.prefill_ms, result.cost_ms)
        self.assertEqual(result.per_token_ms, 0.0)

    def test_diagnostic_path_syncs_around_every_step(self):
        # The counterpart invariant: --per-step-timing deliberately pays the
        # per-step sync, which is why its numbers live in their own section.
        harness = _FakeHarness(batch_size=2, seq_len=8)
        diag = run_step_diag(
            harness,
            mode=PHASE_DECODE,
            batch_size=2,
            seq_len=8,
            num_decode_steps=4,
            skip_prefill_forward=False,
        )
        self.assertEqual(len(harness.steps), 5)
        self.assertEqual(harness.sync_count, 2 * 5)
        self.assertEqual(diag.num_steps, 4)
        self.assertGreaterEqual(diag.sum_step_ms, diag.prefill_step_ms)

    def test_warmup_rounds_are_driven_identically_and_discarded(self):
        harness = _FakeHarness(batch_size=1, seq_len=8)
        result = run_prefill_bench(harness, 1, 8, num_iters=2, num_warmup_iters=3)
        self.assertEqual(len(harness.steps), 5)
        self.assertEqual(result.num_rounds, 2)


# ----------------------------------------------------------------------
# Loud failures
# ----------------------------------------------------------------------


class TestPhaseAssertions(unittest.TestCase):
    """A batch the scheduler split, chunked or shrank must raise.

    These drive the real bench loops rather than the assertion helpers, so
    dropping an ``assert_*`` call from a loop turns them red too — a silently
    split batch is the one failure mode that still produces a plausible number.
    """

    def test_bench_loop_rejects_a_split_batch(self):
        # The scheduler hands back 3 of the 4 requested requests.
        harness = _FakeHarness(batch_size=4, seq_len=8, served_batch_size=3)
        with self.assertRaises(AssertionError):
            run_prefill_bench(harness, 4, 8, num_iters=1, num_warmup_iters=0)

    def test_bench_loop_rejects_a_chunked_prefill(self):
        harness = _FakeHarness(batch_size=4, seq_len=8, served_extend_tokens=16)
        with self.assertRaises(AssertionError):
            run_prefill_bench(harness, 4, 8, num_iters=1, num_warmup_iters=0)

    def test_bench_loop_rejects_a_decode_step_where_prefill_was_expected(self):
        harness = _FakeHarness(batch_size=2, seq_len=8, decode_only=True)
        with self.assertRaises(AssertionError):
            run_decode_bench(
                harness,
                2,
                8,
                2,
                num_iters=1,
                num_warmup_iters=0,
                skip_prefill_forward=False,
            )

    def test_bench_loop_rejects_a_batch_that_shrinks_mid_decode(self):
        # A request finishing early would shrink the decode batch and quietly
        # skew per_token_ms; only the in-loop assert_batch catches it.
        class _ShrinkingHarness(_FakeHarness):
            def _advance_one_step(self):
                batch = super()._advance_one_step()
                if len(self.steps) > 2:
                    self._served_batch_size = 1
                return batch

        harness = _ShrinkingHarness(batch_size=2, seq_len=8, decode_only=True)
        with self.assertRaises(AssertionError):
            run_decode_bench(
                harness,
                2,
                8,
                4,
                num_iters=1,
                num_warmup_iters=0,
                skip_prefill_forward=True,
            )


class TestFakeKvRegistrationAssertions(unittest.TestCase):
    """Before the first measured decode step every request must sit at the same
    KV length, or the decode batch is ragged and the per-token number is not
    comparable across engines."""

    def test_uniform_registration_passes(self):
        reqs = [_StubReq(f"r{i}", prompt_len=8, output_len=1) for i in range(2)]
        harness = _harness_with_scheduler(_StubScheduler(reqs))
        harness._assert_uniform_registration(2, 8)

    def test_reqs_still_in_last_batch_are_counted(self):
        # get_next_batch_to_run merges an extend batch into running_batch only
        # on the *next* call, so right after registration the requests live in
        # last_batch. Missing this made every --partial 1 run fail.
        reqs = [_StubReq(f"r{i}", prompt_len=8, output_len=1) for i in range(2)]
        harness = _harness_with_scheduler(
            _StubScheduler(running_reqs=[], last_batch_reqs=reqs)
        )
        harness._assert_uniform_registration(2, 8)

    def test_missing_request_raises(self):
        reqs = [_StubReq("r0", prompt_len=8, output_len=1)]
        harness = _harness_with_scheduler(_StubScheduler(reqs))
        with self.assertRaises(AssertionError):
            harness._assert_uniform_registration(2, 8)

    def test_ragged_kv_length_raises(self):
        # r1 decoded an extra token during registration.
        reqs = [
            _StubReq("r0", prompt_len=8, output_len=1),
            _StubReq("r1", prompt_len=8, output_len=2),
        ]
        harness = _harness_with_scheduler(_StubScheduler(reqs))
        with self.assertRaises(AssertionError):
            harness._assert_uniform_registration(2, 8)

    def test_non_uniform_prompt_length_raises(self):
        reqs = [
            _StubReq("r0", prompt_len=8, output_len=1),
            _StubReq("r1", prompt_len=9, output_len=1),
        ]
        harness = _harness_with_scheduler(_StubScheduler(reqs))
        with self.assertRaises(AssertionError):
            harness._assert_uniform_registration(2, 8)

    def test_leftover_waiting_queue_raises(self):
        reqs = [_StubReq(f"r{i}", prompt_len=8, output_len=1) for i in range(2)]
        scheduler = _StubScheduler(reqs)
        scheduler.waiting_queue = [_StubReq("r2", prompt_len=8, output_len=0)]
        harness = _harness_with_scheduler(scheduler)
        with self.assertRaises(AssertionError):
            harness._assert_uniform_registration(2, 8)

    def test_chunked_req_raises(self):
        reqs = [_StubReq(f"r{i}", prompt_len=8, output_len=1) for i in range(2)]
        scheduler = _StubScheduler(reqs)
        scheduler.chunked_req = _StubReq("r9", prompt_len=8, output_len=0)
        harness = _harness_with_scheduler(scheduler)
        with self.assertRaises(AssertionError):
            harness._assert_uniform_registration(2, 8)


# ----------------------------------------------------------------------
# Harness contracts
# ----------------------------------------------------------------------


class TestHarnessContracts(unittest.TestCase):
    def test_synthetic_prompts_differ_per_request(self):
        # Radix cache is disabled, but identical prompts would still be a
        # second way to turn every request after the first into a cache hit.
        first = synthetic_input_ids(0, 32)
        second = synthetic_input_ids(1, 32)
        self.assertEqual(len(first), 32)
        self.assertNotEqual(list(first), list(second))
        self.assertTrue(all(1 <= t <= 4096 for t in first))

    def test_generation_batch_result_accepts_the_fabricated_fields(self):
        # Contract check against upstream drift: _fake_process_batch_result
        # constructs exactly these two fields.
        result = GenerationBatchResult(
            next_token_ids=torch.full((2,), FAKE_TOKEN_ID, dtype=torch.long),
            can_run_cuda_graph=False,
        )
        self.assertEqual(result.next_token_ids.tolist(), [FAKE_TOKEN_ID] * 2)
        self.assertFalse(result.can_run_cuda_graph)


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


class TestReporting(unittest.TestCase):
    def test_primary_metric_per_mode(self):
        decode = BenchResult(
            mode=PHASE_DECODE,
            batch_size=1,
            seq_len=128,
            per_token_ms=12.0,
            prefill_ms=99.0,
        )
        prefill = BenchResult(
            mode=PHASE_PREFILL, batch_size=1, seq_len=128, prefill_ms=13.0
        )
        self.assertEqual(decode.primary_ms, 12.0)
        self.assertEqual(prefill.primary_ms, 13.0)

    def test_csv_layout_matches_the_vllm_patch(self):
        results = [
            BenchResult(
                mode=PHASE_DECODE,
                batch_size=4,
                seq_len=128,
                num_rounds=3,
                cost_ms=1.234,
                cost_mean_ms=1.5,
                prefill_ms=2.0,
                prefill_mean_ms=2.5,
                per_token_ms=3.0,
                per_token_mean_ms=3.5,
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.csv")
            write_csv(results, path)
            with open(path, newline="") as f:
                rows = list(csv.reader(f))
        self.assertEqual(
            rows[0],
            [
                "mode",
                "batch_size",
                "seq_len",
                "num_rounds",
                "cost_ms",
                "cost_mean_ms",
                "prefill_ms",
                "prefill_mean_ms",
                "per_token_ms",
                "per_token_mean_ms",
            ],
        )
        self.assertEqual(
            rows[1],
            ["decode", "4", "128", "3", "1.23", "1.50", "2.00", "2.50", "3.00", "3.50"],
        )

    def test_clean_run_has_no_failure_reasons(self):
        payloads = [
            RankPayload(tp_rank=0, dp_rank=0, status="ok"),
            RankPayload(tp_rank=1, dp_rank=0, status="ok"),
        ]
        self.assertEqual(
            failure_reasons(payloads=payloads, expected_ranks=2, process_failures=[]),
            [],
        )

    def test_failing_rank_is_reported(self):
        payloads = [
            RankPayload(tp_rank=0, dp_rank=0, status="ok"),
            RankPayload(tp_rank=1, dp_rank=0, status="error", error="boom\n"),
        ]
        reasons = failure_reasons(
            payloads=payloads, expected_ranks=2, process_failures=[]
        )
        self.assertEqual(len(reasons), 1)
        self.assertIn("dp0 tp1", reasons[0])
        self.assertIn("boom", reasons[0])

    def test_missing_rank_is_reported(self):
        payloads = [RankPayload(tp_rank=0, dp_rank=0, status="ok")]
        reasons = failure_reasons(
            payloads=payloads, expected_ranks=2, process_failures=["rank #1 exit 1"]
        )
        self.assertEqual(len(reasons), 2)
        self.assertIn("only 1 of 2", reasons[0])
        self.assertIn("exit 1", reasons[1])


if __name__ == "__main__":
    unittest.main()
