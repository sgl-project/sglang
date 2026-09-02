"""Unit tests for the serving_coverage warmup, no server, no model."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints import warmup as warmup_module
from sglang.srt.entrypoints.warmup import (
    _SERVING_COVERAGE_MAX_COHORT,
    _SERVING_COVERAGE_MAX_DECODE,
    _SERVING_COVERAGE_SAMPLING_VARIANTS,
    _gather_all,
    _serving_coverage_budget,
    _serving_coverage_phases,
    _warmup_registry,
    serving_coverage,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MOD = "sglang.srt.entrypoints.warmup"
CHUNK = 256
MAX_RUNNING = 3
# A production-sized input limit: clamping is always on, never binding here.
INPUT_LIMIT = 65536


class FakeTokenizerManager:
    """Records every request handed to generate_request and its concurrency."""

    def __init__(
        self,
        image_token=None,
        tokenizer=object(),
        max_req_input_len=INPUT_LIMIT,
    ):
        self.max_req_input_len = max_req_input_len
        self.model_config = SimpleNamespace(vocab_size=32000)
        self.tokenizer = tokenizer
        self.mm_processor = (
            SimpleNamespace(mm_tokens=SimpleNamespace(image_token=image_token))
            if image_token is not None
            else None
        )
        self.requests = []
        self.max_in_flight = 0
        self._in_flight = 0

    async def generate_request(self, req, request):
        self.requests.append(req)
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        await asyncio.sleep(0)
        self._in_flight -= 1
        yield {}


def _phases(tokenizer_manager, disaggregation_mode="null"):
    return dict(_serving_coverage_phases(disaggregation_mode, tokenizer_manager))


def _prompt_lens(requests):
    return [len(r.input_ids) for r in requests]


class _PublishedConfig(unittest.IsolatedAsyncioTestCase):
    """The warmup reads the published schedule/parallel bags, like production."""

    max_running_requests = MAX_RUNNING
    chunked_prefill_size = CHUNK
    dp_size = 1

    def setUp(self):
        override = get_context().override_server_args(
            max_running_requests=self.max_running_requests,
            chunked_prefill_size=self.chunked_prefill_size,
            dp_size=self.dp_size,
        )
        override.install()
        self.addCleanup(override.restore)


class TestServingCoveragePhases(_PublishedConfig):
    def test_is_registered_under_its_warmup_name(self):
        self.assertIs(_warmup_registry["serving_coverage"], serving_coverage)

    def test_phase_families_and_order(self):
        names = [
            name for name, _ in _serving_coverage_phases("null", FakeTokenizerManager())
        ]
        self.assertEqual(
            names,
            [
                "cohort",
                "cold-cohort",
                "chunk-multiples",
                "short-prompts",
                "natural-text",
                "sampling",
            ],
        )

    async def test_cohort_issues_max_running_requests_concurrently(self):
        tm = FakeTokenizerManager()
        await _phases(tm)["cohort"]()

        self.assertEqual(len(tm.requests), MAX_RUNNING)
        self.assertEqual(tm.max_in_flight, MAX_RUNNING)
        self.assertEqual(len(set(_prompt_lens(tm.requests))), MAX_RUNNING)
        self.assertEqual(
            len({r.sampling_params["max_new_tokens"] for r in tm.requests}),
            MAX_RUNNING,
        )
        for req in tm.requests:
            self.assertEqual(req.sampling_params["temperature"], 0.0)
            self.assertTrue(req.sampling_params["ignore_eos"])
            self.assertIsNone(req.routed_dp_rank)

    async def test_cold_cohort_prompts_all_exceed_one_chunk(self):
        tm = FakeTokenizerManager()
        await _phases(tm)["cold-cohort"]()

        self.assertEqual(len(tm.requests), MAX_RUNNING)
        self.assertEqual(tm.max_in_flight, MAX_RUNNING)
        for n in _prompt_lens(tm.requests):
            self.assertGreater(n, CHUNK)

    async def test_chunk_multiple_prefills_present(self):
        tm = FakeTokenizerManager()
        await _phases(tm)["chunk-multiples"]()

        lens = _prompt_lens(tm.requests)
        for expected in (CHUNK, 2 * CHUNK, CHUNK + 64, 2 * CHUNK + 64, 3 * CHUNK + 64):
            self.assertIn(expected, lens)
        # Single prefills, one at a time.
        self.assertEqual(tm.max_in_flight, 1)

    async def test_short_prompts_and_one_long_decode(self):
        tm = FakeTokenizerManager()
        await _phases(tm)["short-prompts"]()

        lens = _prompt_lens(tm.requests)
        for expected in (16, 48, 128, 320, 768):
            self.assertIn(expected, lens)
        self.assertEqual(tm.max_in_flight, MAX_RUNNING)
        long_decodes = [
            r for r in tm.requests if r.sampling_params["max_new_tokens"] == 256
        ]
        self.assertEqual(len(long_decodes), 1)

    async def test_natural_text_requests_carry_text(self):
        tm = FakeTokenizerManager()
        await _phases(tm)["natural-text"]()

        self.assertGreaterEqual(len(tm.requests), 1 + MAX_RUNNING)
        for req in tm.requests:
            self.assertIsNone(req.input_ids)
            self.assertIsInstance(req.text, str)
            self.assertTrue(req.text)

    def test_natural_text_is_skipped_without_a_tokenizer(self):
        phases = _phases(FakeTokenizerManager(tokenizer=None))
        self.assertNotIn("natural-text", phases)
        self.assertIn("sampling", phases)

    async def test_sampling_phase_sets_non_greedy_params(self):
        tm = FakeTokenizerManager()
        await _phases(tm)["sampling"]()

        self.assertEqual(
            len(tm.requests), len(_SERVING_COVERAGE_SAMPLING_VARIANTS) + MAX_RUNNING
        )
        for req in tm.requests:
            params = req.sampling_params
            self.assertGreater(params["temperature"], 0.0)
            self.assertTrue(
                {"top_p", "top_k", "min_p"} & set(params),
                f"no sampling knob in {params}",
            )

    async def test_sampling_phase_falls_back_to_ids_without_a_tokenizer(self):
        tm = FakeTokenizerManager(tokenizer=None)
        await _phases(tm)["sampling"]()
        for req in tm.requests:
            self.assertIsNone(req.text)
            self.assertIsNotNone(req.input_ids)

    async def test_image_phase_only_when_the_processor_has_an_image_token(self):
        self.assertNotIn("image", _phases(FakeTokenizerManager()))
        self.assertNotIn("image", _phases(FakeTokenizerManager(image_token="")))

        tm = FakeTokenizerManager(image_token="<image>")
        phases = _phases(tm)
        self.assertIn("image", phases)
        await phases["image"]()

        self.assertEqual(len(tm.requests), 1)
        req = tm.requests[0]
        self.assertIn("<image>", req.text)
        self.assertEqual(len(req.image_data), 1)
        self.assertTrue(req.image_data[0].startswith("data:image/png;base64,"))

    async def test_image_token_list_uses_its_first_entry(self):
        tm = FakeTokenizerManager(image_token=["<img>", "<img_end>"])
        await _phases(tm)["image"]()
        self.assertIn("<img>", tm.requests[0].text)

    async def test_disaggregation_gives_concurrent_requests_distinct_rooms(self):
        tm = FakeTokenizerManager()
        await _phases(tm, disaggregation_mode="prefill")["cohort"]()

        rooms = [r.bootstrap_room for r in tm.requests]
        self.assertEqual(len(set(rooms)), MAX_RUNNING)
        for req in tm.requests:
            self.assertEqual(req.bootstrap_host, FAKE_BOOTSTRAP_HOST)

    async def test_token_ids_stay_inside_the_vocabulary(self):
        tm = FakeTokenizerManager()
        tm.model_config.vocab_size = 100
        await _phases(tm)["short-prompts"]()
        for req in tm.requests:
            self.assertLess(max(req.input_ids), 100)

    async def test_a_failing_phase_does_not_abort_the_others(self):
        ran = []

        async def boom():
            raise RuntimeError("phase failure")

        async def ok():
            ran.append("ok")

        with patch(
            f"{MOD}._serving_coverage_phases",
            return_value=[("boom", boom), ("ok", ok)],
        ):
            # The module names its logger by file path, so pass the object.
            with self.assertLogs(warmup_module.logger, level="ERROR") as cm:
                await serving_coverage("null", FakeTokenizerManager())

        self.assertEqual(ran, ["ok"])
        self.assertTrue(any("boom" in line for line in cm.output))

    async def test_full_warmup_drives_every_phase(self):
        tm = FakeTokenizerManager(image_token="<image>")
        await serving_coverage("null", tm)

        self.assertTrue(any(r.image_data for r in tm.requests))
        self.assertTrue(any(r.text and not r.image_data for r in tm.requests))
        self.assertTrue(any(r.sampling_params["temperature"] > 0 for r in tm.requests))
        self.assertEqual(tm.max_in_flight, MAX_RUNNING)
        for req in tm.requests:
            if req.input_ids is not None:
                self.assertLessEqual(len(req.input_ids), INPUT_LIMIT - 384)
            self.assertLessEqual(
                req.sampling_params["max_new_tokens"], _SERVING_COVERAGE_MAX_DECODE
            )


class TestCohortCap(_PublishedConfig):
    max_running_requests = 1000

    async def test_cohort_is_capped(self):
        tm = FakeTokenizerManager()
        await _phases(tm)["cohort"]()
        self.assertEqual(len(tm.requests), _SERVING_COVERAGE_MAX_COHORT)


class TestUnresolvedMaxRunning(_PublishedConfig):
    max_running_requests = None

    async def test_unresolved_max_running_requests_uses_the_cap(self):
        tm = FakeTokenizerManager()
        await _phases(tm)["cohort"]()
        self.assertEqual(len(tm.requests), _SERVING_COVERAGE_MAX_COHORT)


class TestGatherWaitsForSiblings(unittest.IsolatedAsyncioTestCase):
    async def test_a_failing_request_does_not_return_before_its_peers(self):
        finished = []

        async def fails_at_once():
            raise ValueError("rejected")

        async def slow(i):
            for _ in range(3):
                await asyncio.sleep(0)
            finished.append(i)

        with self.assertRaisesRegex(ValueError, "rejected"):
            await _gather_all([fails_at_once(), slow(1), slow(2), slow(3)])

        # The failure surfaced only after every sibling had drained.
        self.assertEqual(sorted(finished), [1, 2, 3])

    async def test_all_successes_raise_nothing(self):
        async def ok():
            return 1

        await _gather_all([ok(), ok()])


class TestPromptClamping(_PublishedConfig):
    def test_budget_reserves_the_decode_length(self):
        self.assertEqual(_serving_coverage_budget(None), (None, 320))
        self.assertEqual(_serving_coverage_budget(0), (None, 320))
        self.assertEqual(_serving_coverage_budget(65536), (65536 - 384, 320))
        self.assertEqual(_serving_coverage_budget(1024), (640, 320))
        # Below 1024 the decode shrinks too, so prompt + decode still fits.
        self.assertEqual(_serving_coverage_budget(512), (320, 128))
        # Below 256 there is no template headroom; the budgets still fit.
        self.assertEqual(_serving_coverage_budget(64), (48, 16))
        self.assertEqual(_serving_coverage_budget(8), (6, 2))
        self.assertEqual(_serving_coverage_budget(4), (3, 1))
        self.assertEqual(_serving_coverage_budget(1), (1, 1))
        for limit in (2, 4, 8, 64, 200, 255, 256, 512, 1023, 1024, 4096):
            max_prompt, max_decode = _serving_coverage_budget(limit)
            self.assertGreaterEqual(max_prompt, 1, limit)
            self.assertGreaterEqual(max_decode, 1, limit)
            self.assertLessEqual(max_prompt + max_decode, limit, limit)

    async def test_tiny_limits_skip_text_phases_and_still_fit(self):
        # Text and image prompts are not clamped (the template adds tokens the
        # warmup cannot count), so below a 32-token prompt budget those phases
        # are skipped and the sampling phase falls back to token ids.
        for limit in (4, 8, 24):
            with self.subTest(limit=limit):
                tm = FakeTokenizerManager(max_req_input_len=limit, image_token="<img>")
                max_prompt, max_decode = _serving_coverage_budget(limit)
                with self.assertLogs(warmup_module.logger, level="INFO") as cm:
                    phases = _phases(tm)
                self.assertNotIn("natural-text", phases)
                self.assertNotIn("image", phases)
                self.assertIn("sampling", phases)
                self.assertTrue(any("text and image" in line for line in cm.output))
                await serving_coverage("null", tm)
                self.assertTrue(tm.requests)
                for req in tm.requests:
                    self.assertIsNone(req.text)
                    self.assertGreaterEqual(len(req.input_ids), 1)
                    self.assertLessEqual(len(req.input_ids), max_prompt)
                    self.assertLessEqual(
                        len(req.input_ids) + req.sampling_params["max_new_tokens"],
                        limit,
                    )

    async def test_a_32_token_budget_keeps_text_phases(self):
        tm = FakeTokenizerManager(max_req_input_len=64, image_token="<img>")
        phases = _phases(tm)
        self.assertIn("natural-text", phases)
        self.assertIn("image", phases)

    async def test_prompts_and_decodes_are_clamped_at_1024(self):
        tm = FakeTokenizerManager(max_req_input_len=1024)
        max_prompt, max_decode = _serving_coverage_budget(1024)
        phases = _phases(tm)
        # chunk (256) + 64 fits in 640, so the multi-chunk phases survive.
        self.assertIn("cold-cohort", phases)
        self.assertIn("chunk-multiples", phases)

        await serving_coverage("null", tm)

        self.assertTrue(tm.requests)
        for req in tm.requests:
            if req.input_ids is not None:
                self.assertLessEqual(len(req.input_ids), max_prompt)
            self.assertLessEqual(req.sampling_params["max_new_tokens"], max_decode)
            self.assertLessEqual(
                len(req.input_ids or []) + req.sampling_params["max_new_tokens"],
                1024,
            )
        # Something was actually clamped: the 768-token short prompt and the
        # 3*chunk+64 = 832 prefill both exceed the 640 budget.
        self.assertIn(max_prompt, _prompt_lens(r for r in tm.requests if r.input_ids))

    async def test_a_zero_input_limit_never_means_unclamped(self):
        # A limit that leaves no headroom above 1024 used to compute a prompt
        # budget of 0 and read it as "do not clamp".
        tm = FakeTokenizerManager(max_req_input_len=1024)
        await _phases(tm)["short-prompts"]()
        self.assertTrue(all(len(r.input_ids) <= 640 for r in tm.requests))


class TestOneTokenChunk(_PublishedConfig):
    chunked_prefill_size = 1

    async def test_no_request_has_an_empty_prompt(self):
        # chunk // 2 is 0 with a one-token chunk; every prompt still has a token.
        tm = FakeTokenizerManager()
        await _phases(tm)["cohort"]()
        self.assertTrue(tm.requests)
        self.assertTrue(all(len(r.input_ids) >= 1 for r in tm.requests))


class TestPhasesThatCannotFitAreSkipped(_PublishedConfig):
    chunked_prefill_size = 4096

    def test_multi_chunk_phases_are_skipped_with_a_reason(self):
        tm = FakeTokenizerManager(max_req_input_len=1024)
        with self.assertLogs(warmup_module.logger, level="INFO") as cm:
            phases = _phases(tm)
        self.assertNotIn("cold-cohort", phases)
        self.assertNotIn("chunk-multiples", phases)
        self.assertIn("cohort", phases)
        self.assertIn("short-prompts", phases)
        skipped = [line for line in cm.output if "skipping" in line]
        self.assertEqual(len(skipped), 2)
        self.assertTrue(all("640" in line and "4096" in line for line in skipped))

    async def test_remaining_phases_still_fit(self):
        tm = FakeTokenizerManager(max_req_input_len=1024)
        await _phases(tm)["cohort"]()
        for req in tm.requests:
            self.assertLessEqual(len(req.input_ids), 640)


class TestDataParallelCoverage(_PublishedConfig):
    dp_size = 3

    async def test_every_dp_rank_gets_the_full_cohort(self):
        tm = FakeTokenizerManager()
        await _phases(tm)["cohort"]()

        self.assertEqual(len(tm.requests), self.dp_size * MAX_RUNNING)
        by_rank = {}
        for req in tm.requests:
            by_rank.setdefault(req.routed_dp_rank, []).append(req)
        self.assertEqual(set(by_rank), set(range(self.dp_size)))
        for rank, reqs in by_rank.items():
            self.assertEqual(len(reqs), MAX_RUNNING, rank)
            self.assertEqual(len(set(_prompt_lens(reqs))), MAX_RUNNING, rank)
        # Ranks run concurrently: the whole fan-out is in flight at once.
        self.assertEqual(tm.max_in_flight, self.dp_size * MAX_RUNNING)

    async def test_single_prefill_phases_stay_alone_per_rank(self):
        tm = FakeTokenizerManager()
        await _phases(tm)["chunk-multiples"]()
        # Concurrency across ranks never exceeds one request per rank.
        self.assertLessEqual(tm.max_in_flight, self.dp_size)
        for rank in range(self.dp_size):
            lens = _prompt_lens(r for r in tm.requests if r.routed_dp_rank == rank)
            self.assertIn(CHUNK, lens)
            self.assertIn(2 * CHUNK, lens)

    async def test_sampling_and_short_prompts_cover_every_rank(self):
        tm = FakeTokenizerManager()
        phases = _phases(tm)
        await phases["short-prompts"]()
        await phases["sampling"]()
        ranks = {r.routed_dp_rank for r in tm.requests}
        self.assertEqual(ranks, set(range(self.dp_size)))
        sampled = [r for r in tm.requests if r.sampling_params["temperature"] > 0]
        self.assertEqual({r.routed_dp_rank for r in sampled}, set(range(self.dp_size)))


class TestWideDataParallelWindows(_PublishedConfig):
    dp_size = 20

    async def test_ranks_run_in_bounded_windows(self):
        from sglang.srt.entrypoints.warmup import (
            _SERVING_COVERAGE_MAX_CONCURRENT_DP_RANKS as WINDOW,
        )

        tm = FakeTokenizerManager()
        await _phases(tm)["cohort"]()
        self.assertEqual(len(tm.requests), self.dp_size * MAX_RUNNING)
        self.assertEqual({r.routed_dp_rank for r in tm.requests}, set(range(20)))
        self.assertLessEqual(tm.max_in_flight, WINDOW * MAX_RUNNING)


if __name__ == "__main__":
    unittest.main()
