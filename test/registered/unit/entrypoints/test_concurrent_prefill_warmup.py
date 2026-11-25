"""The concurrent_prefill warmup must reach the batch shapes serving will see
(one prompt, full concurrency, a round larger than one prefill chunk) without
sending a request the server refuses or can never admit."""

import asyncio
import unittest

from sglang.srt.disaggregation.mode import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints.warmup import (
    ConcurrentPrefillLimits,
    concurrent_prefill,
    concurrent_prefill_limits,
    concurrent_prefill_plan,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _limits(**overrides):
    # The served Muse Spark NVFP4 + DFlash config on 8x H200.
    fields = dict(
        max_running_requests=32,
        max_queued_requests=None,
        chunked_prefill_size=16384,
        context_len=131072,
        max_req_input_len=131066,
        num_reserved_tokens=0,
        kv_pool_tokens=1888043,
        page_size=1,
        serves_input_logprobs=True,
    )
    fields.update(overrides)
    return ConcurrentPrefillLimits(**fields)


def _internal_state(**overrides):
    state = {
        "effective_max_running_requests_per_dp": 32,
        "max_queued_requests": None,
        "chunked_prefill_size": 16384,
        "page_size": 1,
        "memory_usage": {"token_capacity": 1888043},
        "speculative_algorithm": None,
        "mlx_enable_sampling": False,
        "disaggregation_mode": "null",
        "tokenizer_worker_num": 1,
    }
    state.update(overrides)
    return state


class _FakeModelConfig:
    vocab_size = 1000


class _RecordingTokenizerManager:
    is_generation = True
    context_len = 4096
    max_req_input_len = 4090
    num_reserved_tokens = 0
    model_config = _FakeModelConfig()

    def __init__(self):
        self.requests = []

    async def get_internal_state(self):
        return [_internal_state(effective_max_running_requests_per_dp=8)]

    def generate_request(self, req, request):
        self.requests.append(req)

        async def finished():
            yield {}

        return finished()


class TestConcurrentPrefillPlan(CustomTestCase):
    def test_served_config_plan(self):
        """The plan the cold-start TTFT gain was measured with: rounds of 1, 4
        and 32 prompts (40K tokens per share, lengths stepped by 37), then the
        two input-logprob probes."""
        plan = concurrent_prefill_plan(_limits())
        self.assertEqual(
            plan.rounds,
            [
                [40001],
                [10001 + 37 * i for i in range(4)],
                [1251 + 37 * i for i in range(32)],
            ],
        )
        self.assertEqual(plan.logprob_sizes, [513, 512])

    def test_unset_max_running_requests_uses_effective_slots(self):
        """Without --max-running-requests the server-arg is None; the rounds
        must follow the scheduler's effective slot count of every DP rank."""
        states = [
            _internal_state(effective_max_running_requests_per_dp=48),
            _internal_state(effective_max_running_requests_per_dp=48),
        ]
        limits = concurrent_prefill_limits(
            states, context_len=131072, max_req_input_len=131066, num_reserved_tokens=0
        )
        plan = concurrent_prefill_plan(limits)
        self.assertEqual([len(r) for r in plan.rounds], [2, 12, 96])

    def test_each_dp_rank_gets_a_full_round_budget(self):
        """Four 8-slot DP ranks with 16K chunks: the round is sized per rank, so
        every rank (round robin) runs more than two chunks of prompts."""
        states = [
            _internal_state(effective_max_running_requests_per_dp=8) for _ in range(4)
        ]
        limits = concurrent_prefill_limits(
            states, context_len=131072, max_req_input_len=131066, num_reserved_tokens=0
        )
        plan = concurrent_prefill_plan(limits)
        self.assertEqual([len(r) for r in plan.rounds], [4, 32])
        for sizes in plan.rounds:
            for rank in range(4):
                self.assertGreater(sum(sizes[rank::4]), 2 * 16384)

    def test_round_never_exceeds_the_waiting_queue(self):
        """A whole round can land in one scheduler poll, so a round larger than
        --max-queued-requests gets 503s."""
        plan = concurrent_prefill_plan(_limits(max_queued_requests=1))
        self.assertEqual([len(r) for r in plan.rounds], [1])
        self.assertEqual(
            concurrent_prefill_plan(_limits(max_queued_requests=0)).rounds, []
        )

    def test_tokenizer_workers_share_the_waiting_queue(self):
        """Each tokenizer worker runs this warmup at the same time, so their
        requests together (rounds and the one-at-a-time logprob probes) must
        fit in the waiting queue."""
        limits = concurrent_prefill_limits(
            [_internal_state(max_queued_requests=2, tokenizer_worker_num=2)],
            context_len=131072,
            max_req_input_len=131066,
            num_reserved_tokens=0,
        )
        plan = concurrent_prefill_plan(limits)
        self.assertNotEqual(plan.rounds, [])
        self.assertLessEqual(2 * max(len(r) for r in plan.rounds), 2)
        self.assertEqual(plan.logprob_sizes, [513, 512])
        plan = concurrent_prefill_plan(
            _limits(max_queued_requests=1, tokenizer_workers=2)
        )
        self.assertEqual((plan.rounds, plan.logprob_sizes), ([], []))

    def test_pd_prefill_ignores_the_queue_limit(self):
        """A PD prefill scheduler never applies --max-queued-requests, so the
        limit must not shrink or skip the warmup there."""

        def round_sizes(mode):
            limits = concurrent_prefill_limits(
                [_internal_state(max_queued_requests=0, disaggregation_mode=mode)],
                context_len=131072,
                max_req_input_len=131066,
                num_reserved_tokens=0,
            )
            return [len(r) for r in concurrent_prefill_plan(limits).rounds]

        self.assertEqual(round_sizes("null"), [])
        self.assertEqual(round_sizes("prefill"), [1, 4, 32])

    def test_round_budget_follows_chunk_size(self):
        """Each round spans a chunk boundary even with a large chunk, and the
        length variation stays under two budgets even with many slots."""
        plan = concurrent_prefill_plan(
            _limits(max_running_requests=4, chunked_prefill_size=65536)
        )
        for sizes in plan.rounds:
            self.assertGreater(sum(sizes), 65536)
        plan = concurrent_prefill_plan(_limits(max_running_requests=256))
        for sizes in plan.rounds:
            self.assertGreater(sum(sizes), 16384)
            self.assertLess(sum(sizes), 2 * 40000)

    def test_prompt_leaves_reserved_tokens_under_context(self):
        """EAGLE with 16 draft tokens on a 4,096 context: the tokenizer counts
        the 16 reserved slots and the new tokens against the context."""
        limits = _limits(
            context_len=4096, max_req_input_len=4090, num_reserved_tokens=16
        )
        plan = concurrent_prefill_plan(limits)
        longest = max(max(r) for r in plan.rounds)
        self.assertLess(longest + 16, 4096)
        self.assertLessEqual(longest + 16 + 8, 4096)

    def test_prompt_leaves_page_headroom_in_kv_pool(self):
        """A 32,768-token pool with 64-token pages: admission rounds the prompt
        up to pages and reserves one more page plus the new tokens, so a prompt
        sized to max_req_input_len is never admitted."""
        page = 64
        limits = _limits(kv_pool_tokens=32768, max_req_input_len=32762, page_size=page)
        longest = max(max(r) for r in concurrent_prefill_plan(limits).rounds)
        charged = -(-longest // page) * page + page + 8
        self.assertLessEqual(charged, 32768)

    def test_skips_logprob_probes_the_server_refuses(self):
        for overrides in (
            {"speculative_algorithm": "UNO"},
            {"mlx_enable_sampling": True},
        ):
            limits = concurrent_prefill_limits(
                [_internal_state(**overrides)],
                context_len=131072,
                max_req_input_len=131066,
                num_reserved_tokens=0,
            )
            plan = concurrent_prefill_plan(limits)
            self.assertEqual(plan.logprob_sizes, [], overrides)
            self.assertNotEqual(plan.rounds, [], overrides)


class TestConcurrentPrefillRequests(CustomTestCase):
    def test_pd_prefill_requests_get_distinct_bootstrap_rooms(self):
        """A PD prefill server rejects a request without a bootstrap room, and
        concurrent requests sharing one room collide."""
        tm = _RecordingTokenizerManager()
        asyncio.run(concurrent_prefill("prefill", tm))
        self.assertGreater(len(tm.requests), 2)
        self.assertTrue(
            all(r.bootstrap_host == FAKE_BOOTSTRAP_HOST for r in tm.requests)
        )
        rooms = [r.bootstrap_room for r in tm.requests]
        self.assertEqual(len(set(rooms)), len(rooms))

    def test_tokenizer_workers_use_distinct_bootstrap_rooms(self):
        """Every tokenizer worker runs this warmup at the same time against the
        same PD prefill scheduler."""
        workers = [_RecordingTokenizerManager() for _ in range(2)]
        for tm in workers:
            asyncio.run(concurrent_prefill("prefill", tm))
        rooms = [{r.bootstrap_room for r in tm.requests} for tm in workers]
        self.assertEqual(rooms[0] & rooms[1], set())

    def test_decode_server_sends_nothing(self):
        tm = _RecordingTokenizerManager()
        asyncio.run(concurrent_prefill("decode", tm))
        self.assertEqual(tm.requests, [])


if __name__ == "__main__":
    unittest.main()
