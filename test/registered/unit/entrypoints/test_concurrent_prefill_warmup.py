"""The concurrent_prefill warmup must reach the batch shapes serving will see
(one prompt, full concurrency, a round larger than one prefill chunk) without
sending a request the server refuses or can never admit."""

import asyncio
import os
import tempfile
import unittest
from unittest import mock

from sglang.srt.disaggregation.mode import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints import warmup as warmup_mod
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
        "effective_chunked_prefill_size": 16384,
        "load_balance_method": "round_robin",
        "page_size": 1,
        "memory_usage": {
            "token_capacity": 1888043,
            "token_capacity_swa": None,
            "token_capacity_window": {},
        },
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

    def __init__(self, **state_overrides):
        self.requests = []
        self.state = _internal_state(
            effective_max_running_requests_per_dp=8, **state_overrides
        )

    async def get_internal_state(self):
        return [self.state]

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
        limits = concurrent_prefill_limits(
            [_internal_state()],
            context_len=131072,
            max_req_input_len=131066,
            num_reserved_tokens=0,
        )
        self.assertEqual(limits, _limits())
        plan = concurrent_prefill_plan(limits)
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

    def test_tokenizer_workers_plan_as_one_sender(self):
        """With a queue limit only one tokenizer worker sends the warmup, so
        its plan is the single-worker plan."""
        for queue in (None, 1, 64):
            self.assertEqual(
                concurrent_prefill_plan(
                    _limits(max_queued_requests=queue, tokenizer_workers=2)
                ),
                concurrent_prefill_plan(_limits(max_queued_requests=queue)),
            )

    def test_load_based_dp_dispatch_sends_nothing(self):
        """Rounds are sized per DP rank, but total_tokens dispatch balances
        tokens, not requests: a 128-prompt round of uneven lengths over four
        32-slot ranks can put 33 prompts on one rank, over a queue limit of 32."""

        def round_sizes(method, mode="null"):
            states = [
                _internal_state(
                    max_queued_requests=32,
                    load_balance_method=method,
                    disaggregation_mode=mode,
                )
                for _ in range(4)
            ]
            limits = concurrent_prefill_limits(
                states,
                context_len=131072,
                max_req_input_len=131066,
                num_reserved_tokens=0,
            )
            return [len(r) for r in concurrent_prefill_plan(limits).rounds]

        for method in ("total_tokens", "total_requests"):
            self.assertEqual(round_sizes(method), [], method)
        # Without a queue limit an uneven share only waits: the warmup runs.
        states = [_internal_state(load_balance_method="total_tokens")] * 4
        limits = concurrent_prefill_limits(
            states, context_len=131072, max_req_input_len=131066, num_reserved_tokens=0
        )
        self.assertEqual(
            [len(r) for r in concurrent_prefill_plan(limits).rounds], [4, 16, 128]
        )
        self.assertEqual(round_sizes("round_robin"), [4, 16, 128])
        # Consecutive bootstrap rooms spread evenly on a PD prefill server.
        self.assertEqual(
            round_sizes("follow_bootstrap_room", mode="prefill"), [4, 16, 128]
        )

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

    def test_scheduler_without_chunking_gets_nothing(self):
        """A multimodal Transformers model keeps --chunked-prefill-size 16384 in
        the server args, but its scheduler turns chunking off, so a 40,001-token
        prompt must fit a 4,096-token sliding-window pool in one piece and
        waits forever."""
        tm = _RecordingTokenizerManager(
            effective_chunked_prefill_size=None,
            memory_usage={
                "token_capacity": 1888043,
                "token_capacity_swa": 4096,
                "token_capacity_window": {"swa": 4096},
            },
        )
        asyncio.run(concurrent_prefill("null", tm))
        self.assertEqual(tm.requests, [])

    def test_full_attention_model_without_chunking_runs_bounded(self):
        """With chunking off but only the full KV pool, every prompt fits it in
        one piece, so the rounds and the logprob probes still run."""
        tm = _RecordingTokenizerManager(effective_chunked_prefill_size=None)
        asyncio.run(concurrent_prefill("null", tm))
        lens = [len(r.input_ids) for r in tm.requests]
        self.assertTrue(lens)
        self.assertLessEqual(max(lens), _RecordingTokenizerManager.max_req_input_len)

    def test_one_tokenizer_worker_runs_serially_under_a_queue_limit_of_one(self):
        """Two tokenizer workers and --max-queued-requests 1: one worker sends
        the rounds and the logprob probes one request at a time, and the other
        sends nothing, so the waiting queue never holds two warmup requests."""

        class _Worker(_RecordingTokenizerManager):
            def generate_request(self, req, request):
                self.requests.append(req)
                in_flight.append(req)
                peak[0] = max(peak[0], len(in_flight))

                async def finished():
                    await asyncio.sleep(0)
                    in_flight.remove(req)
                    yield {}

                return finished()

        in_flight, peak = [], [0]
        workers = [_Worker(max_queued_requests=1, tokenizer_worker_num=2) for _ in "ab"]
        with tempfile.TemporaryDirectory() as tmp:
            claims = []

            def claim():
                # One process per worker: each holds its own lock descriptor.
                fd = warmup_mod._try_lock(os.path.join(tmp, "warmup.lock"))
                claims.append(fd)
                return fd is not None

            with mock.patch.object(warmup_mod, "_claim_tokenizer_worker_warmup", claim):
                for tm in workers:
                    asyncio.run(concurrent_prefill("null", tm))
            for fd in claims:
                if fd is not None:
                    os.close(fd)
        sent = sorted(len(tm.requests) for tm in workers)
        self.assertEqual(sent[0], 0)
        self.assertGreater(sent[1], 2)  # a round plus the two logprob probes
        self.assertEqual(peak[0], 1)

    def test_worker_lock_is_exclusive_until_its_holder_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "warmup.lock")
            first = warmup_mod._try_lock(path)
            self.assertIsNotNone(first)
            self.assertIsNone(warmup_mod._try_lock(path))
            os.close(first)  # the holder exits
            again = warmup_mod._try_lock(path)
            self.assertIsNotNone(again)
            os.close(again)

    def test_decode_server_sends_nothing(self):
        tm = _RecordingTokenizerManager()
        asyncio.run(concurrent_prefill("decode", tm))
        self.assertEqual(tm.requests, [])


if __name__ == "__main__":
    unittest.main()
