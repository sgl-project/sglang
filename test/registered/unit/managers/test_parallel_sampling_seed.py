import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sglang.srt.managers.tokenizer_manager import (
    TokenizerManager,
    _seed_for_parallel_sample,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(3, "base-a-test-cpu")


class TestSeedForParallelSample(unittest.TestCase):
    def test_parallel_samples_get_distinct_signed_int64_seeds(self):
        cases = [
            (42, True, [42, 43, 44, 45]),
            (None, True, [42, 43, 44, 45]),
            (2**63 - 1, True, [2**63 - 1, -(2**63)]),
            (-(2**63), True, [-(2**63), -(2**63) + 1]),
        ]
        for seed, deterministic, expected in cases:
            with self.subTest(seed=seed, deterministic=deterministic):
                base = SamplingParams(sampling_seed=seed)
                params = [
                    _seed_for_parallel_sample(base, j, deterministic=deterministic)
                    for j in range(len(expected))
                ]
                self.assertEqual([item.sampling_seed for item in params], expected)
                self.assertTrue(all(item is not base for item in params))
                self.assertEqual(base.sampling_seed, seed)

    def test_explicit_seed_is_derived_outside_deterministic_mode(self):
        base = SamplingParams(sampling_seed=7)
        self.assertEqual(
            _seed_for_parallel_sample(base, 2, deterministic=False).sampling_seed, 9
        )

    def test_unseeded_sample_is_returned_unchanged(self):
        base = SamplingParams(sampling_seed=None)
        self.assertIs(_seed_for_parallel_sample(base, 3, deterministic=False), base)


class _Request:
    next_rid = 0

    def __init__(self, rid, sampling_params):
        self.rid = rid
        self.sampling_params = sampling_params
        self.return_prompt_token_ids = False

    def regenerate_rid(self):
        type(self).next_rid += 1
        self.rid = f"child-{type(self).next_rid}"
        return self.rid


class _Batch:
    stream = False

    def __init__(self, requests, parallel_sample_num=3):
        self.requests = requests
        self.batch_size = len(requests)
        self.parallel_sample_num = parallel_sample_num

    def __getitem__(self, index):
        return self.requests[index]


class TestParallelSamplingManagerWiring(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _manager(requests, tokenized, *, deterministic=True):
        states = {
            request.rid: SimpleNamespace(
                time_stats=SimpleNamespace(set_finished_time=lambda: None)
            )
            for request in requests
        }
        sent = []
        manager = object.__new__(TokenizerManager)
        manager.server_args = SimpleNamespace(
            enable_deterministic_inference=deterministic
        )
        manager.rid_to_state = states
        manager._tokenize_one_request = AsyncMock(
            side_effect=lambda request: tokenized[requests.index(request)]
        )

        def init_state(request):
            states[request.rid] = SimpleNamespace(time_stats=SimpleNamespace())

        async def wait_one_response(request, http_request):
            yield {}

        manager._init_req_state = init_state
        manager._send_one_request = sent.append
        manager._wait_one_response = wait_one_response
        manager._collect_batch_responses = AsyncMock(return_value=[])
        manager._should_use_batch_tokenization = lambda batch_size, obj: False
        return manager, sent

    async def test_batch_expansion_is_prompt_major_and_does_not_mutate_inputs(self):
        requests = [
            _Request("prompt-0", SamplingParams(sampling_seed=10)),
            _Request("prompt-1", SamplingParams()),
        ]
        tokenized = [
            SimpleNamespace(
                sampling_params=request.sampling_params,
                input_ids=[index],
                mm_inputs=None,
            )
            for index, request in enumerate(requests)
        ]
        manager, sent = self._manager(requests, tokenized)

        output = [
            item async for item in manager._handle_batch_request(_Batch(requests))
        ]

        self.assertEqual(output, [[]])
        self.assertEqual(
            [item.sampling_params.sampling_seed for item in sent[2:]],
            [10, 11, 12, 42, 43, 44],
        )
        self.assertEqual(
            [item.sampling_params.max_new_tokens for item in sent[:2]], [0, 0]
        )
        self.assertEqual(
            [item.sampling_params.sampling_seed for item in tokenized], [10, None]
        )

    async def test_single_samples_keep_explicit_and_omitted_seeds(self):
        requests = [
            _Request("prompt-0", SamplingParams(sampling_seed=2**63 - 1)),
            _Request("prompt-1", SamplingParams()),
        ]
        tokenized = [
            SimpleNamespace(sampling_params=request.sampling_params, input_ids=[index])
            for index, request in enumerate(requests)
        ]
        manager, sent = self._manager(requests, tokenized)

        output = [
            item
            async for item in manager._handle_batch_request(
                _Batch(requests, parallel_sample_num=1)
            )
        ]

        self.assertEqual(output, [[]])
        self.assertEqual(
            [item.sampling_params.sampling_seed for item in sent],
            [2**63 - 1, None],
        )
        self.assertEqual(sent, tokenized)


if __name__ == "__main__":
    unittest.main()
