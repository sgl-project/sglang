"""A dLLM cache lookup must stop before synthetic or incomplete blocks."""

import unittest
from array import array

import torch
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_req(prompt, output=(), *, dllm=True, logprob_start=-1):
    config = DllmConfig("LowConfidence", {}, 4, 99, 4) if dllm else None
    req = Req(
        rid="prefix-boundary",
        origin_input_text="",
        origin_input_ids=array("q", prompt),
        sampling_params=SamplingParams(max_new_tokens=8, temperature=0),
        vocab_size=128,
        dllm_config=config,
        return_logprob=logprob_start >= 0,
    )
    req.logprob_start_len = logprob_start
    req.output_ids.extend(output)
    return req


class TestDllmPrefixBoundary(unittest.TestCase):
    def test_cache_cannot_consume_prompt_tail_and_appended_masks(self):
        # This is legitimate cached input from a different request that
        # contains literal mask-token IDs. The new request only shares its
        # first five real tokens, so only its first full block is reusable.
        cache = RadixCache.create_simulated(page_size=4)
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3, 4, 5, 99, 99, 99])),
                value=torch.arange(8),
            )
        )
        req = make_req([1, 2, 3, 4, 5])
        req.init_next_round_input(cache)
        self.assertEqual(req.prefix_indices.tolist(), [0, 1, 2, 3])
        self.assertEqual(req.dllm_block_offset, 4)
        self.assertEqual(list(req.full_untruncated_fill_ids[4:8]), [5, 99, 99, 99])

    def test_real_literal_masks_in_complete_blocks_remain_cacheable(self):
        prompt = [1, 99, 2, 99, 3, 99, 4, 99]
        cache = RadixCache.create_simulated(page_size=4)
        cache.insert(
            InsertParams(key=RadixKey(array("q", prompt)), value=torch.arange(8))
        )
        req = make_req(prompt)
        req.init_next_round_input(cache)
        self.assertEqual(len(req.prefix_indices), 8)

    def test_real_token_and_logprob_boundaries(self):
        for prompt_len in (1, 3, 4, 5, 7, 8, 9):
            for output_len in (0, 1, 3, 4, 7):
                for logprob_start in (-1, 0, 3, 4, 5, 8):
                    with self.subTest(
                        prompt=prompt_len, output=output_len, logprob=logprob_start
                    ):
                        req = make_req(
                            range(prompt_len),
                            range(output_len),
                            logprob_start=logprob_start,
                        )
                        req._init_fill_ids_for_dllm()
                        real_len = prompt_len + output_len
                        boundary = (
                            min(real_len, logprob_start)
                            if logprob_start >= 0
                            else real_len
                        )
                        self.assertEqual(
                            req._compute_max_prefix_len(
                                len(req.full_untruncated_fill_ids)
                            ),
                            boundary // 4 * 4,
                        )

    def test_autoregressive_logprob_behavior_is_unchanged(self):
        req = make_req(range(9), dllm=False)
        self.assertEqual(req._compute_max_prefix_len(9), 8)
        req = make_req(range(9), dllm=False, logprob_start=5)
        self.assertEqual(req._compute_max_prefix_len(9), 5)


if __name__ == "__main__":
    unittest.main()
