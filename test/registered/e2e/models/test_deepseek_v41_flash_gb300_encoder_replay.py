"""DeepSeek-V4.1-Flash + DSPARK with encoder SWA bounded replay on four GB300 GPUs.

GSM8K with TP4 + EP4, plus prompt logprobs scored on a replayed prefix hit
against a cold prefill of the same prompt.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.dsv41_flash_common import TP4_EP4_ARGS, DSV41FlashAccuracyMixin
from sglang.test.kl_test_utils import (
    _flush_cache,
    _generate,
    compare_kl_divergence,
    get_input_ids,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=600, stage="base-c", runner_config="4-gpu-gb300")

PROMPT_LEN = 2048
SCORED_LEN = 768
KL_THRESHOLD = 0.02


class TestDSV41FlashEncoderReplayDSpark(DSV41FlashAccuracyMixin, CustomTestCase):
    """Main KV and indexer keys are cached; a hit rebuilds its SWA window."""

    parallel_args = TP4_EP4_ARGS + ["--enable-encoder-swa-bounded-replay"]

    @unittest.skip("encoder SWA replay accepts token-only requests")
    def test_mmmu_pro(self):
        pass

    def test_prompt_logprobs_on_replayed_hit_match_cold_prefill(self):
        """Scoring a cached prompt cuts the hit before the scored rows and replays
        the window in front of them; the scores must match a cold prefill."""
        prompts = [ids[:PROMPT_LEN] for ids in get_input_ids(self.model, num_samples=4)]
        start = PROMPT_LEN - SCORED_LEN
        cold = []
        for ids in prompts:
            _flush_cache(self.base_url)
            cold.extend(self._score([ids], start))
        warm = self._score(prompts, start)
        for i, (before, after) in enumerate(zip(cold, warm)):
            self.assertEqual(before["meta_info"]["cached_tokens"], 0, f"cold[{i}]")
            cached = after["meta_info"]["cached_tokens"]
            self.assertTrue(0 < cached <= start, f"warm[{i}]: cached_tokens={cached}")
        compare_kl_divergence(
            [self._scored_logprobs(r) for r in warm],
            [self._scored_logprobs(r) for r in cold],
            {self.model: {"kl_div": KL_THRESHOLD}},
            self.model,
            "encoder_replay_prompt_logprobs",
        )

    def _score(self, prompts, start):
        return _generate(
            self.base_url,
            prompts,
            max_new_tokens=0,
            return_logprob=True,
            logprob_start_len=start,
        )

    @staticmethod
    def _scored_logprobs(result):
        # The first scored position has no logprob of its own.
        return [
            x[0]
            for x in result["meta_info"]["input_token_logprobs"]
            if x[0] is not None
        ]


if __name__ == "__main__":
    unittest.main()
