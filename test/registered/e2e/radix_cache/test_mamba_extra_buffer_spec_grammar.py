"""A prefix-cache hit on a hybrid model's decode checkpoint must score the
continuation like a cache miss, also when the cached request finished under a
grammar while speculative decoding was on."""

import unittest

import requests
from transformers import AutoTokenizer

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-large")

MODEL = "Qwen/Qwen3.5-0.8B"
# Default --mamba-track-interval: the grid on which decode checkpoints the
# recurrent state for the prefix cache.
TRACK_INTERVAL = 256
PHRASE = "The answer is yes. "
REGEX = "The answer is yes\\. "
FILLER = (
    "The history of the printing press begins in the fifteenth century, when "
    "movable type transformed how books were made and shared across Europe. "
) * 12
FOLLOW_UP = " In summary, the most important consequence of this invention was"
# extra_buffer needs the Triton linear-attention backend; NGRAM drafts from
# the prompt tail, so no draft model is needed.
SERVER_ARGS = [
    "--random-seed",
    "1",
    "--linear-attn-backend",
    "triton",
    "--mamba-backend",
    "triton",
    "--mamba-radix-cache-strategy",
    "extra_buffer",
    "--max-mamba-cache-size",
    "32",
    "--max-total-tokens",
    "16384",
    "--mem-fraction-static",
    "0.3",
    "--grammar-backend",
    "xgrammar",
    "--speculative-algorithm",
    "NGRAM",
    "--speculative-num-draft-tokens",
    "4",
    "--speculative-ngram-max-bfs-breadth",
    "1",
]
# Total absolute logprob difference over the scored suffix. A correct
# checkpoint scores within ~1 nat of a cache miss (bf16 kernel noise); a
# wrong recurrent state scores tens of nats away.
MAX_SCORE_DELTA = 3.0


class TestMambaExtraBufferSpecGrammar(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=SERVER_ARGS,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        cls.phrase_ids = tokenizer.encode(PHRASE, add_special_tokens=False)
        cls.filler_ids = tokenizer.encode(FILLER, add_special_tokens=False)
        cls.follow_up_ids = tokenizer.encode(FOLLOW_UP, add_special_tokens=False)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)

    def _generate(self, body):
        body.setdefault("sampling_params", {}).setdefault("temperature", 0)
        response = requests.post(self.base_url + "/generate", json=body, timeout=120)
        response.raise_for_status()
        return response.json()

    def _flush_cache(self):
        requests.post(self.base_url + "/flush_cache", timeout=60).raise_for_status()

    def _constrained(self, input_ids):
        result = self._generate(
            {
                "input_ids": input_ids,
                "sampling_params": {"max_new_tokens": 32, "regex": REGEX},
            }
        )
        self.assertEqual(result["meta_info"]["finish_reason"]["type"], "stop")
        self.assertGreaterEqual(result["meta_info"]["spec_verify_ct"], 1)
        return result["output_ids"]

    def _score_suffix(self, input_ids):
        result = self._generate(
            {
                "input_ids": input_ids,
                "sampling_params": {"max_new_tokens": 8},
                "return_logprob": True,
                "logprob_start_len": TRACK_INTERVAL,
            }
        )
        logprobs = [
            logprob
            for logprob, _token_id, _text in result["meta_info"]["input_token_logprobs"]
            if logprob is not None
        ]
        return logprobs, result["meta_info"]["cached_tokens"]

    def test_cached_continuation_matches_cache_miss(self):
        # The token the constrained run stops on; repeating "<phrase><stop>"
        # in the prompt tail lets the drafter propose tokens past the
        # terminator, so verify can accept more than the grammar keeps.
        stop_id = self._constrained(self.filler_ids[:64] + self.phrase_ids * 4)[-1]
        pattern = (self.phrase_ids + [stop_id]) * 6

        # Sweep the prompt length so the constrained run's final step lands
        # on every offset around the checkpoint boundary.
        resumed_at_checkpoint = 0
        for prompt_len in range(TRACK_INTERVAL - 10, TRACK_INTERVAL + 2):
            self._flush_cache()
            # Unrelated traffic so recycled cache slots are not blank.
            for k in range(2):
                self._generate(
                    {
                        "text": f"Write a short story about planet number {2 * prompt_len + k}. ",
                        "sampling_params": {"max_new_tokens": 24},
                    }
                )
            prompt = self.filler_ids[: prompt_len - len(pattern)] + pattern
            follow_up = prompt + self._constrained(prompt) + self.follow_up_ids

            hit_logprobs, cached_tokens = self._score_suffix(follow_up)
            self._flush_cache()
            miss_logprobs, _ = self._score_suffix(follow_up)

            resumed_at_checkpoint += cached_tokens == TRACK_INTERVAL
            self.assertEqual(len(hit_logprobs), len(miss_logprobs))
            delta = sum(abs(a - b) for a, b in zip(hit_logprobs, miss_logprobs))
            self.assertLess(
                delta,
                MAX_SCORE_DELTA,
                f"prompt_len={prompt_len}: continuation scored {delta:.2f} nats "
                f"away from the cache miss (cached_tokens={cached_tokens})",
            )
        self.assertGreater(
            resumed_at_checkpoint,
            0,
            "no follow-up resumed from the decode checkpoint; the sweep no "
            "longer reaches the boundary",
        )


if __name__ == "__main__":
    unittest.main()
