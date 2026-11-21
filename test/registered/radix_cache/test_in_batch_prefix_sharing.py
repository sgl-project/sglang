"""--enable-in-batch-prefix-sharing end to end: the samples of one prompt sent
together compute the prompt once and decode as if they read it from the radix
cache."""

import random
import unittest

import sglang as sgl
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase

register_cuda_ci(est_time=90, stage="extra-a", runner_config="1-gpu-small")

MODEL = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
PAGE_SIZE = 16


class TestInBatchPrefixSharing(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = sgl.Engine(
            model_path=MODEL,
            enable_in_batch_prefix_sharing=True,
            page_size=PAGE_SIZE,
            skip_tokenizer_init=True,
            mem_fraction_static=0.6,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine"):
            cls.engine.shutdown()

    def test_samples_of_one_prompt_compute_it_once(self):
        rng = random.Random(0)
        prompt = [rng.randrange(100, 20000) for _ in range(600)]
        sampling_params = {"temperature": 0, "max_new_tokens": 16}
        self.engine.flush_cache()
        samples = self.engine.generate(
            input_ids=[prompt] * 8, sampling_params=sampling_params
        )
        # The first sample computes the prompt; each other one takes its full
        # pages but the last token, which it computes for the logits.
        shared = (len(prompt) - 1) // PAGE_SIZE * PAGE_SIZE
        self.assertEqual(
            [s["meta_info"]["cached_tokens"] for s in samples], [0] + [shared] * 7
        )
        # A request reading the prompt from the radix cache decodes the same.
        (cached,) = self.engine.generate(
            input_ids=[prompt], sampling_params=sampling_params
        )
        self.assertEqual(cached["meta_info"]["cached_tokens"], shared)
        for sample in samples:
            self.assertEqual(sample["output_ids"], cached["output_ids"])


if __name__ == "__main__":
    unittest.main()
