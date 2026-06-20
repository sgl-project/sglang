"""Each parallel sample must get a distinct seed (deterministic inference).

multinomial_with_seed hashes sampling_seed, so an n>1 request with a fixed seed
would otherwise produce N byte-identical completions.
"""

import unittest

from sglang.srt.managers.tokenizer_manager import _seed_for_parallel_sample
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(3, "base-a-test-cpu")


class TestSeedForParallelSample(unittest.TestCase):
    def test_seeded_samples_get_offset_seeds(self):
        base = SamplingParams(sampling_seed=42)
        seeds = [_seed_for_parallel_sample(base, j).sampling_seed for j in range(4)]
        self.assertEqual(seeds, [42, 43, 44, 45])
        # The base params must not be mutated.
        self.assertEqual(base.sampling_seed, 42)

    def test_unseeded_sample_is_returned_unchanged(self):
        base = SamplingParams(sampling_seed=None)
        self.assertIs(_seed_for_parallel_sample(base, 3), base)


if __name__ == "__main__":
    unittest.main()
