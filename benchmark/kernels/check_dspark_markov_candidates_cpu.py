#!/usr/bin/env python3
"""Run DSpark candidate CPU checks with only Torch installed.

Usage from any directory:
    python /path/to/sglang/benchmark/kernels/check_dspark_markov_candidates_cpu.py

Loads fixed repository modules directly, following the standalone kernel
benchmark convention. It does not import SGLang's server dependencies and does
not need Triton, a checkpoint, or a GPU. These checks are not GPU validation.
"""

import importlib.util
import itertools
import math
import sys
import unittest
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
_KERNELS = "python/sglang/kernels/ops/speculative/dspark/"
_MODULE_FILES = (
    (
        "sglang.kernels.ops.speculative.dspark.dspark_markov_topk_reference",
        _KERNELS + "dspark_markov_topk_reference.py",
    ),
    (
        "sglang.kernels.ops.speculative.dspark.dspark_markov_topk",
        _KERNELS + "dspark_markov_topk.py",
    ),
    ("sglang.test.ci.ci_register", "python/sglang/test/ci/ci_register.py"),
    (
        "dspark_candidate_tests",
        "test/registered/spec/dspark/test_dspark_markov_candidates.py",
    ),
)


def _load_modules():
    for name, relative_path in _MODULE_FILES:
        spec = importlib.util.spec_from_file_location(name, _ROOT / relative_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return tuple(sys.modules[name] for name, _ in _MODULE_FILES)


def main():
    reference, kernel, _, tests = _load_modules()

    class ExtraCpuChecks(unittest.TestCase):
        def test_staged_sampler_against_independent_reference(self):
            base, w1, w2, mapping = tests._fixture()
            inputs = tests._inputs(base, mapping)
            for k, m, alpha in itertools.product(
                (1, 3, 5), (0, 2, 5), (-0.5, 0.0, 1.0)
            ):
                with self.subTest(k=k, m=m, alpha=alpha):
                    sampler = kernel.MarkovCandidateSampler(
                        w1,
                        w2,
                        alpha=alpha,
                        topk=k,
                        bias_topk=m,
                        target_vocab_size=7,
                        gamma=3,
                        capacity=2,
                        d2t_offset=inputs["d2t_offset"],
                    )
                    actual = sampler.sample(
                        base,
                        inputs["anchor"],
                        inputs["temperatures"],
                        inputs["greedy_mask"],
                        seeds=inputs["seeds"],
                    )
                    expected = reference.markov_candidates_reference(
                        base,
                        inputs["anchor"],
                        w1,
                        w2,
                        alpha=alpha,
                        topk=k,
                        bias_topk=m,
                        target_vocab_size=7,
                        **{
                            key: value
                            for key, value in inputs.items()
                            if key != "anchor"
                        },
                    )
                    torch.testing.assert_close(actual.tokens, expected.tokens)
                    torch.testing.assert_close(
                        actual.corrected_logits, expected.corrected_logits
                    )

        def test_refresh_preserves_addresses_and_updates_static_values(self):
            base, w1, w2, mapping = tests._fixture()
            inputs = tests._inputs(base, mapping)
            sampler = kernel.MarkovCandidateSampler(
                w1,
                w2,
                alpha=1,
                topk=2,
                bias_topk=2,
                target_vocab_size=7,
                gamma=3,
                capacity=2,
                d2t_offset=inputs["d2t_offset"],
            )
            addresses = sampler._buffer_addresses()
            old_bias = sampler.static_bias.clone()
            new_w1, new_w2 = w1.clone() * 2, w2.flip(0).contiguous()
            sampler.refresh_weights(
                new_w1, new_w2, alpha=1, d2t_offset=inputs["d2t_offset"]
            )
            self.assertEqual(addresses, sampler._buffer_addresses())
            self.assertFalse(torch.equal(old_bias, sampler.static_bias))
            actual = sampler.sample(
                base,
                inputs["anchor"],
                inputs["temperatures"],
                inputs["greedy_mask"],
                seeds=inputs["seeds"],
            )
            expected = reference.markov_candidates_reference(
                base,
                inputs["anchor"],
                new_w1,
                new_w2,
                alpha=1,
                topk=2,
                bias_topk=2,
                target_vocab_size=7,
                **{key: value for key, value in inputs.items() if key != "anchor"},
            )
            torch.testing.assert_close(actual.tokens, expected.tokens)
            torch.testing.assert_close(
                actual.corrected_logits, expected.corrected_logits
            )
            with self.assertRaisesRegex(ValueError, "alpha changed"):
                sampler.refresh_weights(new_w1, new_w2, alpha=-1)
            with self.assertRaisesRegex(ValueError, "mapping changed"):
                sampler.refresh_weights(new_w1, new_w2, d2t_offset=None)

        def test_unsigned_noise_math_and_fixed_sample_statistics(self):
            def scalar_uniform(seed, token):
                x = (int(token) ^ int(seed) ^ 0xA511E9B3) & 0xFFFFFFFF
                x = ((x ^ (x >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
                x = ((x ^ (x >> 15)) * 0x846CA68B) & 0xFFFFFFFF
                x ^= x >> 16
                return ((x >> 9) + 0.5) / 8388608

            seeds = torch.tensor([0, 1, -1, 2**31 - 1, 2**32 - 1, 2**62 - 1])
            ids = torch.tensor([0, 1, 2, 151935, 2**31 - 1, 2**32 - 1])
            actual = reference.candidate_uniform(seeds[:, None], ids[None, :])
            expected = torch.tensor(
                [[scalar_uniform(seed, token) for token in ids] for seed in seeds]
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertTrue(((actual > 0) & (actual < 1)).all())
            # This checks CPU noise arithmetic, not the actual GPU sampler.
            generator = torch.Generator().manual_seed(1909)
            count = 131072
            seeds = torch.randint(0, 2**31, (count, 1), generator=generator)
            score = torch.tensor([0.6, -0.9, 0.0, 1.0, 0.3, -0.4, 0.8, -1.1]) / 0.75
            uniform = reference.candidate_uniform(seeds, torch.arange(8)[None, :])
            chosen = (score - torch.log(-torch.log(uniform))).argmax(-1)
            observed = torch.bincount(chosen, minlength=8) / count
            expected = score.softmax(-1)
            log_term = math.log(2 * 256 / 1e-6)
            limit = (2 * expected * (1 - expected) * log_term / count).sqrt()
            limit += 2 * log_term / (3 * count)
            self.assertTrue(((observed - expected).abs() <= limit).all())

    suite = unittest.TestSuite(
        (
            unittest.defaultTestLoader.loadTestsFromTestCase(
                tests.TestMarkovCandidateReference
            ),
            unittest.defaultTestLoader.loadTestsFromTestCase(ExtraCpuChecks),
        )
    )
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    print(
        "CPU checks only. GPU correctness, CUDA graphs, and performance are unverified."
    )
    return 0 if outcome.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
