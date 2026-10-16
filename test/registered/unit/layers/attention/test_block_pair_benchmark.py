"""Exercise the decoding controller on CPU before the A100 allocation."""

import importlib.util
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

spec = importlib.util.spec_from_file_location(
    "pair_bench",
    Path(__file__).resolve().parents[5]
    / "benchmark/dllm/bench_refresh_denoise_pair.py",
)
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


class Event:
    def __init__(self, **kwargs):
        self.t = 0.0

    def record(self):
        self.t = time.perf_counter()

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return (other.t - self.t) * 1000


class FakeGraph:
    def __init__(self, confidence_logit):
        self.confidence_logit = confidence_logit

    def __call__(self, ids, prefix):
        logits = torch.zeros(ids.numel(), 64)
        predictions = ((torch.arange(ids.shape[1]) + prefix) % 63 + 1).repeat(
            ids.shape[0]
        )
        logits.scatter_(1, predictions[:, None], self.confidence_logit)
        return logits


class TestBenchmark(CustomTestCase):
    def test_native_selection_and_progress_counts(self):
        with (
            patch.object(torch.cuda, "Event", Event),
            patch.object(torch.cuda, "synchronize", lambda: None),
        ):
            for n in (1, 4):
                for block in (4, 16, 32):
                    for threshold, logit, passes in [
                        (0.0, 1.0, 1),
                        (0.95, 1.0, block),
                        (0.95, 15.0, 1),
                    ]:
                        with self.subTest(
                            n=n, block=block, threshold=threshold, logit=logit
                        ):
                            g = FakeGraph(logit)
                            prompt = torch.ones((n, 2 * block), dtype=torch.long)
                            output_tokens = 4 * block
                            separate = B.execute(
                                g,
                                g,
                                prompt,
                                block,
                                threshold,
                                "separate",
                                output_tokens=output_tokens,
                            )
                            paired = B.execute(
                                g,
                                g,
                                prompt,
                                block,
                                threshold,
                                "paired",
                                output_tokens=output_tokens,
                            )
                            self.assertEqual(
                                separate["output_ids"], paired["output_ids"]
                            )
                            self.assertEqual(separate["forward_counts"]["refresh"], 4)
                            self.assertEqual(paired["forward_counts"]["refresh"], 1)
                            self.assertEqual(
                                paired["forward_counts"]["refresh_and_denoise"], 3
                            )
                            self.assertEqual(
                                separate["forward_counts"]["denoising"], 4 * passes
                            )
                            self.assertEqual(
                                paired["forward_counts"]["denoising"], 4 * passes - 3
                            )
                            self.assertTrue(paired["terminal_refresh_retained"])

    def test_pair_row_order_used_by_selection(self):
        ids = torch.full((3, 8), B.MASK, dtype=torch.long)
        for n in range(3):
            ids[n, :4] = n + 1
        logits = FakeGraph(15.0)(ids, 20)
        self.assertTrue(B.select(ids, logits, 4, 0.95))
        self.assertEqual(ids[:, :4].tolist(), [[1] * 4, [2] * 4, [3] * 4])
        self.assertEqual(ids[:, 4:].tolist(), [[25, 26, 27, 28]] * 3)

    def test_no_cuda_initialization(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
