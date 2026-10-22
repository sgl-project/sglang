"""Qwen3 BF16 generation on MLU: eager/graph parity and TP/DP execution."""

import os
import unittest

import torch_mlu  # noqa: F401

from sglang import Engine
from sglang.test.ci.ci_register import register_mlu_ci
from sglang.test.test_utils import CustomTestCase

register_mlu_ci(est_time=500, suite="nightly-test-2-mlu", nightly=True)


class TestQwen3MLU(CustomTestCase):
    model = os.environ.get("MLU_TEST_MODEL", "Qwen/Qwen3-8B")

    def _generate(self, *, tp_size, dp_size, graphs):
        config = (
            {
                "decode": {"backend": "full", "bs": [1, 2, 4]},
                "prefill": {"backend": "breakable", "bs": [16, 64, 256]},
            }
            if graphs
            else {
                "decode": {"backend": "disabled"},
                "prefill": {"backend": "disabled"},
            }
        )
        engine = Engine(
            model_path=self.model,
            device="mlu",
            dtype="bfloat16",
            tp_size=tp_size,
            dp_size=dp_size,
            context_length=1024,
            max_total_tokens=2048,
            max_running_requests=4,
            chunked_prefill_size=256,
            max_prefill_tokens=256,
            mem_fraction_static=float(os.environ.get("MLU_TEST_MEM_FRACTION", "0.8")),
            cuda_graph_config=config,
            skip_server_warmup=True,
            disable_overlap_schedule=True,
        )
        try:
            prefix = "List of world capitals. France: Paris. China: Beijing. Italy: Rome. Japan: Tokyo.\n"
            prompts = [prefix + "France:", prefix + "China:"]
            params = {"temperature": 0, "max_new_tokens": 16}
            outputs = engine.generate(prompts, sampling_params=params)
            texts = [output["text"] for output in outputs]
            self.assertIn("Paris", texts[0])
            self.assertIn("Beijing", texts[1])
            # Repeat a cached prefix with a different batch size.
            repeated = engine.generate([prompts[0]], sampling_params=params)
            self.assertIn("Paris", repeated[0]["text"])
            self.assertGreater(repeated[0]["meta_info"]["cached_tokens"], 0)
            # More than one prefill chunk, also spanning graph token buckets.
            long = engine.generate(
                ["Hello. " * 150 + prompts[0]], sampling_params=params
            )
            self.assertTrue(long[0]["text"])
            return texts
        finally:
            engine.shutdown()

    def test_eager_graph_tp_and_dp(self):
        reference = self._generate(tp_size=1, dp_size=1, graphs=False)
        for tp_size, dp_size in ((1, 1), (2, 1), (1, 2)):
            with self.subTest(tp_size=tp_size, dp_size=dp_size):
                actual = self._generate(tp_size=tp_size, dp_size=dp_size, graphs=True)
                if tp_size == 1 and dp_size == 1:
                    self.assertEqual(actual, reference)


if __name__ == "__main__":
    unittest.main()
