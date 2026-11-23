import unittest

from sglang.benchmark.serving import RequestFuncOutput, calculate_metrics
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class TestRetokenizedITL(unittest.TestCase):
    def test_zero_token_chunks_do_not_drop_elapsed_time(self):
        output = RequestFuncOutput(
            success=True,
            generated_text="first next last ",
            output_len=3,
            latency=0.2,
            ttft=0.02,
            itl=[0.04, 0.06, 0.01],
            text_chunks=[" ", "next last", " "],
        )

        for backend in ("sglang-oai", "sglang-oai-chat"):
            with self.subTest(backend=backend):
                metrics, _ = calculate_metrics(
                    input_requests=None,
                    outputs=[output],
                    dur_s=1.0,
                    tokenizer=WhitespaceTokenizer(),
                    backend=backend,
                    accept_length=2.0,
                )
                self.assertAlmostEqual(metrics.mean_itl_ms, 50.0)
                self.assertAlmostEqual(metrics.max_itl_ms, 50.0)
                self.assertEqual(metrics.completed, 1)


if __name__ == "__main__":
    unittest.main()
