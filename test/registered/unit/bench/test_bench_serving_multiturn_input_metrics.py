"""Unit tests for multi-turn input token totals in calculate_metrics."""

import unittest

from sglang.benchmark.datasets.common import DatasetRow
from sglang.benchmark.serving import (
    RequestFuncInput,
    RequestFuncOutput,
    calculate_metrics,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        if not isinstance(text, str):
            raise ValueError("text input must be of type `str`")
        return text.split()


def _successful_output(prompt_len: int) -> RequestFuncOutput:
    return RequestFuncOutput(
        generated_text="ok",
        success=True,
        latency=0.2,
        ttft=0.05,
        prompt_len=prompt_len,
        output_len=1,
        start_time=0.0,
    )


class TestCalculateMetricsMultiTurnInput(CustomTestCase):
    def _metrics(self, outputs, input_requests=None, dur_s=1.0):
        metrics, _ = calculate_metrics(
            input_requests=input_requests,
            outputs=outputs,
            dur_s=dur_s,
            tokenizer=_FakeTokenizer(),
            backend="sglang-oai-chat",
        )
        return metrics

    def test_multiturn_server_usage_sums_round_prompt_lens(self):
        metrics = self._metrics([_successful_output(100), _successful_output(240)])
        self.assertEqual(metrics.total_input, 340)
        self.assertEqual(metrics.total_input_text, 340)
        self.assertEqual(metrics.total_input_vision, 0)
        self.assertGreater(metrics.input_throughput, 0)

    def test_failed_requests_are_not_counted(self):
        failed = RequestFuncOutput(success=False, prompt_len=999, generated_text="")
        metrics = self._metrics(
            [_successful_output(100), _successful_output(240), failed]
        )
        self.assertEqual(metrics.total_input, 340)
        self.assertEqual(metrics.total_input_text, 340)
        self.assertEqual(metrics.total_input_vision, 0)

    def test_single_turn_totals_come_from_input_requests(self):
        input_requests = [
            DatasetRow(
                prompt="a cat",
                prompt_len=30,
                output_len=8,
                text_prompt_len=20,
                vision_prompt_len=10,
            ),
            DatasetRow(
                prompt="a dog",
                prompt_len=70,
                output_len=8,
                text_prompt_len=40,
                vision_prompt_len=30,
            ),
        ]
        # Server-reported lengths must not replace the dataset split.
        outputs = [_successful_output(1000), _successful_output(2000)]
        metrics = self._metrics(outputs, input_requests=input_requests)
        self.assertEqual(metrics.total_input, 100)
        self.assertEqual(metrics.total_input_text, 60)
        self.assertEqual(metrics.total_input_vision, 40)

    def test_missing_usage_falls_back_to_dataset_row_prompt_len(self):
        """Fallback when the server returns no usage.

        init_new copies the dataset row prompt_len and nothing overwrites it.
        Two successful turns of a 100-token row therefore total 200, which is
        an estimate of the dataset row rather than measured server usage.
        """
        request = RequestFuncInput(
            prompt=["turn-1", "turn-2"],
            api_url="http://127.0.0.1/v1/chat/completions",
            prompt_len=100,
            output_len=8,
            model="dummy",
            lora_name="",
            image_data=None,
            extra_request_body={},
        )
        outputs = []
        for _ in range(2):
            output = RequestFuncOutput.init_new(request)
            output.success = True
            output.generated_text = "ok"
            output.output_len = 1
            output.latency = 0.2
            output.ttft = 0.05
            outputs.append(output)
            self.assertEqual(output.prompt_len, 100)

        metrics = self._metrics(outputs)
        self.assertEqual(metrics.total_input, 200)
        self.assertEqual(metrics.total_input_text, 200)


if __name__ == "__main__":
    unittest.main()
