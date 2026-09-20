import copy
import unittest
from unittest.mock import patch

from sglang.srt.utils.request_logger import ModelOutputTrace
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "base-a-test-cpu")


class TestModelOutputTrace(unittest.TestCase):
    def test_deltas_and_replace_reconstruct_output_without_request_fields(self):
        captured = []
        trace = ModelOutputTrace([], "model")
        with patch(
            "sglang.srt.utils.request_logger.log_json",
            side_effect=lambda targets, event, data: captured.append((event, data)),
        ):
            for index, text in enumerate(["hello", "hello</think>", "hello"]):
                content = {
                    "text": text,
                    "input_ids": [987654],
                    "request_body": "PRIVATE_INPUT",
                    "headers": {"Authorization": "PRIVATE_KEY"},
                    "meta_info": {
                        "id": "request-1",
                        "completion_tokens": index + 1,
                        "finish_reason": {"type": "stop"} if index == 2 else None,
                        "input_token_logprobs": ["PRIVATE_INPUT"],
                    },
                }
                before = copy.deepcopy(content)
                trace.record(content)
                self.assertEqual(content, before)
        self.assertEqual(
            [data["operation"] for _, data in captured], ["append", "append", "replace"]
        )
        self.assertEqual(
            [data["text"] for _, data in captured], ["hello", "</think>", "hello"]
        )
        self.assertEqual(captured[-1][1]["finish_type"], "stop")
        self.assertNotIn("PRIVATE", str(captured))
        self.assertNotIn("input_ids", str(captured))
        self.assertNotIn("headers", str(captured))
        self.assertTrue(
            all(event == "model.output.before_parsers" for event, _ in captured)
        )

    def test_choices_and_requests_have_independent_state(self):
        captured = []
        trace = ModelOutputTrace([], "model")
        with patch(
            "sglang.srt.utils.request_logger.log_json",
            side_effect=lambda targets, event, data: captured.append(data),
        ):
            trace.record({"text": "A", "meta_info": {"id": "r"}}, 0)
            trace.record({"text": "B", "meta_info": {"id": "r"}}, 1)
            trace.record({"text": "AC", "meta_info": {"id": "r"}}, 0)
            trace.record({"text": "D", "meta_info": {"id": "other"}}, 0)
        self.assertEqual([item["text"] for item in captured], ["A", "B", "C", "D"])
        self.assertEqual([item["sequence"] for item in captured], [1, 1, 2, 1])


if __name__ == "__main__":
    unittest.main()
