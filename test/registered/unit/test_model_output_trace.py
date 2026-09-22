import copy
import os
import unittest
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.utils.request_logger import ModelOutputTrace, RequestLogger
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "base-a-test-cpu")


class TestModelOutputTrace(unittest.TestCase):
    def setUp(self):
        enabled = envs.SGLANG_ENABLE_MODEL_OUTPUT_LOGGING.override(True)
        enabled.__enter__()
        self.addCleanup(enabled.__exit__, None, None, None)

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

    def test_record_does_not_emit_or_retain_output_when_disabled(self):
        trace = ModelOutputTrace([], "model")
        with envs.SGLANG_ENABLE_MODEL_OUTPUT_LOGGING.override(False), patch(
            "sglang.srt.utils.request_logger.log_json"
        ) as output:
            trace.record({"text": "PRIVATE_OUTPUT", "meta_info": {"id": "r"}})
        output.assert_not_called()
        self.assertEqual(trace.previous, {})
        self.assertEqual(trace.sequences, {})


class TestModelOutputTraceConfiguration(unittest.TestCase):
    def _logger(self):
        return RequestLogger(
            log_requests=False,
            log_requests_level=0,
            log_requests_format="json",
            log_requests_target=None,
        )

    def test_directory_alone_does_not_enable_trace(self):
        with patch.dict(os.environ):
            os.environ.pop("SGLANG_ENABLE_MODEL_OUTPUT_LOGGING", None)
            os.environ["SGLANG_MODEL_OUTPUT_TRACE_DIR"] = "configured-trace-directory"
            with patch(
                "sglang.srt.utils.request_logger.create_log_targets", return_value=[]
            ) as targets:
                logger = self._logger()
                self.assertIsNone(logger.start_model_output_trace("model"))
                self.assertEqual(logger.output_trace_targets, [])
                targets.assert_called_once_with(
                    targets=None, name_prefix="sglang.srt.utils.request_logger"
                )

    def test_explicit_false_values_do_not_create_trace_targets(self):
        for value in ["0", "false", "no", "n"]:
            with self.subTest(value=value), patch.dict(
                os.environ,
                {
                    "SGLANG_ENABLE_MODEL_OUTPUT_LOGGING": value,
                    "SGLANG_MODEL_OUTPUT_TRACE_DIR": "configured-trace-directory",
                },
            ), patch(
                "sglang.srt.utils.request_logger.create_log_targets", return_value=[]
            ) as targets:
                self.assertIsNone(self._logger().start_model_output_trace("model"))
                self.assertEqual(targets.call_count, 1)

    def test_enable_flag_without_destination_does_not_enable_trace(self):
        with patch.dict(
            os.environ,
            {
                "SGLANG_ENABLE_MODEL_OUTPUT_LOGGING": "1",
                "SGLANG_MODEL_OUTPUT_TRACE_DIR": "",
            },
        ), patch(
            "sglang.srt.utils.request_logger.create_log_targets", return_value=[]
        ) as targets:
            self.assertIsNone(self._logger().start_model_output_trace("model"))
            self.assertEqual(targets.call_count, 1)

    def test_explicit_flag_and_destination_enable_trace(self):
        with patch.dict(
            os.environ,
            {
                "SGLANG_ENABLE_MODEL_OUTPUT_LOGGING": "1",
                "SGLANG_MODEL_OUTPUT_TRACE_DIR": "configured-trace-directory",
            },
        ), patch(
            "sglang.srt.utils.request_logger.create_log_targets",
            return_value=[unittest.mock.Mock()],
        ) as targets, patch(
            "sglang.srt.utils.request_logger.log_json"
        ) as output:
            logger = self._logger()
            trace = logger.start_model_output_trace("model")
            self.assertIsInstance(trace, ModelOutputTrace)
            self.assertEqual(targets.call_count, 2)
            self.assertEqual(
                targets.call_args.kwargs["targets"], ["configured-trace-directory"]
            )
            value = {"text": "original output", "meta_info": {"id": "r"}}
            before = copy.deepcopy(value)
            trace.record(value)
            self.assertEqual(value, before)
            self.assertEqual(output.call_args.args[2]["text"], "original output")
            with envs.SGLANG_ENABLE_MODEL_OUTPUT_LOGGING.override(False):
                self.assertIsNone(logger.start_model_output_trace("another-model"))

    def test_invalid_enable_value_warns_and_stays_disabled(self):
        with patch.dict(
            os.environ,
            {
                "SGLANG_ENABLE_MODEL_OUTPUT_LOGGING": "invalid",
                "SGLANG_MODEL_OUTPUT_TRACE_DIR": "configured-trace-directory",
            },
        ), patch(
            "sglang.srt.utils.request_logger.create_log_targets", return_value=[]
        ) as targets, self.assertWarnsRegex(
            UserWarning, "Invalid value"
        ):
            logger = self._logger()
            self.assertIsNone(logger.start_model_output_trace("model"))
            self.assertEqual(targets.call_count, 1)


if __name__ == "__main__":
    unittest.main()
