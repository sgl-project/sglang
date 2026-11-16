"""Watermark secrets stay redacted in request logs, and the redaction walk is
bounded by max_length instead of the prompt length. With watermarking off, only
requests that carry a watermark field are redacted."""

import json
import unittest
from unittest import mock

import msgspec

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sglang.srt.runtime_context import create_context, use_context
from sglang.srt.sampling.watermarking.core import redact_watermark_secrets
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import request_logger
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

SECRET = "super-secret-watermark-key"


class ExtraOptions(msgspec.Struct):
    watermark_key: str = SECRET
    note: str = "x"


def _request(num_ids: int) -> GenerateReqInput:
    return GenerateReqInput(
        input_ids=list(range(num_ids)),
        sampling_params={"max_new_tokens": 8, "watermark": {"key": SECRET}},
        rid="rid-0",
    )


def _log_all_entry_points(req, chat_req, log_requests_format: str) -> str:
    """Logs req as received and finished and chat_req as an OpenAI request."""
    logger = request_logger.RequestLogger(
        log_requests=True,
        log_requests_level=3,
        log_requests_format=log_requests_format,
        log_requests_target=None,
    )
    target = mock.Mock()
    logger.targets = [target]
    logger.log_received_request(req)
    logger.log_finished_request(req, {"meta_info": {"e2e_latency": 0.1}})
    logger.log_openai_received_request(chat_req)
    return "\n".join(call.args[0] for call in target.info.call_args_list)


class _LongestWalkSpy:
    """Records the longest list redact_watermark_secrets is asked to walk."""

    def __init__(self):
        self.real = request_logger.redact_watermark_secrets
        self.longest = 0

    def __call__(self, value, **kwargs):
        self._scan(value)
        return self.real(value, **kwargs)

    def _scan(self, value):
        if isinstance(value, (list, tuple)):
            self.longest = max(self.longest, len(value))
            for item in value:
                self._scan(item)
        elif isinstance(value, dict):
            for item in value.values():
                self._scan(item)


class TestRequestLogRedaction(CustomTestCase):
    def test_secrets_redacted_in_both_formatters(self):
        for num_ids in (16, 100_000):
            req = _request(num_ids)
            text_out = request_logger._dataclass_to_string_truncated(req)
            json_out = repr(request_logger._transform_data_for_logging(req))
            self.assertNotIn(SECRET, text_out)
            self.assertNotIn(SECRET, json_out)
            self.assertIn("<redacted>", text_out)

    def test_top_level_watermark_field_is_redacted(self):
        req = GenerateReqInput(
            input_ids=[1, 2, 3],
            watermark={"enabled": True, "key": SECRET},
            rid="rid-0",
        )
        text_out = request_logger._dataclass_to_string_truncated(req)
        json_out = repr(request_logger._transform_data_for_logging(req))
        self.assertNotIn(SECRET, text_out)
        self.assertNotIn(SECRET, json_out)

    def test_watermarking_off_checks_only_watermark_fields(self):
        # The received-request log runs before validation rejects a key sent to
        # a server without --enable-watermark, so the key must still be redacted.
        watermark_off = create_context(ServerArgs(model_path="dummy"), role="test")
        self.assertFalse(watermark_off.config_bag("exec").features.enable_watermark)
        messages = [{"role": "user", "content": "hi"}]
        key = {"key": SECRET}
        with_key = [
            GenerateReqInput(input_ids=[1, 2], watermark=key),
            GenerateReqInput(input_ids=[1, 2], sampling_params={"watermark": key}),
            GenerateReqInput(input_ids=[[1], [2]], watermark=[None, key]),
            GenerateReqInput(
                input_ids=[[1], [2]], sampling_params=[{}, {"watermark": key}]
            ),
            EmbeddingReqInput(input_ids=[1, 2], sampling_params={"watermark": key}),
        ]
        without_key = [
            GenerateReqInput(input_ids=[1, 2], sampling_params={"max_new_tokens": 1}),
            GenerateReqInput(input_ids=[[1], [2]]),
            EmbeddingReqInput(input_ids=[1, 2]),
        ]
        for req in with_key + without_key:
            req.normalize_batch_and_arguments()
        chat_with_key = ChatCompletionRequest(
            model="m", messages=messages, watermark=key
        )
        chat_without_key = ChatCompletionRequest(model="m", messages=messages)

        with use_context(watermark_off):
            for log_requests_format in ("text", "json"):
                for req in with_key:
                    out = _log_all_entry_points(req, chat_with_key, log_requests_format)
                    self.assertNotIn(SECRET, out, req)
                    self.assertEqual(out.count("<redacted>"), 3, req)

                # Requests without a watermark field are logged as is, with no
                # redaction copy or walk.
                with mock.patch.object(
                    request_logger, "redact_watermark_secrets"
                ) as redact:
                    for req in without_key:
                        _log_all_entry_points(
                            req, chat_without_key, log_requests_format
                        )
                redact.assert_not_called()

                # An OpenAI request with a watermark config redacts only that
                # config, not the dict holding its prompt.
                with mock.patch.object(
                    request_logger,
                    "redact_watermark_secrets",
                    wraps=request_logger.redact_watermark_secrets,
                ) as redact:
                    _log_all_entry_points(
                        without_key[0], chat_with_key, log_requests_format
                    )
                self.assertEqual([c.args[0] for c in redact.call_args_list], [key])

    def test_struct_under_ordinary_field_is_redacted(self):
        payload = {"rid": "r1", "extra": ExtraOptions()}
        text_out = request_logger._dataclass_to_string_truncated(payload)
        json_out = repr(request_logger._transform_data_for_logging(payload))
        self.assertNotIn(SECRET, text_out)
        self.assertNotIn(SECRET, json_out)
        self.assertIn("<redacted>", text_out)

    def test_truncated_list_halves_stay_redacted(self):
        payload = {"items": [{"watermark_key": SECRET}] * 5000}
        text_out = request_logger._dataclass_to_string_truncated(payload)
        json_out = repr(request_logger._transform_data_for_logging(payload))
        self.assertNotIn(SECRET, text_out)
        self.assertNotIn(SECRET, json_out)
        self.assertIn("<redacted>", text_out)

    def test_truncated_json_list_of_structs_is_serializable(self):
        payload = {"items": [ExtraOptions()] * 5000}
        transformed = request_logger._transform_data_for_logging(payload)
        dumped = json.dumps(transformed)
        self.assertNotIn(SECRET, dumped)

    def test_redaction_walk_is_bounded_by_max_length(self):
        for level in (0, 1, 2, 3):
            max_length, skip_names, _ = request_logger.RequestLogger(
                log_requests=True,
                log_requests_level=level,
                log_requests_format="text",
                log_requests_target=None,
            ).metadata
            if level == 3:
                # Level 3 prints every input id, so the walk is inherently unbounded.
                continue
            bound = min(max_length, 2048)
            for fn in (
                request_logger._dataclass_to_string_truncated,
                request_logger._transform_data_for_logging,
            ):
                spy = _LongestWalkSpy()
                with mock.patch.object(request_logger, "redact_watermark_secrets", spy):
                    fn(_request(100_000), max_length, skip_names)
                self.assertLessEqual(spy.longest, bound, f"level={level} {fn.__name__}")

    def test_scalar_sequences_skip_the_redaction_walk(self):
        ids = list(range(100_000))
        redacted = redact_watermark_secrets(ids)
        self.assertEqual(redacted, ids)
        # The dump path ships the result to a worker thread, so the mutable
        # list must come back as a copy; tuples are immutable and may alias.
        self.assertIsNot(redacted, ids)
        ids_tuple = tuple(ids)
        self.assertIs(redact_watermark_secrets(ids_tuple), ids_tuple)
        redacted = redact_watermark_secrets([{"watermark_key": SECRET}])
        self.assertEqual(redacted, [{"watermark_key": "<redacted>"}])


if __name__ == "__main__":
    unittest.main()
