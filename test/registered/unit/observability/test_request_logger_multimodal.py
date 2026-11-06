"""Request logs summarize media inputs per modality at every log level, and
never carry the media payload at the levels that skip input contents."""

import json
import unittest
from unittest import mock

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.observability import request_logger
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

IMAGE_B64 = "data:image/png;base64," + "A" * 4000
IMAGE_URL = "https://example.com/cat.png"
AUDIO_B64 = "B" * 300


def _logger(level: int, fmt: str) -> tuple:
    logger = request_logger.RequestLogger(
        log_requests=True,
        log_requests_level=level,
        log_requests_format=fmt,
        log_requests_target=None,
    )
    target = mock.Mock()
    logger.targets = [target]
    return logger, target


def _lines(target) -> list:
    return [call.args[0] for call in target.info.call_args_list]


class TestRequestLoggerMultimodal(CustomTestCase):
    def test_text_log_summarizes_media_without_payload_at_level_0(self):
        req = GenerateReqInput(
            text="describe",
            image_data=[IMAGE_B64, IMAGE_URL],
            audio_data=AUDIO_B64,
            rid="rid-0",
        )
        req.normalize_batch_and_arguments()
        logger, target = _logger(level=0, fmt="text")
        logger.log_received_request(req)
        logger.log_finished_request(req, {"meta_info": {"e2e_latency": 0.1}})
        received, finished = _lines(target)
        expected = (
            "multimodal={'image': {'count': 2, 'payload_len': "
            f"{len(IMAGE_B64) + len(IMAGE_URL)}}}, "
            f"'audio': {{'count': 1, 'payload_len': {len(AUDIO_B64)}}}}}"
        )
        for line in (received, finished):
            self.assertIn(expected, line)
            self.assertNotIn("AAAA", line)
            self.assertNotIn("BBBB", line)

    def test_json_log_carries_summary_for_batch_and_omits_it_for_text(self):
        batch = GenerateReqInput(
            text=["a", "b"],
            image_data=[[IMAGE_URL, IMAGE_URL], None],
            rid=["rid-a", "rid-b"],
        )
        batch.normalize_batch_and_arguments()
        plain = GenerateReqInput(text="a", rid="rid-c")
        plain.normalize_batch_and_arguments()
        logger, target = _logger(level=1, fmt="json")
        logger.log_received_request(batch)
        logger.log_received_request(plain)
        batch_line, plain_line = (json.loads(line) for line in _lines(target))
        self.assertEqual(
            batch_line["multimodal"],
            {"image": {"count": 2, "payload_len": 2 * len(IMAGE_URL)}},
        )
        self.assertNotIn("multimodal", plain_line)

    def test_parallel_sampling_counts_media_once(self):
        req = GenerateReqInput(
            text="describe",
            image_data=IMAGE_URL,
            audio_data=AUDIO_B64,
            sampling_params={"n": 4},
            rid="rid-n",
        )
        req.normalize_batch_and_arguments()
        logger, target = _logger(level=0, fmt="json")
        logger.log_received_request(req)
        (line,) = (json.loads(line) for line in _lines(target))
        self.assertEqual(
            line["multimodal"],
            {
                "image": {"count": 1, "payload_len": len(IMAGE_URL)},
                "audio": {"count": 1, "payload_len": len(AUDIO_B64)},
            },
        )

    def test_openai_chat_log_counts_content_parts(self):
        chat = ChatCompletionRequest(
            model="m",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {"type": "image_url", "image_url": {"url": IMAGE_B64}},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": AUDIO_B64, "format": "wav"},
                        },
                    ],
                }
            ],
        )
        # Inline audio validates into an `audio_url` data URI.
        audio_url = chat.messages[0].content[2].audio_url.url
        self.assertIn(AUDIO_B64, audio_url)
        logger, target = _logger(level=2, fmt="json")
        logger.log_openai_received_request(chat)
        (line,) = (json.loads(line) for line in _lines(target))
        self.assertEqual(
            line["multimodal"],
            {
                "image": {"count": 1, "payload_len": len(IMAGE_B64)},
                "audio": {"count": 1, "payload_len": len(audio_url)},
            },
        )


if __name__ == "__main__":
    unittest.main()
