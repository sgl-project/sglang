import json
import unittest

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.utils.request_logger import (
    RequestLogger,
    _dataclass_to_string_truncated,
    _transform_data_for_logging,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_LOGGER = "sglang.srt.utils.request_logger.stdout"
_SECRETS = (
    "SECRET-QUESTION",
    "SECRET-PREFIX",
    "SECRET-TEXT",
    "SECRET-CUSTOM",
)


def _make_obj() -> GenerateReqInput:
    return GenerateReqInput(
        text="SECRET-TEXT",
        input_ids=[1, 2, 3],
        video_data="SECRET-VIDEO",
        video_config={
            "_question": "SECRET-QUESTION",
            "lora_path": "lora-x",
            "fps": 16,
            "max_frames": 8,
        },
        images_config={"prefix": "SECRET-PREFIX", "image_mode": "auto"},
        sampling_params={
            "max_new_tokens": 4,
            "custom_params": {"text": "SECRET-CUSTOM"},
        },
    )


class TestNestedSkipEntries(CustomTestCase):
    """Dotted skip entries hide nested fields at levels 0/1 while
    same-named fields elsewhere (and levels 2/3) stay intact."""

    def _received_log(self, level: int, fmt: str) -> str:
        log = RequestLogger(
            log_requests=True,
            log_requests_level=level,
            log_requests_format=fmt,
            log_requests_target=None,
        )
        with self.assertLogs(_LOGGER, level="INFO") as cm:
            log.log_received_request(_make_obj())
        self.assertEqual(len(cm.records), 1)
        return cm.records[0].getMessage()

    def _received_obj_json(self, level: int) -> dict:
        return json.loads(self._received_log(level, "json"))["obj"]

    def test_level0_json_hides_nested_fields(self):
        msg = self._received_log(0, "json")
        obj = json.loads(msg)["obj"]
        # Bare "lora_path" only skips the top-level field.
        self.assertEqual(
            obj["video_config"], {"lora_path": "lora-x", "fps": 16, "max_frames": 8}
        )
        self.assertEqual(obj["images_config"], {"image_mode": "auto"})
        self.assertNotIn("sampling_params", obj)
        for secret in _SECRETS:
            self.assertNotIn(secret, msg)

    def test_level0_text_hides_nested_fields(self):
        msg = self._received_log(0, "text")
        self.assertIn("'fps': 16", msg)
        self.assertIn("'image_mode': 'auto'", msg)
        for secret in _SECRETS:
            self.assertNotIn(secret, msg)

    def test_level1_json_hides_nested_fields(self):
        msg = self._received_log(1, "json")
        obj = json.loads(msg)["obj"]
        self.assertEqual(
            obj["video_config"], {"lora_path": "lora-x", "fps": 16, "max_frames": 8}
        )
        self.assertEqual(obj["images_config"], {"image_mode": "auto"})
        # Bare top-level names must not skip same-named nested keys.
        self.assertEqual(
            obj["sampling_params"]["custom_params"], {"text": "SECRET-CUSTOM"}
        )
        for secret in ("SECRET-QUESTION", "SECRET-PREFIX", "SECRET-TEXT"):
            self.assertNotIn(secret, msg)

    def test_level2_and_level3_keep_nested_fields(self):
        for level in (2, 3):
            with self.subTest(level=level):
                obj = self._received_obj_json(level)
                self.assertEqual(obj["video_config"]["_question"], "SECRET-QUESTION")
                self.assertEqual(obj["images_config"]["prefix"], "SECRET-PREFIX")
                self.assertEqual(obj["text"], "SECRET-TEXT")

    def test_dotted_skip_propagates_to_depth_three(self):
        data = {"a": {"b": {"c": "SECRET-DEEP", "d": 1}}}
        out = _transform_data_for_logging(data, 2048, {"a.b.c"})
        self.assertEqual(out, {"a": {"b": {"d": 1}}})

    def test_bare_skip_does_not_reach_nested(self):
        out = _transform_data_for_logging(
            {"text": "DROP", "outer": {"text": "KEEP"}}, 2048, {"text"}
        )
        self.assertEqual(out, {"outer": {"text": "KEEP"}})

    def test_two_arg_calls_unchanged(self):
        self.assertEqual(
            _transform_data_for_logging({"a": {"b": 1}}, 2048), {"a": {"b": 1}}
        )
        s = _dataclass_to_string_truncated(_make_obj(), 2048)
        self.assertIn("'_question': 'SECRET-QUESTION'", s)
        self.assertIn("'fps': 16", s)


if __name__ == "__main__":
    unittest.main()
