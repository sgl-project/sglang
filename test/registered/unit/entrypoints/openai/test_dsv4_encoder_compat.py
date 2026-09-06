"""Preserve legacy effort keywords and normalized tool history encoding."""

import json
import unittest

from sglang.srt.entrypoints.openai import encoding_dsv4 as encoding
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")


class TestDsv4EncoderCompatibility(CustomTestCase):
    def test_legacy_profiles_match_flat_efforts(self):
        messages = [{"role": "user", "content": "Solve this."}]
        for profile, effort, flat in [
            ("preview", None, "low"),
            ("preview", "high", "low"),
            ("preview", "max", "high"),
            ("official", None, "low"),
            ("official", "low", "low"),
            ("official", "high", "high"),
            ("official", "max", "max"),
        ]:
            with self.subTest(profile=profile, effort=effort):
                actual = encoding.encode_messages(
                    messages,
                    thinking_mode="thinking",
                    reasoning_effort=effort,
                    reasoning_effort_profile=profile,
                )
                expected = encoding.encode_messages(
                    messages,
                    thinking_mode="thinking",
                    reasoning_effort=flat,
                )
                self.assertEqual(actual, expected)

    def test_invalid_legacy_profile_or_effort(self):
        for profile, effort in [("invalid", "high"), ("preview", "low")]:
            with self.subTest(profile=profile, effort=effort):
                with self.assertRaises(ValueError):
                    encoding.encode_messages(
                        [{"role": "user", "content": "Hi"}],
                        thinking_mode="thinking",
                        reasoning_effort_profile=profile,
                        reasoning_effort=effort,
                    )

    def test_replayed_tool_arguments_match_json_strings(self):
        arguments = {"city": "Paris", "count": 2, "options": {"units": "metric"}}

        def history(value):
            return [
                {"role": "user", "content": "Find the weather."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "weather", "arguments": value},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
                {"role": "user", "content": "And tomorrow?"},
            ]

        actual = encoding.encode_messages(history(arguments), thinking_mode="chat")
        expected = encoding.encode_messages(
            history(json.dumps(arguments)), thinking_mode="chat"
        )
        self.assertEqual(actual, expected)
        self.assertIn('parameter name="city"', actual)
        self.assertNotIn('parameter name="arguments"', actual)


if __name__ == "__main__":
    unittest.main()
