# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for the standalone reasoning-request normalization helpers."""

import unittest

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.reasoning_request import (
    normalize_reasoning_inputs,
    normalize_reasoning_request,
    pop_reasoning_effort_kwarg,
    resolve_v4_reasoning_effort,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


class TestNormalizeReasoningInputs(unittest.TestCase):
    def test_reasoning_enabled_sets_both_toggle_spellings(self):
        values = normalize_reasoning_inputs({"reasoning": {"enabled": True}})
        self.assertIs(values["chat_template_kwargs"]["thinking"], True)
        self.assertIs(values["chat_template_kwargs"]["enable_thinking"], True)

    def test_reasoning_enabled_accepts_string_spellings(self):
        for spelling in ("1", "true", "YES", " on "):
            values = normalize_reasoning_inputs({"reasoning": {"enabled": spelling}})
            self.assertIs(values["chat_template_kwargs"]["thinking"], True)

    def test_explicit_client_toggle_wins(self):
        values = normalize_reasoning_inputs(
            {
                "reasoning": {"enabled": True},
                "chat_template_kwargs": {"thinking": False},
            }
        )
        self.assertIs(values["chat_template_kwargs"]["thinking"], False)
        self.assertIs(values["chat_template_kwargs"]["enable_thinking"], True)

    def test_reasoning_dict_effort_promoted_except_max(self):
        values = normalize_reasoning_inputs({"reasoning": {"effort": "high"}})
        self.assertEqual(values["reasoning_effort"], "high")

        values = normalize_reasoning_inputs({"reasoning": {"effort": "max"}})
        self.assertNotIn("reasoning_effort", values)

    def test_effort_none_disables_both_toggles(self):
        values = normalize_reasoning_inputs({"reasoning_effort": "none"})
        self.assertIs(values["chat_template_kwargs"]["thinking"], False)
        self.assertIs(values["chat_template_kwargs"]["enable_thinking"], False)

    def test_matches_pydantic_validator_path(self):
        payload = {"reasoning": {"enabled": True, "effort": "low"}}
        request = ChatCompletionRequest(
            model="m", messages=[{"role": "user", "content": "hi"}], **payload
        )
        helper_values = normalize_reasoning_inputs(dict(payload))
        self.assertEqual(request.chat_template_kwargs, helper_values["chat_template_kwargs"])
        self.assertEqual(request.reasoning_effort, helper_values["reasoning_effort"])


class TestPopReasoningEffortKwarg(unittest.TestCase):
    def test_none_and_empty_kwargs(self):
        self.assertIsNone(pop_reasoning_effort_kwarg(None))
        self.assertIsNone(pop_reasoning_effort_kwarg({}))

    def test_pops_the_key(self):
        kwargs = {"reasoning_effort": "max", "thinking": True}
        self.assertEqual(pop_reasoning_effort_kwarg(kwargs), "max")
        self.assertNotIn("reasoning_effort", kwargs)
        self.assertIs(kwargs["thinking"], True)


class TestNormalizeReasoningRequest(unittest.TestCase):
    def test_kwargs_spelling_promoted_without_deriving_toggles(self):
        # Mirrors the server order: the validator normalization runs before the
        # serving-path promotion, so a kwargs-spelled effort must not derive a
        # thinking toggle.
        values = normalize_reasoning_request(
            {"chat_template_kwargs": {"reasoning_effort": "max"}}
        )
        self.assertEqual(values["reasoning_effort"], "max")
        self.assertNotIn("reasoning_effort", values["chat_template_kwargs"])
        self.assertNotIn("thinking", values["chat_template_kwargs"])

    def test_kwargs_spelling_overrides_top_level_field(self):
        values = normalize_reasoning_request(
            {
                "reasoning_effort": "low",
                "chat_template_kwargs": {"reasoning_effort": "high"},
            }
        )
        self.assertEqual(values["reasoning_effort"], "high")


class TestResolveV4ReasoningEffort(unittest.TestCase):
    def test_only_max_and_high_render(self):
        self.assertEqual(resolve_v4_reasoning_effort("max"), "max")
        self.assertEqual(resolve_v4_reasoning_effort("high"), "high")
        for effort in (None, "none", "low", "medium"):
            self.assertIsNone(resolve_v4_reasoning_effort(effort))


if __name__ == "__main__":
    unittest.main(verbosity=2)
