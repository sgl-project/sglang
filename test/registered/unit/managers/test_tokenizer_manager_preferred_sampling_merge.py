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

"""Unit tests for merging --preferred-sampling-params with per-request ones."""

import unittest

from sglang.srt.managers.tokenizer_manager import _merge_sampling_kwargs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMergeSamplingKwargs(CustomTestCase):
    def test_no_preferred_params_returns_request_params_unchanged(self):
        request_params = {"temperature": 0.7, "custom_params": None}
        merged = _merge_sampling_kwargs(None, request_params)
        self.assertIs(merged, request_params)

    def test_explicit_request_value_overrides_preferred(self):
        merged = _merge_sampling_kwargs(
            {"temperature": 0.2}, {"temperature": 0.9, "top_p": None}
        )
        self.assertEqual(merged["temperature"], 0.9)
        self.assertNotIn("top_p", merged)

    def test_unset_request_field_does_not_clobber_preferred_default(self):
        """Regression guard: `ChatCompletionRequest.to_sampling_params()`
        always emits every field, so an unset one arrives as an explicit
        `None` rather than being absent. A naive `{**preferred, **request}`
        merge let that `None` silently overwrite a preferred server-side
        default -- exactly the bug this merge exists to avoid."""
        merged = _merge_sampling_kwargs(
            {"temperature": 0.2, "custom_params": {"thinking_budget": 512}},
            {"temperature": None, "custom_params": None},
        )
        self.assertEqual(merged["temperature"], 0.2)
        self.assertEqual(merged["custom_params"], {"thinking_budget": 512})

    def test_custom_params_merged_key_by_key_not_replaced_wholesale(self):
        """A request that only sets an unrelated custom param (e.g. a
        client-provided reasoning flag) must not lose the server's preferred
        `thinking_budget` default, and the request's own key must win on
        conflict."""
        merged = _merge_sampling_kwargs(
            {"custom_params": {"thinking_budget": 512, "foo": "server"}},
            {"custom_params": {"foo": "client", "bar": "request-only"}},
        )
        self.assertEqual(
            merged["custom_params"],
            {"thinking_budget": 512, "foo": "client", "bar": "request-only"},
        )

    def test_no_custom_params_anywhere_key_absent(self):
        merged = _merge_sampling_kwargs(
            {"temperature": 0.2}, {"temperature": None, "top_p": 0.9}
        )
        self.assertNotIn("custom_params", merged)
        self.assertEqual(merged["top_p"], 0.9)


if __name__ == "__main__":
    unittest.main()
