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
"""Tests for sglang.srt.entrypoints.openai.utils"""

import unittest

from sglang.srt.entrypoints.openai.utils import to_openai_style_logprobs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestToOpenAIStyleLogprobs(unittest.TestCase):
    def test_top_logprobs_decode_collision_keeps_highest_ranked(self):
        """Two distinct top-k token ids that decode to the same text (routine
        with byte-fallback tokenizers emitting multiple U+FFFD fragments)
        must not let a lower-ranked candidate silently overwrite the
        higher-ranked one's logprob for that text.
        """
        # Ranked best-logprob-first, as torch.topk returns them: token ids
        # 111 and 222 both decode to "�".
        output_top_logprobs = [
            [
                (-0.5, 111, "�"),
                (-2.5, 222, "�"),
                (-3.0, 333, "ok"),
            ]
        ]

        result = to_openai_style_logprobs(output_top_logprobs=output_top_logprobs)

        self.assertEqual(len(result.top_logprobs), 1)
        entry = result.top_logprobs[0]
        # The colliding text keeps the higher-ranked candidate's logprob.
        self.assertEqual(entry["�"], -0.5)
        self.assertEqual(entry["ok"], -3.0)

    def test_top_logprobs_no_collision_unaffected(self):
        output_top_logprobs = [
            [
                (-0.1, 1, "a"),
                (-0.2, 2, "b"),
            ],
            None,
        ]

        result = to_openai_style_logprobs(output_top_logprobs=output_top_logprobs)

        self.assertEqual(result.top_logprobs, [{"a": -0.1, "b": -0.2}, None])


if __name__ == "__main__":
    unittest.main(verbosity=2)
