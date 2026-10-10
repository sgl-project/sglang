# Copyright 2023-2025 SGLang Team
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

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.elastic_ep.expert_backup_client import extract_layer_and_expert_id
from sglang.srt.elastic_ep.expert_backup_manager import extract_expert_id


class TestExtractExpertId(unittest.TestCase):
    def test_expert_weight(self):
        self.assertEqual(
            extract_expert_id("model.layers.3.mlp.experts.17.gate_proj.weight"), 17
        )

    def test_non_expert_weight(self):
        self.assertEqual(
            extract_expert_id("model.layers.0.self_attn.q_proj.weight"), -1
        )

    def test_multidigit_expert_id(self):
        self.assertEqual(
            extract_expert_id("model.layers.5.mlp.experts.123.up_proj.weight"), 123
        )

    def test_expert_without_leading_dot_no_match(self):
        # The pattern requires a dot before "experts.".
        self.assertEqual(extract_expert_id("experts.5.weight"), -1)


class TestExtractLayerAndExpertId(unittest.TestCase):
    def test_full_name(self):
        layer, expert, weight = extract_layer_and_expert_id(
            "model.layers.3.mlp.experts.7.gate_proj.weight"
        )
        self.assertEqual((layer, expert, weight), (3, 7, "gate_proj"))

    def test_up_proj(self):
        layer, expert, weight = extract_layer_and_expert_id(
            "model.layers.11.mlp.experts.2.up_proj.weight"
        )
        self.assertEqual((layer, expert, weight), (11, 2, "up_proj"))

    def test_no_match(self):
        self.assertEqual(
            extract_layer_and_expert_id("model.layers.0.self_attn.q_proj.weight"),
            (-1, -1, ""),
        )


if __name__ == "__main__":
    unittest.main()
