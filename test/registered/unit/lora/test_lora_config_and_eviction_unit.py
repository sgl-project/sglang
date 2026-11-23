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

import json
import tempfile
import unittest
from unittest.mock import patch

from sglang.srt.lora.eviction_policy import (
    FIFOEvictionPolicy,
    LRUEvictionPolicy,
    get_eviction_policy,
)
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestLoRAConfigAndEvictionPolicy(CustomTestCase):
    def test_remote_lora_config_calls_snapshot_download_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(f"{tmpdir}/adapter_config.json", "w") as f:
                json.dump(
                    {
                        "target_modules": ["q_proj", "v_proj"],
                        "r": 8,
                        "lora_alpha": 16,
                    },
                    f,
                )
            with open(f"{tmpdir}/added_tokens.json", "w") as f:
                json.dump({"<fake_base_tok>": 10}, f)

            with patch(
                "sglang.srt.lora.lora_config.snapshot_download", return_value=tmpdir
            ) as mock_download:
                cfg = LoRAConfig(path="org/remote-lora-adapter", base_vocab_size=32000)

            self.assertEqual(mock_download.call_count, 1)
            self.assertIsNone(cfg.added_tokens_config)
            self.assertEqual(cfg.lora_added_tokens_size, 0)

    def test_eviction_policy_prefers_untracked_adapter_over_none_and_is_case_insensitive(
        self,
    ):
        for policy_name, cls in (("LRU", LRUEvictionPolicy), ("FIFO", FIFOEvictionPolicy)):
            policy = get_eviction_policy(policy_name)
            self.assertIsInstance(policy, cls)
            victim = policy.select_victim({None, "untracked_b", "untracked_a"})
            self.assertEqual(victim, "untracked_a")
            self.assertEqual(policy.eviction_count, 1)


if __name__ == "__main__":
    unittest.main()
