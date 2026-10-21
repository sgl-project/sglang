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

"""Focused contract tests for LoRA adapter eviction policies."""

import unittest

from sglang.srt.lora.eviction_policy import get_eviction_policy
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestLoRAEvictionPolicy(CustomTestCase):
    """Protect the behavioral differences that callers rely on."""

    @staticmethod
    def _make_policy(policy_name, access_sequence):
        policy = get_eviction_policy(policy_name)
        for adapter_id in access_sequence:
            policy.mark_used(adapter_id)
        return policy

    def test_reuse_changes_lru_but_not_fifo(self):
        """A repeated access refreshes LRU recency, not FIFO insertion order."""
        access_sequence = ["lora1", "lora2", "lora3", "lora1"]
        candidates = {"lora1", "lora2", "lora3"}

        lru = self._make_policy("lru", access_sequence)
        fifo = self._make_policy("fifo", access_sequence)

        self.assertEqual(lru.select_victim(candidates), "lora2")
        self.assertEqual(fifo.select_victim(candidates), "lora1")

    def test_selection_skips_older_noncandidates(self):
        """The oldest ineligible adapter must not displace an eligible one."""
        for policy_name in ("lru", "fifo"):
            with self.subTest(policy=policy_name):
                policy = self._make_policy(policy_name, ["lora1", "lora2", "lora3"])
                self.assertEqual(policy.select_victim({"lora2", "lora3"}), "lora2")

    def test_base_model_is_evicted_only_as_last_resort(self):
        """Regression for #14795: keep the base slot while an adapter can move."""
        for policy_name in ("lru", "fifo"):
            with self.subTest(policy=policy_name):
                policy = self._make_policy(policy_name, ["lora1", "lora2"])
                self.assertEqual(
                    policy.select_victim({None, "lora1", "lora2"}), "lora1"
                )
                self.assertIsNone(policy.select_victim({None}))

    def test_remove_excludes_adapter_from_future_selection(self):
        """Unloaded adapters must not remain eligible through stale policy state."""
        for policy_name in ("lru", "fifo"):
            with self.subTest(policy=policy_name):
                policy = self._make_policy(policy_name, ["lora1", "lora2", "lora3"])
                policy.remove("lora1")
                self.assertEqual(
                    policy.select_victim({"lora1", "lora2", "lora3"}), "lora2"
                )

    def test_unknown_policy_is_rejected(self):
        """A configuration typo must not silently select a fallback policy."""
        with self.assertRaisesRegex(
            ValueError, "Unknown eviction policy: invalid_policy"
        ):
            get_eviction_policy("invalid_policy")


if __name__ == "__main__":
    unittest.main(verbosity=2)
