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

"""The eviction policy must honour the candidate set it is given."""

import unittest

from sglang.srt.lora.eviction_policy import get_eviction_policy
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestLoRAEvictionPolicy(CustomTestCase):
    @staticmethod
    def _make_policy(policy_name, access_sequence):
        policy = get_eviction_policy(policy_name)
        for adapter_id in access_sequence:
            policy.mark_used(adapter_id)
        return policy

    def test_selection_skips_older_noncandidates(self):
        """mem_pool keeps current-batch and pinned adapters out of the candidate
        set but they stay resident, so a policy that returned the oldest tracked
        adapter regardless would evict live weights without any error."""
        for policy_name in ("lru", "fifo"):
            with self.subTest(policy=policy_name):
                policy = self._make_policy(policy_name, ["lora1", "lora2", "lora3"])
                self.assertEqual(policy.select_victim({"lora2", "lora3"}), "lora2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
