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

import unittest
from types import SimpleNamespace

from sglang.srt.lora.eviction_policy import get_eviction_policy
from sglang.srt.lora.mem_pool import EMPTY_SLOT, LoRAMemoryPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestLoRAEvictionBatchPrep(CustomTestCase):
    def test_prepare_lora_batch_multi_eviction_and_base_fallback(self):
        """prepare_lora_batch allocates empty slots in order, evicts multiple adapters, and falls back to None last."""
        pool = LoRAMemoryPool.__new__(LoRAMemoryPool)
        pool.max_loras_per_batch = 4
        pool.buffer_id_to_uid = [EMPTY_SLOT] * 4
        pool.uid_to_buffer_id = {}
        pool.eviction_policy = get_eviction_policy("lru")
        loaded = []
        pool.load_lora_weight_to_buffer = lambda uid, buf_id, *args: loaded.append(
            (uid, buf_id)
        )

        # Fill all 4 slots one by one so slot assignment and LRU order are deterministic
        for idx, uid in enumerate([None, "lora1", "lora2", "lora3"]):
            pool.prepare_lora_batch({uid}, {}, [], {}, None, None)
            self.assertEqual(pool.uid_to_buffer_id[uid], idx)
        self.assertEqual(pool.buffer_id_to_uid, [None, "lora1", "lora2", "lora3"])

        # Refresh lora1 so LRU order among LoRA adapters is lora2 (oldest), lora3, lora1; pin lora2
        pool.prepare_lora_batch({"lora1"}, {}, [], {}, None, None)
        lora_refs = {"lora2": SimpleNamespace(pinned=True)}

        # Request 3 new adapters in a single batch:
        # Must skip pinned lora2 (slot 2), evict lora3 (slot 3) and lora1 (slot 1) first, then None (slot 0).
        loaded.clear()
        pool.prepare_lora_batch(
            {"lora4", "lora5", "lora6"}, {}, [], lora_refs, None, None
        )
        self.assertEqual(
            set(pool.uid_to_buffer_id.keys()), {"lora2", "lora4", "lora5", "lora6"}
        )
        self.assertEqual(pool.uid_to_buffer_id["lora2"], 2)
        self.assertEqual([buf_id for _, buf_id in loaded], [3, 1, 0])

        # When all non-current slots are pinned, prepare_lora_batch must raise ValueError
        all_pinned = {
            uid: SimpleNamespace(pinned=True)
            for uid in ("lora2", "lora4", "lora5", "lora6")
        }
        with self.assertRaisesRegex(ValueError, "No available buffer slots found"):
            pool.prepare_lora_batch({"lora7"}, {}, [], all_pinned, None, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
