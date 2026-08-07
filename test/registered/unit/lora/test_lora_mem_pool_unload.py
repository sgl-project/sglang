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

"""Regression for #21380: LoRAManager.unload_lora_adapter must release the GPU
memory pool slot so repeated load/unload cycles don't exhaust the pool."""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=5, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")

import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.lora.mem_pool import EMPTY_SLOT, LoRAMemoryPool


def _make_pool(max_loras_per_batch: int = 2) -> LoRAMemoryPool:
    """Build a bookkeeping-only LoRAMemoryPool with init_buffers patched out (no GPU)."""
    with patch.object(LoRAMemoryPool, "init_buffers", return_value=None):
        hf_config = MagicMock(num_hidden_layers=2, hidden_size=64)
        return LoRAMemoryPool(
            base_hf_config=hf_config,
            max_loras_per_batch=max_loras_per_batch,
            dtype=None,
            tp_size=1,
            tp_rank=0,
            max_lora_rank=8,
            target_modules={"q_proj", "v_proj"},
            base_model=MagicMock(),
            eviction_policy="lru",
            lora_added_tokens_size=0,
        )


class TestReleaseLoraSlot(unittest.TestCase):
    def test_repeated_load_unload_cycles_do_not_exhaust_pool(self):
        """N+2 load/unload cycles on a size-N pool must leave every slot empty
        and every tracker clean (previously: ghost-uid exhaustion crashed eviction)."""
        N = 2
        pool = _make_pool(max_loras_per_batch=N)
        for i in range(N + 2):
            uid, slot = f"uid_{i}", i % N
            pool.uid_to_buffer_id[uid] = slot
            pool.buffer_id_to_uid[slot] = uid
            pool.eviction_policy.mark_used(uid)

            pool.release_lora_slot(uid)

            self.assertNotIn(uid, pool.uid_to_buffer_id)
            self.assertIs(pool.buffer_id_to_uid[slot], EMPTY_SLOT)
            self.assertNotIn(uid, pool.eviction_policy.access_order)


if __name__ == "__main__":
    unittest.main()
