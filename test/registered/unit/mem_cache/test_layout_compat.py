# Copyright 2023-2026 SGLang Team
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
"""``move_kv_cache_native`` -- the stock per-layer 3-D move static-pool
compaction rides on -- must stay byte-exact. CPU-only.

The page-major envelope layout and the per-layer views over it are covered by
``test_unified_mha_views.py``, which pins the view addressing against the
envelope formula byte for byte.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.mem_cache.memory_pool import move_kv_cache_native


class TestMoveKVCacheNative(unittest.TestCase):
    def test_move_kv_cache_3d_path_unchanged(self):
        """The stock per-layer 3-D move must relocate exactly the named token
        rows, byte-identically; compaction on static pools rides on it."""
        k = [torch.zeros((32, 2, 4), dtype=torch.float16) for _ in range(2)]
        v = [torch.zeros((32, 2, 4), dtype=torch.float16) for _ in range(2)]
        for L in range(2):
            k[L][5] = float(L + 1)
            v[L][5] = -float(L + 1)
        move_kv_cache_native(
            k,
            v,
            tgt_loc=torch.tensor([7], dtype=torch.int64),
            src_loc=torch.tensor([5], dtype=torch.int64),
        )
        for L in range(2):
            self.assertTrue(torch.all(k[L][7] == float(L + 1)))
            self.assertTrue(torch.all(v[L][7] == -float(L + 1)))


if __name__ == "__main__":
    unittest.main()
