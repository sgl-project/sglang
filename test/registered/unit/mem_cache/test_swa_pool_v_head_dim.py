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
"""`SWAKVPool.get_v_head_dim()` -- the pool method a mambaish + hybrid-SWA model
reaches on boot.

`TritonAttnBackend.__init__` asks the POOL for `v_head_dim` when the model is
mambaish and its full/SWA value head dims MATCH (Inkling-class), so an
SWA-shaped pool without the method kills backend construction with
AttributeError.
"""

import unittest

import torch

from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_DEV = "cpu"
_FULL_V_HEAD_DIM = 8
_SWA_V_HEAD_DIM = 8  # MATCHING: this is what routes Inkling into the branch


def _swa_pool():
    """Inkling-class layer split: full and SWA layers interleaved, layer 0 NOT a
    full-attention layer, which is why the backend asks the pool at all."""
    return SWAKVPool(
        size=32,
        size_swa=16,
        page_size=1,
        dtype=torch.float16,
        head_num=2,
        head_dim=_FULL_V_HEAD_DIM,
        swa_attention_layer_ids=[0, 2],
        full_attention_layer_ids=[1, 3],
        device=_DEV,
        enable_memory_saver=False,
    )


class TestSWAPoolVHeadDim(unittest.TestCase):
    def test_static_pool_reports_the_full_side_value_head_dim(self):
        pool = _swa_pool()
        self.assertEqual(pool.get_v_head_dim(), _FULL_V_HEAD_DIM)


if __name__ == "__main__":
    unittest.main()
