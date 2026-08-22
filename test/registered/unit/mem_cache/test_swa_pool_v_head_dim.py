"""SWAKVPool must report the full-attention value head dim.

Regression: TritonAttnBackend.__init__ calls token_to_kv_pool.get_v_head_dim()
on its mambaish branch, SWAKVPool did not implement it, and a bare
`launch_server --model-path thinkingmachines/Inkling-Small` on Hopper died with
`AttributeError: 'SWAKVPool' object has no attribute 'get_v_head_dim'`. The
three Inkling tests all pin --attention-backend fa4, so nothing covered it.

The SWA sub-pool gets a different value head dim and layer 0 is an SWA layer, so
returning self.head_dim or indexing layer 0 both fail here too.
"""

import unittest

import torch

from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

HEAD_NUM = 2
HEAD_DIM = 128
V_HEAD_DIM_FULL = 64
V_HEAD_DIM_SWA = 96
SWA_LAYER_IDS = [0, 2]
FULL_LAYER_IDS = [1, 3]


@unittest.skipUnless(torch.cuda.is_available(), "KV pools allocate device buffers")
class TestSWAPoolVHeadDim(CustomTestCase):
    def test_reports_full_attention_v_head_dim(self):
        pool = SWAKVPool(
            size=32,
            size_swa=16,
            page_size=1,
            dtype=torch.bfloat16,
            head_num=HEAD_NUM,
            head_dim=HEAD_DIM,
            v_head_dim=V_HEAD_DIM_FULL,
            swa_v_head_dim=V_HEAD_DIM_SWA,
            swa_attention_layer_ids=SWA_LAYER_IDS,
            full_attention_layer_ids=FULL_LAYER_IDS,
            device="cuda",
        )

        self.assertEqual(
            pool.get_value_buffer(SWA_LAYER_IDS[0]).shape[-1], V_HEAD_DIM_SWA
        )
        self.assertEqual(pool.get_v_head_dim(), V_HEAD_DIM_FULL)


if __name__ == "__main__":
    unittest.main()
