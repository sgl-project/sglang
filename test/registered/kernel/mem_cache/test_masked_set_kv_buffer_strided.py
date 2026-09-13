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
"""``masked_set_kv_buffer_kernel`` must step destination slots by the buffer's
own stride. The unified pool's per-layer K/V buffers are strided views (slot
stride > head_num * head_dim), so a hardcoded ``loc * H * D`` would write into
the wrong slot without any error.

    python -m pytest test/registered/unit/mem_cache/test_masked_set_kv_buffer_strided.py -v
"""

import unittest

import torch

from sglang.srt.mem_cache.memory_pool import masked_set_kv_buffer_kernel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestMaskedSetKvBufferStrided(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_writes_land_at_slot_stride(self):
        torch.manual_seed(0)
        N, H, D = 16, 4, 64
        slots = N + 8
        E = 2 * H * D + 128  # slot stride wider than one K row
        device = "cuda"
        backing_k = torch.zeros(slots * E, dtype=torch.bfloat16, device=device)
        backing_v = torch.zeros_like(backing_k)
        k_buffer = backing_k.as_strided((slots, H, D), (E, D, 1))
        v_buffer = backing_v.as_strided((slots, H, D), (E, D, 1), H * D)
        cache_k = torch.randn(N, H, D, dtype=torch.bfloat16, device=device)
        cache_v = torch.randn(N, H, D, dtype=torch.bfloat16, device=device)
        loc = torch.randperm(slots, device=device)[:N]
        mask = torch.arange(N, device=device) % 3 != 0

        masked_set_kv_buffer_kernel[(N,)](
            cache_k,
            cache_v,
            k_buffer,
            v_buffer,
            loc,
            mask,
            N,
            H,
            D,
            128,
            cache_k.stride(0),
            cache_k.stride(1),
            cache_v.stride(0),
            cache_v.stride(1),
            k_buffer.stride(0),
            v_buffer.stride(0),
        )

        ref_k = torch.zeros_like(k_buffer)
        ref_v = torch.zeros_like(v_buffer)
        ref_k[loc[mask]] = cache_k[mask]
        ref_v[loc[mask]] = cache_v[mask]
        torch.testing.assert_close(k_buffer, ref_k, rtol=0, atol=0)
        torch.testing.assert_close(v_buffer, ref_v, rtol=0, atol=0)
        # Nothing outside the two views was touched.
        gap_k = backing_k.view(slots, E)[:, H * D :]
        gap_v = backing_v.view(slots, E)[:, : H * D]
        self.assertTrue(bool((gap_k == 0).all()) and bool((gap_v == 0).all()))
        self.assertTrue(bool((backing_v.view(slots, E)[:, 2 * H * D :] == 0).all()))


if __name__ == "__main__":
    unittest.main()
