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

    python -m pytest test/registered/kernels/ops/kvcache/test_masked_set_kv_buffer_strided.py -v
"""

import unittest

import torch

from sglang.srt.mem_cache.kv.layout.mha import MHASubPoolSpec, MHAView
from sglang.srt.mem_cache.kv.memory.arena import ArenaBand, KVArena
from sglang.srt.mem_cache.memory_pool import (
    KVWriteLoc,
    MHATokenToKVPool,
    masked_set_kv_buffer_kernel,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def _make_pool(pool_type):
    if pool_type is MHATokenToKVPool:
        return MHATokenToKVPool(
            size=63,
            page_size=1,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=8,
            layer_num=1,
            device="cuda",
            enable_memory_saver=False,
            enable_alt_stream=False,
            kv_cache_layout="nhd",
        )

    spec = MHASubPoolSpec(
        name="full",
        layer_num=1,
        head_num=2,
        head_dim=8,
        store_dtype=torch.bfloat16,
        grow_direction="down",
    )
    arena = KVArena(
        bands=[
            ArenaBand(
                name=spec.name,
                entry_bytes=spec.entry_bytes(),
                unit_entries=1,
                grow_direction=spec.grow_direction,
                zero_on_handout=spec.zero_units_on_handout,
            )
        ],
        total_bytes=64 * spec.entry_bytes(),
        sink_bytes=spec.entry_bytes(),
        device="cuda",
    )
    return MHAView(
        arena=arena,
        spec=spec,
        page_size=1,
        model_layer_ids=(0,),
        enable_alt_stream=False,
    )


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

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_callers_reuse_specialization_across_batch_sizes(self):
        for pool_type in (MHATokenToKVPool, MHAView):
            with self.subTest(pool=pool_type.__name__):
                pool = _make_pool(pool_type)
                key_buffer, value_buffer = pool.get_kv_buffer(0)
                if pool_type is MHAView:
                    self.assertGreater(key_buffer.stride(0), 2 * 8)
                masked_set_kv_buffer_kernel.device_caches.clear()
                try:
                    for n in (17, 33):
                        torch.manual_seed(n)
                        key = torch.randn(n, 2, 8, device="cuda", dtype=torch.bfloat16)
                        value = torch.randn_like(key)
                        key_buffer.fill_(-1)
                        value_buffer.fill_(-1)
                        locations = torch.arange(
                            n, 0, -1, device="cuda", dtype=torch.int64
                        )
                        write_mask = torch.arange(n, device="cuda") % 2 == 0
                        pool.set_kv_buffer(
                            None,
                            KVWriteLoc(locations, physical=True),
                            key,
                            value,
                            layer_id_override=0,
                            dcp_kv_mask=write_mask,
                        )
                        torch.cuda.synchronize()

                        expected_key = torch.full_like(key_buffer, -1)
                        expected_value = torch.full_like(value_buffer, -1)
                        expected_key[locations[write_mask]] = key[write_mask]
                        expected_value[locations[write_mask]] = value[write_mask]
                        torch.testing.assert_close(
                            key_buffer, expected_key, rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            value_buffer, expected_value, rtol=0, atol=0
                        )
                        # Only N and the launch grid change between calls.
                        self.assertEqual(
                            sum(
                                len(cache[0])
                                for cache in masked_set_kv_buffer_kernel.device_caches.values()
                            ),
                            1,
                        )
                finally:
                    masked_set_kv_buffer_kernel.device_caches.clear()


if __name__ == "__main__":
    unittest.main()
