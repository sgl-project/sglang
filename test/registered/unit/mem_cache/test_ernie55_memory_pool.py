# Copyright 2026 Baidu ERNIE Team and The SGLang Team. All rights reserved.
#
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
import types
import unittest

import torch

from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.ernie55_memory_pool import (
    DSA_PAGE_SIZE,
    Ernie5_5TokenToKVPool,
    mla_layer_bytes_per_token,
)
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_PAGE_SIZE = 256
_BASE = ([0x1000], [8192], [2048])
_LATENT_ROW = 656
_INDEX_ROW = 4352


def _pool(num_layers=3, owned=(1,)):
    """CPU stub of what get_contiguous_buf_infos and extra_host_regions read."""
    pool = object.__new__(Ernie5_5TokenToKVPool)
    pool.page_size = _PAGE_SIZE
    pool._stage_start = 0
    pool.mla_kv_pool = types.SimpleNamespace(
        # 0-row placeholders stand in for the HCA layers, like the real sub-pool.
        kv_buffer=[
            torch.zeros(
                (_PAGE_SIZE * 2 if i in owned else 0, 1, _LATENT_ROW),
                dtype=torch.uint8,
            )
            for i in range(num_layers)
        ],
        index_k_with_scale_buffer=[
            torch.zeros((8 if i in owned else 0, _INDEX_ROW), dtype=torch.uint8)
            for i in range(num_layers)
        ],
    )
    return pool


class _StubbedBase(CustomTestCase):
    def setUp(self):
        self._real = DeepSeekV4TokenToKVPool.get_contiguous_buf_infos
        DeepSeekV4TokenToKVPool.get_contiguous_buf_infos = lambda self: (
            list(_BASE[0]),
            list(_BASE[1]),
            list(_BASE[2]),
        )

    def tearDown(self):
        DeepSeekV4TokenToKVPool.get_contiguous_buf_infos = self._real


class TestErnie5_5PdBufInfos(_StubbedBase):
    def test_appends_latent_then_index_regions_per_mla_layer(self):
        ptrs, lens, items = _pool(num_layers=4, owned=(1, 3)).get_contiguous_buf_infos()
        self.assertEqual(ptrs[0], _BASE[0][0], "base regions must come first")
        self.assertEqual(items[0], _BASE[2][0])
        latent_bytes = _PAGE_SIZE * 2 * _LATENT_ROW
        index_bytes = 8 * _INDEX_ROW
        self.assertEqual(
            lens[1:], [latent_bytes, latent_bytes, index_bytes, index_bytes]
        )
        self.assertEqual(len(ptrs), len(lens))
        self.assertEqual(len(ptrs), len(items))

    def test_a_pool_with_no_mla_layer_matches_the_base(self):
        got = _pool(num_layers=2, owned=()).get_contiguous_buf_infos()
        self.assertEqual(got, (list(_BASE[0]), list(_BASE[1]), list(_BASE[2])))


class TestErnie5_5MlaHostRegions(_StubbedBase):
    def test_one_region_per_kind_keyed_by_owning_layer(self):
        regions = _pool(num_layers=4, owned=(1, 3)).extra_host_regions()
        self.assertEqual(
            [r.pool_name for r in regions],
            [PoolName.ERNIE5_5_MLA, PoolName.ERNIE5_5_MLA_INDEXER],
        )
        for region in regions:
            self.assertEqual(region.layer_mapping, {1: 0, 3: 1})
            self.assertEqual(len(region.device_buffers), 2)

    def test_a_host_page_is_one_main_page_of_the_device_buffer(self):
        pool = _pool(num_layers=2, owned=(1,))
        latent, indexer = pool.extra_host_regions()
        self.assertEqual(latent.item_bytes, _LATENT_ROW * _PAGE_SIZE)
        self.assertEqual(indexer.item_bytes, _INDEX_ROW * (_PAGE_SIZE // DSA_PAGE_SIZE))
        device = pool.mla_kv_pool.kv_buffer[1]
        device[_PAGE_SIZE, 0, 0] = 7
        self.assertEqual(latent.device_buffers[0][1, 0].item(), 7)
        self.assertEqual(latent.device_buffers[0].data_ptr(), device.data_ptr())

    def test_items_match_the_pd_enumeration(self):
        pool = _pool(num_layers=4, owned=(1, 3))
        _, _, items = pool.get_contiguous_buf_infos()
        host_items = [r.item_bytes for r in pool.extra_host_regions() for _ in range(2)]
        self.assertEqual(items[1:], host_items)

    def test_a_pool_with_no_mla_layer_mirrors_nothing(self):
        self.assertEqual(_pool(num_layers=2, owned=()).extra_host_regions(), ())


class TestErnie5_5SwaBuffers(CustomTestCase):
    @staticmethod
    def _swa_pool(stage, mla_layer_ids):
        pool = object.__new__(Ernie5_5TokenToKVPool)
        pool._mla_layer_ids = frozenset(mla_layer_ids)
        pool._stage_start, pool._stage_end = stage.start, stage.stop
        pool.stage_layer_ids = tuple(stage)
        pool._swa_buffer_index = {
            layer: index for index, layer in enumerate(pool.swa_layer_ids())
        }
        return pool

    def test_swa_buffers_cover_only_the_hca_layers(self):
        pool = self._swa_pool(range(0, 8), mla_layer_ids=(3, 7))
        self.assertEqual(pool.swa_layer_ids(), [0, 1, 2, 4, 5, 6])
        self.assertEqual(
            [pool._swa_local_layer_id(i) for i in (0, 1, 2, 4, 5, 6)], list(range(6))
        )
        with self.assertRaises(KeyError):
            pool._swa_local_layer_id(3)

    def test_hicache_mapping_is_stage_local_and_matches_the_buffers(self):
        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            _dsv4_transfer_layer_mapping,
        )

        pool = self._swa_pool(range(7, 14), mla_layer_ids=(13,))
        mapping = _dsv4_transfer_layer_mapping(tuple(pool.swa_layer_ids()), 7)
        self.assertEqual(len(mapping), len(pool.swa_layer_ids()))
        self.assertEqual(mapping, {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5})


class TestErnie5_5Fp4IndexBudget(CustomTestCase):
    def test_fp4_index_budget_matches_packed_cache_bytes(self):
        config = types.SimpleNamespace(
            hybrid_mla_kv_lora_rank=512,
            hybrid_mla_qk_rope_head_dim=64,
            index_head_dim=128,
        )
        for kv_dtype, latent_bytes in (("fp8_e4m3", 656), ("bfloat16", 1152)):
            with self.subTest(kv_dtype=kv_dtype):
                for fp4, index_bytes in ((False, 132), (True, 68)):
                    self.assertEqual(
                        DSATokenToKVPool.index_bytes_per_token_for(128, fp4),
                        index_bytes,
                    )
                    self.assertEqual(
                        mla_layer_bytes_per_token(config, kv_dtype, fp4),
                        latent_bytes + index_bytes,
                    )


if __name__ == "__main__":
    unittest.main()
