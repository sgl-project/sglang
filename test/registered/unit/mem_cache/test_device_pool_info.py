import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache.device_pool_info import (
    DevicePoolInfo,
    EncodedPageBuffers,
    IndexKeyBufferInfo,
    IndexPageEncoding,
    MLABufferInfo,
)
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    _build_dsa_device_pool_group,
)
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.mem_cache.pool_buffer_binding import (
    bind_packed_pool_buffers,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _pool_infos(*, layers=(20, 22), page_size=256, ratio=4):
    return (
        DevicePoolInfo(
            pool_name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            layer_ids=(20, 21, 22),
            buffer_info=MLABufferInfo(
                page_size=page_size,
                buffers=tuple(torch.empty((1024, 1, 160)) for _ in range(3)),
            ),
        ),
        DevicePoolInfo(
            pool_name=PoolName.INDEXER,
            indices_from_pool=PoolName.KV,
            layer_ids=layers,
            buffer_info=IndexKeyBufferInfo(
                page_size=page_size,
                compress_ratio=ratio,
                buffers=EncodedPageBuffers(
                    buffers=tuple(
                        torch.empty((4, page_size // ratio * 132), dtype=torch.uint8)
                        for _ in layers
                    ),
                    encoding=IndexPageEncoding.DSA_FP8,
                ),
            ),
        ),
    )


class TestDevicePoolInfo(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(
            patch(
                "sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler.get_parallel",
                return_value=SimpleNamespace(dcp_enabled=False),
            )
        )

    def test_model_layers_match_physical_layers_for_each_buffer_format(self):
        for info in _pool_infos():
            with self.subTest(pool=info.pool_name):
                with self.assertRaisesRegex(ValueError, "model layers must match"):
                    DevicePoolInfo(
                        pool_name=info.pool_name,
                        indices_from_pool=info.indices_from_pool,
                        layer_ids=info.layer_ids[:-1],
                        buffer_info=info.buffer_info,
                    )

    def test_packed_mla_rejects_strided_rows_before_linker_construction(self):
        from msgspec.structs import replace

        target = _pool_infos()[0]
        strided = tuple(buffer[::2] for buffer in target.buffer_info.buffers)
        target = replace(
            target, buffer_info=replace(target.buffer_info, buffers=strided)
        )
        with self.assertRaisesRegex(ValueError, "matching packed rows"):
            bind_packed_pool_buffers(
                target=target,
                drafts=(),
                model_to_transfer_layer={20: 0, 21: 1, 22: 2},
                target_layer_num=3,
            )

    def test_dcp_index_host_keeps_legacy_page_geometry(self):
        from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerHostPoolBuilder

        pool = DSATokenToKVPool.__new__(DSATokenToKVPool)
        pool.get_device_pool_infos = Mock(
            side_effect=AssertionError("DCP provider was called")
        )
        decl = SimpleNamespace(device_pool=pool)
        anchor = SimpleNamespace(_is_dummy=False, dcp_size=2, page_size=64)
        with (
            patch("sglang.srt.mem_cache.pool_host.dsa._is_cuda", True),
            patch("sglang.srt.mem_cache.pool_host.dsa.DSAIndexerPoolHost") as legacy,
        ):
            host = DSAIndexerHostPoolBuilder().build(
                decl=decl,
                anchor_host=anchor,
                allocator_type="default",
                packed_draft_device_pools=(),
            )
        self.assertIs(host, legacy.return_value)
        legacy.assert_called_once_with(
            decl=decl,
            anchor_host=anchor,
            packed_draft_device_pools=(),
            allocator_type="default",
        )
        pool.get_device_pool_infos.assert_not_called()

    def test_packed_layer_coordinates_preserve_source_context(self):
        target = _pool_infos()[1]
        draft = _pool_infos(layers=(0,))[1]
        packed, mapping = bind_packed_pool_buffers(
            target=target,
            drafts=(draft, draft),
            model_to_transfer_layer={20: 0, 21: 1, 22: 2},
            target_layer_num=3,
        )
        self.assertEqual(mapping, {0: 0, 2: 1, 3: 2, 4: 3})
        self.assertIs(packed.buffers.buffers[1], target.buffer_info.buffers.buffers[1])
        self.assertIs(packed.buffers.buffers[2], draft.buffer_info.buffers.buffers[0])
        self.assertEqual(target.layer_ids, (20, 22))
        self.assertEqual(draft.layer_ids, (0,))

    def test_packed_pages_do_not_match_by_bytes_alone(self):
        target = _pool_infos()[1]
        draft = _pool_infos(layers=(0,), page_size=128, ratio=2)[1]
        self.assertEqual(
            target.buffer_info.buffers.buffers[0].shape,
            draft.buffer_info.buffers.buffers[0].shape,
        )
        with self.assertRaisesRegex(ValueError, "page format differs"):
            bind_packed_pool_buffers(
                target=target,
                drafts=(draft,),
                model_to_transfer_layer={20: 0, 21: 1, 22: 2},
                target_layer_num=3,
            )

    def test_index_page_coverage_rejects_wrong_compression(self):
        info = _pool_infos()[1].buffer_info
        wrong = IndexKeyBufferInfo(
            page_size=info.page_size, buffers=info.buffers, compress_ratio=1
        )
        with self.assertRaisesRegex(ValueError, "index pages of 33792 bytes"):
            wrong.validate()

    def test_linker_compiles_model_layers_without_pool_buffer_reads(self):
        pool = DSATokenToKVPool.__new__(DSATokenToKVPool)
        pool.page_size = 256
        pool.layer_num = 3
        pool.start_layer = 20
        infos = _pool_infos()
        with (
            patch.object(pool, "get_device_pool_infos", return_value=infos),
            patch("sglang.srt.utils.is_cuda", return_value=True),
        ):
            group = _build_dsa_device_pool_group(pool, page_size=256)
        index = group.entry_map[PoolName.INDEXER]
        self.assertIsNone(index.get_prepared_layer_range_meta([1], 1))
        ptrs, sizes, offsets = index.get_prepared_layer_range_meta([1], 2)
        self.assertEqual(
            ptrs, [[infos[1].buffer_info.buffers.buffers[1][1].data_ptr()]]
        )
        self.assertEqual(sizes, [[8448]])
        self.assertEqual(offsets, [[8448]])

    def test_linker_packed_drafts_keep_compact_target_layer_offsets(self):
        target = DSATokenToKVPool.__new__(DSATokenToKVPool)
        target.page_size, target.layer_num, target.start_layer = 256, 3, 20
        target.get_device_pool_infos = Mock(return_value=_pool_infos())
        drafts = []
        for _ in range(2):
            draft = DSATokenToKVPool.__new__(DSATokenToKVPool)
            draft.page_size, draft.layer_num, draft.start_layer = 256, 1, 0
            kv, index = _pool_infos(layers=(0,))
            draft.get_device_pool_infos = Mock(
                return_value=(
                    DevicePoolInfo(
                        pool_name=PoolName.KV,
                        indices_from_pool=PoolName.KV,
                        layer_ids=(0,),
                        buffer_info=MLABufferInfo(
                            page_size=256, buffers=kv.buffer_info.buffers[:1]
                        ),
                    ),
                    index,
                )
            )
            drafts.append(draft)
        with patch("sglang.srt.utils.is_cuda", return_value=True):
            group = _build_dsa_device_pool_group(target, 256, tuple(drafts))
        index = group.entry_map[PoolName.INDEXER]
        self.assertEqual(group.num_layers, 3)
        self.assertEqual(index.layer_mapping, {0: (0, 2), 1: 3, 2: 1})
        for layer, buffer_ids in ((0, (0, 2)), (1, (3,)), (2, (1,))):
            pointers, sizes, offsets = index.get_prepared_layer_range_meta([1], layer)
            self.assertEqual(
                pointers, [[index.components[0][i][1].data_ptr() for i in buffer_ids]]
            )
            self.assertEqual(sizes, [[8448] * len(buffer_ids)])
            self.assertEqual(offsets, [[8448 * i for i in buffer_ids]])

    def test_linker_omits_all_shared_index_without_empty_pages(self):
        pool = DSATokenToKVPool.__new__(DSATokenToKVPool)
        pool.page_size = 256
        pool.layer_num = 3
        pool.start_layer = 20
        with (
            patch.object(pool, "get_device_pool_infos", return_value=_pool_infos()[:1]),
            patch("sglang.srt.utils.is_cuda", return_value=True),
        ):
            group = _build_dsa_device_pool_group(pool, page_size=256)
        self.assertEqual(set(group.entry_map), {PoolName.KV})

    def test_provider_uses_model_coordinates_and_omits_placeholders(self):
        from types import SimpleNamespace

        pool = DSATokenToKVPool.__new__(DSATokenToKVPool)
        pool.start_layer = 20
        pool.model_layer_ids = (20, 21, 22)
        pool.layer_num = 3
        pool.page_size = 256
        pool.index_kpool = 4
        pool.kv_buffer = [torch.empty((1024, 1, 160)) for _ in range(3)]
        pool.skip_topk_layers = [False, True, False]
        pool.index_key_cache = SimpleNamespace(
            buffer=[
                torch.empty((4, 8448), dtype=torch.uint8),
                torch.empty((0, 8448), dtype=torch.uint8),
                torch.empty((4, 8448), dtype=torch.uint8),
            ]
        )
        kv, index = pool.get_device_pool_infos()
        self.assertEqual(kv.layer_ids, (20, 21, 22))
        self.assertEqual(index.layer_ids, (20, 22))
        self.assertIs(
            index.buffer_info.buffers.buffers[1], pool.index_key_cache.buffer[2]
        )
        with patch.object(pool, "skip_topk_layers", [True] * 3):
            self.assertEqual(len(pool.get_device_pool_infos()), 1)


if __name__ == "__main__":
    unittest.main()
