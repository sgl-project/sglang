import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache.device_pool_info import (
    DevicePoolInfo,
    IndexKeyBufferInfo,
    IndexKeyFormat,
    MLABufferInfo,
    PagedLayerBufferInfo,
)
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    _build_dsa_device_pool_group,
)
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.mem_cache.pool_buffer_binding import (
    pack_host_pool_buffers,
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
                buffers=tuple(
                    torch.empty((4, page_size // ratio * 132), dtype=torch.uint8)
                    for _ in layers
                ),
                format=IndexKeyFormat.DSA_FP8,
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

    def test_page_views_preserve_bytes_and_alias_complete_pages(self):
        # Linker must transfer original bytes without copying or including a tail.
        mla = MLABufferInfo(
            page_size=4,
            buffers=(torch.arange(27, dtype=torch.bfloat16).reshape(9, 1, 3),),
        )
        for buffers in (mla, _pool_infos()[1].buffer_info):
            descriptor: PagedLayerBufferInfo = buffers
            descriptor.validate()
            for original, pages in zip(descriptor.buffers, descriptor.page_views()):
                with self.subTest(descriptor=type(descriptor).__name__):
                    self.assertEqual(pages.dtype, torch.uint8)
                    self.assertEqual(pages.shape[1], descriptor.page_bytes)
                    self.assertEqual(pages.data_ptr(), original.data_ptr())
                    self.assertTrue(pages.is_contiguous())
            self.assertEqual(len(descriptor.page_views()), len(descriptor.buffers))
        self.assertEqual(mla.page_views()[0].shape, (2, 24))
        self.assertTrue(
            torch.equal(
                mla.page_views()[0].flatten(),
                mla.buffers[0][:8].view(torch.uint8).flatten(),
            )
        )

    def test_mla_validation_identifies_bad_buffer_and_field(self):
        good = torch.empty((8, 1, 3))
        cases = (
            (torch.empty((8, 3)), "shape="),
            (torch.empty((8, 1, 4)), "row shape="),
            (torch.empty((8, 1, 3), dtype=torch.float16), "dtype="),
            (torch.empty((16, 1, 3))[::2], "stride="),
        )
        for bad, field in cases:
            with self.subTest(field=field), self.assertRaises(ValueError) as error:
                MLABufferInfo(page_size=4, buffers=(good, bad)).validate()
            self.assertIn("MLA buffer[1]", str(error.exception))
            self.assertIn(field, str(error.exception))

    def test_model_layers_match_physical_layers_for_each_buffer_format(self):
        from msgspec.structs import replace

        for info in _pool_infos():
            for layers in (info.layer_ids[:-1], (*info.layer_ids, 23)):
                with (
                    self.subTest(pool=info.pool_name, layers=layers),
                    self.assertRaisesRegex(ValueError, "model layers must match"),
                ):
                    pack_host_pool_buffers(
                        target=replace(info, layer_ids=layers),
                        drafts=(),
                        target_model_layer_ids=(20, 21, 22, 23),
                    )

    def test_packed_draft_layers_match_physical_layers_for_each_buffer_format(self):
        from msgspec.structs import replace

        for target in _pool_infos():
            draft = replace(target, layer_ids=(0,))
            with (
                self.subTest(pool=target.pool_name),
                self.assertRaisesRegex(ValueError, "model layers must match"),
            ):
                pack_host_pool_buffers(
                    target=target,
                    drafts=(draft,),
                    target_model_layer_ids=(20, 21, 22),
                )

    def test_packed_mla_rejects_strided_rows_before_linker_construction(self):
        from msgspec.structs import replace

        target = _pool_infos()[0]
        strided = tuple(buffer[::2] for buffer in target.buffer_info.buffers)
        target = replace(
            target, buffer_info=replace(target.buffer_info, buffers=strided)
        )
        with self.assertRaisesRegex(ValueError, "contiguous"):
            pack_host_pool_buffers(
                target=target,
                drafts=(),
                target_model_layer_ids=(20, 21, 22),
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
        packed, mapping = pack_host_pool_buffers(
            target=target,
            drafts=(draft, draft),
            target_model_layer_ids=(20, 21, 22),
        )
        self.assertEqual(mapping, {0: 0, 2: 1, 3: 2, 4: 3})
        self.assertIs(packed.buffers[1], target.buffer_info.buffers[1])
        self.assertIs(packed.buffers[2], draft.buffer_info.buffers[0])
        self.assertEqual(target.layer_ids, (20, 22))
        self.assertEqual(draft.layer_ids, (0,))

    def test_packed_draft_errors_identify_the_entry_and_field(self):
        from msgspec.structs import replace

        target = _pool_infos()[1]
        draft = _pool_infos(layers=(0,))[1]
        cases = (
            (replace(draft, pool_name=PoolName.KV), "pool_name="),
            (replace(draft, indices_from_pool=PoolName.DRAFT), "indices_from_pool="),
            (
                replace(draft, buffer_info=_pool_infos()[0].buffer_info),
                "buffer_info type=",
            ),
            (_pool_infos(layers=(0, 1))[1], "layer_ids=(0, 1)"),
            (replace(draft, shared_layer_to_owner=((1, 0),)), "shared_layer_to_owner"),
            (_pool_infos(layers=(0,), page_size=128, ratio=2)[1], "page_size="),
        )
        for invalid, field in cases:
            with self.subTest(field=field), self.assertRaises(ValueError) as error:
                pack_host_pool_buffers(
                    target=target,
                    drafts=(draft, invalid),
                    target_model_layer_ids=(20, 21, 22),
                )
            self.assertIn("draft[1]", str(error.exception))
            self.assertIn(field, str(error.exception))
            self.assertIn("got", str(error.exception))

    def test_packed_pages_do_not_match_by_bytes_alone(self):
        target = _pool_infos()[1]
        draft = _pool_infos(layers=(0,), page_size=128, ratio=2)[1]
        self.assertEqual(
            target.buffer_info.buffers[0].shape,
            draft.buffer_info.buffers[0].shape,
        )
        with self.assertRaisesRegex(ValueError, "page coverage differs"):
            pack_host_pool_buffers(
                target=target,
                drafts=(draft,),
                target_model_layer_ids=(20, 21, 22),
            )

    def test_index_page_coverage_rejects_wrong_compression(self):
        info = _pool_infos()[1].buffer_info
        wrong = IndexKeyBufferInfo(
            page_size=info.page_size,
            buffers=info.buffers,
            compress_ratio=1,
            format=info.format,
        )
        with self.assertRaisesRegex(ValueError, "index pages of 33792 bytes"):
            wrong.validate()

    def test_index_pages_require_scale_bytes(self):
        from msgspec.structs import replace

        for ratio in (1, 4):
            info = _pool_infos(ratio=ratio)[1].buffer_info
            info.validate()
            keys_only = torch.empty(
                (4, info.page_size // ratio * 128), dtype=torch.uint8
            )
            with (
                self.subTest(compress_ratio=ratio),
                self.assertRaisesRegex(ValueError, "index pages of"),
            ):
                replace(info, buffers=(keys_only,)).validate()

    def test_unknown_index_format_has_no_default_page_size(self):
        from enum import Enum

        class OtherFormat(Enum):
            OTHER = "other"

        with self.assertRaisesRegex(ValueError, "unsupported index key format"):
            IndexKeyFormat.page_bytes(OtherFormat.OTHER, 64)

    def test_linker_preserves_heterogeneous_draft_page_metadata(self):
        from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
            _build_legacy_dsa_device_pool_group,
        )

        def pool(layers, width, dtype, extra_rows=0):
            pool = DSATokenToKVPool.__new__(DSATokenToKVPool)
            pool.page_size = 64
            pool.index_kpool = 1
            pool.start_layer = 20
            pool.layer_num = layers
            pool.layer_shard_enabled = False
            pool.model_layer_ids = tuple(range(20, 20 + layers))
            pool.skip_topk_layers = [False] * layers
            pool.kv_buffer = [
                torch.empty((128 + extra_rows, 1, width), dtype=dtype)
                for _ in range(layers)
            ]
            pool.index_key_cache = SimpleNamespace(
                buffer=[
                    torch.empty((2, 8448), dtype=torch.uint8) for _ in range(layers)
                ]
            )
            return pool

        target = pool(2, 576, torch.bfloat16, extra_rows=17)
        drafts = (pool(1, 656, torch.uint8), pool(1, 576, torch.float32))
        legacy = _build_legacy_dsa_device_pool_group(target, 64, drafts)
        with patch("sglang.srt.utils.is_cuda", return_value=True):
            native = _build_dsa_device_pool_group(target, 64, drafts)
        for name, entry in native.entry_map.items():
            self.assertIsNone(entry.device_pool)
            previous = legacy.entry_map[name]
            for page in (0, 1):
                indices = torch.arange(page * 64, (page + 1) * 64)
                self.assertEqual(
                    entry.get_page_buffer_meta(indices),
                    previous.get_page_buffer_meta(indices),
                )
                for layer in (0, 1):
                    self.assertEqual(
                        entry.get_prepared_layer_range_meta(
                            entry.prepare_locations(indices), layer
                        ),
                        previous.get_prepared_layer_range_meta(
                            previous.prepare_locations(indices), layer
                        ),
                    )
        with self.assertRaisesRegex(
            ValueError, r"MLA buffer\[2\]: expected row shape="
        ):
            pack_host_pool_buffers(
                target=target.get_device_pool_infos()[0],
                drafts=tuple(draft.get_device_pool_infos()[0] for draft in drafts),
                target_model_layer_ids=target.model_layer_ids,
            )

    def test_linker_compiles_noncontiguous_model_layers_without_pool_buffer_reads(self):
        pool = DSATokenToKVPool.__new__(DSATokenToKVPool)
        pool.layer_shard_enabled = False
        pool.page_size = 256
        pool.layer_num = 3
        pool.start_layer = 20
        from msgspec.structs import replace

        kv, index = _pool_infos()
        infos = (
            replace(kv, layer_ids=(21, 25, 29)),
            replace(index, layer_ids=(21, 29)),
        )
        with (
            patch.object(pool, "get_device_pool_infos", return_value=infos),
            patch("sglang.srt.utils.is_cuda", return_value=True),
        ):
            group = _build_dsa_device_pool_group(pool, page_size=256)
        index = group.entry_map[PoolName.INDEXER]
        self.assertIsNone(index.get_prepared_layer_range_meta([1], 1))
        ptrs, sizes, offsets = index.get_prepared_layer_range_meta([1], 2)
        self.assertEqual(ptrs, [[infos[1].buffer_info.buffers[1][1].data_ptr()]])
        self.assertEqual(sizes, [[8448]])
        self.assertEqual(offsets, [[8448]])

    def test_linker_packed_drafts_keep_compact_target_layer_offsets(self):
        target = DSATokenToKVPool.__new__(DSATokenToKVPool)
        target.layer_shard_enabled = False
        target.page_size, target.layer_num, target.start_layer = 256, 3, 20
        target.get_device_pool_infos = Mock(return_value=_pool_infos())
        drafts = []
        for _ in range(2):
            draft = DSATokenToKVPool.__new__(DSATokenToKVPool)
            draft.layer_shard_enabled = False
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
        pool.layer_shard_enabled = False
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
        pool.layer_shard_enabled = False
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
        self.assertIs(index.buffer_info.buffers[1], pool.index_key_cache.buffer[2])
        with patch.object(pool, "skip_topk_layers", [True] * 3):
            self.assertEqual(len(pool.get_device_pool_infos()), 1)


if __name__ == "__main__":
    unittest.main()
