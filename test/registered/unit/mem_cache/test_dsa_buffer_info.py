import unittest

import torch

from sglang.srt.mem_cache.device_pool_info import (
    DevicePoolInfo,
    IndexKeyBufferInfo,
    IndexKeyFormat,
    MLABufferInfo,
)
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.host_pool_config import prepare_host_pool_config
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    build_host_pool_group,
    build_kv_host_pool,
)
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, HybridLinearKVPool
from sglang.srt.mem_cache.pool_buffer_binding import (
    pack_host_pool_buffers,
)
from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=12, stage="base-b", runner_config="1-gpu-small")


class TestDSABufferInfoTransfer(CustomTestCase):
    def test_format_constructors_need_no_device_pool(self):
        # Model coordinates and buffer ownership must suffice without a DSA object.
        def kv(layers):
            return DevicePoolInfo(
                pool_name=PoolName.KV,
                indices_from_pool=PoolName.KV,
                layer_ids=layers,
                buffer_info=MLABufferInfo(
                    page_size=64,
                    buffers=tuple(
                        torch.zeros((320, 1, 160), device="cuda", dtype=torch.bfloat16)
                        for _ in layers
                    ),
                ),
            )

        def index(layers):
            return DevicePoolInfo(
                pool_name=PoolName.INDEXER,
                indices_from_pool=PoolName.KV,
                layer_ids=layers,
                buffer_info=IndexKeyBufferInfo(
                    page_size=64,
                    compress_ratio=1,
                    format=IndexKeyFormat.DSA_FP8,
                    buffers=tuple(
                        torch.zeros((5, 8448), device="cuda", dtype=torch.uint8)
                        for _ in layers
                    ),
                ),
            )

        target, draft = kv((21, 25)), kv((0,))
        host = MLATokenToKVPoolHost.from_pool_infos(
            target=target,
            drafts=(draft,),
            target_model_layer_ids=(21, 25),
            device_capacity=256,
            host_to_device_ratio=2,
            host_size=0,
            layout="layer_first",
            pin_memory=False,
        )
        try:
            self.assertIsNone(host.device_pool)
            self.assertEqual(host.layer_num, 3)
            self.assertEqual(host.target_layer_num, 2)
            self.assertIs(host._buffer_info.buffers[2], draft.buffer_info.buffers[0])
            index_host = DSAIndexerPoolHost.from_pool_infos(
                target=index((25,)),
                drafts=(index((0,)),),
                target_model_layer_ids=(21, 25),
                transfer_page_size=64,
                num_host_pages=host.page_num,
                layout=host.layout,
                pin_memory=False,
            )
            try:
                self.assertIsNone(index_host.device_pool)
                self.assertEqual(index_host._device_to_host_layer, {1: 0, 2: 1})
                self.assertEqual(index_host.target_layer_num, 1)
                self.assertEqual(index_host.size, host.size)
            finally:
                index_host.destroy()
        finally:
            host.destroy()

    def test_hybrid_keeps_model_ids_separate_from_compact_transfer_layers(self):
        pool = HybridLinearKVPool(
            size=256,
            dtype=torch.bfloat16,
            page_size=64,
            head_num=1,
            head_dim=160,
            full_attention_layer_ids=[21, 25, 29],
            device="cuda",
            mamba_pool=None,
            use_mla=True,
            use_dsa=True,
            kv_lora_rank=128,
            qk_rope_head_dim=32,
            index_head_dim=128,
            kv_cache_dim=160,
            skip_topk_layers=[False, True, False],
            start_layer=20,
        )
        kv, index = pool.full_kv_pool.get_device_pool_infos()
        self.assertEqual(kv.layer_ids, (21, 25, 29))
        self.assertEqual(index.layer_ids, (21, 29))
        self.assertEqual(pool.host_pool_decls()[1].owned_device_layers, (0, 2))
        packed, mapping = pack_host_pool_buffers(
            target=index,
            drafts=(),
            target_model_layer_ids=kv.layer_ids,
        )
        self.assertEqual(mapping, {0: 0, 2: 1})
        self.assertEqual(len(packed.buffers), 2)

    def test_hybrid_preserves_hisparse_pool_constructor(self):
        # Alternate DSA pools retain their existing constructor contract.
        from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool

        pool = HybridLinearKVPool(
            size=256,
            dtype=torch.bfloat16,
            page_size=64,
            head_num=1,
            head_dim=160,
            full_attention_layer_ids=[1, 3],
            device="cuda",
            mamba_pool=None,
            use_mla=True,
            use_dsa=True,
            kv_lora_rank=128,
            qk_rope_head_dim=32,
            index_head_dim=128,
            kv_cache_dim=160,
            full_kv_pool_class=HiSparseDSATokenToKVPool,
        )
        self.assertIsInstance(pool.full_kv_pool, HiSparseDSATokenToKVPool)
        self.assertEqual(pool.get_key_buffer(3).shape, (320, 1, 160))

    def _pool(self, *, ratio, layers, start=0, skip=None, fp8=False):
        return DSATokenToKVPool(
            size=64 * ratio * 4,
            page_size=64 * ratio,
            kv_lora_rank=512 if fp8 else 128,
            dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16,
            qk_rope_head_dim=64 if fp8 else 32,
            layer_num=layers,
            device="cuda",
            enable_memory_saver=False,
            kv_cache_dim=656 if fp8 else 160,
            index_head_dim=128,
            index_kpool=ratio,
            start_layer=start,
            end_layer=start + layers - 1,
            skip_topk_layers=skip,
        )

    def test_subclass_index_hosts_keep_the_legacy_constructor(self):
        class AlternateDSAPool(DSATokenToKVPool):
            pass

        publish(ServerArgs(model_path="dummy", hicache_ratio=2), role="test")
        self.addCleanup(reset_context)
        alternate = AlternateDSAPool(
            size=256,
            page_size=64,
            kv_lora_rank=128,
            dtype=torch.bfloat16,
            qk_rope_head_dim=32,
            layer_num=1,
            device="cuda",
            enable_memory_saver=False,
            kv_cache_dim=160,
            index_head_dim=128,
        )
        for target, drafts in (
            (alternate, ()),
            (self._pool(ratio=1, layers=2), (alternate,)),
        ):
            with self.subTest(target=type(target), draft=bool(drafts)):
                config = prepare_host_pool_config(
                    decls=target.host_pool_decls(),
                    full_layer_mapping={i: i for i in range(target.layer_num)},
                    transfer_layer_id_max=target.layer_num,
                    transfer_page_size=64,
                    packed_draft_device_pools=drafts,
                )
                group = build_host_pool_group(config=config)
                try:
                    self.assertIs(group.get_pool(PoolName.INDEXER).device_pool, target)
                finally:
                    group.destroy()

    def test_assembly_preserves_capacity_and_packed_transfer_routes(self):
        publish(
            ServerArgs(
                model_path="dummy", hicache_ratio=2, hicache_mem_layout="layer_first"
            ),
            role="scheduler",
        )
        self.addCleanup(reset_context)
        pool = self._pool(ratio=4, layers=3, skip=[False, True, False])
        drafts = tuple(self._pool(ratio=4, layers=1) for _ in range(2))
        config = prepare_host_pool_config(
            decls=pool.host_pool_decls(),
            full_layer_mapping={0: 0, 1: 1, 2: 2},
            transfer_layer_id_max=3,
            transfer_page_size=256,
            packed_draft_device_pools=drafts,
        )
        group = build_host_pool_group(config=config)
        for entry in group.entries:
            self.addCleanup(entry.host_pool.destroy)
        kv = group.get_pool(PoolName.KV)
        index = group.get_pool(PoolName.INDEXER)
        self.assertEqual(kv.page_num, 9)
        self.assertEqual(index.page_num, kv.page_num)
        self.assertEqual(index.size, kv.size)
        self.assertEqual(index.layer_num, 4)
        self.assertEqual(index.target_layer_num, 2)
        self.assertEqual(index._live_target_layers, [0, 2])
        self.assertEqual(index.end_layer, 2)
        self.assertEqual(index.indexer_page_stride_size, 64 * 132)
        self.assertIsNone(kv.device_pool)
        self.assertIsNone(index.device_pool)
        self.assertEqual(index._device_to_host_layer, {0: 0, 2: 1, 3: 2, 4: 3})
        entry = next(entry for entry in group.entries if entry.name == PoolName.INDEXER)
        self.assertIsNone(entry.layer_mapper(1))
        self.assertEqual(entry.layer_mapper(2), 2)
        self.assertEqual(entry.layer_mapper(4), 4)

    def test_main_kv_preserves_bytes_and_capacity_without_pool_reads(self):
        for ratio in (1, 4):
            for fp8 in (False, True):
                for layout, backend in (
                    ("layer_first", "kernel"),
                    ("page_first", "kernel"),
                    ("page_first_direct", "direct"),
                ):
                    with self.subTest(ratio=ratio, fp8=fp8, layout=layout):
                        publish(
                            ServerArgs(
                                model_path="dummy",
                                hicache_ratio=2,
                                hicache_mem_layout=layout,
                            ),
                            role="scheduler",
                        )
                        self.addCleanup(reset_context)
                        target = self._pool(ratio=ratio, layers=3, start=20, fp8=fp8)
                        drafts = (self._pool(ratio=ratio, layers=1, fp8=fp8),)
                        host = build_kv_host_pool(
                            kv_pool=target,
                            page_size=target.page_size,
                            mtp_draft_device_pools=drafts,
                            pool_label="draft",
                        )
                        legacy = MLATokenToKVPoolHost(
                            target,
                            2,
                            0,
                            target.page_size,
                            layout,
                            override_kv_cache_dim=target.kv_cache_dim,
                            mtp_draft_device_pools=drafts,
                            pool_label="draft",
                        )
                        try:
                            self.assertIsNone(host.device_pool)
                            self.assertEqual(host.pool_label, legacy.pool_label)
                            self.assertEqual(host.size, legacy.size)
                            self.assertEqual(host.dtype, legacy.dtype)
                            self.assertEqual(
                                host.kv_buffer.shape, legacy.kv_buffer.shape
                            )
                            self.assertEqual(host.start_layer, legacy.start_layer)
                            page_size = target.page_size
                            source = torch.arange(page_size, page_size * 2).cuda()
                            destination = torch.arange(
                                page_size * 3, page_size * 4
                            ).cuda()
                            host_indices = torch.arange(page_size * 5, page_size * 6)
                            buffers = host._buffer_info.buffers
                            expected = []
                            for i, buffer in enumerate(buffers):
                                buffer.copy_(
                                    (
                                        torch.arange(buffer.numel(), device="cuda") % 83
                                        + i
                                    )
                                    .reshape_as(buffer)
                                    .to(buffer.dtype)
                                )
                                wanted = buffer.clone()
                                wanted[destination] = wanted[source]
                                expected.append(wanted)
                            stream = torch.cuda.Stream()
                            stream.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(stream):
                                host.backup_from_device_all_layer(
                                    object(),
                                    host_indices.cuda()
                                    if layout == "layer_first"
                                    else host_indices,
                                    source.cpu() if backend == "direct" else source,
                                    backend,
                                )
                                for layer in range(host.layer_num):
                                    host.load_to_device_per_layer(
                                        object(),
                                        host_indices
                                        if backend == "direct"
                                        else host_indices.cuda(),
                                        destination.cpu()
                                        if backend == "direct"
                                        else destination,
                                        layer,
                                        backend,
                                        is_draft=layer >= 3,
                                    )
                            stream.synchronize()
                            for actual, wanted in zip(buffers, expected):
                                self.assertTrue(torch.equal(actual, wanted))
                        finally:
                            host.destroy()
                            legacy.destroy()

    def test_restore_sparse_layers_and_packed_drafts_without_pool_reads(self):
        for ratio in (1, 4):
            for layout in ("layer_first", "page_first"):
                with self.subTest(ratio=ratio, layout=layout):
                    target = self._pool(
                        ratio=ratio, layers=3, start=20, skip=[False, True, False]
                    )
                    drafts = [self._pool(ratio=ratio, layers=1) for _ in range(2)]
                    target_info = target.get_device_pool_infos()[1]
                    self.assertEqual(target_info.pool_name, PoolName.INDEXER)
                    info, mapping = pack_host_pool_buffers(
                        target=target_info,
                        drafts=tuple(
                            pool.get_device_pool_infos()[1] for pool in drafts
                        ),
                        target_model_layer_ids=(20, 21, 22),
                    )
                    host = DSAIndexerPoolHost.from_buffer_info(
                        info,
                        layer_mapping=mapping,
                        target_device_layer_num=3,
                        num_host_pages=9,
                        layout=layout,
                    )
                    self.addCleanup(host.destroy)
                    expected = []
                    for layer, buffer in enumerate(info.buffers):
                        values = torch.arange(
                            buffer.numel(), device="cuda", dtype=torch.int64
                        ).reshape_as(buffer)
                        buffer.copy_((values + layer * 37).to(torch.uint8))
                        expected.append(buffer.clone())
                    page_size = info.page_size
                    offsets = torch.arange(page_size, device="cuda")
                    source_indices = (
                        torch.tensor([1, 3], device="cuda")[:, None] * page_size
                        + offsets
                    ).flatten()
                    host_indices = (
                        torch.tensor([2, 5], device="cuda")[:, None] * page_size
                        + offsets
                    ).flatten()
                    host.backup_from_device_all_layer(
                        object(),
                        host_indices.cpu() if layout == "page_first" else host_indices,
                        source_indices,
                        "kernel",
                    )
                    torch.cuda.synchronize()
                    for buffer in info.buffers:
                        buffer.fill_(17)
                    for callback in (0, 1, 2, 3, 4):
                        host.load_to_device_per_layer(
                            object(),
                            host_indices,
                            source_indices,
                            callback,
                            "kernel",
                            is_draft=callback >= 3,
                        )
                    torch.cuda.synchronize()
                    for layer, buffer in enumerate(info.buffers):
                        self.assertTrue(
                            torch.equal(buffer[[1, 3]], expected[layer][[1, 3]])
                        )
                        self.assertTrue(torch.all(buffer[[0, 2, 4]] == 17))
                    self.assertIsNone(host.device_pool)
                    self.assertEqual(host.size, 9 * page_size)


if __name__ == "__main__":
    unittest.main()
