"""CPU regressions for hybrid DSA index ownership and HiCache assembly."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _build_hybrid_dsa_index_entry,
    _MambaStrategy,
    build_hybrid_mamba_stack,
    build_pool_entry,
)
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    HybridLinearKVPool,
    HybridReqToTokenPool,
)
from sglang.srt.mem_cache.pool_host import HostPoolGroup
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType, MambaComponent
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    _compressed_index_tree_params,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def dsa_pool(*, live_layers=(True, True), compress=True, kpool=4):
    pool = object.__new__(DSATokenToKVPool)
    pool.page_size = 64
    pool.index_kpool = kpool
    pool.kpool_use_compress = compress
    pool.layer_num = len(live_layers)
    pool.index_key_cache = SimpleNamespace(
        buffer=[
            torch.zeros((8 if live else 0, 64 * 132), dtype=torch.uint8)
            for live in live_layers
        ]
    )
    return pool


def cache_params(pool, *, page_size=64):
    hybrid = object.__new__(HybridLinearKVPool)
    hybrid.full_kv_pool = pool
    return CacheInitParams(
        disable=False,
        req_to_token_pool=SimpleNamespace(),
        token_to_kv_pool_allocator=SimpleNamespace(
            get_kvcache=lambda: hybrid,
            device=torch.device("cpu"),
            size=2048,
            free=MagicMock(),
            free_segment=MagicMock(),
        ),
        page_size=page_size,
        tree_components=(ComponentType.FULL,),
    )


class TestCompressedIndexOwnership(unittest.TestCase):
    def setUp(self):
        server_args = ServerArgs(model_path="dummy")
        server_args._mamba_cache_chunk_size = 64
        publish(server_args, role="scheduler")
        self.addCleanup(reset_context)

    def test_tree_alignment_does_not_change_transfer_params(self):
        params = cache_params(dsa_pool())
        tree_params = _compressed_index_tree_params(params)
        self.assertEqual(tree_params.page_size, 256)
        self.assertEqual(params.page_size, 64)
        self.assertIs(
            tree_params.token_to_kv_pool_allocator, params.token_to_kv_pool_allocator
        )
        self.assertEqual(
            _compressed_index_tree_params(
                cache_params(dsa_pool(), page_size=512)
            ).page_size,
            512,
        )

    def test_uncompressed_hybrid_is_unchanged(self):
        params = cache_params(dsa_pool(compress=False))
        self.assertIs(_compressed_index_tree_params(params), params)
        params = cache_params(SimpleNamespace())
        self.assertIs(_compressed_index_tree_params(params), params)

    def test_mamba_checkpoint_grid_follows_index_ownership_not_transfer_pages(self):
        params = cache_params(dsa_pool())
        params.req_to_token_pool = object.__new__(HybridReqToTokenPool)
        params.enable_mamba_extra_buffer = True
        tree_params = _compressed_index_tree_params(params)
        component = MambaComponent(MagicMock(), tree_params)
        self.assertEqual(component.mamba_cache_chunk_size, 64)
        self.assertEqual(component.mamba_checkpoint_grid, 256)
        self.assertEqual(params.page_size, 64)

    def test_branch_inside_group_cannot_share_first_index_page(self):
        for common in (64, 128, 192, 320, 384, 448):
            with self.subTest(common=common):
                cache = UnifiedRadixCache(cache_params(dsa_pool()))
                a = [1] * 512
                cache.insert(InsertParams(key=RadixKey(a), value=torch.arange(64, 576)))
                b = [1] * common + [2] * (512 - common)
                match = cache.match_prefix(MatchPrefixParams(key=RadixKey(b)))
                self.assertEqual(len(match.device_indices), common // 256 * 256)
                # Neither matching nor inserting the branch may create a node
                # that divides one compressed index row between two children.
                cache.insert(
                    InsertParams(key=RadixKey(b), value=torch.arange(640, 1152))
                )
                for node in cache.tree_core._node_arena.values():
                    self.assertEqual(len(node.key) % 256, 0)

    def test_partial_group_is_not_published_then_extended_in_place(self):
        cache = UnifiedRadixCache(cache_params(dsa_pool()))
        for length in (64, 128, 192):
            key = RadixKey([1] * length)
            cache.insert(InsertParams(key=key, value=torch.arange(64, 64 + length)))
            match = cache.match_prefix(MatchPrefixParams(key=key))
            self.assertEqual(len(match.device_indices), 0)
        cache.insert(InsertParams(key=RadixKey([1] * 256), value=torch.arange(64, 320)))
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey([1] * 320)))
        self.assertEqual(len(match.device_indices), 256)

    def test_runtime_storage_attach_rejects_different_hash_page_size(self):
        cache = UnifiedRadixCache(cache_params(dsa_pool()))
        cache.cache_controller = SimpleNamespace(page_size=64)
        cache._storage_attachment = MagicMock()
        success, message = cache.attach_storage_backend("file")
        self.assertFalse(success)
        self.assertIn("L2 HiCache only", message)
        cache._storage_attachment.attach.assert_not_called()

    def test_external_linker_rejects_different_hash_page_size(self):
        cache = UnifiedRadixCache(cache_params(dsa_pool()))
        with self.assertRaisesRegex(ValueError, "external cache linker"):
            cache.init_cache_linker(MagicMock())

    def test_startup_storage_rejects_different_hash_page_size(self):
        params = cache_params(dsa_pool())
        cache = UnifiedRadixCache(params)
        with (
            patch(
                "sglang.srt.mem_cache.unified_radix_cache.get_memory",
                return_value=SimpleNamespace(
                    hicache_host_memory_mode="cache", hicache_storage_backend="file"
                ),
            ),
            self.assertRaisesRegex(ValueError, "L2 HiCache only"),
        ):
            cache.init_hicache(ServerArgs(model_path="dummy"), params)


class TestHybridIndexSidecar(unittest.TestCase):
    def setUp(self):
        self.prefix = "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
        self.memory = SimpleNamespace(hicache_mem_layout="layer_first")

    def build_entry(self, pool, *, drafts=(), mapping=None, layer_num=6):
        if mapping is None:
            mapping = {0: 0, 3: 1}
        with (
            patch(self.prefix + "get_memory", return_value=self.memory),
            patch(self.prefix + "_get_allocator_type", return_value="default"),
            patch(self.prefix + "DeepSeekV4PagedHostPool") as host_cls,
        ):
            entry = _build_hybrid_dsa_index_entry(
                kv_pool=pool,
                kv_host_pool=SimpleNamespace(page_num=16),
                layer_mapping=mapping,
                transfer_layer_num=layer_num + len(drafts),
                draft_pools=drafts,
            )
        return entry, host_cls

    def test_non_dsa_has_no_sidecar(self):
        entry, host_cls = self.build_entry(SimpleNamespace())
        self.assertIsNone(entry)
        host_cls.assert_not_called()

    def test_real_stack_registers_indexer_within_fixed_host_budget(self):
        pool = dsa_pool()
        draft = dsa_pool(live_layers=(True,))
        for item in (pool, draft):
            item.kv_cache_dim = 656
            item.store_dtype = torch.uint8
        wrapper = object.__new__(HybridLinearKVPool)
        wrapper.full_kv_pool = draft
        params = cache_params(pool)
        params.req_to_token_pool.mamba_allocator = MagicMock()
        params.mtp_draft_device_pools = (wrapper,)
        memory = SimpleNamespace(
            hicache_size=32,
            hicache_ratio=2.0,
            hicache_mem_layout="layer_first",
            hicache_write_policy="write_through",
            hicache_io_backend="direct",
            hicache_host_memory_mode=None,
        )
        with (
            patch(self.prefix + "get_memory", return_value=memory),
            patch(self.prefix + "_get_allocator_type", return_value="default"),
            patch(self.prefix + "_split_hicache_size", return_value=(20.0, 12.0)),
            patch(self.prefix + "build_kv_host_pool") as build_kv,
            patch(self.prefix + "MambaPoolHost"),
            patch(self.prefix + "DeepSeekV4PagedHostPool"),
            patch(self.prefix + "HybridCacheController"),
        ):
            build_kv.return_value.page_num = 16
            group, _ = build_hybrid_mamba_stack(
                params=params,
                kv_pool=pool,
                mamba_pool=MagicMock(),
                full_layer_mapping={0: 0, 3: 1},
                mamba_layer_mapping={1: 0, 2: 1},
                load_cache_event=None,
                storage_backend=None,
                use_mla=True,
            )
        self.assertIn(PoolName.INDEXER, group.entry_map)
        self.assertEqual(build_kv.call_args.kwargs["page_size"], 64)
        kv_budget = build_kv.call_args.kwargs["host_size"]
        self.assertAlmostEqual(kv_budget, 20.0 * 656 / (656 + 132))
        self.assertAlmostEqual(kv_budget * (1 + 132 / 656), 20.0)
        self.assertEqual(group.get_entry(PoolName.INDEXER).layer_mapper(4), 2)

    def test_complete_group_roundtrip_with_noncontiguous_pages(self):
        # Exercise the real host pool and physical-page index conversion. Only
        # the CUDA copy primitive is replaced by an equivalent CPU byte copy.
        def allocate(shape, *, dtype, device, **_):
            return torch.empty(shape, dtype=dtype, device=device)

        def copy_pages(*, src_layers, dst_layers, src_indices, dst_indices, page_size):
            self.assertEqual(page_size, 1)
            for src, dst in zip(src_layers, dst_layers, strict=True):
                dst[dst_indices] = src[src_indices]

        from collections import defaultdict

        from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
            compute_pooled_write_locs,
        )

        pool = dsa_pool()
        draft = dsa_pool(live_layers=(True,))
        with (
            patch(self.prefix + "get_memory", return_value=self.memory),
            patch(self.prefix + "_get_allocator_type", return_value="default"),
            patch(
                "sglang.srt.mem_cache.memory_pool_host.ALLOC_MEMORY_FUNCS",
                defaultdict(lambda: allocate),
            ),
        ):
            entry = _build_hybrid_dsa_index_entry(
                kv_pool=pool,
                kv_host_pool=SimpleNamespace(page_num=16),
                layer_mapping={0: 0, 3: 1, 6: 2},
                transfer_layer_num=7,
                draft_pools=(draft,),
            )
        host = entry.host_pool
        originals = []
        for layer, buffer in enumerate(host.device_buffers):
            buffer.copy_(torch.arange(buffer.numel()).reshape_as(buffer) + layer)
            originals.append(buffer.clone())
        src_pages = torch.tensor([5, 1, 7, 3, 0, 6, 2, 4])
        host_pages = torch.tensor([9, 4, 13, 1, 8, 14, 3, 7])
        dst_pages = torch.tensor([2, 6, 4, 0, 7, 3, 5, 1])

        def tokens(pages):
            return (pages[:, None] * 64 + torch.arange(64)[None, :]).flatten()

        with patch(
            "sglang.srt.mem_cache.memory_pool_host.transfer_kv_direct",
            side_effect=copy_pages,
            create=True,
        ):
            # Backup prefix, then its extension without revisiting the prefix.
            for start in (0, 4):
                host.backup_from_device_all_layer(
                    pool,
                    tokens(host_pages[start : start + 4]),
                    tokens(src_pages[start : start + 4]),
                    "direct",
                )
            for buffer in host.device_buffers:
                buffer.fill_(173)
            for layer in range(len(host.device_buffers)):
                host.load_to_device_per_layer(
                    pool, tokens(host_pages), tokens(dst_pages), layer, "direct"
                )
                self.assertTrue(
                    torch.equal(
                        host.device_buffers[layer][dst_pages],
                        originals[layer][src_pages],
                    )
                )
            # The pooled-key reader selects each restored group's first page.
            locs = compute_pooled_write_locs(dst_pages, torch.arange(128), 4)
            self.assertTrue(torch.equal(locs[:64], 2 * 64 + torch.arange(64)))
            self.assertTrue(torch.equal(locs[64:], 7 * 64 + torch.arange(64)))

    def test_skipped_target_layers_and_packed_draft_are_layer_mapped(self):
        pool = dsa_pool(live_layers=(True, False, True))
        draft = dsa_pool(live_layers=(True,))
        entry, host_cls = self.build_entry(
            pool, drafts=(draft,), mapping={0: 0, 2: 1, 4: 2, 6: 3}
        )
        buffers = host_cls.call_args.kwargs["device_buffers"]
        self.assertEqual(len(buffers), 3)
        self.assertIs(buffers[0], pool.index_k_with_scale_buffer[0])
        self.assertIs(buffers[1], pool.index_k_with_scale_buffer[2])
        self.assertIs(buffers[2], draft.index_k_with_scale_buffer[0])
        self.assertEqual(entry.layer_mapper(0), 0)
        self.assertIsNone(entry.layer_mapper(2))
        self.assertEqual(entry.layer_mapper(4), 1)
        self.assertEqual(entry.layer_mapper(6), 2)
        self.assertEqual(host_cls.call_args.kwargs["slot_page_size"], 64)
        self.assertTrue(host_cls.call_args.kwargs["page_aligned_only"])

        # Exercise the actual load-transfer builder: the draft's index must
        # restore on its first transfer layer, not the target's last layer.
        anchor = build_pool_entry(
            name=PoolName.KV,
            host_pool=MagicMock(layout="layer_first"),
            device_pool=pool,
            layer_mapping={0: 0},
            transfer_layer_num=6,
            is_anchor=True,
        )
        controller = object.__new__(HybridCacheController)
        controller.mem_pool_host = HostPoolGroup([anchor, entry])
        controller.layer_num = 6
        indices = torch.arange(64)
        transfers = controller._l2_load_transfers(
            indices,
            indices,
            [
                PoolTransfer(
                    PoolName.INDEXER, host_indices=indices, device_indices=indices
                )
            ],
        )
        draft_transfers = [t for t in transfers if t.is_draft]
        self.assertEqual(len(draft_transfers), 1)
        self.assertEqual(draft_transfers[0].layer_mapper(0), 2)
        self.assertIsNone(draft_transfers[0].layer_mapper(1))

    def test_strategy_registers_the_index_transfer_spec(self):
        pool = dsa_pool()
        entry, _ = self.build_entry(pool)
        group = MagicMock()
        group.entry_map = {PoolName.INDEXER: entry}
        params = MagicMock()
        params.req_to_token_pool.mamba_map = {1: 0, 2: 1}
        kvcache = SimpleNamespace(
            full_attention_layer_id_mapping={0: 0, 3: 1},
            full_kv_pool=pool,
            use_mla=True,
        )
        with patch(
            self.prefix + "build_hybrid_mamba_stack", return_value=(group, MagicMock())
        ):
            result = _MambaStrategy().build(
                cache=MagicMock(),
                kvcache=kvcache,
                params=params,
                server_args=MagicMock(),
                load_cache_event=None,
            )
        self.assertEqual(len(result.sidecars), 1)
        self.assertEqual(result.sidecars[0].pool_name, PoolName.INDEXER)
        self.assertEqual(result.sidecars[0].indices_from_pool, PoolName.KV)


if __name__ == "__main__":
    unittest.main()
