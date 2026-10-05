import tempfile
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.mem_cache.mla_host_dedup import (
    MLAHostDedupBroadcaster,
    MLAHostDedupContext,
    MLAHostDedupLayerOwners,
    maybe_create_hicache_mla_host_dedup,
    maybe_create_mla_host_dedup_context,
)
from sglang.srt.mem_cache.pool_host.dsa import (
    DSAIndexerPoolHost,
    make_dsa_indexer_pool_decl,
)
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _device_pool_stub(*, layer_num: int, **fields) -> SimpleNamespace:
    return SimpleNamespace(
        layer_num=layer_num,
        layer_shard_enabled=False,
        **fields,
    )


class _FakeStream:
    pass


class TestMLAHostDedupPrimitives(unittest.TestCase):
    def test_disabled_flag_is_a_noop(self):
        with mock.patch(
            "sglang.srt.mem_cache.mla_host_dedup.mla_host_dedup_eligible"
        ) as eligible:
            context = maybe_create_mla_host_dedup_context(
                object(), object(), None, None, None, enabled=False
            )

        self.assertIsNone(context)
        eligible.assert_not_called()

    def test_dummy_host_pools_keep_allocator_metadata_only(self):
        mla_device_pool = _device_pool_stub(
            layer_num=2,
            store_dtype=torch.float16,
            kv_lora_rank=4,
            qk_rope_head_dim=2,
            size=8,
            start_layer=0,
            end_layer=2,
        )
        mla_host = MLATokenToKVPoolHost(
            mla_device_pool,
            host_to_device_ratio=2,
            host_size=0,
            page_size=2,
            layout="page_first",
            pin_memory=False,
            is_dummy=True,
        )

        self.assertTrue(mla_host._is_dummy)
        self.assertIsNone(mla_host.kv_buffer)
        self.assertIsNone(mla_host.data_ptrs)
        self.assertEqual(mla_host.get_contiguous_buf_infos(), ([], [], []))
        slots = mla_host.alloc(2)
        self.assertEqual(slots.tolist(), [0, 1])
        with self.assertRaisesRegex(AssertionError, "load on a dummy"):
            mla_host.load_to_device_per_layer(
                mla_device_pool, slots, slots, layer_id=0, io_backend="kernel"
            )

        dsa_device_pool = _device_pool_stub(
            layer_num=2,
            store_dtype=torch.float16,
            size=8,
            start_layer=0,
            end_layer=2,
            index_head_dim=8,
            quant_block_size=4,
            skip_topk_layers=[False] * 2,
        )
        indexer_host = DSAIndexerPoolHost(
            decl=make_dsa_indexer_pool_decl(dsa_device_pool),
            anchor_host=mla_host,
            pin_memory=False,
            is_dummy=True,
        )

        self.assertTrue(indexer_host._is_dummy)
        self.assertIsNone(indexer_host.index_k_with_scale_buffer)
        self.assertIsNone(indexer_host.index_k_device_ptrs)
        self.assertEqual(indexer_host.size, mla_host.size)
        with self.assertRaisesRegex(AssertionError, "load on a dummy"):
            indexer_host.load_to_device_per_layer(
                dsa_device_pool, slots, slots, layer_id=0, io_backend="kernel"
            )

    def test_layer_broadcast_reuses_full_staging_capacity(self):
        broadcaster = MLAHostDedupBroadcaster.__new__(MLAHostDedupBroadcaster)
        broadcaster.is_src = True
        broadcaster.src_global_rank = 0
        broadcaster.group = object()

        layer_buffers = [
            torch.arange(24, dtype=torch.float32).reshape(6, 1, 4),
            torch.arange(24, 48, dtype=torch.float32).reshape(6, 1, 4),
        ]
        target = torch.tensor([0, 2, 5], dtype=torch.int64)
        staging = torch.empty(2 * 3 * 4, dtype=torch.float32)

        with mock.patch.object(torch.distributed, "broadcast") as broadcast:
            broadcaster._bcast_layer(layer_buffers, staging, target, 4, layer_id=1)

        broadcast.assert_called_once()
        expected = layer_buffers[1].index_select(0, target)
        torch.testing.assert_close(
            staging[: expected.numel()].view_as(expected), expected
        )

        broadcaster.is_src = False
        received = [torch.zeros_like(layer) for layer in layer_buffers]
        with mock.patch.object(torch.distributed, "broadcast"):
            broadcaster._bcast_layer(received, staging, target, 4, layer_id=1)
        torch.testing.assert_close(received[1].index_select(0, target), expected)

    def test_chunk_tokens_uses_environment(self):
        device_pool = _device_pool_stub(
            layer_num=2,
            device=torch.device("cpu"),
            kv_cache_dim=4,
            kv_buffer=[torch.empty(3, 1, 4), torch.empty(3, 1, 4)],
        )

        with (
            envs.SGLANG_MLA_DEDUP_CHUNK_TOKENS.override(7),
            mock.patch(
                "sglang.srt.mem_cache.mla_host_dedup.mla_dedup_rank_and_size",
                return_value=(0, 2),
            ),
        ):
            broadcaster = MLAHostDedupBroadcaster(
                device_pool, group=object(), src_global_rank=0
            )

        self.assertEqual(broadcaster.chunk_tokens, 7)
        self.assertEqual(broadcaster.kv_staging.numel(), 2 * 7 * 4)

    def test_chunk_tokens_must_be_positive(self):
        device_pool = _device_pool_stub(
            layer_num=2,
            device=torch.device("cpu"),
            kv_cache_dim=4,
            kv_buffer=[torch.empty(3, 1, 4), torch.empty(3, 1, 4)],
        )

        with (
            envs.SGLANG_MLA_DEDUP_CHUNK_TOKENS.override(0),
            mock.patch(
                "sglang.srt.mem_cache.mla_host_dedup.mla_dedup_rank_and_size",
                return_value=(0, 2),
            ),
            self.assertRaisesRegex(ValueError, "must be positive"),
        ):
            MLAHostDedupBroadcaster(device_pool, group=object(), src_global_rank=0)

    def test_build_eagerly_warms_dedicated_nccl_group(self):
        tp_group = object()
        dedicated_group = object()
        device_pool = _device_pool_stub(
            layer_num=2,
            device=torch.device("cpu"),
            kv_cache_dim=4,
            kv_buffer=[torch.empty(3, 1, 4)],
        )

        with (
            mock.patch(
                "sglang.srt.mem_cache.mla_host_dedup.is_dp_attention_enabled",
                return_value=False,
            ),
            mock.patch(
                "sglang.srt.mem_cache.mla_host_dedup.mla_dedup_rank_and_size",
                return_value=(0, 2),
            ),
            mock.patch.object(
                torch.distributed,
                "get_process_group_ranks",
                return_value=[4, 5],
            ),
            mock.patch(
                "sglang.srt.distributed.parallel_state.create_custom_parallel_group",
                return_value=dedicated_group,
            ) as create_group,
            mock.patch.object(torch.distributed, "broadcast") as broadcast,
            mock.patch.object(torch.cuda, "synchronize") as synchronize,
        ):
            broadcaster = MLAHostDedupBroadcaster.build(
                device_pool, tp_group, attn_tp_group=None
            )

        create_group.assert_called_once_with(group_ranks=[4, 5], backend="nccl")
        broadcast.assert_called_once()
        self.assertEqual(broadcast.call_args.args[0].numel(), 1)
        self.assertIs(broadcast.call_args.kwargs["group"], dedicated_group)
        self.assertEqual(broadcast.call_args.kwargs["src"], 4)
        synchronize.assert_called_once_with(device_pool.device)
        self.assertIs(broadcaster.group, dedicated_group)

    def test_indexer_pages_preserve_logical_order(self):
        broadcaster = MLAHostDedupBroadcaster.__new__(MLAHostDedupBroadcaster)
        broadcaster.device = torch.device("cpu")
        broadcaster.device_pool = SimpleNamespace(page_size=4)
        broadcaster.idx_bufs = [object()]

        device_indices = torch.tensor([8, 9, 10, 11, 0, 1, 2, 3])
        prepared_indices, page_indices = broadcaster.prepare_broadcast(
            device_indices, _FakeStream()
        )

        self.assertIs(prepared_indices, device_indices)
        torch.testing.assert_close(page_indices, torch.tensor([2, 0]))

    def test_indexer_rejects_partial_pages(self):
        broadcaster = MLAHostDedupBroadcaster.__new__(MLAHostDedupBroadcaster)
        broadcaster.device = torch.device("cpu")
        broadcaster.device_pool = SimpleNamespace(page_size=4)
        broadcaster.idx_bufs = [object()]

        with self.assertRaisesRegex(ValueError, "page-aligned device indices"):
            broadcaster.prepare_broadcast(torch.arange(7), _FakeStream())

    def test_context_destroys_all_owned_process_groups(self):
        broadcaster = mock.Mock()
        hit_group = object()
        completion_group = object()
        context = MLAHostDedupContext(
            broadcaster=broadcaster,
            prefetch_hits_sync_groups=[hit_group],
            prefetch_completion_sync_groups=[completion_group],
        )

        with mock.patch.object(torch.distributed, "destroy_process_group") as destroy:
            context.destroy()

        broadcaster.destroy.assert_called_once()
        self.assertEqual(
            destroy.call_args_list,
            [mock.call(hit_group), mock.call(completion_group)],
        )
        self.assertIsNone(context.prefetch_hits_sync_groups)
        self.assertIsNone(context.prefetch_completion_sync_groups)


class _DSAPoolStub(DSATokenToKVPool):
    """CPU stand-in with the buffers the broadcaster and host pools read."""

    def __init__(self, *, layer_num=5, skip=(), page_size=2, pages=8, seed=0):
        self.layer_num = layer_num
        self.start_layer, self.end_layer = 0, layer_num
        self.page_size = page_size
        self.size = pages * page_size
        self.device = torch.device("cpu")
        self.layer_shard_enabled = False
        self.store_dtype = torch.uint8
        self.kv_lora_rank, self.qk_rope_head_dim = 4, 2
        self.kv_cache_dim = 6
        self.index_head_dim, self.quant_block_size = 4, 4
        self.skip_topk_layers = [i in skip for i in range(layer_num)]
        g = torch.Generator().manual_seed(seed)
        rows = self.size + page_size
        self.kv_buffer = [
            torch.randint(0, 256, (rows, 1, 6), dtype=torch.uint8, generator=g)
            for _ in range(layer_num)
        ]
        self.data_ptrs = torch.tensor(
            [buf.data_ptr() for buf in self.kv_buffer], dtype=torch.uint64
        )
        page_bytes = page_size * (4 + 4)
        self.index_key_cache = SimpleNamespace(
            buffer=[
                torch.randint(
                    0,
                    256,
                    (0 if s else pages + 1, page_bytes),
                    dtype=torch.uint8,
                    generator=g,
                )
                for s in self.skip_topk_layers
            ]
        )


class _RecordingEvent:
    def __init__(self):
        self.records = 0

    def record(self):
        self.records += 1


def _rotating_broadcaster(device_pool, owners, group_ranks):
    with (
        envs.SGLANG_MLA_DEDUP_CHUNK_TOKENS.override(1),
        mock.patch(
            "sglang.srt.mem_cache.mla_host_dedup.mla_dedup_rank_and_size",
            return_value=(owners.rank, owners.size),
        ),
    ):
        return MLAHostDedupBroadcaster(
            device_pool,
            group=None,
            src_global_rank=group_ranks[0],
            owners=owners,
            group_ranks=group_ranks,
        )


def _layer_hook_restore_worker(rank, world_size, rendezvous):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        skip = (1, 3)
        source = _DSAPoolStub(layer_num=7, skip=skip, seed=1)
        # Each rank holds only the layers it owns; the rest is stale.
        pool = _DSAPoolStub(layer_num=7, skip=skip, seed=100 + rank)
        owners = MLAHostDedupLayerOwners(rank, world_size, pool.layer_num)
        live = [i for i in range(pool.layer_num) if i not in skip]
        for layer in range(pool.layer_num):
            if owners.kv_owner(layer) == rank:
                pool.kv_buffer[layer].copy_(source.kv_buffer[layer])
        for ordinal, layer in enumerate(live):
            if owners.indexer_owner(ordinal) == rank:
                pool.index_k_with_scale_buffer[layer].copy_(
                    source.index_k_with_scale_buffer[layer]
                )
        context = MLAHostDedupContext(
            broadcaster=_rotating_broadcaster(pool, owners, list(range(world_size))),
            prefetch_hits_sync_groups=None,
            prefetch_completion_sync_groups=None,
        )
        pages = torch.tensor([5, 1, 6])
        indices = (pages[:, None] * pool.page_size + torch.arange(2)).flatten()
        finish_event = _RecordingEvent()
        context.start_load(2, indices, finish_event)
        # The forward waits per KV/indexer access: repeats, and layers 1, 4 and 5
        # are never waited for, so later waits must catch them up in order.
        for threshold in (0, 0, 2, 2, 3, 6):
            assert finish_event.records == 0
            context.broadcast_ready_layers(2, threshold)
        assert finish_event.records == 1 and not context.pending_loads
        for layer in range(pool.layer_num):
            torch.testing.assert_close(
                pool.kv_buffer[layer][indices], source.kv_buffer[layer][indices]
            )
            if layer in live:
                torch.testing.assert_close(
                    pool.index_k_with_scale_buffer[layer][pages],
                    source.index_k_with_scale_buffer[layer][pages],
                )
    finally:
        torch.distributed.destroy_process_group()


def _host_pools(owners_by_rank, device_pool, draft_pool=None):
    drafts = () if draft_pool is None else (draft_pool,)
    hosts = []
    for owners in owners_by_rank:
        mla_host = MLATokenToKVPoolHost(
            device_pool,
            host_to_device_ratio=2,
            host_size=1e-5,
            page_size=2,
            layout="page_first_direct",
            pin_memory=False,
            override_kv_cache_dim=6,
            mtp_draft_device_pools=drafts,
            dedup_owners=owners,
        )
        indexer_host = DSAIndexerPoolHost(
            decl=make_dsa_indexer_pool_decl(device_pool),
            anchor_host=mla_host,
            packed_draft_device_pools=drafts,
            pin_memory=False,
        )
        hosts.append((mla_host, indexer_host))
    return hosts


_HOST_ALLOC_PATCHES = (
    mock.patch(
        "sglang.srt.mem_cache.pool_host.common.alloc_mmap",
        side_effect=lambda dims, dtype: torch.zeros(dims, dtype=dtype),
    ),
    mock.patch(
        "sglang.srt.mem_cache.pool_host.base.host_memory_budget_bytes",
        return_value=1 << 30,
    ),
    mock.patch(
        "sglang.srt.mem_cache.pool_host.dsa.host_memory_budget_bytes",
        return_value=1 << 30,
    ),
)


def _with_host_alloc_patches(test):
    for patch in reversed(_HOST_ALLOC_PATCHES):
        test = patch(test)
    return test


class TestMLAHostDedupRotatingOwners(unittest.TestCase):
    def test_owners_rotate_kv_layers_then_indexer_layers(self):
        owners = [MLAHostDedupLayerOwners(rank, 4, 10) for rank in range(4)]

        self.assertEqual(owners[1].owned_kv_layers(), [1, 5, 9])
        self.assertEqual(owners[3].owned_kv_layers(), [3, 7])
        self.assertEqual(
            sorted(layer for o in owners for layer in o.owned_kv_layers()),
            list(range(10)),
        )
        self.assertEqual({o.max_kv_layers for o in owners}, {3})
        # Indexer layers continue after the last KV layer's owner.
        self.assertEqual([owners[0].indexer_owner(i) for i in range(4)], [2, 3, 0, 1])

    def test_rotating_broadcast_sources_each_layer_from_its_owner(self):
        pool = _DSAPoolStub(layer_num=5, skip=(1,))
        owners = MLAHostDedupLayerOwners(rank=2, size=4, kv_layer_num=5)
        broadcaster = _rotating_broadcaster(pool, owners, [8, 9, 10, 11])
        before = [buf.clone() for buf in pool.kv_buffer]
        indices = torch.tensor([2, 3, 6, 7])
        prepared = broadcaster.prepare_broadcast(indices, _FakeStream())

        with mock.patch.object(torch.distributed, "broadcast") as broadcast:
            for layer in range(pool.layer_num):
                broadcaster.broadcast_loaded_layer(layer, prepared)

        sources = [call.kwargs["src"] for call in broadcast.call_args_list]
        # One chunk per buffer. KV layer i comes from rank i % 4; index layers
        # (layer 1 is a placeholder) continue the rotation after the 5 KV layers.
        self.assertEqual(sources, [8, 9, 9, 10, 10, 11, 11, 8, 8])
        # Rank 2 owns KV layer 2: it sends it and keeps its own rows.
        torch.testing.assert_close(pool.kv_buffer[2], before[2])

    def test_index_staging_is_sized_in_pages(self):
        pool = _DSAPoolStub(layer_num=4, page_size=2)
        with (
            envs.SGLANG_MLA_DEDUP_CHUNK_TOKENS.override(6),
            mock.patch(
                "sglang.srt.mem_cache.mla_host_dedup.mla_dedup_rank_and_size",
                return_value=(0, 2),
            ),
        ):
            broadcaster = MLAHostDedupBroadcaster(
                pool, group=object(), src_global_rank=0
            )

        # Both staging buffers cover layer_num * chunk_tokens = 24 tokens.
        self.assertEqual(broadcaster.kv_staging.numel(), 24 * pool.kv_cache_dim)
        self.assertEqual(broadcaster.idx_staging.numel(), 12 * broadcaster.idx_elem)

    def test_build_warms_every_source_when_rotating(self):
        pool = _DSAPoolStub(layer_num=3)
        owners = MLAHostDedupLayerOwners(rank=0, size=3, kv_layer_num=3)
        with (
            mock.patch(
                "sglang.srt.mem_cache.mla_host_dedup.is_dp_attention_enabled",
                return_value=False,
            ),
            mock.patch(
                "sglang.srt.mem_cache.mla_host_dedup.mla_dedup_rank_and_size",
                return_value=(0, 3),
            ),
            mock.patch.object(
                torch.distributed, "get_process_group_ranks", return_value=[4, 5, 6]
            ),
            mock.patch(
                "sglang.srt.distributed.parallel_state.create_custom_parallel_group",
                return_value=object(),
            ),
            mock.patch.object(torch.distributed, "broadcast") as broadcast,
            mock.patch.object(torch.cuda, "synchronize"),
        ):
            broadcaster = MLAHostDedupBroadcaster.build(
                pool, object(), attn_tp_group=None, owners=owners
            )

        self.assertEqual([c.kwargs["src"] for c in broadcast.call_args_list], [4, 5, 6])
        self.assertEqual(broadcaster.group_ranks, [4, 5, 6])
        self.assertIs(broadcaster.owners, owners)

    def test_layer_hook_restores_pages_on_every_rank(self):
        with tempfile.TemporaryDirectory(prefix="mla-dedup-rotate-") as directory:
            torch.multiprocessing.spawn(
                _layer_hook_restore_worker,
                args=(3, f"{directory}/store"),
                nprocs=3,
                join=True,
            )

    @_with_host_alloc_patches
    def test_rotating_host_pools_map_only_owned_layers(self, *_):
        pool = _DSAPoolStub(layer_num=5, skip=(1,))
        draft = _DSAPoolStub(layer_num=1)
        owners = [MLAHostDedupLayerOwners(rank, 2, pool.layer_num) for rank in range(2)]
        (kv0, idx0), (kv1, idx1) = _host_pools(owners, pool, draft)

        # Host layers: the owned target layers, then the packed draft layer.
        self.assertEqual((kv0.kv_buffer.shape[1], kv1.kv_buffer.shape[1]), (4, 3))
        # Index layers 0, 2, 3, 4 rotate from rank 5 % 2 = 1.
        self.assertEqual(
            (idx0._live_target_layers, idx1._live_target_layers), ([2, 4], [0, 3])
        )
        self.assertTrue(kv1._is_device_layer_owned(pool, 3))
        self.assertFalse(kv1._is_device_layer_owned(pool, 2))

        loads = []
        with mock.patch(
            "sglang.srt.mem_cache.pool_host.mla.transfer_kv_per_layer_direct_pf_lf",
            side_effect=lambda **kw: loads.append(kw["layer_id"]),
            create=True,
        ):
            slots = torch.arange(2)
            for layer in range(pool.layer_num):
                kv1.load_to_device_per_layer(pool, slots, slots, layer, "direct")
            kv1.load_to_device_per_layer(
                draft, slots, slots, pool.layer_num, "direct", is_draft=True
            )
        # Device layers 1 and 3, then the draft, read host layers 0, 1 and 2.
        self.assertEqual(loads, [0, 1, 2])

    @_with_host_alloc_patches
    def test_rotating_host_capacity_is_sized_by_largest_share(self, *_):
        pool = _DSAPoolStub(layer_num=5)
        draft = _DSAPoolStub(layer_num=1)
        owners = [MLAHostDedupLayerOwners(rank, 2, pool.layer_num) for rank in range(2)]
        (kv0, idx0), (kv1, idx1) = _host_pools(owners, pool, draft)
        ((unsharded, _),) = _host_pools([None], pool, draft)

        # Ranks own 3 and 2 KV layers; both size by the larger share plus the
        # draft, so every rank's host pool holds the same tokens.
        self.assertEqual({kv0.size_per_token, kv1.size_per_token}, {(3 + 1) * 6})
        self.assertEqual(kv0.size, kv1.size)
        self.assertEqual((idx0.size, idx1.size), (kv0.size, kv1.size))
        self.assertEqual(unsharded.size_per_token, (5 + 1) * 6)
        self.assertGreater(kv0.size, unsharded.size)

    def test_dp_attention_rotates_over_the_attention_tp_group(self):
        pool = _DSAPoolStub(layer_num=6)
        attn_tp_group, tp_group = object(), object()
        memory = SimpleNamespace(
            hicache_io_backend="direct",
            hicache_mem_layout="page_first_direct",
            hicache_storage_backend=None,
            enable_hisparse=False,
        )
        parallel = SimpleNamespace(
            tp_rank=5,
            tp_size=8,
            attn_tp_rank=1,
            attn_tp_size=4,
            attn_cp_size=1,
            dcp_enabled=False,
            pp_size=1,
            nnodes=1,
        )
        group_ranks = {id(tp_group): list(range(8)), id(attn_tp_group): [4, 5, 6, 7]}
        module = "sglang.srt.mem_cache.mla_host_dedup"
        with (
            mock.patch(f"{module}.get_memory", return_value=memory),
            mock.patch(f"{module}.get_parallel", return_value=parallel),
            mock.patch(
                f"{module}.get_disagg",
                return_value=SimpleNamespace(disaggregation_mode="null"),
            ),
            mock.patch(f"{module}.is_dp_attention_enabled", return_value=True),
            mock.patch(f"{module}.is_cuda", return_value=True),
            mock.patch.object(
                torch.distributed,
                "get_process_group_ranks",
                side_effect=lambda group: group_ranks[id(group)],
            ),
            mock.patch(
                "sglang.srt.distributed.parallel_state.create_custom_parallel_group",
                return_value=object(),
            ),
            mock.patch.object(torch.distributed, "broadcast") as broadcast,
            mock.patch.object(torch.cuda, "synchronize"),
        ):
            context = maybe_create_hicache_mla_host_dedup(
                pool,
                SimpleNamespace(
                    tp_cache_group=tp_group,
                    attn_cp_cache_group=None,
                    attn_tp_cache_group=attn_tp_group,
                ),
                enabled=True,
            )

        self.assertEqual(context.owners, MLAHostDedupLayerOwners(1, 4, 6))
        self.assertEqual(context.broadcaster.group_ranks, [4, 5, 6, 7])
        self.assertEqual(
            [c.kwargs["src"] for c in broadcast.call_args_list], [4, 5, 6, 7]
        )
        self.assertFalse(context.is_dummy_rank)


if __name__ == "__main__":
    unittest.main()
