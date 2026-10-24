import tempfile
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.mla_host_dedup import (
    MLAHostDedupBroadcaster,
    MLAHostDedupContext,
    maybe_create_mla_host_dedup_context,
)
from sglang.srt.mem_cache.pool_host.dsa import (
    DSAIndexerPoolHost,
    make_dsa_indexer_pool_decl,
)
from sglang.srt.mem_cache.pool_host.mla import MLAHostAllocation, MLATokenToKVPoolHost
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
            MLAHostAllocation(
                dtype=mla_device_pool.store_dtype,
                kv_lora_rank=mla_device_pool.kv_lora_rank,
                qk_rope_head_dim=mla_device_pool.qk_rope_head_dim,
                kv_cache_dim=(
                    mla_device_pool.kv_lora_rank + mla_device_pool.qk_rope_head_dim
                ),
                size_per_token=(
                    mla_device_pool.kv_lora_rank + mla_device_pool.qk_rope_head_dim
                )
                * mla_device_pool.store_dtype.itemsize
                * (mla_device_pool.layer_num),
            ),
            host_to_device_ratio=2,
            host_size=0,
            page_size=2,
            layout="page_first",
            pin_memory=False,
            device_capacity_tokens=mla_device_pool.size,
            start_layer=mla_device_pool.start_layer,
            end_layer=mla_device_pool.end_layer,
            target_layer_num=mla_device_pool.layer_num,
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
            page_size=2,
            index_page_size=2,
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

    def test_dummy_and_physical_hosts_share_capacity_geometry(self):
        """Dedup peers must allocate identical IDs despite different host payloads."""
        for dcp_size in (1, 2):
            for host_size in (0, 0.000002):
                with self.subTest(dcp_size=dcp_size, host_size=host_size):
                    pool = MLATokenToKVPool(
                        size=8,
                        page_size=2,
                        layer_num=2,
                        dtype=torch.float16,
                        kv_lora_rank=4,
                        qk_rope_head_dim=2,
                        start_layer=7,
                        end_layer=9,
                        device="cpu",
                        enable_memory_saver=False,
                    )
                    pool.host_capacity_tokens = 16
                    kwargs = dict(
                        host_to_device_ratio=2,
                        host_size=host_size,
                        page_size=4,
                        layout="page_first",
                        pin_memory=False,
                        dcp_size=dcp_size,
                    )
                    host = MLATokenToKVPoolHost(
                        pool.get_device_pool_infos()[0].buffer_info,
                        device_capacity_tokens=(
                            pool.size
                            if pool.host_capacity_tokens is None
                            else pool.host_capacity_tokens
                        ),
                        start_layer=pool.start_layer,
                        end_layer=pool.end_layer,
                        target_layer_num=(
                            (pool.layer_num + pool.layer_shard_size - 1)
                            // pool.layer_shard_size
                            if pool.layer_shard_enabled
                            else pool.layer_num
                        ),
                        layer_shard_enabled=pool.layer_shard_enabled,
                        **kwargs,
                    )
                    dummy = MLATokenToKVPoolHost(
                        MLAHostAllocation(
                            dtype=pool.store_dtype,
                            kv_lora_rank=pool.kv_lora_rank,
                            qk_rope_head_dim=pool.qk_rope_head_dim,
                            kv_cache_dim=(pool.kv_lora_rank + pool.qk_rope_head_dim),
                            size_per_token=(pool.kv_lora_rank + pool.qk_rope_head_dim)
                            * pool.store_dtype.itemsize
                            * (pool.layer_num),
                        ),
                        device_capacity_tokens=pool.host_capacity_tokens,
                        start_layer=pool.start_layer,
                        end_layer=pool.end_layer,
                        target_layer_num=pool.layer_num,
                        **kwargs,
                    )
                    self.addCleanup(host.destroy)
                    self.addCleanup(dummy.destroy)
                    self.assertEqual(
                        (host.size, host.page_num, host.page_size, host.size_per_token),
                        (
                            dummy.size,
                            dummy.page_num,
                            dummy.page_size,
                            dummy.size_per_token,
                        ),
                    )
                    self.assertIsNone(dummy.kv_buffer)
                    count = host.logical_size // 4 * 4
                    self.assertEqual(
                        host.alloc(count).tolist(), dummy.alloc(count).tolist()
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


_MODULE = "sglang.srt.mem_cache.mla_host_dedup"


class _DSAPoolStub(DSATokenToKVPool):
    """CPU stand-in with the buffers the broadcaster and host pools read."""

    def __init__(self, *, layer_num=5, skip=(), page_size=2, pages=8, seed=0):
        self.layer_num = layer_num
        self.start_layer, self.end_layer = 0, layer_num
        self.page_size = page_size
        self.index_page_size = page_size
        self.size = pages * page_size
        self.device = torch.device("cpu")
        self.layer_shard_enabled = False
        self.store_dtype = torch.uint8
        self.dsa_kv_cache_store_fp8 = False
        self.kv_scale_buffer = None
        self.kv_lora_rank, self.qk_rope_head_dim = 4, 2
        self.kv_cache_dim = 6
        self.index_head_dim, self.quant_block_size = 4, 4
        self.skip_topk_layers = [i in skip for i in range(layer_num)]
        g = torch.Generator().manual_seed(seed)
        self.kv_buffer = [
            torch.randint(
                0, 256, (self.size + page_size, 1, 6), dtype=torch.uint8, generator=g
            )
            for _ in range(layer_num)
        ]
        self.data_ptrs = torch.tensor(
            [buf.data_ptr() for buf in self.kv_buffer], dtype=torch.uint64
        )
        self.index_key_cache = SimpleNamespace(
            buffer=[
                torch.randint(
                    0,
                    256,
                    (0 if s else pages + 1, page_size * 8),
                    dtype=torch.uint8,
                    generator=g,
                )
                for s in self.skip_topk_layers
            ]
        )


def _rotating_broadcaster(device_pool, rank, size):
    with (
        envs.SGLANG_MLA_DEDUP_CHUNK_TOKENS.override(1),
        mock.patch(f"{_MODULE}.mla_dedup_rank_and_size", return_value=(rank, size)),
    ):
        return MLAHostDedupBroadcaster(
            device_pool, group=None, src_global_rank=0, rotate_owners=True
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
        for layer in range(rank, pool.layer_num, world_size):
            pool.kv_buffer[layer].copy_(source.kv_buffer[layer])
        live = [i for i in range(pool.layer_num) if i not in skip]
        for ordinal, layer in enumerate(live):
            if (pool.layer_num + ordinal) % world_size == rank:
                pool.index_k_with_scale_buffer[layer].copy_(
                    source.index_k_with_scale_buffer[layer]
                )
        context = MLAHostDedupContext(
            broadcaster=_rotating_broadcaster(pool, rank, world_size),
            prefetch_hits_sync_groups=None,
            prefetch_completion_sync_groups=None,
        )
        pages = torch.tensor([5, 1, 6])
        indices = (pages[:, None] * pool.page_size + torch.arange(2)).flatten()
        finish_event = mock.Mock()
        context.start_load(index=2, device_indices=indices, finish_event=finish_event)
        # The forward waits per KV/indexer access: repeats, and layers 1, 4 and 5
        # are never waited for, so later waits must catch them up in order.
        for threshold in (0, 0, 2, 2, 3, 6):
            assert finish_event.record.call_count == 0
            context.broadcast_ready_layers(2, threshold)
        assert finish_event.record.call_count == 1 and not context.pending_loads
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


class TestMLAHostDedupRotatingOwners(unittest.TestCase):
    def test_rotating_broadcast_sources_each_layer_from_its_owner(self):
        pool = _DSAPoolStub(layer_num=5, skip=(1,))
        broadcaster = _rotating_broadcaster(pool, rank=2, size=4)
        before = [buf.clone() for buf in pool.kv_buffer]
        prepared = broadcaster.prepare_broadcast(
            torch.tensor([2, 3, 6, 7]), _FakeStream()
        )

        with mock.patch.object(torch.distributed, "broadcast") as broadcast:
            for layer in range(pool.layer_num):
                broadcaster.broadcast_loaded_layer(layer, prepared)

        sources = [call.kwargs["group_src"] for call in broadcast.call_args_list]
        # One chunk per buffer. KV layer i comes from rank i % 4; index layers
        # (layer 1 is a placeholder) continue the rotation after the 5 KV layers.
        self.assertEqual(sources, [0, 1, 1, 2, 2, 3, 3, 0, 0])
        # Rank 2 owns KV layer 2: it sends it and keeps its own rows.
        torch.testing.assert_close(pool.kv_buffer[2], before[2])

    def test_index_staging_is_sized_in_pages(self):
        pool = _DSAPoolStub(layer_num=4, page_size=2)
        with (
            envs.SGLANG_MLA_DEDUP_CHUNK_TOKENS.override(6),
            mock.patch(f"{_MODULE}.mla_dedup_rank_and_size", return_value=(0, 2)),
        ):
            broadcaster = MLAHostDedupBroadcaster(
                pool, group=object(), src_global_rank=0
            )

        # Both staging buffers cover layer_num * chunk_tokens = 24 tokens.
        self.assertEqual(broadcaster.kv_staging.numel(), 24 * pool.kv_cache_dim)
        self.assertEqual(broadcaster.idx_staging.numel(), 12 * broadcaster.idx_elem)

    def test_layer_hook_restores_pages_on_every_rank(self):
        with tempfile.TemporaryDirectory(prefix="mla-dedup-rotate-") as directory:
            torch.multiprocessing.spawn(
                _layer_hook_restore_worker,
                args=(3, f"{directory}/store"),
                nprocs=3,
                join=True,
            )

    def test_dp_attention_rotates_over_the_attention_tp_group(self):
        attn_tp_group, tp_group = object(), object()
        parallel = SimpleNamespace(tp_rank=5, tp_size=8, attn_tp_rank=1, attn_tp_size=4)
        group_ranks = {id(tp_group): list(range(8)), id(attn_tp_group): [4, 5, 6, 7]}
        with (
            mock.patch(f"{_MODULE}.get_parallel", return_value=parallel),
            mock.patch(f"{_MODULE}.is_dp_attention_enabled", return_value=True),
            mock.patch(f"{_MODULE}.is_cuda", return_value=True),
            mock.patch.object(
                torch.distributed,
                "get_process_group_ranks",
                side_effect=lambda group: group_ranks[id(group)],
            ),
            mock.patch(
                "sglang.srt.distributed.parallel_state.create_custom_parallel_group",
                return_value=object(),
            ) as create_group,
            mock.patch.object(torch.distributed, "broadcast"),
            mock.patch.object(torch.cuda, "synchronize"),
        ):
            context = maybe_create_mla_host_dedup_context(
                _DSAPoolStub(layer_num=6),
                tp_group,
                None,
                attn_tp_group,
                storage_backend=None,
                enabled=True,
                rotate_owners=True,
            )

        create_group.assert_called_once_with(group_ranks=[4, 5, 6, 7], backend="nccl")
        broadcaster = context.broadcaster
        self.assertEqual((broadcaster.dedup_rank, broadcaster.owner_size), (1, 4))
        self.assertFalse(context.is_dummy_rank)


@mock.patch(
    "sglang.srt.mem_cache.pool_host.common.alloc_mmap",
    side_effect=lambda dims, dtype: torch.zeros(dims, dtype=dtype),
)
@mock.patch(
    "sglang.srt.mem_cache.pool_host.base.host_memory_budget_bytes",
    return_value=1 << 30,
)
@mock.patch(
    "sglang.srt.mem_cache.pool_host.dsa.host_memory_budget_bytes",
    return_value=1 << 30,
)
class TestMLAHostDedupRotatingHostPools(unittest.TestCase):
    @staticmethod
    def _host_pools(pool, draft, rank, size):
        # Mirrors build_kv_host_pool's MLA kwargs.
        mla_host = MLATokenToKVPoolHost(
            pool._mla_buffer_info(),
            host_to_device_ratio=2,
            host_size=1e-5,
            page_size=2,
            layout="page_first_direct",
            pin_memory=False,
            override_kv_cache_dim=6,
            draft_buffer_infos=(draft._mla_buffer_info(),),
            device_capacity_tokens=pool.size,
            start_layer=pool.start_layer,
            end_layer=pool.end_layer,
            target_layer_num=pool.layer_num,
            dedup_rank=rank,
            dedup_size=size,
        )
        indexer_host = DSAIndexerPoolHost(
            decl=make_dsa_indexer_pool_decl(pool),
            anchor_host=mla_host,
            packed_draft_device_pools=(draft,),
            pin_memory=False,
        )
        return mla_host, indexer_host

    def test_host_pools_map_only_owned_layers(self, *_):
        pool = _DSAPoolStub(layer_num=5, skip=(1,))
        draft = _DSAPoolStub(layer_num=1)
        kv0, idx0 = self._host_pools(pool, draft, rank=0, size=2)
        kv1, idx1 = self._host_pools(pool, draft, rank=1, size=2)

        # Host layers: the owned target layers, then the packed draft layer.
        self.assertEqual((kv0.kv_buffer.shape[1], kv1.kv_buffer.shape[1]), (4, 3))
        # Index layers 0, 2, 3, 4 rotate from rank 5 % 2 = 1.
        self.assertEqual(
            (idx0._live_target_layers, idx1._live_target_layers), ([2, 4], [0, 3])
        )
        self.assertEqual(
            (kv0._target_layer_indices, kv1._target_layer_indices), ((0, 2, 4), (1, 3))
        )
        # The all-layer backup reads the owned target layers, then the draft.
        self.assertEqual(
            [t.data_ptr() for t in kv1.device_kv_buffers],
            [
                t.data_ptr()
                for t in (pool.kv_buffer[1], pool.kv_buffer[3], draft.kv_buffer[0])
            ],
        )
        self.assertEqual(
            kv1.device_data_ptrs.tolist(), [t.data_ptr() for t in kv1.device_kv_buffers]
        )

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

    def test_host_capacity_is_sized_by_largest_share(self, *_):
        pool = _DSAPoolStub(layer_num=5)
        draft = _DSAPoolStub(layer_num=1)
        kv0, idx0 = self._host_pools(pool, draft, rank=0, size=2)
        kv1, idx1 = self._host_pools(pool, draft, rank=1, size=2)
        unsharded, _ = self._host_pools(pool, draft, rank=0, size=1)

        # Ranks own 3 and 2 KV layers; both size by the larger share plus the
        # draft, so every rank's host pool holds the same tokens.
        self.assertEqual({kv0.size_per_token, kv1.size_per_token}, {(3 + 1) * 6})
        self.assertEqual(kv0.size, kv1.size)
        self.assertEqual((idx0.size, idx1.size), (kv0.size, kv1.size))
        self.assertEqual(unsharded.size_per_token, (5 + 1) * 6)
        self.assertGreater(kv0.size, unsharded.size)


if __name__ == "__main__":
    unittest.main()
