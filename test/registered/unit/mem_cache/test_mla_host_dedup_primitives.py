import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
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
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


def _device_pool_stub(*, layer_num: int, **fields) -> SimpleNamespace:
    return SimpleNamespace(
        layer_num=layer_num,
        layer_shard_enabled=False,
        **fields,
    )


class _FakeStream:
    pass


class TestMLAHostDedupPrimitives(CustomTestCase):
    def test_packed_dsa_broadcast_preserves_rank_local_rows_and_pages(self):
        class DSAPool(SimpleNamespace):
            pass

        def make(rank, chunk_tokens):
            pool = DSAPool(
                layer_num=1,
                device=torch.device("cpu"),
                page_size=2,
                kv_cache_dim=3,
                kv_buffer=[torch.full((14, 1, 3), -1, dtype=torch.bfloat16)],
                index_k_with_scale_buffer=[
                    torch.full((7, 2, 4), 255, dtype=torch.uint8)
                ],
            )
            with (
                envs.SGLANG_MLA_DEDUP_CHUNK_TOKENS.override(chunk_tokens),
                mock.patch(
                    "sglang.srt.mem_cache.mla_host_dedup.DSATokenToKVPool", DSAPool
                ),
                mock.patch(
                    "sglang.srt.mem_cache.mla_host_dedup.mla_dedup_rank_and_size",
                    return_value=(rank, 2),
                ),
            ):
                return MLAHostDedupBroadcaster(pool, object(), 0)

        source_rows = torch.tensor([4, 5, 0, 1, 10, 11, 6, 7, 2, 3])
        peer_rows = torch.tensor([8, 9, 6, 7, 12, 13, 2, 3, 0, 1])
        for chunk_tokens in (8, 1):
            owner = make(0, chunk_tokens)
            peer = make(1, chunk_tokens)
            owner.device_pool.kv_buffer[0].copy_(torch.arange(42).view(14, 1, 3))
            owner.idx_bufs[0].copy_(torch.arange(56).view(7, 2, 4))
            for n in (10, 4):
                with self.subTest(chunk_tokens=chunk_tokens, tokens=n):
                    packets = []
                    with mock.patch.object(
                        torch.distributed,
                        "broadcast",
                        side_effect=lambda tensor, **kwargs: packets.append(
                            tensor.clone()
                        ),
                    ):
                        owner.broadcast_loaded_layer(
                            0, (source_rows[:n], source_rows[:n:2] // 2)
                        )
                    self.assertEqual(
                        len(packets), (n + 7) // 8 if chunk_tokens == 8 else n + n // 2
                    )
                    if chunk_tokens == 8:
                        self.assertTrue(all(p.dtype == torch.uint8 for p in packets))
                        self.assertEqual(
                            sum(p.numel() for p in packets), n * 6 + n // 2 * 8
                        )
                    peer.device_pool.kv_buffer[0].fill_(-1)
                    peer.idx_bufs[0].fill_(255)
                    with mock.patch.object(
                        torch.distributed,
                        "broadcast",
                        side_effect=lambda tensor, **kwargs: tensor.copy_(
                            packets.pop(0)
                        ),
                    ):
                        peer.broadcast_loaded_layer(
                            0, (peer_rows[:n], peer_rows[:n:2] // 2)
                        )
                    self.assertEqual(packets, [])
                    torch.testing.assert_close(
                        peer.device_pool.kv_buffer[0][peer_rows[:n]],
                        owner.device_pool.kv_buffer[0][source_rows[:n]],
                    )
                    torch.testing.assert_close(
                        peer.idx_bufs[0][peer_rows[:n:2] // 2],
                        owner.idx_bufs[0][source_rows[:n:2] // 2],
                    )
                    untouched = torch.ones(14, dtype=torch.bool)
                    untouched[peer_rows[:n]] = False
                    self.assertTrue(
                        (peer.device_pool.kv_buffer[0][untouched] == -1).all()
                    )
                    untouched = torch.ones(7, dtype=torch.bool)
                    untouched[peer_rows[:n:2] // 2] = False
                    self.assertTrue((peer.idx_bufs[0][untouched] == 255).all())

    def test_storage_sync_groups_cover_pp_and_deduplicate_rank_sets(self):
        from sglang.srt.distributed import parallel_state
        from sglang.srt.mem_cache.mla_host_dedup import _prebuild_prefetch_sync_groups

        rank_sets = {"tp": [0, 1, 2, 3], "cp": [0, 1], "attn": [0, 1], "pp": [0, 2]}
        cpu_group = object()
        with (
            mock.patch.object(
                torch.distributed,
                "get_world_size",
                side_effect=lambda group: len(rank_sets[group]),
            ),
            mock.patch.object(
                torch.distributed,
                "get_process_group_ranks",
                side_effect=rank_sets.__getitem__,
            ),
            mock.patch.object(
                parallel_state,
                "get_world_group",
                return_value=SimpleNamespace(cpu_group=cpu_group),
            ),
            mock.patch(
                "sglang.srt.mem_cache.mla_host_dedup.get_server_args",
                return_value=SimpleNamespace(dist_timeout=1200),
            ),
            mock.patch.object(
                parallel_state,
                "create_custom_parallel_group",
                side_effect=["hit-attn", "hit-pp"],
            ) as create,
        ):
            groups = _prebuild_prefetch_sync_groups("tp", "cp", "attn", "pp")
        self.assertEqual(groups, ["hit-attn", "hit-pp"])
        self.assertEqual(
            create.call_args_list,
            [
                mock.call(
                    group_ranks=[0, 1],
                    backend="gloo",
                    metadata_group=cpu_group,
                    timeout=timedelta(seconds=1200),
                ),
                mock.call(
                    group_ranks=[0, 2],
                    backend="gloo",
                    metadata_group=cpu_group,
                    timeout=timedelta(seconds=1200),
                ),
            ],
        )

    def test_broadcast_preparation_translates_rank_local_physical_indices(self):
        broadcaster = MLAHostDedupBroadcaster.__new__(MLAHostDedupBroadcaster)
        broadcaster.device = "cpu"
        broadcaster.idx_bufs = [torch.empty(1)]
        broadcaster.device_pool = SimpleNamespace(
            page_size=2,
            host_transfer_translate=lambda indices: indices + 4,
        )
        indices, pages = broadcaster.prepare_broadcast(torch.tensor([2, 3, 0, 1]), None)
        self.assertEqual(indices.tolist(), [6, 7, 4, 5])
        self.assertEqual(pages.tolist(), [3, 2])

    def test_shared_topk_skips_empty_indexer_but_broadcasts_kv(self):
        broadcaster = MLAHostDedupBroadcaster.__new__(MLAHostDedupBroadcaster)
        broadcaster.is_src = True
        broadcaster.src_global_rank = 0
        broadcaster.group = object()
        broadcaster.device_pool = SimpleNamespace(
            kv_cache_dim=2,
            kv_buffer=[torch.full((4, 1, 2), float(i)) for i in range(3)],
        )
        broadcaster.kv_staging = torch.empty(8)
        broadcaster.idx_bufs = [
            torch.full((2, 4), 10.0),
            torch.empty(0, 4),
            torch.full((2, 4), 12.0),
        ]
        broadcaster.idx_elem = 4
        broadcaster.idx_staging = torch.empty(8)
        payloads = []
        with mock.patch.object(
            torch.distributed,
            "broadcast",
            side_effect=lambda tensor, **kw: payloads.append(tensor.clone()),
        ):
            for layer in range(3):
                broadcaster.broadcast_loaded_layer(
                    layer, (torch.tensor([0, 1]), torch.tensor([0]))
                )
        self.assertEqual(len(payloads), 5)
        for payload, expected in zip(payloads, (0, 10, 1, 2, 12)):
            torch.testing.assert_close(payload, torch.full_like(payload, expected))

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
        self.assertEqual(broadcaster.kv_staging.numel(), 3 * 4)

    def test_indexer_staging_is_bounded_by_physical_pages(self):
        class DSAPool(SimpleNamespace):
            pass

        pool = DSAPool(
            layer_num=3,
            device=torch.device("cpu"),
            page_size=8,
            kv_cache_dim=4,
            kv_buffer=[torch.empty(16, 1, 4) for _ in range(3)],
            index_k_with_scale_buffer=[
                torch.empty(0, 8),
                torch.empty(2, 8),
                torch.empty(2, 8),
            ],
        )
        with (
            envs.SGLANG_MLA_DEDUP_CHUNK_TOKENS.override(1024),
            mock.patch("sglang.srt.mem_cache.mla_host_dedup.DSATokenToKVPool", DSAPool),
            mock.patch(
                "sglang.srt.mem_cache.mla_host_dedup.mla_dedup_rank_and_size",
                return_value=(0, 2),
            ),
        ):
            broadcaster = MLAHostDedupBroadcaster(pool, object(), 0)
        self.assertEqual(broadcaster.kv_staging.numel(), 16 * 4)
        self.assertEqual(broadcaster.idx_staging.numel(), 2 * 8)
        # A shared-topk placeholder at layer zero must not suppress later pages.
        indices = torch.tensor([1, 0])
        pool.index_k_with_scale_buffer[1].copy_(torch.arange(16).view(2, 8))
        with mock.patch.object(torch.distributed, "broadcast") as broadcast:
            broadcaster._bcast_layer(
                pool.index_k_with_scale_buffer,
                broadcaster.idx_staging,
                indices,
                broadcaster.idx_elem,
                1,
            )
        broadcast.assert_called_once()
        torch.testing.assert_close(
            broadcaster.idx_staging.view(2, 8),
            pool.index_k_with_scale_buffer[1][indices],
        )

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
        cpu_group = object()
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
            mock.patch(
                "sglang.srt.distributed.parallel_state.get_world_group",
                return_value=SimpleNamespace(cpu_group=cpu_group),
            ),
            mock.patch(
                "sglang.srt.mem_cache.mla_host_dedup.get_server_args",
                return_value=SimpleNamespace(dist_timeout=1200),
            ),
            mock.patch.object(torch.distributed, "broadcast") as broadcast,
            mock.patch.object(torch.cuda, "synchronize") as synchronize,
        ):
            broadcaster = MLAHostDedupBroadcaster.build(
                device_pool, tp_group, attn_tp_group=None
            )

        create_group.assert_called_once_with(
            group_ranks=[4, 5],
            backend="nccl",
            metadata_group=cpu_group,
            timeout=timedelta(seconds=1200),
        )
        broadcast.assert_called_once()
        self.assertEqual(broadcast.call_args.args[0].numel(), 1)
        self.assertIs(broadcast.call_args.kwargs["group"], dedicated_group)
        self.assertEqual(broadcast.call_args.kwargs["src"], 4)
        synchronize.assert_called_once_with(device_pool.device)
        self.assertIs(broadcaster.group, dedicated_group)

    def test_custom_group_uses_cpu_metadata_and_preserves_partition_order(self):
        from sglang.srt.distributed import parallel_state

        cpu_group = object()
        first, second = object(), object()

        def gather(output, config, *, group):
            self.assertIs(group, cpu_group)
            self.assertEqual(config, [2, 3])
            output[:] = [[0, 1], [0, 1], [2, 3], [2, 3]]

        with (
            mock.patch.object(torch.distributed, "is_initialized", return_value=True),
            mock.patch.object(torch.distributed, "get_world_size", return_value=4),
            mock.patch.object(torch.distributed, "get_rank", return_value=2),
            mock.patch.object(torch.distributed, "get_backend", return_value="gloo"),
            mock.patch.object(
                torch.distributed, "all_gather_object", side_effect=gather
            ),
            mock.patch.object(torch.distributed, "monitored_barrier") as barrier,
            mock.patch.object(
                torch.distributed,
                "new_group",
                side_effect=[first, second, first, second],
            ) as create,
            mock.patch.object(parallel_state, "_is_npu", False),
        ):
            result = parallel_state.create_custom_parallel_group(
                [3, 2], "nccl", metadata_group=cpu_group
            )
            timed_result = parallel_state.create_custom_parallel_group(
                [3, 2],
                "nccl",
                metadata_group=cpu_group,
                timeout=timedelta(seconds=1200),
            )
        self.assertIs(result, second)
        self.assertIs(timed_result, second)
        barrier.assert_called_once_with(
            group=cpu_group, timeout=timedelta(seconds=1200), wait_all_ranks=True
        )
        self.assertEqual(
            create.call_args_list,
            [
                mock.call(ranks=[0, 1], backend="nccl"),
                mock.call(ranks=[2, 3], backend="nccl"),
                mock.call(
                    ranks=[0, 1], backend="nccl", timeout=timedelta(seconds=1200)
                ),
                mock.call(
                    ranks=[2, 3], backend="nccl", timeout=timedelta(seconds=1200)
                ),
            ],
        )

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


if __name__ == "__main__":
    unittest.main()
