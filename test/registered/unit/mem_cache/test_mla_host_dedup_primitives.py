import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.mla_host_dedup import (
    MLAHostDedupBroadcaster,
    MLAHostDedupContext,
    maybe_create_mla_host_dedup_context,
)
from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


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
        )
        indexer_host = DSAIndexerPoolHost(
            dsa_device_pool,
            mla_host,
            layout="page_first",
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


if __name__ == "__main__":
    unittest.main()
