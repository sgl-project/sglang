"""Host restore must preserve shared compressed KV and every FP4 index page."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _DeepSeekV4Strategy,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

V41_RATIOS = [0] * 2 + [2] * 18 + [1] * 20
V41_SOURCES = [2, 8, 14, 20]


@unittest.skipUnless(torch.cuda.is_available(), "requires device/host transfers")
class TestDeepseekV41HiCache(CustomTestCase):
    def test_restore_relocated_shared_pages(self):
        self._restore_layouts(encoder_replay=False)

    def test_encoder_replay_restores_only_persistent_pages(self):
        self._restore_layouts(encoder_replay=True)

    def _restore_layouts(self, encoder_replay):
        for layout, backend in (
            ("layer_first", "kernel"),
            ("page_first", "kernel"),
            ("layer_first", "direct"),
            ("page_first_direct", "direct"),
        ):
            with self.subTest(layout=layout, backend=backend):
                # CacheController transfers run on dedicated, non-default streams.
                with torch.cuda.stream(torch.cuda.Stream()):
                    self._restore(layout, backend, encoder_replay)

    def _restore(self, layout, backend, encoder_replay):
        page = 256
        args = ServerArgs(
            model_path="dummy",
            page_size=page,
            hicache_mem_layout=layout,
            hicache_io_backend=backend,
            hicache_ratio=2,
            enable_encoder_swa_bounded_replay=encoder_replay,
        )
        set_global_server_args_for_scheduler(args)
        pool = DeepSeekV4TokenToKVPool(
            max_num_reqs=1,
            swa_size=0 if encoder_replay else 512,
            c4_size=0,
            c128_size=0,
            c4_state_pool_size=0,
            c128_state_pool_size=0,
            page_size=page,
            swa_page_size=128,
            dtype=torch.float8_e4m3fn,
            c4_state_dtype=torch.float32,
            c128_state_dtype=torch.float32,
            qk_nope_head_dim=448,
            qk_rope_head_dim=64,
            indexer_head_dim=128,
            layer_num=40,
            start_layer=0,
            end_layer=40,
            device="cuda",
            enable_memory_saver=False,
            compression_ratios=V41_RATIOS,
            kv_source_layers=V41_SOURCES,
            full_size=512,
        )
        params = SimpleNamespace(
            page_size=page,
            mtp_draft_device_pools=(),
            token_to_kv_pool_allocator=SimpleNamespace(
                size_full=512, swa_attn_allocator=SimpleNamespace(alloc=None, free=None)
            ),
            tp_cache_group=None,
            attn_cp_cache_group=None,
            attn_tp_cache_group=None,
            pp_cache_group=None,
        )
        with patch(
            "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler.HybridCacheController"
        ):
            result = _DeepSeekV4Strategy().build(
                cache=None,
                kvcache=pool,
                params=params,
                server_args=args,
                load_cache_event=None,
            )
        self.addCleanup(result.host_pool_group.destroy)
        if encoder_replay:
            self.assertNotIn(PoolName.SWA, result.host_pool_group.entry_map)
            for buffer in pool.request_window.state.kv_buffer:
                buffer.fill_(37)
        expected = {
            PoolName.DEEPSEEK_V4_C1,
            PoolName.DEEPSEEK_V4_C1_INDEXER,
            PoolName.DEEPSEEK_V4_C2,
            PoolName.DEEPSEEK_V4_C2_INDEXER,
        }
        self.assertEqual({spec.pool_name for spec in result.sidecars}, expected)
        source_count = 0
        for entry in result.host_pool_group.entries:
            if entry.name == PoolName.KV:
                continue
            host = entry.host_pool
            slots = host.slot_page_size
            device_indices = torch.arange(
                2 * slots, 3 * slots, device="cuda", dtype=torch.int64
            )
            target_indices = torch.arange(
                slots, 2 * slots, device="cuda", dtype=torch.int64
            )
            host_indices = torch.arange(slots, dtype=torch.int64, device="cuda")
            expected_rows = []
            for layer, buffer in enumerate(host.device_buffers):
                row = torch.randint(
                    1, 255, buffer[2].shape, dtype=torch.uint8, device="cuda"
                )
                buffer[2].copy_(row)
                expected_rows.append(row)
            backup_host_indices = (
                host_indices.cpu() if host.can_use_write_back_jit else host_indices
            )
            if backend == "direct":
                host_indices = host_indices.cpu()
                backup_host_indices = host_indices
                device_indices = device_indices.cpu()
                target_indices = target_indices.cpu()
            host.backup_from_device_all_layer(
                entry.device_pool, backup_host_indices, device_indices, backend
            )
            torch.cuda.synchronize()
            for buffer in host.device_buffers:
                buffer.zero_()
            for layer in range(40):
                mapped = entry.layer_mapper(layer)
                if mapped is not None:
                    if entry.name != PoolName.SWA:
                        self.assertIn(layer, V41_SOURCES)
                        source_count += 1
                    host.load_to_device_per_layer(
                        entry.device_pool, host_indices, target_indices, mapped, backend
                    )
            torch.cuda.synchronize()
            for buffer, expected_row in zip(host.device_buffers, expected_rows):
                torch.testing.assert_close(buffer[1], expected_row, rtol=0, atol=0)
                self.assertEqual(torch.count_nonzero(buffer[2]).item(), 0)
        # Four KV sources and four index-K sources, not 38 consumers each.
        self.assertEqual(source_count, 8)
        if encoder_replay:
            for buffer in pool.request_window.state.kv_buffer:
                self.assertTrue(torch.all(buffer == 37).item())


class TestEncoderReplayHiCacheOrdering(CustomTestCase):
    def test_consumer_is_selected_before_encoder_replay(self):
        from sglang.srt.managers.tp_worker import TpModelWorker

        events = []
        worker = SimpleNamespace(
            set_hicache_consumer=lambda index: events.append(("consumer", index))
        )
        batch = SimpleNamespace(hicache_consumer_index=2)

        class ReplayReached(Exception):
            pass

        def replay(worker, batch):
            assert events == [("consumer", 2)]
            raise ReplayReached

        with (
            patch(
                "sglang.srt.managers.tp_worker.get_exec",
                return_value=SimpleNamespace(
                    features=SimpleNamespace(enable_encoder_swa_bounded_replay=True)
                ),
            ),
            patch(
                "sglang.srt.model_executor.encoder_swa_replay.run_encoder_swa_replay",
                side_effect=replay,
            ),
            self.assertRaises(ReplayReached),
        ):
            TpModelWorker.forward_batch_generation(worker, batch)


if __name__ == "__main__":
    unittest.main()
