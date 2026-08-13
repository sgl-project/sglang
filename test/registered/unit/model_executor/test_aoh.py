"""CPU unit tests for AoH sidecars, eviction, and KV-pool sizing."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.common import free_aoh_out_of_window_slots
from sglang.srt.mem_cache.unified_cache.components import ComponentType, SWAComponent
from sglang.srt.model_executor.aoh import (
    AoHConfig,
    get_aoh_cacheable_prefix_len,
    get_aoh_kv_group,
)
from sglang.srt.model_executor.pool_configurator import AoHPoolConfigurator
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RecordingAllocator:
    def __init__(self):
        self.freed = []

    def free_swa(self, slots):
        self.freed.append(slots.clone())


class TestAoHConfig(CustomTestCase):
    def _sidecar(self, payload):
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / "aoh.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.addCleanup(tmp.cleanup)
        return path

    def test_load_and_route_kv_groups(self):
        config = AoHConfig.load(
            self._sidecar(
                {
                    "version": 1,
                    "layers": {"3": ["streaming", "retrieval"]},
                }
            )
        )
        self.assertEqual(config.mode_for(3, 0), "streaming")
        self.assertEqual(config.mode_for(3, 1), "retrieval")
        with self.assertRaisesRegex(ValueError, "KV-group modes"):
            config.mode_for(3, 2)

    def test_rejects_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "invalid modes"):
            AoHConfig.load(
                self._sidecar({"version": 1, "layers": {"3": ["streaming", "dense"]}})
            )

    def test_qwen_gqa_tp_routes_one_whole_kv_group_per_rank(self):
        groups = [
            get_aoh_kv_group(
                total_kv_heads=2,
                kv_tp_size=8,
                kv_tp_rank=rank,
                local_kv_heads=1,
            )
            for rank in range(8)
        ]
        self.assertEqual(groups, [0, 0, 0, 0, 1, 1, 1, 1])

    def test_radix_cache_only_shares_the_permanent_anchor(self):
        self.assertEqual(get_aoh_cacheable_prefix_len(64, 128), 64)
        self.assertEqual(get_aoh_cacheable_prefix_len(128, 128), 128)
        self.assertEqual(get_aoh_cacheable_prefix_len(4096, 128), 128)

    def test_radix_cache_anchor_ignores_the_request_swa_frontier(self):
        component = SWAComponent.__new__(SWAComponent)
        component.cache = SimpleNamespace(aoh_radix_anchor_only=True)
        insert_params = SimpleNamespace()
        req = SimpleNamespace(kv=SimpleNamespace(swa_evicted_seqlen=1024))

        component.prepare_for_caching_req(
            req, insert_params, token_ids_len=128, is_finished=True
        )
        self.assertEqual(insert_params.swa_evicted_seqlen, 0)

        validator = component.create_match_validator()
        cached_anchor = SimpleNamespace(
            component_data={ComponentType.SWA: SimpleNamespace(value=torch.tensor([1]))}
        )
        self.assertTrue(validator(cached_anchor))


class TestAoHEviction(CustomTestCase):
    def test_keeps_anchors_and_moves_tail_frontier(self):
        req = SimpleNamespace(
            req_pool_idx=0,
            kv=SimpleNamespace(swa_evicted_seqlen=0),
        )
        req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(64, dtype=torch.int64).view(1, -1)
        )
        allocator = _RecordingAllocator()

        free_aoh_out_of_window_slots(
            req,
            20,
            sink_size=4,
            recent_size=8,
            page_size=4,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
        )
        free_aoh_out_of_window_slots(
            req,
            28,
            sink_size=4,
            recent_size=8,
            page_size=4,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
        )

        self.assertEqual(
            [x.tolist() for x in allocator.freed],
            [list(range(4, 12)), list(range(12, 20))],
        )
        self.assertEqual(req.kv.swa_evicted_seqlen, 20)


class TestAoHPoolConfigurator(CustomTestCase):
    def test_stream_pool_is_fixed_and_remaining_memory_grows_retrieval_pool(self):
        model_config = SimpleNamespace(
            head_dim=256,
            v_head_dim=256,
            get_num_kv_heads=lambda tp_size: 1,
        )
        server_args = SimpleNamespace(
            max_running_requests=8,
            chunked_prefill_size=256,
            disable_overlap_schedule=True,
            aoh_sink_size=128,
            aoh_recent_size=256,
        )
        kvc = SimpleNamespace(
            is_aoh=True,
            server_args=server_args,
            page_size=128,
            aoh_retrieval_layer_ids=[3, 7, 11, 15, 19],
            aoh_streaming_layer_ids=[23, 27, 31, 35, 39],
            kv_cache_dtype=torch.bfloat16,
            model_config=model_config,
            ps=SimpleNamespace(attn_dp_size=1),
        )
        with get_parallel().override(attn_tp_size=2):
            configurator = AoHPoolConfigurator(kvc)
            # stream capacity: (128 + 256 + 128) * 8 + 256 + 128 = 4480.
            config = configurator.calculate_pool_sizes(
                available_bytes=(4480 * 5 + 10000 * 5) * 1024,
                page_size=128,
            )

        self.assertEqual(config.swa_max_total_num_tokens, 4480)
        self.assertEqual(config.full_max_total_num_tokens, 9984)
        self.assertEqual(config.max_total_num_tokens, 9984)


if __name__ == "__main__":
    unittest.main()
