import asyncio
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
)
from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.utils import get_dsv41_spec_layout
from sglang.srt.mem_cache.common import retraction_backup
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def make_layout():
    args = SimpleNamespace(
        mla_compression_ratios=[0, 2, 1],
        kv_layer_ids=[1, 2],
        kv_item_lens=[512, 1024],
        state_types=[StateType.SWA, StateType.DSV4_REQUEST_STATE, StateType.SWA],
        state_item_lens=[[512], [32768], [512]],
    )
    with get_context().override_server_args(
        speculative_algorithm="DSPARK", speculative_num_draft_tokens=6
    ):
        return get_dsv41_spec_layout(args)


class TestDSV41DSparkPD(CustomTestCase):
    def test_bootstrap_validates_before_caching(self):
        layout = make_layout()
        cases = [("matching", layout, layout, 4, True), ("legacy", None, None, 2, True)]
        for key, value in (
            ("num_draft_tokens", 5),
            ("kv_layer_ids", [2, 1]),
            ("kv_item_lens", [256, 1024]),
            ("state_types", ["swa", "c128_state"]),
            ("state_item_lens", [[512], [8192], [512]]),
        ):
            different = copy.deepcopy(layout)
            different[key] = value
            cases.append((key, layout, different, 4, False))
        cases += [
            ("prefill_only", None, layout, 4, False),
            ("decode_only_or_old_prefill", layout, None, 4, False),
            ("tp_mismatch", layout, layout, 2, False),
        ]
        for name, local, peer, tp_size, supported in cases:
            with self.subTest(name=name):
                manager = object.__new__(CommonKVManager)
                manager.prefill_info_table = {}
                manager.kv_args = SimpleNamespace(page_size=256)
                manager.kv_cache_dtype_str = "fp8_e4m3"
                manager.dsv41_spec_layout = local
                manager.attn_tp_size = 4
                manager.dcp_size = 1
                manager._resolve_rank_mapping = Mock()
                response = Mock(status_code=200)
                response.json.return_value = dict(
                    attn_tp_size=tp_size,
                    attn_cp_size=1,
                    dp_size=1,
                    pp_size=1,
                    page_size=256,
                    kv_cache_dtype="fp8_e4m3",
                    follow_bootstrap_room=True,
                    dsv41_spec_layout=peer,
                )
                with patch(
                    "sglang.srt.disaggregation.common.conn.requests.get",
                    return_value=response,
                ) as fetch:
                    if supported:
                        self.assertTrue(
                            manager.try_ensure_parallel_info("prefill:8998")
                        )
                        self.assertTrue(
                            manager.try_ensure_parallel_info("prefill:8998")
                        )
                        fetch.assert_called_once()
                    else:
                        with self.assertRaisesRegex(
                            RuntimeError, "DeepSeek-V4.1 DSpark PD"
                        ):
                            manager.try_ensure_parallel_info("prefill:8998")
                        self.assertFalse(manager.prefill_info_table)
                        manager._resolve_rank_mapping.assert_not_called()

    def test_python_bootstrap_preserves_layout_and_rejects_mixed_ranks(self):
        with patch.object(CommonKVBootstrapServer, "run"):
            server = CommonKVBootstrapServer("127.0.0.1", 8998)
        layout = make_layout()
        payload = dict(
            attn_tp_size=1,
            attn_tp_rank=0,
            attn_cp_size=1,
            attn_cp_rank=0,
            attn_dp_size=1,
            attn_dp_rank=0,
            pp_size=1,
            pp_rank=0,
            system_dp_size=1,
            system_dp_rank=0,
            rank_ip="127.0.0.1",
            rank_port=1234,
            page_size=256,
            kv_cache_dtype="fp8_e4m3",
            dsv41_spec_layout=layout,
        )
        request = Mock(json=AsyncMock(return_value=payload))
        self.assertEqual(asyncio.run(server._handle_route_put(request)).status, 200)
        query = Mock(
            query={
                key: "-1"
                for key in (
                    "prefill_dp_rank",
                    "prefill_cp_rank",
                    "target_tp_rank",
                    "target_pp_rank",
                )
            }
        )
        response = asyncio.run(server._handle_route_get(query))
        self.assertEqual(json.loads(response.text)["dsv41_spec_layout"], layout)
        payload["dsv41_spec_layout"] = None
        self.assertEqual(asyncio.run(server._handle_route_put(request)).status, 400)
        self.assertEqual(server._registered_count, 1)
        self.assertEqual(server.dsv41_spec_layout, layout)

    def test_retraction_recomputes_from_prefill_and_replays_boundary_token(self):
        pool = object.__new__(DeepSeekV4TokenToKVPool)
        pool.compression_ratios = [0, 2, 1]
        pool.device = "cuda"
        allocator = Mock(get_kvcache=Mock(return_value=pool))
        for algorithm in (None, "DSPARK"):
            with (
                self.subTest(algorithm=algorithm),
                get_context().override_server_args(speculative_algorithm=algorithm),
                patch("torch.get_device_module") as device_module,
            ):
                req = SimpleNamespace(
                    output_ids=[7, 8],
                    bootstrap_host="prefill",
                    time_stats=Mock(),
                    offload_kv_cache=Mock(),
                )
                request_pool = Mock()
                self.assertTrue(
                    retraction_backup(
                        req, Mock(), request_pool, allocator, "cpu_tensor"
                    )
                )
                queue = SimpleNamespace(
                    token_to_kv_pool_allocator=allocator,
                    _check_if_req_exceed_kv_capacity=Mock(return_value=False),
                    _create_receiver_and_enqueue=Mock(),
                    _resolve_prefill_dp_rank=Mock(return_value=0),
                    retracted_queue=[],
                    pending_reqs=[],
                )
                DecodePreallocQueue.add(queue, req, is_retracted=True)
                if algorithm == "DSPARK":
                    req.offload_kv_cache.assert_not_called()
                    device_module.return_value.synchronize.assert_called_once_with(
                        "cuda"
                    )
                    self.assertEqual(req.output_ids, [7])
                    self.assertEqual(req.pd_rebootstrap_forced_output_id, 8)
                    self.assertTrue(req.pd_rebootstrap_in_progress)
                    queue._create_receiver_and_enqueue.assert_called_once_with(
                        req, is_rebootstrap=True
                    )
                    self.assertFalse(queue.retracted_queue)
                else:
                    req.offload_kv_cache.assert_called_once_with(
                        request_pool, allocator
                    )
                    device_module.assert_not_called()
                    self.assertEqual(req.output_ids, [7, 8])
                    self.assertEqual(queue.retracted_queue, [req])


class TestDSV41CPPDHandshake(CustomTestCase):
    def make(self, rank=0, hybrid=True):
        m = object.__new__(CommonKVManager)
        m.prefill_info_table = {}
        m.kv_args = SimpleNamespace(page_size=256, engine_rank=rank)
        m.kv_cache_dtype_str = "fp8_e4m3"
        m.dsv41_spec_layout = {"kv_item_lens": [512], "state_item_lens": [[32768]]}
        m.attn_tp_size = 4
        m.attn_cp_size = 1
        m.attn_cp_rank = 0
        m.dcp_size = 1
        m.is_mla_backend = False
        m.is_hybrid_mla_backend = hybrid
        m.enable_all_cp_ranks_for_transfer = True
        m.pp_size = 1
        m.pp_rank = 0
        return m

    def fetch(self, m, tp, cp, layout=None):
        response = Mock(status_code=200)
        response.json.return_value = dict(
            attn_tp_size=tp,
            attn_cp_size=cp,
            dp_size=1,
            pp_size=1,
            page_size=256,
            kv_cache_dtype="fp8_e4m3",
            follow_bootstrap_room=True,
            dsv41_spec_layout=layout or m.dsv41_spec_layout,
        )
        with patch(
            "sglang.srt.disaggregation.common.conn.requests.get", return_value=response
        ):
            return m.try_ensure_parallel_info("prefill:8761")

    def test_cp4_maps_all_shards_to_each_decode_rank(self):
        for rank in range(4):
            m = self.make(rank)
            self.assertTrue(self.fetch(m, 1, 4))
            info = m.prefill_info_table["prefill:8761"]
            self.assertEqual(info.target_tp_ranks, [0])
            self.assertEqual(info.target_cp_ranks, [0, 1, 2, 3])
            self.assertEqual(info.required_prefill_response_num, 4)
            self.assertEqual(info.required_dst_info_num, 4)

    def test_dsv4_pool_is_classified_as_mla(self):
        from sglang.srt.disaggregation.utils import is_mla_backend
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

        pool = object.__new__(DeepSeekV4TokenToKVPool)
        self.assertTrue(is_mla_backend(pool))
        m = self.make(hybrid=False)
        m.is_mla_backend = is_mla_backend(pool)
        self.assertTrue(self.fetch(m, 1, 4))
        self.assertEqual(
            m.prefill_info_table["prefill:8761"].required_prefill_response_num, 4
        )

    def test_cp2_tp2_maps_corresponding_tp_and_both_cp_ranks(self):
        for rank in range(4):
            m = self.make(rank)
            self.assertTrue(self.fetch(m, 2, 2))
            info = m.prefill_info_table["prefill:8761"]
            self.assertEqual(info.target_tp_ranks, [rank // 2])
            self.assertEqual(info.target_cp_ranks, [0, 1])
            self.assertEqual(info.required_prefill_response_num, 2)

    def test_plain_tp4_unchanged(self):
        m = self.make(3)
        self.assertTrue(self.fetch(m, 4, 1))
        info = m.prefill_info_table["prefill:8761"]
        self.assertEqual(info.target_tp_ranks, [3])
        self.assertEqual(info.target_cp_ranks, [0])

    def test_unequal_model_tp_rejected(self):
        for tp, cp in [(2, 1), (1, 2), (1, 8)]:
            m = self.make()
            with self.assertRaisesRegex(RuntimeError, "same TP size"):
                self.fetch(m, tp, cp)
            self.assertFalse(m.prefill_info_table)

    def test_nonhybrid_cp_mismatch_rejected(self):
        m = self.make(hybrid=False)
        with self.assertRaisesRegex(RuntimeError, "same TP size"):
            self.fetch(m, 1, 4)

    def test_layout_mismatch_still_rejected(self):
        m = self.make()
        with self.assertRaisesRegex(RuntimeError, "layout mismatch"):
            self.fetch(m, 1, 4, {"kv_item_lens": [1024], "state_item_lens": [[32768]]})
        self.assertFalse(m.prefill_info_table)


if __name__ == "__main__":
    unittest.main()
