import asyncio
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import torch

from sglang.srt.arg_groups.deepseek_v4_hook import validate_deepseek_v41_features
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
)
from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.utils import get_dsv41_spec_layout
from sglang.srt.environ import envs
from sglang.srt.mem_cache.common import retraction_backup
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.dspark_disaggregation import build_dspark_disagg_draft_input
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def make_layout():
    args = SimpleNamespace(
        mla_compression_ratios=[0, 2, 1],
        kv_layer_ids=[1, 2],
        kv_item_lens=[512, 1024],
        state_types=[StateType.SWA, StateType.C128_STATE, StateType.SWA],
        state_item_lens=[[512], [32768], [512]],
    )
    with get_context().override_server_args(
        speculative_algorithm="DSPARK", speculative_num_draft_tokens=6
    ):
        return get_dsv41_spec_layout(args)


class TestDSV41DSparkPD(CustomTestCase):
    def test_feature_gate(self):
        base = dict(
            speculative_algorithm="DSPARK",
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="mooncake",
            enable_hisparse=False,
            enable_two_batch_overlap=False,
            pp_size=1,
            dp_size=1,
            enable_dp_attention=False,
            attn_cp_size=1,
            dcp_size=1,
            enable_prefill_context_parallel=False,
            enable_decoder_swa_bounded_replay=False,
            cuda_graph_config=CudaGraphConfig(prefill=PhaseConfig(backend="disabled")),
        )
        model = SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v41"))
        cases = [
            ({}, "static", True),
            ({"disaggregation_mode": "decode"}, "static", True),
            ({}, "compact", False),
            ({}, "cap-accept", False),
            ({"disaggregation_transfer_backend": "nixl"}, "static", False),
            ({"dp_size": 2}, "static", False),
            ({"enable_dp_attention": True}, "static", False),
            ({"enable_prefill_context_parallel": True}, "static", False),
            ({"attn_cp_size": 2}, "static", False),
            ({"dcp_size": 2}, "static", False),
            ({"pp_size": 2}, "static", False),
            ({"speculative_algorithm": "EAGLE"}, "static", False),
        ]
        for overrides, mode, supported in cases:
            with (
                self.subTest(overrides=overrides, mode=mode),
                envs.SGLANG_RAGGED_VERIFY_MODE.override(mode),
                patch(
                    "sglang.srt.arg_groups.deepseek_v4_hook.model_config_of",
                    return_value=model,
                ),
                get_context().override_server_args(**(base | overrides)) as args,
                patch(
                    "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
                    return_value=False,
                ),
            ):
                if supported:
                    validate_deepseek_v41_features(args)
                else:
                    with self.assertRaises(ValueError):
                        validate_deepseek_v41_features(args)

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

    def test_prebuilt_seeds_bonus_and_committed_lengths(self):
        for overlap in (False, True):
            with self.subTest(overlap=overlap):
                batch = SimpleNamespace(
                    seq_lens=torch.tensor([255, 256, 257], dtype=torch.int32),
                    req_pool_indices=torch.tensor([3, 1, 7]),
                    enable_overlap=overlap,
                )
                bonus_tokens = torch.tensor([11, 12, 13])
                future_map = Mock()
                draft = build_dspark_disagg_draft_input(batch, bonus_tokens, future_map)
                self.assertTrue(torch.equal(draft.bonus_tokens, bonus_tokens))
                if overlap:
                    future_map.publish.assert_called_once_with(
                        batch.req_pool_indices, batch.seq_lens
                    )
                    payload = future_map.stash.call_args.args[1]
                    self.assertTrue(torch.equal(payload.bonus_tokens, bonus_tokens))
                else:
                    future_map.publish.assert_not_called()
                    future_map.stash.assert_not_called()


if __name__ == "__main__":
    unittest.main()
