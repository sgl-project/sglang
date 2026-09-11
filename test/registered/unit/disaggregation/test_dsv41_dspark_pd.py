import asyncio
import copy
import json
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import torch

from sglang.srt.arg_groups.deepseek_v4_hook import (
    _dsv41_dspark_pd_parallelism_supported,
    validate_deepseek_v41_features,
)
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVSender,
)
from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.utils import (
    get_dsv41_spec_layout,
    get_dsv4_request_state_indices,
)
from sglang.srt.mem_cache.common import retraction_backup
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.runtime_context import get_context
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
    def test_prefill_cp_parallelism_is_supported(self):
        base = dict(
            disaggregation_mode="prefill",
            dp_size=1,
            dcp_size=1,
            enable_dp_attention=False,
            enable_prefill_cp=False,
            enable_prefill_context_parallel=False,
            attn_cp_size=1,
        )
        cases = (
            (
                "prefill_cp",
                dict(enable_prefill_cp=True, enable_dp_attention=True, attn_cp_size=8),
                True,
            ),
            ("prefill_tp", {}, True),
            ("prefill_dp_attention", dict(enable_dp_attention=True), False),
            ("prefill_noncanonical_cp", dict(attn_cp_size=8), False),
            ("decode_tp", dict(disaggregation_mode="decode"), True),
            (
                "decode_cp",
                dict(
                    disaggregation_mode="decode",
                    enable_prefill_cp=True,
                    enable_dp_attention=True,
                    attn_cp_size=8,
                ),
                False,
            ),
            ("decode_dcp", dict(disaggregation_mode="decode", dcp_size=2), False),
        )
        for name, overrides, supported in cases:
            with self.subTest(name=name):
                values = base | overrides
                self.assertEqual(
                    _dsv41_dspark_pd_parallelism_supported(SimpleNamespace(**values)),
                    supported,
                )

    def test_prefill_cp_allows_decoder_bounded_replay(self):
        cfg = SimpleNamespace(
            enable_encoder_swa_bounded_replay=False,
            enable_decoder_swa_bounded_replay=True,
            enable_hisparse=False,
            enable_unified_memory=False,
            enable_two_batch_overlap=False,
            enable_dp_attention=True,
            enable_prefill_cp=True,
            enable_prefill_context_parallel=False,
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="mooncake",
            speculative_algorithm="DSPARK",
            dp_size=1,
            attn_cp_size=8,
            dcp_size=1,
            pp_size=1,
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend=Backend.DISABLED, max_seq_len=None)
            ),
        )
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(model_type="deepseek_v41")
        )
        with (
            patch(
                "sglang.srt.arg_groups.deepseek_v4_hook.resolving_view",
                return_value=cfg,
            ),
            patch(
                "sglang.srt.arg_groups.deepseek_v4_hook.model_config_of",
                return_value=model_config,
            ),
        ):
            validate_deepseek_v41_features(SimpleNamespace())

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
                manager.attn_cp_size = 1
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

    def test_bootstrap_accepts_prefill_cp_fan_in(self):
        manager = object.__new__(CommonKVManager)
        manager.prefill_info_table = {}
        manager.kv_args = SimpleNamespace(page_size=256, engine_rank=3)
        manager.kv_cache_dtype_str = "fp8_e4m3"
        manager.dsv41_spec_layout = make_layout()
        manager.attn_tp_size = 8
        manager.attn_cp_size = 1
        manager.attn_cp_rank = 0
        manager.dcp_size = 1
        manager.pp_size = 1
        manager.pp_rank = 0
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.enable_all_cp_ranks_for_transfer = True
        response = Mock(status_code=200)
        response.json.return_value = dict(
            attn_tp_size=1,
            attn_cp_size=8,
            dp_size=1,
            pp_size=1,
            page_size=256,
            kv_cache_dtype="fp8_e4m3",
            follow_bootstrap_room=True,
            dsv41_spec_layout=manager.dsv41_spec_layout,
        )
        with patch(
            "sglang.srt.disaggregation.common.conn.requests.get",
            return_value=response,
        ):
            self.assertTrue(manager.try_ensure_parallel_info("prefill:8998"))

        info = manager.prefill_info_table["prefill:8998"]
        self.assertEqual(info.target_tp_rank, 0)
        self.assertEqual(info.target_tp_ranks, [0])
        self.assertEqual(info.target_cp_ranks, list(range(8)))
        self.assertEqual(info.required_dst_info_num, 8)
        self.assertEqual(info.required_prefill_response_num, 8)

    def test_bootstrap_rejects_mismatched_total_attention_width(self):
        manager = object.__new__(CommonKVManager)
        manager.prefill_info_table = {}
        manager.kv_args = SimpleNamespace(page_size=256)
        manager.kv_cache_dtype_str = "fp8_e4m3"
        manager.dsv41_spec_layout = make_layout()
        manager.attn_tp_size = 8
        manager.attn_cp_size = 1
        manager.dcp_size = 1
        manager._resolve_rank_mapping = Mock()
        response = Mock(status_code=200)
        response.json.return_value = dict(
            attn_tp_size=1,
            attn_cp_size=4,
            dp_size=1,
            pp_size=1,
            page_size=256,
            kv_cache_dtype="fp8_e4m3",
            follow_bootstrap_room=True,
            dsv41_spec_layout=manager.dsv41_spec_layout,
        )
        with (
            patch(
                "sglang.srt.disaggregation.common.conn.requests.get",
                return_value=response,
            ),
            self.assertRaisesRegex(RuntimeError, "attention parallel width"),
        ):
            manager.try_ensure_parallel_info("prefill:8998")
        manager._resolve_rank_mapping.assert_not_called()

    def test_prefill_cp_partitions_every_kv_page_once(self):
        pages = np.arange(11, dtype=np.int32)
        owned = []
        with get_context().override_server_args(enable_dsa_cache_layer_split=False):
            for cp_rank in range(4):
                manager = SimpleNamespace(
                    enable_all_cp_ranks_for_transfer=True,
                    is_dummy_cp_rank=False,
                    attn_cp_rank=cp_rank,
                    attn_cp_size=4,
                )
                sender = SimpleNamespace()
                sender.kv_mgr = manager
                sender.curr_idx = 0
                sender.num_kv_indices = len(pages)
                local_pages, _, is_last, should_skip = (
                    CommonKVSender._prepare_send_indices(sender, pages)
                )
                self.assertTrue(is_last)
                self.assertFalse(should_skip)
                owned.extend(local_pages.tolist())
        self.assertEqual(sorted(owned), pages.tolist())
        self.assertEqual(len(owned), len(set(owned)))

    def test_c2_handoff_keeps_request_ring_indexing(self):
        pool = SimpleNamespace(kv_pools={2: object()})
        self.assertEqual(
            get_dsv4_request_state_indices(pool, req_pool_idx=7, seq_len=101).tolist(),
            [7],
        )
        self.assertEqual(
            get_dsv4_request_state_indices(pool, req_pool_idx=7, seq_len=100).tolist(),
            [],
        )

    def test_multimodal_request_is_rejected_at_runtime_under_cp(self):
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

        model = object.__new__(DeepseekV4ForCausalLM)
        torch.nn.Module.__init__(model)
        model.vision = torch.nn.Identity()
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode=lambda: False),
            mm_inputs=[object()],
        )
        with (
            patch(
                "sglang.srt.models.deepseek_v4.get_parallel",
                return_value=SimpleNamespace(attn_cp_size=2),
            ),
            self.assertRaisesRegex(ValueError, "multimodal requests"),
        ):
            model.forward(
                torch.tensor([1]),
                torch.tensor([0]),
                forward_batch,
            )

    def test_text_request_can_enter_v41_model_with_cp(self):
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

        model = object.__new__(DeepseekV4ForCausalLM)
        torch.nn.Module.__init__(model)
        model.vision = torch.nn.Identity()
        model.config = SimpleNamespace(image_token_id=-1)
        model.dsa_enable_prefill_cp = False
        model.pp_group = SimpleNamespace(is_last_rank=False)
        model.model = Mock()
        model.model.forward.return_value = "hidden"
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_decode=lambda: False,
                is_decode_or_idle=lambda: False,
                is_target_verify=lambda: False,
            ),
            mm_inputs=[None],
        )
        with (
            patch(
                "sglang.srt.models.deepseek_v4.get_parallel",
                return_value=SimpleNamespace(attn_cp_size=2),
            ),
            patch(
                "sglang.srt.models.deepseek_v4.get_attn_tp_context"
            ) as get_attn_tp_context,
        ):
            get_attn_tp_context.return_value.maybe_input_scattered.return_value.__enter__.return_value = None
            get_attn_tp_context.return_value.maybe_input_scattered.return_value.__exit__.return_value = False
            self.assertEqual(
                model.forward(
                    torch.tensor([1]),
                    torch.tensor([0]),
                    forward_batch,
                ),
                "hidden",
            )

    def test_cp_v2_preserves_decoder_replay_hidden_indices(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import LateLayerTail
        from sglang.srt.model_executor.runner.eager_runner import EagerRunner
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

        token_indices = torch.tensor([2, 3])
        tail_cp_metadata = object()
        tail = LateLayerTail(
            token_indices=torch.tensor([0, 1]),
            positions=torch.tensor([2, 3]),
            extend_seq_lens=torch.tensor([2], dtype=torch.int32),
            extend_seq_lens_cpu=[2],
            swa_out_cache_loc=torch.tensor([2, 3]),
            cp_metadata=tail_cp_metadata,
            global_token_indices=token_indices,
        )
        output = SimpleNamespace(hidden_states_token_indices=None)
        model = object.__new__(DeepseekV4ForCausalLM)
        torch.nn.Module.__init__(model)
        model.capture_aux_hidden_states = True
        model.pp_group = SimpleNamespace(is_last_rank=True)
        model.model = Mock()
        model.model.late_layer_start = 21
        model.model.return_value = (
            (torch.ones(2, 4), torch.ones(2, 4)),
            [torch.ones(2, 4)],
        )
        model.lm_head = None
        model.logits_processor = Mock(return_value=output)
        runner = object.__new__(EagerRunner)
        runner.model_runner = SimpleNamespace(
            model=model,
            attn_backend=SimpleNamespace(
                tail_forward_metadata=SimpleNamespace(late_layer_tail=tail)
            ),
        )
        full_cp_metadata = object()
        forward_batch = SimpleNamespace(
            input_ids=torch.tensor([1, 2, 3, 4]),
            positions=torch.tensor([0, 1]),
            attn_cp_metadata=full_cp_metadata,
        )
        gather_metadata = []

        def gather(value, batch, *_):
            gather_metadata.append(batch.attn_cp_metadata)
            return value

        with (
            patch(
                "sglang.srt.model_executor.runner.eager_runner.cp_shard_model_inputs",
                return_value=nullcontext((torch.ones(2, 4), forward_batch.positions)),
            ),
            patch(
                "sglang.srt.model_executor.runner.eager_runner.cp_gather_after_forward",
                side_effect=gather,
            ),
            patch(
                "sglang.srt.layers.logits_processor.LogitsMetadata.from_forward_batch",
                return_value=SimpleNamespace(),
            ),
            patch("torch.cuda.current_stream", return_value=Mock()),
        ):
            result = EagerRunner._execute_extend_cp_v2(
                runner,
                forward_batch,
                {"input_embeds": torch.ones(2, 4)},
            )
        self.assertIs(result, output)
        self.assertIs(result.hidden_states_token_indices, token_indices)
        self.assertTrue(all(x is tail_cp_metadata for x in gather_metadata))
        self.assertIs(forward_batch.attn_cp_metadata, full_cp_metadata)
        self.assertTrue(
            torch.equal(model.logits_processor.call_args.args[0], torch.tensor([3, 4]))
        )

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


if __name__ == "__main__":
    unittest.main()
