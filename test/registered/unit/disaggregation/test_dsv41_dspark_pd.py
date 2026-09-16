"""Static DSpark PD validation and heterogeneous attention-TP rank mapping."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.utils import get_dsv41_spec_layout
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
    def test_dp_attention_allowed_with_static_mooncake_pd(self):
        from sglang.srt.arg_groups import deepseek_v4_hook as hook
        from sglang.srt.model_executor.cuda_graph_config import Backend
        from sglang.srt.speculative.ragged_verify import RaggedVerifyMode

        cfg = SimpleNamespace(
            enable_encoder_swa_bounded_replay=False,
            enable_decoder_swa_bounded_replay=True,
            enable_dp_attention=True,
            dp_size=8,
            speculative_algorithm="DSPARK",
            enable_hisparse=False,
            dsv4_attn_backend="auto",
            enable_two_batch_overlap=False,
            pp_size=1,
            attn_cp_size=1,
            dcp_size=1,
            disaggregation_mode="decode",
            disaggregation_transfer_backend="mooncake",
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend=Backend.DISABLED, max_seq_len=4096)
            ),
        )
        model = SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v41"))
        with (
            patch.object(hook, "resolving_view", return_value=cfg),
            patch.object(hook, "model_config_of", return_value=model),
            patch(
                "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
                return_value=False,
            ),
            patch(
                "sglang.srt.speculative.ragged_verify.read_ragged_verify_mode",
                return_value=RaggedVerifyMode.STATIC,
            ) as verify_mode,
        ):
            for role in ("prefill", "decode"):
                for dp_size in (1, 8):
                    with self.subTest(role=role, dp_size=dp_size):
                        cfg.disaggregation_mode = role
                        cfg.dp_size = dp_size
                        hook.validate_deepseek_v41_features(object())

            for name, value in (
                ("disaggregation_transfer_backend", "nixl"),
                ("attn_cp_size", 2),
                ("dcp_size", 2),
            ):
                with self.subTest(unsupported=name), patch.object(cfg, name, value):
                    with self.assertRaisesRegex(ValueError, "DSpark PD requires"):
                        hook.validate_deepseek_v41_features(object())
            verify_mode.return_value = RaggedVerifyMode.COMPACT
            with self.assertRaisesRegex(ValueError, "DSpark PD requires"):
                hook.validate_deepseek_v41_features(object())

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
            ("smaller_prefill_tp", layout, layout, 2, True),
            ("larger_prefill_tp", layout, layout, 8, True),
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
                        manager._resolve_rank_mapping.assert_called_once()
                    else:
                        with self.assertRaisesRegex(
                            RuntimeError, "DeepSeek-V4.1 DSpark PD"
                        ):
                            manager.try_ensure_parallel_info("prefill:8998")
                        self.assertFalse(manager.prefill_info_table)
                        manager._resolve_rank_mapping.assert_not_called()

    def test_mla_heterogeneous_tp_bootstrap_rank_mapping(self):
        layout = make_layout()
        for prefill_tp, decode_tp in ((8, 1), (8, 4), (1, 8), (4, 8)):
            for rank in range(decode_tp):
                with self.subTest(
                    prefill_tp=prefill_tp, decode_tp=decode_tp, rank=rank
                ):
                    manager = object.__new__(CommonKVManager)
                    manager.prefill_info_table = {}
                    manager.kv_args = SimpleNamespace(page_size=256, engine_rank=rank)
                    manager.kv_cache_dtype_str = "fp8_e4m3"
                    manager.dsv41_spec_layout = layout
                    manager.attn_tp_size = decode_tp
                    manager.attn_cp_size = manager.dcp_size = manager.pp_size = 1
                    manager.attn_cp_rank = manager.pp_rank = 0
                    manager.is_mla_backend = True
                    manager.is_hybrid_mla_backend = False
                    response = Mock(status_code=200)
                    response.json.return_value = dict(
                        attn_tp_size=prefill_tp,
                        attn_cp_size=1,
                        dp_size=1,
                        pp_size=1,
                        page_size=256,
                        kv_cache_dtype="fp8_e4m3",
                        follow_bootstrap_room=True,
                        dsv41_spec_layout=layout,
                    )
                    with patch(
                        "sglang.srt.disaggregation.common.conn.requests.get",
                        return_value=response,
                    ):
                        self.assertTrue(
                            manager.try_ensure_parallel_info("prefill:8998")
                        )
                    info = manager.prefill_info_table["prefill:8998"]
                    self.assertEqual(
                        info.target_tp_rank, rank * prefill_tp // decode_tp
                    )
                    self.assertEqual(info.required_prefill_response_num, 1)
                    self.assertEqual(
                        info.required_dst_info_num, max(1, decode_tp // prefill_tp)
                    )
                    self.assertEqual(
                        len(info.target_tp_ranks), max(1, prefill_tp // decode_tp)
                    )


if __name__ == "__main__":
    unittest.main()
