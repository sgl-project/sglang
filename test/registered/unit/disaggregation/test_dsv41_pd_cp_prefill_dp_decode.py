"""DeepSeek-V4.1 PD topology: CP prefill and DP-attention decode."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import (
    CommonKVManager,
    PrefillServerInfo,
)
from sglang.srt.disaggregation.utils import get_dsv41_spec_layout
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_spec_layout():
    kv_args = SimpleNamespace(
        mla_compression_ratios=[0, 2, 1],
        kv_layer_ids=[1, 2],
        kv_item_lens=[512, 1024],
        state_types=[StateType.SWA, StateType.DSV4_REQUEST_STATE, StateType.SWA],
        state_item_lens=[[512], [32768], [512]],
    )
    with get_context().override_server_args(
        speculative_algorithm="DSPARK", speculative_num_draft_tokens=6
    ):
        return get_dsv41_spec_layout(kv_args)


def _make_feature_config(*, role: str):
    from sglang.srt.model_executor.cuda_graph_config import Backend

    is_prefill = role == "prefill"
    return SimpleNamespace(
        enable_encoder_swa_bounded_replay=False,
        enable_decoder_swa_bounded_replay=is_prefill,
        enable_dp_attention=True,
        enable_prefill_cp=is_prefill,
        dp_size=1 if is_prefill else 8,
        speculative_algorithm="DSPARK",
        enable_hisparse=False,
        dsv4_attn_backend="auto",
        enable_two_batch_overlap=False,
        pp_size=1,
        attn_cp_size=8 if is_prefill else 1,
        dcp_size=1,
        disaggregation_mode=role,
        disaggregation_transfer_backend="mooncake",
        moe_a2a_backend="megamoe",
        cuda_graph_config=SimpleNamespace(
            prefill=SimpleNamespace(backend=Backend.DISABLED, max_seq_len=4096)
        ),
    )


class TestDSV41PDPrefillCPDecodeDP(CustomTestCase):
    def _validate_features(self, cfg):
        from sglang.srt.arg_groups import deepseek_v4_hook as hook
        from sglang.srt.speculative.ragged_verify import RaggedVerifyMode

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
            ),
        ):
            hook.validate_deepseek_v41_features(object())

    def test_role_asymmetric_parallelism_is_accepted(self):
        # Prefill uses interleaved CP. enable_dp_attention=True is the
        # DeepSeek-V4 CP implementation detail here; dp_size=1 means no real DP.
        self._validate_features(_make_feature_config(role="prefill"))

        # Decode uses real DP attention and MegaMoE, without replaying the
        # prompt tail locally.
        self._validate_features(_make_feature_config(role="decode"))

    def test_server_local_cp_dp_and_decode_cp_are_rejected(self):
        prefill = _make_feature_config(role="prefill")
        prefill.dp_size = 8
        prefill.enable_dp_attention = False
        with self.assertRaisesRegex(ValueError, r"no server-local CP\+DP"):
            self._validate_features(prefill)

        decode = _make_feature_config(role="decode")
        decode.attn_cp_size = 2
        with self.assertRaisesRegex(ValueError, "decode CP=1"):
            self._validate_features(decode)

    def test_decode_does_not_enable_bounded_replay(self):
        decode = _make_feature_config(role="decode")
        decode.enable_decoder_swa_bounded_replay = True
        with self.assertRaisesRegex(ValueError, "DP attention"):
            self._validate_features(decode)

    def test_bootstrap_maps_every_decode_dp_replica_to_cp_rank_zero(self):
        layout = _make_spec_layout()
        manager = object.__new__(CommonKVManager)
        manager.prefill_info_table = {}
        manager.kv_args = SimpleNamespace(page_size=256, engine_rank=0)
        manager.kv_cache_dtype_str = "fp8_e4m3"
        manager.dsv41_spec_layout = layout
        manager.attn_tp_size = 1
        manager.attn_cp_size = 1
        manager.attn_cp_rank = 0
        manager.dcp_size = 1
        manager.pp_size = 1
        manager.pp_rank = 0
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.enable_all_cp_ranks_for_transfer = False

        response = Mock(status_code=200)
        response.json.return_value = dict(
            attn_tp_size=1,
            attn_cp_size=8,
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
            self.assertTrue(manager.try_ensure_parallel_info("prefill:8998"))

        info = manager.prefill_info_table["prefill:8998"]
        self.assertEqual(info.target_tp_ranks, [0])
        self.assertEqual(info.target_cp_ranks, [0])
        self.assertEqual(info.required_dst_info_num, 1)
        self.assertEqual(info.required_prefill_response_num, 1)

    def test_all_cp_transfer_fans_in_when_explicitly_enabled(self):
        manager = object.__new__(CommonKVManager)
        manager.kv_args = SimpleNamespace(engine_rank=0)
        manager.attn_tp_size = 1
        manager.attn_cp_size = 1
        manager.attn_cp_rank = 0
        manager.dcp_size = 1
        manager.pp_size = 1
        manager.pp_rank = 0
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.enable_all_cp_ranks_for_transfer = True
        info = PrefillServerInfo(
            attn_tp_size=1,
            attn_cp_size=8,
            dp_size=1,
            pp_size=1,
            page_size=256,
            kv_cache_dtype="fp8_e4m3",
            follow_bootstrap_room=True,
        )

        manager._resolve_rank_mapping(info)

        self.assertEqual(info.target_cp_ranks, list(range(8)))
        self.assertEqual(info.required_prefill_response_num, 8)

    def test_only_cp_rank_zero_sends_replicated_swa_state(self):
        manager = object.__new__(CommonKVManager)
        manager.attn_cp_size = 8
        parallel = SimpleNamespace(enable_dsa_cache_layer_split=False)
        with patch(
            "sglang.srt.disaggregation.common.conn.get_parallel",
            return_value=parallel,
        ):
            manager.attn_cp_rank = 0
            self.assertFalse(manager._should_skip_cp_replicated_state_transfer())
            manager.attn_cp_rank = 1
            self.assertTrue(manager._should_skip_cp_replicated_state_transfer())

            # Layer-split CP owns disjoint state layers, so every rank sends.
            parallel.enable_dsa_cache_layer_split = True
            self.assertFalse(manager._should_skip_cp_replicated_state_transfer())

    def test_dspark_layout_allows_heterogeneous_attention_tp(self):
        layout = _make_spec_layout()
        manager = object.__new__(CommonKVManager)
        manager.prefill_info_table = {}
        manager.kv_args = SimpleNamespace(page_size=256, engine_rank=0)
        manager.kv_cache_dtype_str = "fp8_e4m3"
        manager.dsv41_spec_layout = layout
        manager.attn_tp_size = 1
        manager.attn_cp_size = 1
        manager.attn_cp_rank = 0
        manager.dcp_size = 1
        manager.pp_size = 1
        manager.pp_rank = 0
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.enable_all_cp_ranks_for_transfer = False

        response = Mock(status_code=200)
        response.json.return_value = dict(
            attn_tp_size=2,
            attn_cp_size=4,
            dp_size=1,
            pp_size=1,
            page_size=256,
            kv_cache_dtype="fp8_e4m3",
            follow_bootstrap_room=True,
            dsv41_spec_layout=copy.deepcopy(layout),
        )
        with patch(
            "sglang.srt.disaggregation.common.conn.requests.get",
            return_value=response,
        ):
            self.assertTrue(manager.try_ensure_parallel_info("prefill:8998"))

        info = manager.prefill_info_table["prefill:8998"]
        self.assertEqual(info.target_tp_ranks, [0, 1])
        self.assertEqual(info.target_cp_ranks, [0])
        # MLA needs one real response; the second TP peer is connected with a
        # dummy request so its KVPoll can finish.
        self.assertEqual(info.required_prefill_response_num, 1)


if __name__ == "__main__":
    unittest.main()
