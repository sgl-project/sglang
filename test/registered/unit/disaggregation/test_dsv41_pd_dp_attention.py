"""V4.1 DSpark PD with DP attention and no context parallelism."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.arg_groups import deepseek_v4_hook as hook
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.speculative.ragged_verify import RaggedVerifyMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _feature_config(role: str, *, dp_size: int = 4):
    return SimpleNamespace(
        enable_encoder_swa_bounded_replay=False,
        enable_decoder_swa_bounded_replay=False,
        enable_dp_attention=dp_size > 1,
        enable_prefill_cp=False,
        dp_size=dp_size,
        speculative_algorithm="DSPARK",
        enable_hisparse=False,
        dsv4_attn_backend="auto",
        enable_two_batch_overlap=False,
        pp_size=1,
        attn_cp_size=1,
        dcp_size=1,
        disaggregation_mode=role,
        disaggregation_transfer_backend="mooncake",
        cuda_graph_config=SimpleNamespace(
            prefill=SimpleNamespace(backend=Backend.DISABLED, max_seq_len=4096)
        ),
    )


class TestDSV41PDDPAttention(CustomTestCase):
    def _validate(self, cfg):
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

    def test_dp_prefill_and_decode_are_accepted(self):
        self._validate(_feature_config("prefill"))
        self._validate(_feature_config("decode"))

    def test_tp_prefill_and_dp_decode_are_accepted(self):
        self._validate(_feature_config("prefill", dp_size=1))
        self._validate(_feature_config("decode"))

    def test_cp_is_not_part_of_dp_only_pd(self):
        cfg = _feature_config("prefill", dp_size=1)
        cfg.enable_prefill_cp = True
        with self.assertRaisesRegex(ValueError, "CP=1"):
            self._validate(cfg)

        cfg.enable_prefill_cp = False
        cfg.attn_cp_size = 4
        with self.assertRaisesRegex(ValueError, "CP=1"):
            self._validate(cfg)

    def _manager(self):
        manager = object.__new__(CommonKVManager)
        manager.prefill_info_table = {}
        manager.kv_args = SimpleNamespace(page_size=256, engine_rank=0)
        manager.kv_cache_dtype_str = "fp8_e4m3"
        manager.dsv41_spec_layout = {"block_size": 128, "target_kv": "fp8"}
        manager.attn_tp_size = 1
        manager.attn_cp_size = 1
        manager.attn_cp_rank = 0
        manager.dcp_size = 1
        manager.pp_size = 1
        manager.pp_rank = 0
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.enable_all_cp_ranks_for_transfer = False
        return manager

    def _response(self, manager, *, attn_cp_size=1, layout=None):
        response = Mock(status_code=200)
        response.json.return_value = dict(
            attn_tp_size=2,
            attn_cp_size=attn_cp_size,
            dp_size=1,
            pp_size=1,
            page_size=256,
            kv_cache_dtype="fp8_e4m3",
            follow_bootstrap_room=True,
            dsv41_spec_layout=(manager.dsv41_spec_layout if layout is None else layout),
        )
        return response

    def test_non_cp_mla_tp_to_dp_rank_mapping(self):
        manager = self._manager()
        with patch(
            "sglang.srt.disaggregation.common.conn.requests.get",
            return_value=self._response(manager),
        ):
            self.assertTrue(manager.try_ensure_parallel_info("prefill:8998"))

        info = manager.prefill_info_table["prefill:8998"]
        self.assertEqual(info.target_tp_ranks, [0, 1])
        self.assertEqual(info.target_cp_ranks, [0])
        self.assertEqual(info.required_prefill_response_num, 1)

    def test_pd_layout_mismatch_is_still_rejected(self):
        manager = self._manager()
        with patch(
            "sglang.srt.disaggregation.common.conn.requests.get",
            return_value=self._response(manager, layout={"block_size": 64}),
        ):
            with self.assertRaisesRegex(RuntimeError, "layout mismatch"):
                manager.try_ensure_parallel_info("prefill:8998")

    def test_cp_bootstrap_is_not_allowed_by_dp_only_mapping(self):
        manager = self._manager()
        with patch(
            "sglang.srt.disaggregation.common.conn.requests.get",
            return_value=self._response(manager, attn_cp_size=4),
        ):
            with self.assertRaisesRegex(RuntimeError, "CP=1"):
                manager.try_ensure_parallel_info("prefill:8998")


if __name__ == "__main__":
    unittest.main()
