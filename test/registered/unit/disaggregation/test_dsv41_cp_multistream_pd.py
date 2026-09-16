"""DeepSeek-V4.1 PD prefill CP4 with eager multi-stream prepare."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _feature_config(*, role: str):
    from sglang.srt.model_executor.cuda_graph_config import Backend

    is_prefill = role == "prefill"
    return SimpleNamespace(
        enable_encoder_swa_bounded_replay=False,
        enable_decoder_swa_bounded_replay=is_prefill,
        enable_dp_attention=is_prefill,
        enable_prefill_cp=is_prefill,
        dp_size=1,
        speculative_algorithm="DSPARK",
        enable_hisparse=False,
        dsv4_attn_backend="auto",
        enable_two_batch_overlap=False,
        pp_size=1,
        attn_cp_size=4 if is_prefill else 1,
        dcp_size=1,
        disaggregation_mode=role,
        disaggregation_transfer_backend="mooncake",
        moe_a2a_backend="none",
        cuda_graph_config=SimpleNamespace(
            prefill=SimpleNamespace(backend=Backend.DISABLED, max_seq_len=4096)
        ),
    )


class TestDSV41CPMultiStreamPD(CustomTestCase):
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

    def test_dspark_pd_accepts_cp4_prefill_and_tp4_decode(self):
        self._validate_features(_feature_config(role="prefill"))
        self._validate_features(_feature_config(role="decode"))

        invalid = _feature_config(role="decode")
        invalid.attn_cp_size = 4
        with self.assertRaisesRegex(ValueError, "decode CP=1"):
            self._validate_features(invalid)

    def test_bootstrap_maps_cp4_prefill_to_tp4_decode(self):
        manager = object.__new__(CommonKVManager)
        manager.prefill_info_table = {}
        manager.kv_args = SimpleNamespace(page_size=256, engine_rank=0)
        manager.kv_cache_dtype_str = "fp8_e4m3"
        manager.dsv41_spec_layout = {"block_size": 5}
        manager.attn_tp_size = 4
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
            attn_cp_size=4,
            dp_size=1,
            pp_size=1,
            page_size=256,
            kv_cache_dtype="fp8_e4m3",
            follow_bootstrap_room=True,
            dsv41_spec_layout={"block_size": 5},
        )
        with patch(
            "sglang.srt.disaggregation.common.conn.requests.get",
            return_value=response,
        ):
            self.assertTrue(manager.try_ensure_parallel_info("prefill:8998"))

        info = manager.prefill_info_table["prefill:8998"]
        self.assertEqual(info.target_tp_ranks, [0])
        self.assertEqual(info.target_cp_ranks, [0])
        self.assertEqual(info.required_dst_info_num, 4)
        self.assertEqual(info.required_prefill_response_num, 1)

    def test_v41_text_only_pd_is_allowed(self):
        from sglang.srt.arg_groups import model_hook

        cfg = SimpleNamespace(
            language_model_only=True,
            encoder_only=False,
            language_only=False,
            enable_prefix_mm_cache=False,
            enable_broadcast_mm_inputs_process=False,
            mm_enable_dp_encoder=False,
            disaggregation_mode="prefill",
        )
        args = SimpleNamespace(
            LANGUAGE_MODEL_ONLY_ARCHITECTURES=("DeepseekV4ForCausalLM",)
        )
        model = SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="deepseek_v41",
                architectures=["DeepseekV4ForCausalLM"],
            )
        )
        with (
            patch.object(model_hook, "resolving_view", return_value=cfg),
            patch.object(model_hook, "model_config_of", return_value=model),
        ):
            model_hook.handle_language_model_only(args)

    def test_cp_prepares_mm_ids_before_sharding(self):
        from sglang.srt.managers.schedule_batch import MM_PAD_SHIFT_VALUE
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

        embeds = torch.zeros((3, 4))
        model = SimpleNamespace(
            vision=object(),
            config=SimpleNamespace(image_token_id=17),
            _prepare_mm_embeddings=Mock(return_value=embeds),
        )
        mode = SimpleNamespace(
            is_decode=lambda: False,
            is_target_verify=lambda: False,
            is_decode_or_idle=lambda: False,
        )
        batch = SimpleNamespace(forward_mode=mode, mm_inputs=[object()])
        scheduler_ids = torch.tensor([3, MM_PAD_SHIFT_VALUE + 5, 4])
        model_ids, model_embeds = DeepseekV4ForCausalLM.prepare_language_model_inputs(
            model, scheduler_ids, batch
        )
        self.assertEqual(model_ids.tolist(), [3, 17, 4])
        self.assertEqual(scheduler_ids.tolist(), [3, MM_PAD_SHIFT_VALUE + 5, 4])
        self.assertIs(model_embeds, embeds)

    def test_fake_transfer_keeps_chunk_kv_without_radix_insert(self):
        from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
        from sglang.srt.mem_cache.common import maybe_cache_unfinished_req

        req = SimpleNamespace(
            skip_radix_cache_insert=True,
            bootstrap_host=FAKE_BOOTSTRAP_HOST,
            kv=SimpleNamespace(req_pool_idx=0),
            get_fill_ids=lambda: [1, 2, 3],
            prefix_indices=torch.empty((0,), dtype=torch.int64),
        )
        cache = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.tensor([[11, 12, 13, 14]], dtype=torch.int32)
            ),
            cache_unfinished_req=Mock(),
        )
        maybe_cache_unfinished_req(req, cache)
        self.assertEqual(req.prefix_indices.tolist(), [11, 12, 13])
        cache.cache_unfinished_req.assert_not_called()

        # A different skip caller, such as beam search, must not gain this KV.
        req.bootstrap_host = "ordinary-skip"
        req.prefix_indices = torch.empty((0,), dtype=torch.int64)
        maybe_cache_unfinished_req(req, cache)
        self.assertEqual(req.prefix_indices.numel(), 0)


if __name__ == "__main__":
    unittest.main()
