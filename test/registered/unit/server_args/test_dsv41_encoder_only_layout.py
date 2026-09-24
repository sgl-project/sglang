import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.deepseek_v4_hook import (
    _validate_dsv41_encoder_only_ratio_layout,
    validate_deepseek_v41_features,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _official_config(**overrides):
    values = dict(
        num_hidden_layers=40,
        num_nextn_predict_layers=3,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _encoder_only_args(**overrides):
    values = dict(
        dsv41_encoder_only_prefill=True,
        disable_overlap_schedule=False,
        disaggregation_mode="prefill",
        disaggregation_transfer_backend="mooncake",
        pp_size=1,
        dp_size=1,
        attn_cp_size=1,
        dcp_size=1,
        enable_prefill_cp=False,
        enable_two_batch_overlap=False,
        enable_mixed_chunk=False,
        enable_lora=False,
        enable_hisparse=False,
        enable_dp_attention=False,
        enable_unified_cache_external_linker=False,
        enable_unified_memory=False,
        enable_session_radix_cache=False,
        speculative_algorithm=None,
        dsv4_attn_backend="auto",
        enable_decoder_swa_bounded_replay=True,
        max_running_requests=64,
        chunked_prefill_size=8192,
        enable_encoder_swa_bounded_replay=False,
        cuda_graph_config=SimpleNamespace(prefill=SimpleNamespace(backend="disabled")),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class TestDsv41EncoderOnlyRatioLayout(unittest.TestCase):
    def test_accepts_official_target_plus_bundled_mtp_layout(self):
        _validate_dsv41_encoder_only_ratio_layout(_official_config())

    def test_rejects_missing_bundled_mtp_ratios(self):
        with self.assertRaisesRegex(ValueError, "expected 40\\+3"):
            _validate_dsv41_encoder_only_ratio_layout(
                _official_config(compress_ratios=[0, 0] + [2] * 18 + [1] * 20)
            )

    def test_rejects_nonzero_bundled_mtp_ratio(self):
        ratios = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 1]
        with self.assertRaisesRegex(ValueError, "MTP layers to have ratio 0"):
            _validate_dsv41_encoder_only_ratio_layout(
                _official_config(compress_ratios=ratios)
            )

    def test_rejects_invalid_target_ratio(self):
        ratios = [0, 0] + [2] * 18 + [1] * 20 + [0] * 3
        ratios[10] = 4
        with self.assertRaisesRegex(ValueError, "target-layer ratio-0/1/2"):
            _validate_dsv41_encoder_only_ratio_layout(
                _official_config(compress_ratios=ratios)
            )

    def test_rejects_nonproducer_boundary(self):
        ratios = [0, 0] + [2] * 18 + [1] * 20 + [0] * 3
        ratios[20] = 0
        with self.assertRaisesRegex(ValueError, "layer 20"):
            _validate_dsv41_encoder_only_ratio_layout(
                _official_config(compress_ratios=ratios)
            )

    @patch("sglang.srt.arg_groups.deepseek_v4_hook.get_platform")
    @patch("sglang.srt.arg_groups.deepseek_v4_hook.model_config_of")
    @patch("sglang.srt.arg_groups.deepseek_v4_hook.resolving_view")
    def test_encoder_only_accepts_regular_overlap_scheduler(
        self, resolving_view, model_config_of, get_platform
    ):
        hf_config = _official_config(
            model_type="deepseek_v41",
            hc_pre_from_prev_sublayer=True,
            kv_source_layer_ids=[2, 8, 14, 20],
            index_source_layer_ids=[2, 8, 14, 20],
            engram_layer_ids=[],
        )
        cfg = _encoder_only_args(disable_overlap_schedule=False)
        resolving_view.return_value = cfg
        model_config_of.return_value = SimpleNamespace(hf_config=hf_config)
        get_platform.return_value.is_cuda = True

        validate_deepseek_v41_features(SimpleNamespace())

    @patch("sglang.srt.arg_groups.deepseek_v4_hook.get_platform")
    @patch("sglang.srt.arg_groups.deepseek_v4_hook.model_config_of")
    @patch("sglang.srt.arg_groups.deepseek_v4_hook.resolving_view")
    def test_encoder_only_accepts_decode_local_dspark_only(
        self, resolving_view, model_config_of, get_platform
    ):
        hf_config = _official_config(
            model_type="deepseek_v41",
            hc_pre_from_prev_sublayer=True,
            kv_source_layer_ids=[2, 8, 14, 20],
            index_source_layer_ids=[2, 8, 14, 20],
            engram_layer_ids=[],
        )
        model_config_of.return_value = SimpleNamespace(hf_config=hf_config)
        get_platform.return_value.is_cuda = True

        for role, accepted in (("prefill", False), ("decode", True)):
            with self.subTest(role=role):
                resolving_view.return_value = _encoder_only_args(
                    disaggregation_mode=role,
                    speculative_algorithm="DSPARK",
                )
                if accepted:
                    validate_deepseek_v41_features(SimpleNamespace())
                else:
                    with self.assertRaisesRegex(ValueError, "Decode-local DSpark"):
                        validate_deepseek_v41_features(SimpleNamespace())

    @patch("sglang.srt.arg_groups.deepseek_v4_hook.get_platform")
    @patch("sglang.srt.arg_groups.deepseek_v4_hook.model_config_of")
    @patch("sglang.srt.arg_groups.deepseek_v4_hook.resolving_view")
    def test_encoder_only_still_rejects_two_batch_overlap(
        self, resolving_view, model_config_of, get_platform
    ):
        hf_config = _official_config(
            model_type="deepseek_v41",
            hc_pre_from_prev_sublayer=True,
            kv_source_layer_ids=[2, 8, 14, 20],
            index_source_layer_ids=[2, 8, 14, 20],
            engram_layer_ids=[],
        )
        cfg = _encoder_only_args(enable_two_batch_overlap=True)
        resolving_view.return_value = cfg
        model_config_of.return_value = SimpleNamespace(hf_config=hf_config)
        get_platform.return_value.is_cuda = True

        with self.assertRaisesRegex(ValueError, "two-batch overlap"):
            validate_deepseek_v41_features(SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
