"""Checkpoint-layout regressions for the encoder-only startup guard."""

import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.deepseek_v4_hook import validate_deepseek_v41_features
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _official_config(**overrides):
    values = dict(
        model_type="deepseek_v41",
        num_hidden_layers=40,
        num_nextn_predict_layers=3,
        hc_pre_from_prev_sublayer=True,
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20],
        window_size=128,
        vision_n_layers=0,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class TestDsv41EncoderOnlyLayout(CustomTestCase):
    def validate(self, config, *, role="prefill", **fields):
        args = ServerArgs(
            model_path="dummy",
            dsv41_encoder_only_prefill=True,
            disaggregation_mode=role,
            disaggregation_transfer_backend="mooncake",
            attn_dp_size=1,
            max_running_requests=8,
            chunked_prefill_size=8192,
            cuda_graph_config=CudaGraphConfig(prefill=PhaseConfig(backend="disabled")),
            **fields,
        )
        # Supply checkpoint metadata without loading weights or downloading a model.
        args._model_config = SimpleNamespace(hf_config=config)
        with override_platform(is_cuda=True):
            validate_deepseek_v41_features(args)

    def test_bundled_mtp_ratios_are_separate_from_target_layers(self):
        """Official checkpoints include three MTP ratios beyond the 40 target layers."""
        config = _official_config()
        for role in ("prefill", "decode"):
            with self.subTest(role=role):
                self.validate(config, role=role)
        for ratios in (config.compress_ratios[:40], config.compress_ratios[:-1] + [1]):
            with self.subTest(ratios=ratios):
                with self.assertRaisesRegex(ValueError, "bundled MTP"):
                    self.validate(_official_config(compress_ratios=ratios))

    def test_window_and_vision_match_the_replay_boundary(self):
        for fields, message in (
            ({"window_size": 256}, "window_size=128"),
            ({"vision_n_layers": 24}, "text-only model"),
        ):
            with self.subTest(**fields):
                with self.assertRaisesRegex(ValueError, message):
                    self.validate(_official_config(**fields))
        self.validate(_official_config(vision_n_layers=24, language_model_only=True))

    def test_dspark_is_decode_local(self):
        self.validate(_official_config(), role="decode", speculative_algorithm="DSPARK")
        with self.assertRaisesRegex(ValueError, "Decode-local DSpark"):
            self.validate(_official_config(), speculative_algorithm="DSPARK")


if __name__ == "__main__":
    unittest.main()
