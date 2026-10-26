"""Startup validation of --dsv41-encoder-only-prefill."""

import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.deepseek_v4_hook import (
    validate_deepseek_v41_encoder_only_prefill,
)
from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    Phase,
    PhaseConfig,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _v41_config(**overrides):
    values = dict(
        model_type="deepseek_v41",
        num_hidden_layers=40,
        hc_pre_from_prev_sublayer=True,
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 23, 27, 31, 35],
        engram_layer_ids=[1, 13],
        window_size=128,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class TestDsv41EncoderOnlyPrefillArgs(CustomTestCase):
    def make_args(self, role, config=None, **fields):
        fields.setdefault("enable_decoder_swa_bounded_replay", role == "decode")
        fields.setdefault("disaggregation_transfer_backend", "mooncake")
        args = ServerArgs(
            model_path="dummy",
            dsv41_encoder_only_prefill=True,
            disaggregation_mode=role,
            cuda_graph_config=CudaGraphConfig(prefill=PhaseConfig(backend="disabled")),
            **fields,
        )
        # Checkpoint metadata without loading weights or downloading a model.
        args._model_config = SimpleNamespace(
            hf_config=config or _v41_config(), context_len=1048576
        )
        # As the parsed CLI would record no explicit graph settings.
        args._cuda_graph_config_locked = set()
        return args

    def test_roles_accept_the_official_layout(self):
        args = self.make_args("prefill", enable_decoder_swa_bounded_replay=True)
        validate_deepseek_v41_encoder_only_prefill(args)
        # Prefill never runs the decoder layers, so it must not split at them.
        self.assertFalse(resolving_view(args).enable_decoder_swa_bounded_replay)
        args = self.make_args("decode")
        validate_deepseek_v41_encoder_only_prefill(args)
        self.assertTrue(resolving_view(args).enable_decoder_swa_bounded_replay)
        validate_deepseek_v41_encoder_only_prefill(
            self.make_args("prefill", attn_dp_size=2, tp_size=2)
        )
        # Decode drafts from its own tail extend.
        validate_deepseek_v41_encoder_only_prefill(
            self.make_args("decode", speculative_algorithm="DSPARK")
        )

    def test_decode_graphs_tails_at_any_context(self):
        args = self.make_args("decode")
        validate_deepseek_v41_encoder_only_prefill(args)
        prefill_graph = resolving_view(args).cuda_graph_config.prefill
        self.assertEqual(prefill_graph.backend, Backend.BREAKABLE)
        self.assertEqual(prefill_graph.max_seq_len, 1048576)
        # An explicit setting stands.
        args = self.make_args("decode")
        args._cuda_graph_config_locked = {(Phase.PREFILL, "backend")}
        validate_deepseek_v41_encoder_only_prefill(args)
        prefill_graph = resolving_view(args).cuda_graph_config.prefill
        self.assertEqual(prefill_graph.backend, Backend.DISABLED)

    def test_decode_requires_decoder_replay(self):
        args = self.make_args("decode", enable_decoder_swa_bounded_replay=False)
        with self.assertRaisesRegex(ValueError, "decoder SWA bounded replay"):
            validate_deepseek_v41_encoder_only_prefill(args)

    def test_rejects_unsupported_features(self):
        for role, fields, message in (
            ("prefill", {"speculative_algorithm": "DSPARK"}, "speculative"),
            ("decode", {"disaggregation_decode_enable_radix_cache": True}, "radix"),
            ("prefill", {"disaggregation_transfer_backend": "nixl"}, "Mooncake"),
        ):
            with self.subTest(role=role, **fields):
                with self.assertRaisesRegex(ValueError, message):
                    validate_deepseek_v41_encoder_only_prefill(
                        self.make_args(role, **fields)
                    )

    def test_rejects_other_layouts(self):
        for config in (
            _v41_config(model_type="deepseek_v4"),
            _v41_config(engram_layer_ids=[1, 20]),
            _v41_config(index_source_layer_ids=[2, 8, 14]),
            # A c4 layer would hold a pending block at the split.
            _v41_config(compress_ratios=[0, 0, 4] + [2] * 17 + [1] * 20 + [0] * 3),
        ):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    validate_deepseek_v41_encoder_only_prefill(
                        self.make_args("prefill", config)
                    )


if __name__ == "__main__":
    unittest.main()
