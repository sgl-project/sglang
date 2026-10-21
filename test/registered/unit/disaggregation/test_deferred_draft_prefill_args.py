"""Deferred draft prefill fails at startup for unsupported ownership layouts."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.pd_disaggregation_hook import (
    _validate_deferred_draft_prefill,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _config(**overrides):
    values = dict(
        disaggregation_mode="prefill",
        disaggregation_transfer_backend="mooncake",
        speculative_algorithm="EAGLE",
        enable_multi_layer_eagle=False,
        speculative_adaptive=False,
        tp_size=1,
        pp_size=4,
        dp_size=1,
        dcp_size=1,
        attn_cp_size=1,
        enable_pd_role_switch=False,
        enable_pdmux=False,
        encoder_only=False,
        language_only=False,
        encoder_urls=[],
        grpc_mode=False,
        smg_grpc_mode=False,
        enable_lora=False,
        lora_paths=None,
        ep_size=1,
        is_embedding=False,
        decoupled_spec_role="null",
        disable_radix_cache=True,
        disaggregation_decode_enable_radix_cache=False,
        enable_hierarchical_cache=False,
        enable_hisparse=False,
        enable_unified_memory=False,
        disaggregation_decode_host_receive_threshold=1,
        disaggregation_decode_enable_offload_kvcache=False,
        optimistic_prefill_attempts=0,
        disaggregation_decode_allocation_policy="early",
        disaggregation_draft_prefill_max_input_tokens=8192,
        disaggregation_draft_prefill_slots=8,
        max_prefill_tokens=2048,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _model(architecture="Glm5NextForConditionalGeneration", nextn_layers=1):
    return SimpleNamespace(
        hf_config=SimpleNamespace(architectures=[architecture], index_topk=3),
        hf_text_config=SimpleNamespace(num_nextn_predict_layers=nextn_layers),
        hidden_size=4,
        spec_hidden_size=4,
    )


class TestDeferredDraftPrefillArgs(CustomTestCase):
    def test_pp_prefill_and_tp_decode_are_supported(self):
        _validate_deferred_draft_prefill(_config(), _model())
        _validate_deferred_draft_prefill(
            _config(
                disaggregation_mode="decode",
                tp_size=4,
                pp_size=1,
                max_prefill_tokens=8192,
            ),
            _model(),
        )

    def test_unsupported_layouts_fail_before_buffers_are_registered(self):
        for changes, message in (
            ({"disaggregation_mode": "null"}, "aggregated serving"),
            ({"disaggregation_transfer_backend": "nixl"}, "Mooncake"),
            ({"speculative_algorithm": None}, "single-layer EAGLE"),
            ({"enable_multi_layer_eagle": True}, "multi-layer"),
            ({"tp_size": 4}, "prefill tensor parallelism"),
            ({"disaggregation_mode": "decode"}, "decode pipeline parallelism"),
            ({"disable_radix_cache": False}, "prefix caching"),
            ({"dcp_size": 2}, "context parallelism"),
            ({"enable_unified_memory": True}, "unified memory"),
            ({"optimistic_prefill_attempts": 1}, "optimistic prefill"),
            ({"encoder_only": True}, "encoder disaggregation"),
            ({"grpc_mode": True}, "gRPC"),
            ({"enable_lora": True}, "LoRA"),
            ({"ep_size": 4}, "expert parallelism"),
        ):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, message):
                    _validate_deferred_draft_prefill(_config(**changes), _model())

    def test_deferred_transfer_drain_tracking_is_required(self):
        with patch(
            "sglang.srt.arg_groups.pd_disaggregation_hook.envs."
            "SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE.get",
            return_value=False,
        ):
            with self.assertRaisesRegex(
                ValueError, "disabled deferred decode KV release"
            ):
                _validate_deferred_draft_prefill(_config(), _model())

    def test_explicit_positive_staging_budget_is_required(self):
        for changes in (
            {"disaggregation_draft_prefill_max_input_tokens": None},
            {"disaggregation_draft_prefill_max_input_tokens": 0},
            {"disaggregation_draft_prefill_slots": 0},
        ):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, "requires positive"):
                    _validate_deferred_draft_prefill(_config(**changes), _model())

    def test_unknown_draft_architecture_is_not_silently_enabled(self):
        for model in (_model("LlamaForCausalLM"), _model(nextn_layers=2)):
            with self.assertRaisesRegex(ValueError, "one DSA NextN draft layer"):
                _validate_deferred_draft_prefill(_config(), model)

    def test_decode_scratch_budget_must_cover_full_draft_initialization(self):
        for budget in (None, 2048, 8192):
            with self.subTest(max_prefill_tokens=budget):
                with self.assertRaisesRegex(ValueError, "--max-prefill-tokens"):
                    _validate_deferred_draft_prefill(
                        _config(
                            disaggregation_mode="decode",
                            tp_size=4,
                            pp_size=1,
                            disaggregation_draft_prefill_max_input_tokens=9216,
                            max_prefill_tokens=budget,
                        ),
                        _model(),
                    )
        _validate_deferred_draft_prefill(
            _config(
                disaggregation_mode="decode",
                tp_size=4,
                pp_size=1,
                disaggregation_draft_prefill_max_input_tokens=9216,
                max_prefill_tokens=9216,
            ),
            _model(),
        )
        # P still computes small chunks and accumulates hidden rows across them.
        _validate_deferred_draft_prefill(_config(max_prefill_tokens=2048), _model())

    def test_expanded_target_capture_is_rejected_before_allocating_h_rows(self):
        model = _model()
        model.spec_hidden_size = 16
        with self.assertRaisesRegex(ValueError, "expanded speculative target hidden"):
            _validate_deferred_draft_prefill(_config(), model)


if __name__ == "__main__":
    unittest.main()
