"""Guardrails for the Gemma 4 Frozen-KV MTP MLX prototype."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from sglang.srt.arg_groups.speculative_hook import (
    _handle_frozen_kv_mtp,
    _resolve_speculative_algorithm_alias,
)
from sglang.srt.hardware_backend.mlx.spec_config import (
    validate_mlx_frozen_kv_mtp_args,
    validate_mlx_frozen_kv_mtp_request,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")


def _target_config():
    text = SimpleNamespace(
        model_type="gemma4_text",
        hidden_size=1536,
        num_hidden_layers=35,
        vocab_size=262144,
        num_kv_shared_layers=20,
        layer_types=["sliding_attention"] * 34 + ["full_attention"],
    )
    hf = SimpleNamespace(
        model_type="gemma4",
        architectures=["Gemma4ForConditionalGeneration"],
        text_config=text,
    )
    return SimpleNamespace(hf_config=hf, context_len=2048, is_multimodal=False)


def _assistant_config() -> dict:
    return {
        "model_type": "gemma4_assistant",
        "architectures": ["Gemma4AssistantForCausalLM"],
        "backbone_hidden_size": 1536,
        "use_ordered_embeddings": True,
        "num_centroids": 2048,
        "centroid_intermediate_top_k": 32,
        "tie_word_embeddings": True,
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "vocab_size": 262144,
            "num_kv_shared_layers": 4,
            "layer_types": [
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
            "sliding_window": 512,
            "tie_word_embeddings": True,
        },
    }


def _server_args(**overrides):
    values = dict(
        speculative_algorithm="FROZEN_KV_MTP",
        speculative_draft_model_path="assistant",
        speculative_draft_model_revision="revision",
        speculative_eagle_topk=1,
        speculative_num_steps=1,
        speculative_num_draft_tokens=2,
        speculative_use_rejection_sampling=False,
        max_running_requests=1,
        disable_overlap_schedule=True,
        disable_radix_cache=True,
        chunked_prefill_size=-1,
        enable_mixed_chunk=False,
        context_length=2048,
        max_total_tokens=2048,
        tp_size=1,
        pp_size=1,
        dp_size=1,
        nnodes=1,
        disaggregation_mode="null",
        enable_dp_attention=False,
        model_path="target",
        revision="target-revision",
        trust_remote_code=False,
        get_model_config=_target_config,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class TestMlxGemma4Alias(unittest.TestCase):
    def test_local_unknown_transformers_config_resolves_before_get_config(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(
                json.dumps(_assistant_config()), encoding="utf-8"
            )
            with mock.patch(
                "sglang.srt.utils.hf_transformers_utils.get_config",
                side_effect=AssertionError("Transformers must not load this config"),
            ):
                self.assertEqual(
                    _resolve_speculative_algorithm_alias("NEXTN", directory),
                    "FROZEN_KV_MTP",
                )

    def test_eagle3_is_rejected_for_gemma4_assistant(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(
                json.dumps(_assistant_config()), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "EAGLE3"):
                _resolve_speculative_algorithm_alias("EAGLE3", directory)


class TestMlxWorkerDispatch(unittest.TestCase):
    def test_mlx_frozen_kv_uses_mlx_worker(self):
        with mock.patch("sglang.srt.utils.tensor_bridge.use_mlx", return_value=True):
            worker = SpeculativeAlgorithm.FROZEN_KV_MTP.create_worker(_server_args())
        self.assertEqual(worker.__name__, "MlxFrozenKVMTPWorker")

    def test_non_mlx_frozen_kv_keeps_generic_worker(self):
        with mock.patch("sglang.srt.utils.tensor_bridge.use_mlx", return_value=False):
            worker = SpeculativeAlgorithm.FROZEN_KV_MTP.create_worker(_server_args())
        self.assertEqual(worker.__name__, "FrozenKVMTPWorkerV2")


class TestMlxServerGuardrails(unittest.TestCase):
    def test_handler_normalizes_safe_defaults(self):
        args = _server_args(
            max_running_requests=None,
            disable_overlap_schedule=False,
            speculative_eagle_topk=None,
            speculative_num_steps=None,
            speculative_num_draft_tokens=None,
        )
        with (
            mock.patch(
                "sglang.srt.arg_groups.speculative_hook.use_mlx", return_value=True
            ),
            mock.patch(
                "sglang.srt.hardware_backend.mlx.spec_config.load_assistant_config_dict",
                return_value=_assistant_config(),
            ),
            self.assertLogs(level="WARNING"),
        ):
            _handle_frozen_kv_mtp(args)
        self.assertEqual(args.max_running_requests, 1)
        self.assertTrue(args.disable_overlap_schedule)
        self.assertEqual(args.speculative_eagle_topk, 1)
        self.assertEqual(args.speculative_num_steps, 1)
        self.assertEqual(args.speculative_num_draft_tokens, 2)

    def test_supported_shape_passes(self):
        with mock.patch(
            "sglang.srt.hardware_backend.mlx.spec_config.load_assistant_config_dict",
            return_value=_assistant_config(),
        ):
            validate_mlx_frozen_kv_mtp_args(_server_args())

    def test_each_unsupported_server_dimension_fails(self):
        cases = [
            ({"speculative_draft_model_path": None}, "draft-model-path"),
            ({"disable_radix_cache": False}, "disable-radix-cache"),
            ({"chunked_prefill_size": 64}, "chunked-prefill-size -1"),
            ({"enable_mixed_chunk": True}, "mixed chunked prefill"),
            ({"speculative_eagle_topk": 2}, "topk"),
            ({"speculative_num_steps": 2}, "num_steps"),
            ({"speculative_num_draft_tokens": 3}, "draft_tokens"),
            ({"max_running_requests": 2}, "max-running-requests 1"),
            ({"context_length": 2049}, "2,048"),
            ({"max_total_tokens": 2049}, "max-total-tokens"),
            ({"tp_size": 2}, "tp-size 1"),
            ({"dp_size": 2}, "dp-size 1"),
            ({"pp_size": 2}, "pp-size 1"),
            ({"nnodes": 2}, "one Apple host"),
            ({"enable_dp_attention": True}, "DP attention"),
            ({"disaggregation_mode": "prefill"}, "disaggregation"),
            ({"disable_overlap_schedule": False}, "synchronous scheduling"),
            ({"speculative_use_rejection_sampling": True}, "rejection sampling"),
        ]
        for changes, message in cases:
            with self.subTest(changes=changes):
                with (
                    mock.patch(
                        "sglang.srt.hardware_backend.mlx.spec_config.load_assistant_config_dict",
                        return_value=_assistant_config(),
                    ),
                    self.assertRaisesRegex((ValueError, NotImplementedError), message),
                ):
                    validate_mlx_frozen_kv_mtp_args(_server_args(**changes))

    def test_wrong_target_or_assistant_fails_before_load(self):
        wrong_target = _target_config()
        wrong_target.hf_config.text_config.hidden_size = 2048
        with (
            mock.patch(
                "sglang.srt.hardware_backend.mlx.spec_config.load_assistant_config_dict",
                return_value=_assistant_config(),
            ),
            self.assertRaisesRegex(ValueError, "E2B"),
        ):
            validate_mlx_frozen_kv_mtp_args(
                _server_args(get_model_config=lambda: wrong_target)
            )

        wrong_assistant = _assistant_config()
        wrong_assistant["model_type"] = "gemma4"
        with (
            mock.patch(
                "sglang.srt.hardware_backend.mlx.spec_config.load_assistant_config_dict",
                return_value=wrong_assistant,
            ),
            self.assertRaisesRegex(ValueError, "gemma4_assistant"),
        ):
            validate_mlx_frozen_kv_mtp_args(_server_args())


class TestMlxRequestGuardrails(unittest.TestCase):
    def _request(self, sampling_params=None, **overrides):
        values = dict(
            sampling_params=sampling_params or SamplingParams(temperature=0),
            return_logprob=False,
            return_hidden_states=False,
            return_sampling_mask=False,
            custom_logit_processor=None,
            session=None,
            session_id=None,
            multimodal_inputs=None,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_greedy_request_passes(self):
        self.assertIsNone(validate_mlx_frozen_kv_mtp_request(self._request()))

    def test_normalized_empty_grammar_fields_are_not_constraints(self):
        params = SamplingParams(temperature=0)
        # Endpoint/model defaults may retain irrelevant nucleus values after
        # temperature=0 normalizes top_k to one.
        params.top_p = 0.95
        params.min_p = 0.1
        params.stop_regex_strs = []
        self.assertIsNone(validate_mlx_frozen_kv_mtp_request(self._request(params)))

    def test_unsupported_request_features_return_actionable_errors(self):
        cases = [
            (SamplingParams(temperature=0.5), {}, "temperature=0"),
            (SamplingParams(temperature=0, frequency_penalty=0.5), {}, "penalties"),
            (SamplingParams(temperature=0, presence_penalty=0.5), {}, "penalties"),
            (SamplingParams(temperature=0, repetition_penalty=1.1), {}, "penalties"),
            (SamplingParams(temperature=0, min_new_tokens=1), {}, "penalties"),
            (SamplingParams(temperature=0, logit_bias={"1": 1.0}), {}, "logit bias"),
            (SamplingParams(temperature=0, regex="a+"), {}, "grammar"),
            (SamplingParams(temperature=0, n=2), {}, "one completion"),
            (
                SamplingParams(temperature=0, custom_params={"sampler": "custom"}),
                {},
                "custom sampling",
            ),
            (SamplingParams(temperature=0), {"return_logprob": True}, "logprobs"),
            (
                SamplingParams(temperature=0),
                {"return_hidden_states": True},
                "hidden states",
            ),
            (
                SamplingParams(temperature=0),
                {"return_sampling_mask": True},
                "sampling masks",
            ),
            (
                SamplingParams(temperature=0),
                {"custom_logit_processor": "processor"},
                "custom logits",
            ),
            (SamplingParams(temperature=0), {"session_id": "session"}, "sessions"),
            (SamplingParams(temperature=0), {"lora_id": "adapter"}, "LoRA"),
            (
                SamplingParams(temperature=0),
                {"multimodal_inputs": object()},
                "text-only",
            ),
        ]
        for params, overrides, message in cases:
            with self.subTest(message=message):
                error = validate_mlx_frozen_kv_mtp_request(
                    self._request(params, **overrides)
                )
                self.assertIsNotNone(error)
                self.assertIn(message, error)


if __name__ == "__main__":
    unittest.main()
