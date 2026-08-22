import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.arg_groups.speculative_hook import (
    _draft_config_kwargs,
    _resolve_speculative_algorithm_alias,
)
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _server_args(**kwargs) -> ServerArgs:
    server_args = ServerArgs.__new__(ServerArgs)
    defaults = {
        "model_path": "target",
        "trust_remote_code": False,
        "revision": None,
        "context_length": None,
        "json_model_override_args": "{}",
        "speculative_draft_model_override_args": None,
        "speculative_draft_model_quantization": None,
        "quantization": None,
        "decrypted_draft_config_file": None,
        "decrypted_config_file": None,
        "is_embedding": False,
        "enable_multimodal": False,
        "dtype": "auto",
        "model_impl": "auto",
        "sampling_defaults": "model",
        "quantize_and_serve": None,
        "enable_multi_layer_eagle": False,
        "language_only": False,
        "language_model_only": False,
        "encoder_only": False,
        "_speculative_draft_quantization_explicitly_set": False,
        "disable_hybrid_swa_memory": False,
        "model_config_parser": None,
        "speculative_algorithm": None,
    }
    defaults.update(kwargs)
    for name, value in defaults.items():
        setattr(server_args, name, value)
    return server_args


class TestDraftModelOverrideArgs(unittest.TestCase):
    def test_draft_override_falls_back_to_target_override(self) -> None:
        server_args = _server_args(
            json_model_override_args='{"rope_scaling": "target"}'
        )

        self.assertEqual(
            server_args.get_draft_model_override_args(),
            server_args.json_model_override_args,
        )

    def test_explicit_draft_override_wins(self) -> None:
        server_args = _server_args(
            json_model_override_args='{"rope_scaling": "target"}',
            speculative_draft_model_override_args='{"draft_window_size": 2048}',
        )

        self.assertEqual(
            server_args.get_draft_model_override_args(),
            '{"draft_window_size": 2048}',
        )

    def test_model_config_uses_target_and_draft_overrides_independently(
        self,
    ) -> None:
        target_override = '{"rope_scaling": "target"}'
        draft_override = '{"draft_window_size": 2048}'
        server_args = _server_args(
            json_model_override_args=target_override,
            speculative_draft_model_override_args=draft_override,
        )

        with mock.patch.object(ModelConfig, "__init__", return_value=None) as init:
            ModelConfig.from_server_args(server_args, is_draft_model=False)
            target_kwargs = init.call_args.kwargs
            ModelConfig.from_server_args(
                server_args, model_path="draft", is_draft_model=True
            )
            draft_kwargs = init.call_args.kwargs

        self.assertEqual(target_kwargs["model_override_args"], target_override)
        self.assertEqual(draft_kwargs["model_override_args"], draft_override)

    def test_algorithm_detection_receives_draft_overrides(self) -> None:
        server_args = _server_args(
            json_model_override_args='{"architectures": ["TargetModel"]}',
            speculative_draft_model_override_args=(
                '{"architectures": ["Gemma4AssistantForCausalLM"]}'
            ),
            decrypted_draft_config_file="  /tmp/draft-config.json  ",
        )

        expected_kwargs = {
            "model_override_args": {"architectures": ["Gemma4AssistantForCausalLM"]},
            "_configuration_file": "/tmp/draft-config.json",
        }
        with mock.patch(
            "sglang.srt.utils.hf_transformers_utils.get_config",
            return_value=SimpleNamespace(architectures=["Gemma4AssistantForCausalLM"]),
        ) as get_config:
            resolved = _resolve_speculative_algorithm_alias(
                "EAGLE",
                "draft",
                trust_remote_code=False,
                kwargs=_draft_config_kwargs(server_args),
            )

        self.assertEqual(resolved, "FROZEN_KV_MTP")
        get_config.assert_called_once_with(
            "draft", trust_remote_code=False, **expected_kwargs
        )


if __name__ == "__main__":
    unittest.main()
