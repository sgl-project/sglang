"""Model facts and derived configs follow their actual constructor inputs."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    model_config_of,
    model_metadata_of,
    resolving_view,
)
from sglang.srt.configs import model_config as config_module
from sglang.srt.configs.model_config import ModelConfig, register_model_config_factory
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class ExtensionArgs(ServerArgs):
    source_tag: str = "first"


class ExtensionConfig(ModelConfig):
    def __init__(self, *, source_tag, **kwargs):
        self.source_tag = source_tag
        super().__init__(**kwargs)

    @staticmethod
    def get_config_inputs(server_args, **kwargs):
        return {
            **ModelConfig.get_config_inputs(server_args, **kwargs),
            "source_tag": resolving_view(server_args).source_tag,
        }


class TestModelConfigInputCache(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.model_path = directory.name
        self.config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "num_hidden_layers": 2,
            "vocab_size": 128,
            "max_position_embeddings": 2048,
            "torch_dtype": "bfloat16",
        }
        (Path(self.model_path) / "config.json").write_text(json.dumps(self.config))
        for patcher in (
            patch.dict(config_module._MODEL_CONFIG_FACTORIES, clear=True),
            patch.dict(os.environ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_same_path_changes_rebuild_from_complete_inputs(self):
        register_model_config_factory(
            ExtensionArgs, ExtensionConfig, inputs=ExtensionConfig.get_config_inputs
        )
        args = ExtensionArgs(model_path=self.model_path, dtype="bfloat16")
        first = model_config_of(args)
        self.assertIs(model_config_of(args), first)

        declare_resolution(args, "dtype_policy", dtype="float32", context_length=1024)
        second = model_config_of(args)
        self.assertIsNot(first, second)
        self.assertEqual((second.dtype, second.context_len), (torch.float32, 1024))
        self.assertEqual(first.dtype, torch.bfloat16)

        declare_resolution(args, "external_policy", source_tag="second")
        third = model_config_of(args)
        self.assertIsNot(second, third)
        self.assertEqual(third.source_tag, "second")
        self.assertEqual(second.source_tag, "first")
        self.assertIs(model_config_of(args), third)

    def test_metadata_is_reused_without_sharing_mutations(self):
        args = ServerArgs(model_path=self.model_path)
        with patch.object(
            config_module, "get_config", wraps=config_module.get_config
        ) as read:
            facts = model_metadata_of(args)
            facts.architectures.append("ModifiedByPolicy")
            first = model_config_of(args)
            self.assertEqual(first.hf_config.architectures, ["LlamaForCausalLM"])
            self.assertEqual(read.call_count, 1)

            declare_resolution(
                args, "overrides", json_model_override_args='{"hidden_size": 32}'
            )
            second = model_config_of(args)
            self.assertIsNot(first, second)
            self.assertEqual(second.hf_config.hidden_size, 32)
            self.assertEqual(first.hf_config.hidden_size, 16)
            self.assertEqual(
                model_metadata_of(args, apply_model_overrides=False).hidden_size, 16
            )
            self.assertEqual(read.call_count, 2)

            other = model_config_of(ServerArgs(model_path=self.model_path))
            other.hf_config.architectures.append("ModifiedByOtherRecord")
            self.assertEqual(first.hf_config.architectures, ["LlamaForCausalLM"])
            self.assertEqual(read.call_count, 3)

    def test_target_and_draft_use_their_own_decrypted_metadata(self):
        target = dict(self.config, hidden_size=32)
        draft = dict(self.config, hidden_size=64)
        for name, config in (("target.json", target), ("draft.json", draft)):
            (Path(self.model_path) / name).write_text(json.dumps(config))
        args = ServerArgs(
            model_path=self.model_path,
            decrypted_config_file="target.json",
            decrypted_draft_config_file="draft.json",
        )
        with patch.object(
            config_module, "get_config", wraps=config_module.get_config
        ) as read:
            self.assertEqual(model_metadata_of(args).hidden_size, 32)
            self.assertEqual(
                model_metadata_of(args, is_draft_model=True).hidden_size, 64
            )
            self.assertEqual(model_config_of(args).hf_config.hidden_size, 32)
            draft_config = ModelConfig.from_server_args(args, is_draft_model=True)
            self.assertEqual(draft_config.hf_config.hidden_size, 64)
            self.assertTrue(draft_config.is_draft_model)
            self.assertEqual(read.call_count, 2)


if __name__ == "__main__":
    unittest.main()
