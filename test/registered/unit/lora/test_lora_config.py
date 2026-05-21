"""Unit tests for LoRAConfig."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import json
import os
import tempfile
import unittest
import unittest.mock as mock

from sglang.srt.lora.lora_config import LoRAConfig
from sglang.test.test_utils import CustomTestCase


def write_adapter_files(tmpdir, config=None, added_tokens=None):
    config = config or {
        "target_modules": ["q_proj", "v_proj"],
        "r": 8,
        "lora_alpha": 16,
    }
    with open(os.path.join(tmpdir, "adapter_config.json"), "w") as f:
        json.dump(config, f)

    if added_tokens is not None:
        with open(os.path.join(tmpdir, "added_tokens.json"), "w") as f:
            json.dump(added_tokens, f)

    return config


class TestLoRAConfig(CustomTestCase):
    def test_from_dict_sets_config_fields(self):
        config = {
            "target_modules": ["q_proj", "v_proj"],
            "r": 8,
            "lora_alpha": 16,
        }

        lora_config = LoRAConfig.from_dict(config)

        self.assertEqual(lora_config.target_modules, ["q_proj", "v_proj"])
        self.assertEqual(lora_config.r, 8)
        self.assertEqual(lora_config.lora_alpha, 16)
        self.assertEqual(lora_config.lora_added_tokens_size, 0)

    def test_loads_adapter_config_from_local_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            write_adapter_files(
                tmpdir,
                config={
                    "target_modules": ["q_proj", "k_proj", "v_proj"],
                    "r": 16,
                    "lora_alpha": 32,
                },
            )

            lora_config = LoRAConfig(path=tmpdir)

        self.assertEqual(lora_config.target_modules, ["q_proj", "k_proj", "v_proj"])
        self.assertEqual(lora_config.r, 16)
        self.assertEqual(lora_config.lora_alpha, 32)

    def test_filters_base_vocab_tokens_from_added_tokens(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            write_adapter_files(
                tmpdir,
                added_tokens={"<copied_token_0>": 10, "<copied_token_1>": 99},
            )

            lora_config = LoRAConfig(path=tmpdir, base_vocab_size=100)

        self.assertEqual(lora_config.lora_added_tokens_size, 0)

    def test_raises_when_adapter_has_unsupported_added_tokens(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            write_adapter_files(
                tmpdir,
                added_tokens={"<copied_token>": 10, "<new_token>": 100},
            )

            with self.assertRaisesRegex(ValueError, "1 added tokens"):
                LoRAConfig(path=tmpdir, base_vocab_size=100)

    @mock.patch("sglang.srt.lora.lora_config.snapshot_download")
    def test_downloaded_adapter_path_loads_config(self, mock_download):
        with tempfile.TemporaryDirectory() as tmpdir:
            write_adapter_files(
                tmpdir,
                config={
                    "target_modules": ["o_proj"],
                    "r": 4,
                    "lora_alpha": 8,
                },
            )
            mock_download.return_value = tmpdir

            lora_config = LoRAConfig(path="hf-adapter/repo")

        mock_download.assert_any_call("hf-adapter/repo", allow_patterns=["*.json"])
        self.assertEqual(lora_config.target_modules, ["o_proj"])
        self.assertEqual(lora_config.r, 4)
        self.assertEqual(lora_config.lora_alpha, 8)


if __name__ == "__main__":
    unittest.main()
