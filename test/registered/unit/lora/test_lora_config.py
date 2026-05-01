"""Unit tests for LoRAConfig.

Tests LoRA adapter configuration loading and validation without
requiring a real adapter or HuggingFace Hub connection.

Usage:
    pytest test/registered/unit/lora/test_lora_config.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import json
import os
import tempfile
import unittest
import unittest.mock as mock

from sglang.srt.lora.lora_config import LoRAConfig
from sglang.test.test_utils import CustomTestCase


class TestLoRAConfigInit(CustomTestCase):
    """Test LoRAConfig.__init__ behavior."""

    def test_init_with_config_dict_basic(self):
        """Direct initialization with config_dict sets all attributes."""
        config = {
            "target_modules": ["q_proj", "v_proj"],
            "r": 8,
            "lora_alpha": 16,
        }
        lora_config = LoRAConfig(config_dict=config)

        self.assertEqual(lora_config.target_modules, ["q_proj", "v_proj"])
        self.assertEqual(lora_config.r, 8)
        self.assertEqual(lora_config.lora_alpha, 16)
        self.assertEqual(lora_config.lora_added_tokens_size, 0)

    def test_init_with_config_dict_single_target_module(self):
        """Single target_module in config."""
        config = {
            "target_modules": ["q_proj"],
            "r": 4,
            "lora_alpha": 8,
        }
        lora_config = LoRAConfig(config_dict=config)

        self.assertEqual(lora_config.target_modules, ["q_proj"])
        self.assertEqual(lora_config.r, 4)

    def test_init_without_added_tokens(self):
        """No added_tokens_config results in lora_added_tokens_size = 0."""
        config = {
            "target_modules": ["q_proj"],
            "r": 8,
            "lora_alpha": 16,
        }
        lora_config = LoRAConfig(config_dict=config)

        self.assertEqual(lora_config.lora_added_tokens_size, 0)

    def test_init_with_added_tokens_no_base_vocab_raises(self):
        """Added tokens without base_vocab_size are not filtered and raise."""
        config = {
            "target_modules": ["q_proj"],
            "r": 8,
            "lora_alpha": 16,
        }
        added_tokens = {"<new_token>": 32000}

        with self.assertRaises(ValueError) as ctx:
            LoRAConfig(config_dict=config, added_tokens_config=added_tokens)

        self.assertIn("1 added tokens", str(ctx.exception))

    def test_init_filters_added_tokens_by_base_vocab(self):
        """Added tokens with ID < base_vocab_size are filtered out."""
        config = {
            "target_modules": ["q_proj"],
            "r": 8,
            "lora_alpha": 16,
        }
        # token_id 50 < base_vocab_size 100 -> filtered
        # token_id 100, 150 >= base_vocab_size 100 -> kept
        added_tokens = {
            "<new_token1>": 50,
            "<new_token2>": 100,
            "<new_token3>": 150,
        }

        with self.assertRaises(ValueError) as ctx:
            LoRAConfig(
                config_dict=config,
                added_tokens_config=added_tokens,
                base_vocab_size=100,
            )

        # After filtering: 2 tokens remain (100, 150)
        self.assertIn("2 added tokens", str(ctx.exception))

    def test_init_filters_all_added_tokens_when_base_vocab_large(self):
        """All added tokens filtered when base_vocab_size > all token IDs."""
        config = {
            "target_modules": ["q_proj"],
            "r": 8,
            "lora_alpha": 16,
        }
        added_tokens = {"<token1>": 50, "<token2>": 60}

        # base_vocab_size=100 > all token IDs, so all filtered
        lora_config = LoRAConfig(
            config_dict=config,
            added_tokens_config=added_tokens,
            base_vocab_size=100,
        )

        # All filtered, size = 0, no error raised
        self.assertEqual(lora_config.lora_added_tokens_size, 0)

    def test_init_with_added_tokens_and_base_vocab_none_no_filter(self):
        """When base_vocab_size is None, no filtering happens."""
        config = {
            "target_modules": ["q_proj"],
            "r": 8,
            "lora_alpha": 16,
        }
        added_tokens = {"<token>": 50}

        # base_vocab_size=None -> no filtering -> raises
        with self.assertRaises(ValueError):
            LoRAConfig(
                config_dict=config,
                added_tokens_config=added_tokens,
                base_vocab_size=None,
            )

    def test_init_missing_required_key_raises(self):
        """Missing required key in config_dict raises KeyError."""
        config_missing_target = {"r": 8, "lora_alpha": 16}

        with self.assertRaises(KeyError):
            LoRAConfig(config_dict=config_missing_target)

    def test_init_missing_r_raises(self):
        """Missing 'r' in config_dict raises KeyError."""
        config_missing_r = {"target_modules": ["q_proj"], "lora_alpha": 16}

        with self.assertRaises(KeyError):
            LoRAConfig(config_dict=config_missing_r)

    def test_init_missing_lora_alpha_raises(self):
        """Missing 'lora_alpha' in config_dict raises KeyError."""
        config_missing_alpha = {"target_modules": ["q_proj"], "r": 8}

        with self.assertRaises(KeyError):
            LoRAConfig(config_dict=config_missing_alpha)


class TestLoRAConfigFromDict(CustomTestCase):
    """Test LoRAConfig.from_dict factory method."""

    def test_from_dict_creates_instance(self):
        """from_dict returns LoRAConfig instance."""
        config = {
            "target_modules": ["q_proj"],
            "r": 8,
            "lora_alpha": 16,
        }
        lora_config = LoRAConfig.from_dict(config)

        self.assertIsInstance(lora_config, LoRAConfig)
        self.assertEqual(lora_config.r, 8)
        self.assertEqual(lora_config.lora_alpha, 16)

    def test_from_dict_with_added_tokens_config(self):
        """from_dict accepts added_tokens_config parameter."""
        config = {
            "target_modules": ["q_proj"],
            "r": 8,
            "lora_alpha": 16,
        }
        added_tokens = {"<special>": 32000}

        with self.assertRaises(ValueError):
            LoRAConfig.from_dict(config, added_tokens_config=added_tokens)

    def test_from_dict_with_base_vocab_size(self):
        """from_dict accepts base_vocab_size parameter for filtering."""
        config = {
            "target_modules": ["q_proj"],
            "r": 8,
            "lora_alpha": 16,
        }
        added_tokens = {"<token>": 50}

        # base_vocab_size=100 filters out token with ID=50
        lora_config = LoRAConfig.from_dict(
            config, added_tokens_config=added_tokens, base_vocab_size=100
        )

        self.assertEqual(lora_config.lora_added_tokens_size, 0)


class TestGetLoraConfig(CustomTestCase):
    """Test get_lora_config file loading."""

    def test_get_lora_config_from_local_dir(self):
        """Load config from local directory using real temp files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "adapter_config.json")
            with open(config_path, "w") as f:
                json.dump({"target_modules": ["q_proj"], "r": 8, "lora_alpha": 16}, f)

            lora_config = LoRAConfig(path=tmpdir)

            self.assertEqual(lora_config.r, 8)
            self.assertEqual(lora_config.target_modules, ["q_proj"])
            self.assertEqual(lora_config.lora_alpha, 16)

    def test_get_lora_config_multiple_target_modules(self):
        """Load config with multiple target modules."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "adapter_config.json")
            with open(config_path, "w") as f:
                json.dump(
                    {
                        "target_modules": ["q_proj", "v_proj", "k_proj"],
                        "r": 16,
                        "lora_alpha": 32,
                    },
                    f,
                )

            lora_config = LoRAConfig(path=tmpdir)

            self.assertEqual(lora_config.target_modules, ["q_proj", "v_proj", "k_proj"])
            self.assertEqual(lora_config.r, 16)

    @mock.patch("sglang.srt.lora.lora_config.snapshot_download")
    def test_get_lora_config_calls_snapshot_download(self, mock_download):
        """Non-directory path triggers snapshot_download."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "adapter_config.json")
            with open(config_path, "w") as f:
                json.dump({"target_modules": ["v_proj"], "r": 4, "lora_alpha": 8}, f)

            mock_download.return_value = tmpdir

            lora_config = LoRAConfig(path="hf-adapter/repo")

            # snapshot_download is called by both get_lora_config and get_added_tokens_config
            self.assertEqual(mock_download.call_count, 2)
            expected_call = mock.call("hf-adapter/repo", allow_patterns=["*.json"])
            mock_download.assert_has_calls([expected_call, expected_call])
            self.assertEqual(lora_config.r, 4)
            self.assertEqual(lora_config.target_modules, ["v_proj"])

    def test_get_lora_config_dummy_raises_not_implemented(self):
        """get_lora_config with dummy=True raises NotImplementedError."""
        lora_config = LoRAConfig.__new__(LoRAConfig)
        lora_config.path = "/fake/path"

        with self.assertRaises(NotImplementedError):
            lora_config.get_lora_config(dummy=True)


class TestGetAddedTokensConfig(CustomTestCase):
    """Test get_added_tokens_config file loading."""

    def test_returns_none_when_file_not_exists(self):
        """Return None if added_tokens.json doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create adapter_config.json so get_lora_config works
            config_path = os.path.join(tmpdir, "adapter_config.json")
            with open(config_path, "w") as f:
                json.dump({"target_modules": ["q_proj"], "r": 8, "lora_alpha": 16}, f)

            # Don't create added_tokens.json - it should return None
            lora_config = LoRAConfig.__new__(LoRAConfig)
            lora_config.path = tmpdir

            result = lora_config.get_added_tokens_config()

            self.assertIsNone(result)

    def test_returns_dict_when_file_exists(self):
        """Return dict if added_tokens.json exists and is valid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tokens_path = os.path.join(tmpdir, "added_tokens.json")
            with open(tokens_path, "w") as f:
                json.dump({"<token>": 32000}, f)

            lora_config = LoRAConfig.__new__(LoRAConfig)
            lora_config.path = tmpdir

            result = lora_config.get_added_tokens_config()

            self.assertEqual(result, {"<token>": 32000})

    def test_returns_none_on_json_decode_error(self):
        """Return None and log warning when JSON decode fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tokens_path = os.path.join(tmpdir, "added_tokens.json")
            with open(tokens_path, "w") as f:
                f.write("invalid json{{{")

            lora_config = LoRAConfig.__new__(LoRAConfig)
            lora_config.path = tmpdir

            result = lora_config.get_added_tokens_config()

            self.assertIsNone(result)

    def test_returns_multiple_tokens(self):
        """Return dict with multiple tokens."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tokens_path = os.path.join(tmpdir, "added_tokens.json")
            with open(tokens_path, "w") as f:
                json.dump({"<token1>": 32000, "<token2>": 32001}, f)

            lora_config = LoRAConfig.__new__(LoRAConfig)
            lora_config.path = tmpdir

            result = lora_config.get_added_tokens_config()

            self.assertEqual(result, {"<token1>": 32000, "<token2>": 32001})

    @mock.patch("sglang.srt.lora.lora_config.snapshot_download")
    def test_calls_snapshot_download_for_non_dir_path(self, mock_download):
        """Non-directory path triggers snapshot_download."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # No added_tokens.json in the downloaded path
            mock_download.return_value = tmpdir

            lora_config = LoRAConfig.__new__(LoRAConfig)
            lora_config.path = "hf-adapter/repo"

            result = lora_config.get_added_tokens_config()

            mock_download.assert_called_once_with(
                "hf-adapter/repo", allow_patterns=["*.json"]
            )
            self.assertIsNone(result)  # File doesn't exist in downloaded path


class TestLoRAConfigIntegration(CustomTestCase):
    """Integration tests combining multiple methods."""

    def test_full_init_from_path_without_added_tokens(self):
        """Full initialization from path with no added tokens file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "adapter_config.json")
            with open(config_path, "w") as f:
                json.dump({"target_modules": ["q_proj"], "r": 8, "lora_alpha": 16}, f)

            # Don't create added_tokens.json

            lora_config = LoRAConfig(path=tmpdir)

            self.assertEqual(lora_config.r, 8)
            self.assertEqual(lora_config.lora_added_tokens_size, 0)

    def test_full_init_from_path_with_added_tokens_filtered(self):
        """Full init with added tokens that get filtered by base_vocab_size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "adapter_config.json")
            with open(config_path, "w") as f:
                json.dump({"target_modules": ["q_proj"], "r": 8, "lora_alpha": 16}, f)

            tokens_path = os.path.join(tmpdir, "added_tokens.json")
            with open(tokens_path, "w") as f:
                # Token IDs below base_vocab_size=100 will be filtered
                json.dump({"<token>": 50}, f)

            # base_vocab_size=100 filters token with ID=50, so no error
            lora_config = LoRAConfig(path=tmpdir, base_vocab_size=100)

            self.assertEqual(lora_config.r, 8)
            self.assertEqual(lora_config.lora_added_tokens_size, 0)

    def test_full_init_from_path_with_added_tokens_raises(self):
        """Full init with added tokens that are not filtered raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "adapter_config.json")
            with open(config_path, "w") as f:
                json.dump({"target_modules": ["q_proj"], "r": 8, "lora_alpha": 16}, f)

            tokens_path = os.path.join(tmpdir, "added_tokens.json")
            with open(tokens_path, "w") as f:
                # Token ID 32000 >= any reasonable base_vocab_size, won't be filtered
                json.dump({"<token>": 32000}, f)

            with self.assertRaises(ValueError) as ctx:
                LoRAConfig(path=tmpdir, base_vocab_size=100)

            self.assertIn("1 added tokens", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
