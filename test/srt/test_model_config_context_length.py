# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for context-length derivation diagnostics.

These are pure-CPU unit tests: they exercise the config-derivation logic only,
without loading a model or touching a GPU.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.utils.hf_transformers_utils import CONTEXT_LENGTH_KEYS

LOGGER_PATH = "sglang.srt.configs.model_config.logger"


def _rendered(call_args):
    """Render a %-style logging call the way logging would."""
    args = call_args[0]
    return args[0] % tuple(args[1:])


def _fake_model_config(**text_config_fields):
    """Build the minimal object that `_derive_context_length` touches."""
    return SimpleNamespace(
        is_draft_model=False,
        hf_text_config=SimpleNamespace(rope_scaling=None, **text_config_fields),
        hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
    )


class TestDeriveContextLengthDiagnostics(unittest.TestCase):
    def test_no_context_length_field_warns_about_hardcoded_fallback(self):
        cfg = _fake_model_config()

        with patch(LOGGER_PATH) as logger:
            ModelConfig._derive_context_length(cfg, None)

        # get_context_length() falls back to a hard-coded 2048 when none of the
        # context-length fields exist; that must not stay silent.
        self.assertEqual(cfg.context_len, 2048)

        logger.warning.assert_called_once()
        rendered = _rendered(logger.warning.call_args)
        self.assertIn("2048", rendered)
        self.assertIn("--context-length", rendered)
        for key in CONTEXT_LENGTH_KEYS:
            self.assertIn(key, rendered)
        logger.info.assert_not_called()

    def test_derived_from_config_field_is_reported(self):
        cfg = _fake_model_config(max_position_embeddings=4096)

        with patch(LOGGER_PATH) as logger:
            ModelConfig._derive_context_length(cfg, None)

        self.assertEqual(cfg.context_len, 4096)

        logger.info.assert_called_once()
        rendered = _rendered(logger.info.call_args)
        self.assertIn("4096", rendered)
        self.assertIn("--context-length", rendered)
        logger.warning.assert_not_called()

    def test_explicit_context_length_stays_quiet(self):
        """The already-safe path must keep its existing behaviour."""
        cfg = _fake_model_config(max_position_embeddings=4096)

        with patch(LOGGER_PATH) as logger:
            ModelConfig._derive_context_length(cfg, 1024)

        self.assertEqual(cfg.context_len, 1024)
        logger.info.assert_not_called()
        logger.warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
