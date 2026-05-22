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

"""
Unit tests for LoRA utility functions that handle multimodal model configs
(e.g., composite HF configs where num_hidden_layers/hidden_size live under
text_config instead of the top-level config).

Regression test for https://github.com/sgl-project/sglang/issues/21876
"""

import unittest
from types import SimpleNamespace

from sglang.srt.lora.utils import get_hidden_size, get_num_layers


def _make_standard_config(**kwargs):
    """Create a mock config mimicking a standard HF model (e.g., Llama)."""
    defaults = {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_multimodal_config(text_kwargs=None, top_kwargs=None):
    """Create a mock config mimicking a multimodal/composite HF model
    (e.g., Qwen3-VL-30B-A3B) where text attributes live under text_config."""
    text_defaults = {
        "num_hidden_layers": 48,
        "hidden_size": 3072,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "intermediate_size": 8192,
    }
    if text_kwargs:
        text_defaults.update(text_kwargs)

    text_config = SimpleNamespace(**text_defaults)

    top_defaults = {
        "text_config": text_config,
        "model_type": "qwen3_vl",
    }
    if top_kwargs:
        top_defaults.update(top_kwargs)

    return SimpleNamespace(**top_defaults)


class TestGetNumLayers(unittest.TestCase):
    """Tests for get_num_layers()."""

    def test_standard_config(self):
        config = _make_standard_config(num_hidden_layers=32)
        self.assertEqual(get_num_layers(config), 32)

    def test_multimodal_config(self):
        config = _make_multimodal_config(text_kwargs={"num_hidden_layers": 48})
        self.assertEqual(get_num_layers(config), 48)


class TestGetHiddenSize(unittest.TestCase):
    """Tests for get_hidden_size()."""

    def test_standard_config(self):
        config = _make_standard_config(hidden_size=4096)
        self.assertEqual(get_hidden_size(config), 4096)

    def test_multimodal_config(self):
        config = _make_multimodal_config(text_kwargs={"hidden_size": 3072})
        self.assertEqual(get_hidden_size(config), 3072)


if __name__ == "__main__":
    unittest.main(warnings="ignore", verbosity=2)
