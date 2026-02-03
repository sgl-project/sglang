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
Tests for Qwen3-TTS model support.
"""

import unittest

import torch


class TestQwen3TTSConfig(unittest.TestCase):
    """Test Qwen3-TTS configuration."""

    def test_config_creation(self):
        """Test that config can be created with default values."""
        from sglang.srt.models.qwen3_tts import Qwen3TTSConfig

        config = Qwen3TTSConfig()
        self.assertEqual(config.model_type, "qwen3_tts")
        self.assertEqual(config.tts_pad_token_id, 151671)
        self.assertEqual(config.tts_bos_token_id, 151672)
        self.assertEqual(config.tts_eos_token_id, 151673)

    def test_config_with_talker_config(self):
        """Test config with custom talker configuration."""
        from sglang.srt.models.qwen3_tts import Qwen3TTSConfig

        talker_config = {
            "hidden_size": 1024,
            "num_hidden_layers": 20,
            "num_attention_heads": 16,
        }
        config = Qwen3TTSConfig(talker_config=talker_config)
        self.assertEqual(config.talker_config["hidden_size"], 1024)
        self.assertEqual(config.talker_config["num_hidden_layers"], 20)


class TestQwen3TTSModelComponents(unittest.TestCase):
    """Test individual Qwen3-TTS model components."""

    def test_mlp_forward(self):
        """Test MLP layer forward pass."""
        from sglang.srt.models.qwen3_tts import Qwen3TTSTalkerMLP

        mlp = Qwen3TTSTalkerMLP(
            hidden_size=64,
            intermediate_size=128,
        )

        x = torch.randn(2, 10, 64)
        output = mlp(x)
        self.assertEqual(output.shape, (2, 10, 64))

    def test_text_projection(self):
        """Test text projection module."""
        from sglang.srt.models.qwen3_tts import Qwen3TTSTextProjection

        proj = Qwen3TTSTextProjection(
            input_size=256,
            intermediate_size=256,
            output_size=128,
        )

        x = torch.randn(2, 10, 256)
        output = proj(x)
        self.assertEqual(output.shape, (2, 10, 128))


class TestQwen3TTSSpeakerEncoder(unittest.TestCase):
    """Test speaker encoder for voice cloning."""

    def test_speaker_encoder_forward(self):
        """Test speaker encoder forward pass."""
        from sglang.srt.models.qwen3_tts import Qwen3TTSSpeakerEncoder

        config = {
            "mel_dim": 128,
            "enc_dim": 192,
            "enc_channels": [256, 256, 256],
            "enc_kernel_sizes": [5, 3, 1],
            "enc_dilations": [1, 2, 1],
        }
        encoder = Qwen3TTSSpeakerEncoder(config)

        # Simulate mel spectrogram input
        mel = torch.randn(2, 100, 128)  # (batch, time, mel_dim)
        output = encoder(mel)
        self.assertEqual(output.shape[0], 2)  # batch size
        self.assertEqual(output.shape[1], 192)  # enc_dim


class TestQwen3TTSCodePredictor(unittest.TestCase):
    """Test code predictor model."""

    def test_code_predictor_creation(self):
        """Test code predictor can be created."""
        from sglang.srt.models.qwen3_tts import Qwen3TTSCodePredictor

        config = {
            "hidden_size": 256,
            "vocab_size": 2048,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_code_groups": 32,
            "rms_norm_eps": 1e-6,
            "intermediate_size": 512,
        }
        predictor = Qwen3TTSCodePredictor(config, talker_hidden_size=256)

        # Check embedding count
        self.assertEqual(len(predictor.codec_embedding), 31)  # num_code_groups - 1
        self.assertEqual(len(predictor.lm_head), 31)


class TestQwen3TTSModelRegistry(unittest.TestCase):
    """Test that model is properly registered."""

    def test_entry_class_exists(self):
        """Test that EntryClass is defined."""
        from sglang.srt.models import qwen3_tts

        self.assertTrue(hasattr(qwen3_tts, "EntryClass"))
        self.assertEqual(
            qwen3_tts.EntryClass.__name__, "Qwen3TTSForConditionalGeneration"
        )


if __name__ == "__main__":
    unittest.main()
