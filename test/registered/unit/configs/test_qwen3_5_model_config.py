"""Unit tests for Qwen3.5 ModelConfig defaults."""

import json
import tempfile
from pathlib import Path

from sglang.srt.configs.model_config import ModelConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _write_qwen3_5_config(model_dir: Path, architecture: str):
    config = {
        "architectures": [architecture],
        "model_type": "qwen3_5_moe"
        if architecture == "Qwen3_5MoeForConditionalGeneration"
        else "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_moe_text"
            if architecture == "Qwen3_5MoeForConditionalGeneration"
            else "qwen3_5_text",
            "hidden_size": 64,
            "intermediate_size": 128,
            "max_position_embeddings": 1024,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "rope_parameters": {"rope_type": "default"},
            "vocab_size": 128,
        },
        "vision_config": {"model_type": "qwen3_5"},
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    (model_dir / "generation_config.json").write_text(
        json.dumps({"eos_token_id": 1, "pad_token_id": 0})
    )


class TestQwen3_5ModelConfig(CustomTestCase):
    def test_qwen3_5_text_architectures_disable_multimodal_by_default(self):
        for architecture in [
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
        ]:
            with self.subTest(architecture=architecture):
                with tempfile.TemporaryDirectory() as tmpdir:
                    model_dir = Path(tmpdir)
                    _write_qwen3_5_config(model_dir, architecture)

                    model_config = ModelConfig(str(model_dir), trust_remote_code=True)

                    self.assertFalse(model_config.is_multimodal)
                    self.assertFalse(model_config.is_image_understandable_model)

    def test_qwen3_5_can_still_opt_in_to_multimodal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            _write_qwen3_5_config(
                model_dir, "Qwen3_5MoeForConditionalGeneration"
            )

            model_config = ModelConfig(
                str(model_dir), trust_remote_code=True, enable_multimodal=True
            )

            self.assertTrue(model_config.is_multimodal)
            self.assertTrue(model_config.is_image_understandable_model)


if __name__ == "__main__":
    import unittest

    unittest.main()
