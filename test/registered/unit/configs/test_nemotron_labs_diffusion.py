import json
import tempfile
import unittest
from pathlib import Path

from sglang.srt.configs.nemotron_labs_diffusion import NemotronLabsDiffusionConfig
from sglang.srt.hf_transformers import get_config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestNemotronLabsDiffusionConfig(CustomTestCase):
    def test_native_config_does_not_execute_checkpoint_code(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = {
                "model_type": "nemotron_labs_diffusion",
                "architectures": ["NemotronLabsDiffusionModel"],
                "auto_map": {"AutoConfig": "not_installed.Config"},
                "rope_scaling": {
                    "type": "yarn",
                    "factor": 16,
                    "original_max_position_embeddings": 16384,
                },
            }
            (Path(directory) / "config.json").write_text(json.dumps(payload))
            config = get_config(directory, trust_remote_code=False)
            self.assertIsInstance(config, NemotronLabsDiffusionConfig)
            self.assertEqual(config.rope_parameters["rope_type"], "yarn")
            self.assertNotIn("type", config.rope_parameters)


if __name__ == "__main__":
    unittest.main()
