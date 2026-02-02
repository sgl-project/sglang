"""
Usage:
python3 -m unittest test_autoround_quantization
"""

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_loader.loader import get_model_loader
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=120, suite="stage-b-test-large-1-gpu")


class TestAutoRound(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.output_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls.output_dir):
            shutil.rmtree(cls.output_dir)

    def test_online_quant(self):
        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--trust-remote-code", "--quantization", "auto-round-int8"],
        )

        try:
            args = SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name="mmlu",
                num_examples=32,
                num_threads=32,
                device="auto",
            )
            metrics = run_eval(args)
            self.assertGreaterEqual(metrics["score"], 0.7)
        finally:
            kill_process_tree(process.pid)
            print(f"[INFO] Server for {self.model} stopped.")

    def test_offline_quant(self):

        # Configure model with inc quantization and saving
        model_config = ModelConfig(
            model_path=self.model,
            quantization="auto-round-int8",
            trust_remote_code=True,
        )

        load_config = LoadConfig(
            inc_save_path=self.output_dir,
        )
        device_config = DeviceConfig(device="cuda")
        # Load and quantize the model
        model_loader = get_model_loader(load_config, model_config)
        quantized_model = model_loader.load_model(
            model_config=model_config,
            device_config=device_config,
        )
        assert os.path.exists(os.path.join(self.output_dir, "config.json"))


if __name__ == "__main__":
    unittest.main()

