import time
import unittest
from types import SimpleNamespace

import requests
from test_eagle_infer import TestEAGLEEngine, TestEAGLEServer

import sglang as sgl
from sglang.srt.speculative.eagle_mab import MABConfig
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
    DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestEAGLEMABConfig(unittest.TestCase):
    """Test the MAB configuration validation and parsing functionality."""

    def test_validate_configs(self):
        """Test that valid MAB configurations are properly validated."""
        valid_configs = "1_1_1,2_2_4,3_4_8"
        result = MABConfig.validate_configs(valid_configs)
        self.assertEqual(result, ["1_1_1", "2_2_4", "3_4_8"])

    def test_validate_configs_invalid_format(self):
        """Test that invalid MAB configurations raise ValueError."""
        invalid_configs = "1_1,2_2_4,3_4_8"
        with self.assertRaises(ValueError):
            MABConfig.validate_configs(invalid_configs)

    def test_validate_configs_negative_values(self):
        """Test that MAB configurations with negative values raise ValueError."""
        invalid_configs = "1_1_1,2_-2_4,3_4_8"
        with self.assertRaises(ValueError):
            MABConfig.validate_configs(invalid_configs)

    def test_parse_config(self):
        """Test parsing a single MAB configuration string."""
        config = "3_4_8"
        steps, topk, draft_tokens = MABConfig.parse_config(config)
        self.assertEqual(steps, 3)
        self.assertEqual(topk, 4)
        self.assertEqual(draft_tokens, 8)

    def test_format_config(self):
        """Test formatting parameters into a MAB configuration string."""
        config_str = MABConfig.format_config(3, 4, 8)
        self.assertEqual(config_str, "3_4_8")


class TestEAGLEMABEngine(TestEAGLEEngine):
    """Test the EAGLE MAB engine functionality by extending TestEAGLEEngine."""

    BASE_CONFIG = {
        **TestEAGLEEngine.BASE_CONFIG,
        "speculative_eagle_mab_algorithm": "EG",
        "speculative_eagle_mab_configs": ["2_2_4", "3_4_8"],
        "speculative_mab_window_size": 100,
        "mem_fraction_static": 0.65,
    }
    NUM_CONFIGS = 1  # Only test the MAB config


class TestEAGLEMABServer(TestEAGLEServer):
    """Test the EAGLE MAB server functionality by extending TestEAGLEServer."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-draft-model-path",
                DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
                "--speculative-num-steps",
                "5",
                "--speculative-eagle-topk",
                "4",
                "--speculative-num-draft-tokens",
                "8",
                "--speculative-eagle-mab-algorithm",
                "EG",
                "--speculative-eagle-mab-configs",
                "2_2_4,5_4_8",
                "--speculative-mab-window-size",
                "100",
                "--mem-fraction-static",
                "0.65",
                "--cuda-graph-max-bs",
                "5",
            ],
        )

    def test_gsm8k(self):
        """Override test_gsm8k to use a lower acceptance length threshold for MAB."""
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )

        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["accuracy"], 0.20)

        server_info = requests.get(self.base_url + "/get_server_info")
        avg_spec_accept_length = server_info.json()["avg_spec_accept_length"]
        print(f"{avg_spec_accept_length=}")

        # Use a lower threshold for MAB since it's learning which strategies work best
        self.assertGreater(avg_spec_accept_length, 2.0)

        # Wait a little bit so that the memory check happens.
        time.sleep(4)


if __name__ == "__main__":
    unittest.main()
