import json
import logging
import os
import time
import unittest
from pathlib import Path

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_30B_A3B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Common configuration (adjusted based on actual logs)
COMMON_CONFIG = {
    "model": QWEN3_30B_A3B_WEIGHTS_PATH,
    "base_url": DEFAULT_URL_FOR_TEST,
    "metrics_dir": os.path.abspath("."),
    # SGLang built-in defaults (matched with logs)
    "SGLANG_BUILTIN_DEFAULTS": {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    },
    "MODEL_GEN_DEFAULTS": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    },
    "base_server_args": [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--mem-fraction-static",
        0.8,
        "--tp-size",
        2,
        "--export-metrics-to-file",
        "--export-metrics-to-file-dir",
        os.path.abspath("."),
    ],
    "request_timeout": 60,
}


class BaseSamplingTest(CustomTestCase):
    """Testcase: Verify the sampling defaults is set correctly based on different --sampling-defaults.

    [Test Category] Parameter
    [Test Target] --sampling-defaults, --export-metrics-to-file, --export-metrics-to-file-dir
    """

    server_process = None

    @classmethod
    def setUpClass(cls):
        """Class-level initialization: Execute only once"""
        # Verify current directory is writable
        metrics_dir = Path(COMMON_CONFIG["metrics_dir"])
        test_file = metrics_dir / "test_write_perm.txt"
        try:
            test_file.write_text("test")
            test_file.unlink()
            logger.info(f"Current directory is writable: {metrics_dir}")
        except PermissionError:
            raise RuntimeError(
                f"No write permission for current directory: {metrics_dir}"
            )

        # Launch server
        cls._launch_server()

        logger.info(f"\n=== {cls.__name__} initialization completed ===")
        logger.info(f"Model default parameters: {COMMON_CONFIG['MODEL_GEN_DEFAULTS']}")
        logger.info(
            f"SGLang built-in default parameters: {COMMON_CONFIG['SGLANG_BUILTIN_DEFAULTS']}"
        )
        logger.info(
            f"Initial files in current directory: {[f.name for f in metrics_dir.glob('*') if f.is_file()]}"
        )

    @classmethod
    def tearDownClass(cls):
        if cls.server_process:
            kill_process_tree(cls.server_process.pid)
            time.sleep(1)
            logger.info(f"\n=== {cls.__name__} server has been shut down ===")
            metrics_dir = Path(COMMON_CONFIG["metrics_dir"])
            logger.info(
                f"Files in current directory after test completion: {[f.name for f in metrics_dir.glob('*') if f.is_file()]}"
            )

    def setUp(self):
        """Before each test method: Do not delete log files, only print current file list"""
        metrics_dir = Path(COMMON_CONFIG["metrics_dir"])
        time.sleep(0.5)
        logger.info(
            f"\nFiles in directory before test method execution: {[f.name for f in metrics_dir.glob('*') if f.is_file()]}"
        )

    @classmethod
    def _launch_server(cls):
        """Launch server (subclasses must implement this method)"""
        raise NotImplementedError("Subclasses must implement _launch_server method")

    def _call_chat(self, custom_params: dict = None):
        """Call API (only adjust timeout, no retry)"""
        req_body = {
            "model": COMMON_CONFIG["model"],
            "messages": [
                {"role": "user", "content": "Test sampling parameters: 1+1=？"}
            ],
        }
        if custom_params:
            req_body.update(custom_params)

        # Call API
        response = requests.post(
            f"{COMMON_CONFIG['base_url']}/v1/chat/completions",
            json=req_body,
            timeout=COMMON_CONFIG["request_timeout"],
        )
        self.assertEqual(response.status_code, 200, f"API call failed: {response.text}")

        # Extend waiting time for log writing
        time.sleep(3)
        return self._get_sampling_params_from_metrics()

    def _get_sampling_params_from_metrics(self):
        """Extract sampling parameters from metrics (adapt to actual log format)"""
        metrics_dir = Path(COMMON_CONFIG["metrics_dir"])
        # Match actual metrics file naming: sglang-request-metrics-*
        metrics_files = list(metrics_dir.glob("sglang-request-metrics-*.log"))
        logger.info(f"\nMatched metrics files: {[f.name for f in metrics_files]}")

        if not metrics_files:
            self.fail(
                f"No sglang-request-metrics-*.log files found! Current directory files: {[f.name for f in metrics_dir.glob('*') if f.is_file()]}"
            )

        # Get latest metrics file
        latest_file = max(metrics_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"Reading latest metrics file: {latest_file.name}")

        # Read and clean log content (fix line break/space issues)
        with open(latest_file, "r", encoding="utf-8") as f:
            log_content = f.read()
            # Split by line (log may contain multiple JSON entries)
            log_lines = [
                line.strip() for line in log_content.split("\n") if line.strip()
            ]
            # Get last valid log entry (latest request)
            last_log = log_lines[-1] if log_lines else ""
            # Clean line breaks and extra spaces
            clean_content = last_log.replace("\n", "").replace("  ", " ").strip()
            logger.info(f"\nCleaned latest log content:\n{clean_content[:800]}...")

        # Parse JSON (core adaptation: sampling_params nested in request_parameters)
        try:
            # Parse outer JSON
            log_data = json.loads(clean_content)
            # Parse request_parameters field (string to JSON)
            req_params = json.loads(log_data["request_parameters"])
            # Extract sampling_params
            sampling_params = req_params.get("sampling_params", {})
            logger.info(f"Parsed sampling_params: {sampling_params}")
        except json.JSONDecodeError as e:
            self.fail(
                f"JSON parsing failed: {e}, Original content: {clean_content[:500]}"
            )

        # Extract core sampling parameters (fill missing parameters with default values)
        core_params = {}
        for key in COMMON_CONFIG["SGLANG_BUILTIN_DEFAULTS"].keys():
            core_params[key] = sampling_params.get(
                key, COMMON_CONFIG["SGLANG_BUILTIN_DEFAULTS"][key]
            )
        logger.info(f"Final extracted core sampling parameters: {core_params}")
        return core_params


class TestSamplingDefaultsModel(BaseSamplingTest):
    """Test --sampling-defaults=model mode"""

    @classmethod
    def _launch_server(cls):
        """Launch server in model mode"""
        server_args = COMMON_CONFIG["base_server_args"] + [
            "--sampling-defaults",
            "model",
        ]
        logger.info(f"\n=== Launching server in model mode ===")
        logger.info(f"Launch parameters: {server_args}")

        cls.server_process = popen_launch_server(
            COMMON_CONFIG["model"],
            COMMON_CONFIG["base_url"],
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=server_args,
        )

    def test_default_params(self):
        """Model mode - Default parameters (no manual configuration)"""
        logger.info("\n=== Testing model mode default parameters ===")
        sampling_params = self._call_chat()

        # Precise assertion: match model default parameters in logs
        self.assertEqual(
            sampling_params["temperature"],
            COMMON_CONFIG["MODEL_GEN_DEFAULTS"]["temperature"],
            f"temperature mismatch: expected={COMMON_CONFIG['MODEL_GEN_DEFAULTS']['temperature']}, actual={sampling_params['temperature']}",
        )
        self.assertEqual(
            sampling_params["top_p"],
            COMMON_CONFIG["MODEL_GEN_DEFAULTS"]["top_p"],
            f"top_p mismatch: expected={COMMON_CONFIG['MODEL_GEN_DEFAULTS']['top_p']}, actual={sampling_params['top_p']}",
        )
        self.assertEqual(
            sampling_params["top_k"],
            COMMON_CONFIG["MODEL_GEN_DEFAULTS"]["top_k"],
            f"top_k mismatch: expected={COMMON_CONFIG['MODEL_GEN_DEFAULTS']['top_k']}, actual={sampling_params['top_k']}",
        )
        self.assertEqual(
            sampling_params["min_p"],
            COMMON_CONFIG["MODEL_GEN_DEFAULTS"]["min_p"],
            f"min_p mismatch: expected={COMMON_CONFIG['MODEL_GEN_DEFAULTS']['min_p']}, actual={sampling_params['min_p']}",
        )
        self.assertEqual(
            sampling_params["repetition_penalty"],
            COMMON_CONFIG["MODEL_GEN_DEFAULTS"]["repetition_penalty"],
            f"repetition_penalty mismatch: expected={COMMON_CONFIG['MODEL_GEN_DEFAULTS']['repetition_penalty']}, "
            f"actual={sampling_params['repetition_penalty']}",
        )
        logger.info("Model mode default parameters assertion passed!")

    def test_custom_params(self):
        """Model mode - Manually customized parameters"""
        logger.info("\n=== Testing model mode manual parameters ===")
        # Manually configured parameters (matched with actual values in logs)
        custom_params = {
            "temperature": 0.6,
            "top_p": 0.75,
            "top_k": 100,
            "min_p": 0.2,
            "repetition_penalty": 1.1,
        }
        logger.info(f"Manually configured parameters: {custom_params}")
        sampling_params = self._call_chat(custom_params)

        # Precise assertion: manual parameters take full effect
        for key, expected_value in custom_params.items():
            self.assertEqual(
                sampling_params[key],
                expected_value,
                f"Manual parameter {key} not effective: expected={expected_value}, actual={sampling_params[key]}",
            )
        logger.info("Model mode manual parameters assertion passed!")


class TestSamplingDefaultsOpenAI(BaseSamplingTest):
    """Test --sampling-defaults=openai mode"""

    @classmethod
    def _launch_server(cls):
        """Launch server in openai mode"""
        server_args = COMMON_CONFIG["base_server_args"] + [
            "--sampling-defaults",
            "openai",
        ]
        logger.info(f"\n=== Launching server in openai mode ===")
        logger.info(f"Launch parameters: {server_args}")

        cls.server_process = popen_launch_server(
            COMMON_CONFIG["model"],
            COMMON_CONFIG["base_url"],
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=server_args,
        )

    def test_default_params(self):
        """openai mode - Default parameters (no manual configuration)"""
        logger.info("\n=== Testing openai mode default parameters ===")
        sampling_params = self._call_chat()

        # Precise assertion: match openai default parameters in logs (consistent with SGLang built-in)
        self.assertEqual(
            sampling_params["temperature"],
            COMMON_CONFIG["SGLANG_BUILTIN_DEFAULTS"]["temperature"],
            f"temperature mismatch: expected={COMMON_CONFIG['SGLANG_BUILTIN_DEFAULTS']['temperature']}, actual={sampling_params['temperature']}",
        )
        self.assertEqual(
            sampling_params["top_p"],
            COMMON_CONFIG["SGLANG_BUILTIN_DEFAULTS"]["top_p"],
            f"top_p mismatch: expected={COMMON_CONFIG['SGLANG_BUILTIN_DEFAULTS']['top_p']}, actual={sampling_params['top_p']}",
        )
        self.assertEqual(
            sampling_params["top_k"],
            COMMON_CONFIG["SGLANG_BUILTIN_DEFAULTS"]["top_k"],
            f"top_k mismatch: expected={COMMON_CONFIG['SGLANG_BUILTIN_DEFAULTS']['top_k']}, actual={sampling_params['top_k']}",
        )
        self.assertEqual(
            sampling_params["min_p"],
            COMMON_CONFIG["SGLANG_BUILTIN_DEFAULTS"]["min_p"],
            f"min_p mismatch: expected={COMMON_CONFIG['SGLANG_BUILTIN_DEFAULTS']['min_p']}, actual={sampling_params['min_p']}",
        )
        self.assertEqual(
            sampling_params["repetition_penalty"],
            COMMON_CONFIG["SGLANG_BUILTIN_DEFAULTS"]["repetition_penalty"],
            f"repetition_penalty mismatch: expected={COMMON_CONFIG['SGLANG_BUILTIN_DEFAULTS']['repetition_penalty']}, actual={sampling_params['repetition_penalty']}",
        )
        logger.info("openai mode default parameters assertion passed!")

    def test_custom_params(self):
        """openai mode - Manually customized parameters"""
        logger.info("\n=== Testing openai mode manual parameters ===")
        # Manually configured parameters (matched with actual values in logs)
        custom_params = {
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 50,
            "min_p": 0.1,
            "repetition_penalty": 1.3,
        }
        logger.info(f"Manually configured parameters: {custom_params}")
        sampling_params = self._call_chat(custom_params)

        # Precise assertion: manual parameters take full effect
        for key, expected_value in custom_params.items():
            self.assertEqual(
                sampling_params[key],
                expected_value,
                f"Manual parameter {key} not effective: expected={expected_value}, actual={sampling_params[key]}",
            )
        logger.info("openai mode manual parameters assertion passed!")


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)
