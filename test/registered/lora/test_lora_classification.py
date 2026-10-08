"""Real CUDA server parity for locally generated PEFT classifiers and generation."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-small")


class TestLoRAClassificationServer(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="classification-http-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.example = (
            Path(__file__).resolve().parents[3]
            / "examples/runtime/lora_classification.py"
        )
        cls.env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        subprocess.run(
            [sys.executable, str(cls.example), "create", cls.directory.name],
            env=cls.env,
            check=True,
            timeout=120,
        )
        cls.base_dir = str(Path(cls.directory.name) / "base")
        cls.process = popen_launch_server(
            cls.base_dir,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env=cls.env,
            other_args=[
                "--served-model-name",
                "tiny-classifier",
                "--enable-lora",
                "--lora-target-modules",
                "o_proj",
                "--max-lora-rank",
                "4",
                "--max-loras-per-batch",
                "4",
                "--return-hidden-states-mode",
                "last",
                "--tp",
                "1",
                "--dtype",
                "float16",
                "--attention-backend",
                "torch_native",
                "--lora-backend",
                "triton",
                "--disable-cuda-graph",
                "--context-length",
                "64",
                "--max-total-tokens",
                "512",
                "--max-running-requests",
                "8",
            ],
        )
        cls.addClassCleanup(kill_process_tree, cls.process.pid)

    def test_saved_classifiers_share_generation_engine(self):
        subprocess.run(
            [
                sys.executable,
                str(self.example),
                "check",
                "--base-dir",
                self.base_dir,
                "--url",
                DEFAULT_URL_FOR_TEST,
                "--tolerance",
                "0.003",
            ],
            env=self.env,
            check=True,
            timeout=300,
        )


if __name__ == "__main__":
    unittest.main()
