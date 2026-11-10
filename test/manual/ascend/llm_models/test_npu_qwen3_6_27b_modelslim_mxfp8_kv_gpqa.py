import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_accuracy_utils import assert_metrics
from sglang.test.test_utils import (
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)


@unittest.skipUnless(
    os.environ.get("QWEN3_6_MODELSLIM_MODEL_PATH"),
    "Set QWEN3_6_MODELSLIM_MODEL_PATH to a local ModelSlim MXFP8 checkpoint",
)
class TestQwen3_6_27BModelSlimMXFP8KVGPQA(CustomTestCase):
    model = os.environ.get("QWEN3_6_MODELSLIM_MODEL_PATH")
    accuracy = 0.878
    datasets = ["gpqa_diamond"]
    served_model_name = "qwen3.6-27b-modelslim-mxfp8-kv"
    base_url = "http://127.0.0.1:6677"
    env = {
        **os.environ,
        "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
        "ASCEND_USE_FIA": "1",
        "HCCL_SOCKET_IFNAME": "lo",
        "GLOO_SOCKET_IFNAME": "lo",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "SGLANG_SET_CPU_AFFINITY": "1",
        "STREAMS_PER_DEVICE": "32",
    }
    other_args = [
        "--served-model-name",
        served_model_name,
        "--device",
        "npu",
        "--tp-size",
        "1",
        "--attention-backend",
        "ascend",
        "--quantization",
        "modelslim",
        "--page-size",
        "512",
        "--dtype",
        "bfloat16",
        "--mamba-ssm-dtype",
        "bfloat16",
        "--chunked-prefill-size",
        "8192",
        "--max-running-requests",
        "64",
        "--mem-fraction-static",
        "0.85",
        "--trust-remote-code",
        "--base-gpu-id",
        os.environ.get("BASE_GPU_ID", "0"),
        "--mamba-track-interval",
        "512",
        "--reasoning-parser",
        "qwen3",
        "--kv-cache-dtype",
        "fp8_e4m3",
    ]
    generation_config = {
        "temperature": 1.0,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "max_tokens": 81920,
        "timeout": 3600,
        # Forward SGLang sampling extensions through the OpenAI API client.
        "extra_body": {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0},
    }

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=3600,
            other_args=cls.other_args,
            env=cls.env,
            device="npu",
        )
        cls.addClassCleanup(terminate_and_kill_process_tree, cls.process)

    def test_gpqa(self):
        output_root = Path(
            os.environ.get("GPQA_OUTPUT_DIR", "./outputs/gpqa_qwen36_27b_mxfp8_kv")
        )
        output_root.mkdir(parents=True, exist_ok=True)
        # Never reuse answers cached by an earlier evaluation.
        work_dir = Path(tempfile.mkdtemp(prefix="mxfp8_kv_", dir=output_root))
        command = [
            "evalscope",
            "eval",
            "--model",
            self.served_model_name,
            "--api-url",
            f"{self.base_url}/v1/chat/completions",
            "--api-key",
            "EMPTY",
            "--eval-type",
            "openai_api",
            "--datasets",
            "gpqa_diamond",
            "--dataset-args",
            json.dumps(
                {
                    "gpqa_diamond": {
                        "local_path": os.environ.get(
                            "GPQA_DATASET_PATH", "./gpqa_diamond"
                        )
                    }
                }
            ),
            "--eval-batch-size",
            "64",
            "--generation-config",
            json.dumps(self.generation_config),
            "--work-dir",
            str(work_dir),
        ]
        report_path = None
        with (
            (work_dir / "evalscope.log").open("w", encoding="utf-8") as log,
            subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            ) as process,
        ):
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    match = re.search(r"Dump report to:\s*(\S+)", line)
                    if match:
                        report_path = Path(match.group(1))
                self.assertEqual(process.wait(), 0, f"See {work_dir}/evalscope.log")
            finally:
                if process.poll() is None:
                    kill_process_tree(process.pid)
        self.assertIsNotNone(report_path, f"No EvalScope report: {work_dir}")
        with report_path.open(encoding="utf-8") as report_file:
            score = float(json.load(report_file)["score"])
        self.assertTrue(0.0 <= score <= 1.0, f"Invalid GPQA score: {score}")
        print(f"GPQA Diamond: score={score}, report={report_path}", flush=True)
        assert_metrics(self, {"accuracy": score})


if __name__ == "__main__":
    unittest.main()
