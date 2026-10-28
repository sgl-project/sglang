"""Kimi K3 C8 GPQA on two hosts with eight A5 NPUs each, without DSpark.

Load the NPU runtime and set HCCL_SOCKET_IFNAME/GLOO_SOCKET_IFNAME on both hosts.
Set KIMI_K3_DIST_INIT_ADDR=<node-0-ip>:5000 on both hosts, then run this file:
    Node 1: KIMI_K3_NODE_RANK=1 python <this_file>
    Node 0: python <this_file> -v

Run node 0 from the evaluation directory containing ./gpqa_diamond. EvalScope
must already be installed. Reports are retained under ./outputs/gpqa_kimik3.
KIMI_K3_MODELSLIM_MODEL_PATH can override the checkpoint path on both hosts.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    assert_metrics,
    get_accuracy_threshold,
    get_max_retries,
)
from sglang.test.test_utils import (
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)


class TestKimiK3ModelSlimMLAFP8(CustomTestCase):
    model = os.environ.get(
        "KIMI_K3_MODELSLIM_MODEL_PATH",
        "/home/l00951279/weights/Kimi-K3-w4a8c8-mxfp-quarot-0817",
    )
    # Match the existing NPU Kimi GPQA baseline and its 5/198 tolerance.
    accuracy = 0.935
    datasets = ["gpqa_diamond"]
    base_url = "http://127.0.0.1:6677"
    env = {
        **os.environ,
        "TORCH_NPU_DEVICE_CAPABILITY": os.environ.get(
            "TORCH_NPU_DEVICE_CAPABILITY", "0.0"
        ),
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "HCCL_BUFFSIZE": "2000",
        "SGLANG_SET_CPU_AFFINITY": "1",
        "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS": "1",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
        "STREAMS_PER_DEVICE": "32",
        "ASCEND_USE_FIA": "1",
        "SGLANG_USE_FIA_NZ": "1",
        "SGLANG_NPU_USE_MLAPO": "1",
        "SGLANG_NPU_MODELSLIM_KV_QUANT_TYPE_OVERRIDE": "Q_FP8_DYNAMIC_KV_FP8",
    }
    other_args = [
        "--served-model-name",
        model,
        "--trust-remote-code",
        "--model-loader-extra-config",
        '{"enable_multithread_load": true}',
        "--device",
        "npu",
        "--attention-backend",
        "ascend",
        "--linear-attn-backend",
        "triton",
        "--quantization",
        "modelslim",
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--tp-size",
        "16",
        "--nnodes",
        "2",
        "--moe-a2a-backend",
        "none",
        "--page-size",
        "128",
        "--max-running-requests",
        "4",
        "--mem-fraction-static",
        "0.86",
        "--cuda-graph-bs-decode",
        "1",
        "2",
        "4",
        "--reasoning-parser",
        "kimi_k3",
        "--watchdog-timeout",
        "9000",
        "--weight-loader-disable-mmap",
        "--disable-radix-cache",
    ]
    generation_config = {
        "temperature": 1.0,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "max_tokens": 131072,
        "timeout": 10000,
        "stream": True,
        "extra_body": {"reasoning_effort": "max", "repetition_penalty": 1.0},
    }

    @classmethod
    def server_args(cls, node_rank):
        return [
            *cls.other_args,
            "--dist-init-addr",
            os.environ["KIMI_K3_DIST_INIT_ADDR"],
            "--node-rank",
            str(node_rank),
        ]

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=3600,
            other_args=cls.server_args(0),
            env=cls.env,
            device="npu",
        )
        cls.addClassCleanup(terminate_and_kill_process_tree, cls.process)

    def test_gpqa(self):
        output_root = Path("./outputs/gpqa_kimik3")
        output_root.mkdir(parents=True, exist_ok=True)
        threshold = get_accuracy_threshold(self.datasets, self.accuracy)
        best_score = 0.0
        for attempt in range(get_max_retries(self.datasets)):
            # Each attempt generates fresh answers; never reuse an older run's cache.
            work_dir = Path(tempfile.mkdtemp(prefix="mla_fp8_", dir=output_root))
            command = [
                "evalscope",
                "eval",
                "--model",
                self.model,
                "--api-url",
                f"{self.base_url}/v1/chat/completions",
                "--api-key",
                "EMPTY",
                "--eval-type",
                "openai_api",
                "--datasets",
                "gpqa_diamond",
                "--dataset-args",
                json.dumps({"gpqa_diamond": {"local_path": "./gpqa_diamond"}}),
                "--eval-batch-size",
                "4",
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
            print(f"GPQA attempt {attempt + 1}: score={score}, threshold={threshold}")
            best_score = max(best_score, score)
            if best_score >= threshold:
                break
        assert_metrics(self, {"accuracy": best_score})


if __name__ == "__main__":
    if os.environ.get("KIMI_K3_NODE_RANK", "0") == "1":
        test = TestKimiK3ModelSlimMLAFP8
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            test.model,
            *test.server_args(1),
            "--host",
            "127.0.0.1",
            "--port",
            "6677",
        ]
        os.execvpe(sys.executable, command, test.env)
    unittest.main()
