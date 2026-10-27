"""Minimal native/1A1F S2 serving parity. Use two owned CUDA devices.

AFD_E2E_MODEL selects an already cached checkpoint; AFD_E2E_LOG_DIR keeps logs.
The larger eager/graph, topology, and lifecycle matrix lives in extended tests.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, terminate_and_kill_process_tree
from sglang.utils import wait_for_server

register_cuda_ci(est_time=600, stage="weekly", runner_config="4-gpu-b200")


class TestQwen3MoEAFDServing(CustomTestCase):
    def test_native_and_role_graph(self):
        model = os.environ.get("AFD_E2E_MODEL", "Qwen/Qwen3-30B-A3B")
        backend = os.environ.get("AFD_E2E_ATTENTION_BACKEND", "fa4")
        port = int(os.environ.get("AFD_E2E_PORT", "23450"))
        url = f"http://127.0.0.1:{port}"
        common = (
            f"--attention-backend {backend} --moe-runner-backend triton "
            "--moe-a2a-backend none --context-length 2048 --max-total-tokens 4096 "
            "--max-running-requests 2 --disable-radix-cache "
            "--cuda-graph-backend-prefill disabled --cuda-graph-backend-decode full "
            "--cuda-graph-bs-decode 2 --tp-size 1"
        ).split()
        config = json.dumps(
            dict(
                lanes=1,
                stages=2,
                attention_backend=backend,
                rendezvous_port=port + 25,
                close_timeout_seconds=10,
            )
        )
        reference = None
        with tempfile.TemporaryDirectory(prefix="afd-smoke-") as temporary:
            root = Path(os.environ.get("AFD_E2E_LOG_DIR", temporary))
            root.mkdir(parents=True, exist_ok=True)
            for name in ("native", "afd"):
                with self.subTest(mode=name), ExitStack() as stack:
                    processes = []
                    roles = ("ffn", "attention") if name == "afd" else ("off",)
                    for role in roles:
                        log = stack.enter_context(
                            (root / f"{name}-{role}.log").open("w")
                        )
                        args = common + [
                            "--base-gpu-id",
                            str(int(role == "ffn")),
                            "--nccl-port",
                            str(port + (2 if role == "ffn" else 1)),
                        ]
                        if role != "off":
                            args += [
                                "--afd-execution-mode",
                                role,
                                "--afd-config",
                                config,
                            ]
                        if role != "ffn":
                            args += ["--host", "127.0.0.1", "--port", str(port)]
                        process = subprocess.Popen(
                            [
                                sys.executable,
                                "-m",
                                "sglang.launch_server",
                                "--model-path",
                                model,
                                *args,
                            ],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                        )
                        stack.callback(terminate_and_kill_process_tree, process)
                        processes.append(process)
                    wait_for_server(url, timeout=1200, process=processes[-1])
                    for _ in range(2):
                        self.assertTrue(all(p.poll() is None for p in processes))
                        response = requests.post(
                            url + "/generate",
                            json={
                                "text": ["The capital of France is"] * 2,
                                "sampling_params": {
                                    "temperature": 0,
                                    "max_new_tokens": 8,
                                    "ignore_eos": True,
                                },
                            },
                            timeout=120,
                        )
                        response.raise_for_status()
                        outputs = response.json()
                        self.assertEqual(len(outputs), 2)
                        self.assertTrue(
                            all(
                                x["meta_info"]["completion_tokens"] == 8
                                for x in outputs
                            )
                        )
                        texts = [x["text"] for x in outputs]
                        self.assertTrue(all(texts))
                        if reference is None:
                            reference = texts
                        self.assertEqual(texts, reference)
                    self.assertTrue(all(p.poll() is None for p in processes))


if __name__ == "__main__":
    unittest.main()
