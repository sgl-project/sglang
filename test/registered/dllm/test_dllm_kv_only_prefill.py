"""Compare ordinary and KV-only LLaDA2 prompt forwards through HTTP serving."""

import os
import tempfile
import unittest
from pathlib import Path

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=240, stage="extra-a", runner_config="1-gpu-large")


class TestDllmKvOnlyPrefill(CustomTestCase):
    def test_graph_and_eager_prompt_blocks(self):
        model = os.environ.get("SGLANG_TEST_LLADA2_MODEL", "inclusionAI/LLaDA2.0-mini")
        # Token IDs are fixed test fixtures; the mask ID is a literal prompt token
        # in the final case, which exercises the unoptimized prompt path.
        tokens = list(range(1000, 1128))
        inputs = [
            tokens[:31],
            tokens[:32],
            tokens[:33],
            tokens[:64],
            tokens[:97],
            [tokens[:64], tokens[1:65], tokens[2:66], tokens[3:67]],
            tokens[:16] + [156895] + tokens[17:64],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            outputs = {}
            for eager in (False, True):
                for enabled in (False, True):
                    log_path = Path(tmp) / f"eager-{eager}-kv-only-{enabled}.log"
                    args = [
                        "--trust-remote-code",
                        "--dtype",
                        "bfloat16",
                        "--tp-size",
                        "1",
                        "--mem-fraction-static",
                        "0.8",
                        "--context-length",
                        "2048",
                        "--max-total-tokens",
                        "2048",
                        "--max-running-requests",
                        "4",
                        "--chunked-prefill-size",
                        "512",
                        "--max-prefill-tokens",
                        "512",
                        "--attention-backend",
                        "flashinfer",
                        "--moe-runner-backend",
                        "triton",
                        "--dllm-algorithm",
                        "LowConfidence",
                        "--dllm-fdfo",
                        "--cuda-graph-bs-decode",
                        "1",
                        "2",
                        "3",
                        "4",
                        "--random-seed",
                        "42",
                    ]
                    if eager:
                        args.append("--disable-cuda-graph")
                    with log_path.open("w") as log:
                        process = popen_launch_server(
                            model,
                            DEFAULT_URL_FOR_TEST,
                            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                            other_args=args,
                            env={
                                "SGLANG_ENABLE_DLLM_KV_ONLY_PREFILL": str(int(enabled)),
                                "SGLANG_LOG_DECODE_GRAPH_KEY": "1",
                            },
                            return_stdout_stderr=(log, log),
                        )
                        try:
                            rows = []
                            for ids in inputs:
                                requests.post(
                                    DEFAULT_URL_FOR_TEST + "/flush_cache", timeout=30
                                ).raise_for_status()
                                rows.append(self.generate(ids))
                            # Reuse a complete prompt block, then encode its suffix.
                            requests.post(
                                DEFAULT_URL_FOR_TEST + "/flush_cache", timeout=30
                            ).raise_for_status()
                            self.generate(tokens[:64])
                            rows.append(self.generate(tokens[:97]))
                            outputs[eager, enabled] = rows
                        finally:
                            kill_process_tree(process.pid)
                    if enabled and not eager:
                        self.assertIn("variant=dllm_kv_only", log_path.read_text())
                self.assertEqual(outputs[eager, False], outputs[eager, True])

    def generate(self, input_ids):
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 64,
                    "ignore_eos": True,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        rows = result if isinstance(result, list) else [result]
        for row in rows:
            self.assertEqual(len(row["output_ids"]), 64)
        return [row["output_ids"] for row in rows]


if __name__ == "__main__":
    unittest.main()
