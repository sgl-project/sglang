"""Compare PP2 generation over the Gloo CPU staging with eager and CUDA graph execution.

Run directly on a Linux host with two CUDA GPUs. TDX/CC network protection is
deployment-specific and must additionally be validated across protected hosts.
"""

import re
import tempfile
import unittest
from pathlib import Path

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
    wait_for_server_output,
)

register_cuda_ci(est_time=180, stage="base-b", runner_config="2-gpu-large")


class TestPPCPUTransport(CustomTestCase):
    def _generate(self, graph):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            with path.open("w") as log:
                return self._generate_with_log(graph, path, log)

    def _generate_with_log(self, graph, path, log):
        process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            return_stdout_stderr=(log, log),
            other_args=[
                "--tp-size",
                "1",
                "--pp-size",
                "2",
                "--disable-overlap-schedule",
                "--skip-server-warmup",
                "--cuda-graph-backend-decode",
                graph,
                "--cuda-graph-max-bs-decode",
                "4",
                "--decode-log-interval",
                "1",
                "--pp-transport",
                "gloo",
                "--mem-fraction-static",
                "0.6",
                "--random-seed",
                "0",
            ],
        )
        try:
            outputs = []
            # Repeated requests exercise static output-buffer reuse and keep
            # the same process groups alive across several scheduler iterations.
            for prompt in (
                "The capital of France is",
                "Write a Python function that adds two integers.",
                "Explain why the sky looks blue.",
                "The capital of France is",
            ):
                response = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/generate",
                    json={
                        "text": prompt,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 32,
                            "ignore_eos": True,
                        },
                    },
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()
                self.assertEqual(data["meta_info"]["completion_tokens"], 32)
                outputs.append(data["output_ids"])
        finally:
            try:
                terminate_and_kill_process_tree(process)
            finally:
                wait_for_server_output(process)

        if graph == "full":
            # Warmup generation is disabled. Inspect only after both output pipes
            # reach EOF, including decode stats emitted after the final HTTP reply.
            text = path.read_text()
            decode_lines = [
                line for line in text.splitlines() if "Decode batch" in line
            ]
            replay_stages = {
                int(match.group(1))
                for line in decode_lines
                if "cuda graph: True" in line
                for match in [re.search(r"\bPP(\d+)\b", line)]
                if match is not None
            }
            self.assertEqual(
                replay_stages, {0, 1}, f"{graph}: no replay on both PP stages\n{text}"
            )
            self.assertFalse(
                any("cuda graph: False" in line for line in decode_lines),
                f"{graph}: decode fell back to eager\n{text}",
            )
        return outputs

    def test_gloo_graph_matches_eager(self):
        self.assertEqual(self._generate("disabled"), self._generate("full"))


if __name__ == "__main__":
    unittest.main()
