import re
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
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

register_cuda_ci(est_time=300, stage="base-b", runner_config="1-gpu-large")


class TestDllmPrefillBreakableParity(CustomTestCase):
    """Exercise multi-block dLLM prefill through eager and Breakable serving."""

    model = "inclusionAI/LLaDA2.0-mini"
    base_url = DEFAULT_URL_FOR_TEST
    block_size = 32
    shared_prefix_len = 77

    @classmethod
    def setUpClass(cls):
        def token_ids(start: int, length: int):
            return [start + index % 97 for index in range(length)]

        # Pre-tokenized IDs keep the scheduler shape exact and avoid making the
        # regression depend on tokenizer downloads or tokenizer-version details.
        cls.shared_prefix = token_ids(1000, cls.shared_prefix_len)
        cls.batch_input_ids = [
            cls.shared_prefix + token_ids(2000, 96),
            cls.shared_prefix + token_ids(3000, 160),
        ]

        assert cls.shared_prefix_len % cls.block_size != 0
        assert [len(ids) for ids in cls.batch_input_ids] == [173, 237]

    def _collect_outputs(self, prefill_backend: str):
        with tempfile.TemporaryDirectory(
            prefix=f"sglang_dllm_{prefill_backend}_"
        ) as log_dir:
            stdout_path = Path(log_dir) / "stdout.log"
            stderr_path = Path(log_dir) / "stderr.log"
            with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
                process = popen_launch_server(
                    self.model,
                    self.base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=[
                        "--trust-remote-code",
                        "--skip-server-warmup",
                        "--tp-size",
                        "1",
                        "--mem-fraction-static",
                        "0.9",
                        "--max-running-requests",
                        "2",
                        "--max-prefill-tokens",
                        "512",
                        "--attention-backend",
                        "flashinfer",
                        "--dllm-algorithm",
                        "LowConfidence",
                        "--dllm-prefill-block-size",
                        "128",
                        "--cuda-graph-backend-prefill",
                        prefill_backend,
                        "--cuda-graph-backend-decode",
                        "disabled",
                    ],
                    return_stdout_stderr=(stdout, stderr),
                )
                try:
                    # Seed an unaligned logical prefix. The dLLM radix page size is
                    # 32, so the following batch must reuse exactly 64 cached tokens
                    # and prefill the remaining 13-token shared tail itself.
                    warmup = requests.post(
                        f"{self.base_url}/generate",
                        json={
                            "input_ids": self.shared_prefix,
                            "sampling_params": {
                                "temperature": 0,
                                "max_new_tokens": 1,
                            },
                        },
                        timeout=120,
                    )
                    self.assertEqual(warmup.status_code, 200, warmup.text)

                    # Pause before submitting the batch so both subrequests reach
                    # the scheduler queue before either can be admitted.
                    pause = requests.post(
                        f"{self.base_url}/pause_generation",
                        json={"mode": "in_place"},
                        timeout=30,
                    )
                    self.assertEqual(pause.status_code, 200, pause.text)

                    def send_batch():
                        return requests.post(
                            f"{self.base_url}/generate",
                            json={
                                "input_ids": self.batch_input_ids,
                                "sampling_params": {
                                    "temperature": 0,
                                    "max_new_tokens": self.block_size,
                                    "ignore_eos": True,
                                },
                            },
                            timeout=240,
                        )

                    with ThreadPoolExecutor(max_workers=1) as executor:
                        response_future = executor.submit(send_batch)
                        try:
                            time.sleep(1)
                        finally:
                            resume = requests.post(
                                f"{self.base_url}/continue_generation",
                                json={"torch_empty_cache": False},
                                timeout=30,
                            )
                            self.assertEqual(resume.status_code, 200, resume.text)
                        response = response_future.result(timeout=240)

                    self.assertEqual(response.status_code, 200, response.text)
                    results = response.json()
                    self.assertEqual(len(results), len(self.batch_input_ids))
                    outputs = [result["output_ids"] for result in results]
                finally:
                    kill_process_tree(process.pid)

                stdout.flush()
                stderr.flush()

            logs = stdout_path.read_text(errors="replace") + stderr_path.read_text(
                errors="replace"
            )
            return outputs, logs

    def _assert_mixed_length_prefill(self, logs: str, *, used_graph: bool):
        match = re.search(
            r"Prefill batch.*#new-seq: 2, #new-token: 224, "
            r"#cached-token: 128,.*cuda graph: (True|False)",
            logs,
        )
        self.assertIsNotNone(
            match,
            "Expected one shared-prefix mixed-length prefill batch with "
            "prefix_lens=[64, 64] and extend_lens=[96, 128].",
        )
        self.assertEqual(match.group(1) == "True", used_graph)

    def test_eager_matches_breakable_with_unaligned_shared_prefix(self):
        eager_outputs, eager_logs = self._collect_outputs("disabled")
        breakable_outputs, breakable_logs = self._collect_outputs("breakable")

        self._assert_mixed_length_prefill(eager_logs, used_graph=False)
        self._assert_mixed_length_prefill(breakable_logs, used_graph=True)
        self.assertEqual(
            breakable_outputs,
            eager_outputs,
            "Breakable multi-block prefill output must match eager execution.",
        )


if __name__ == "__main__":
    unittest.main()
