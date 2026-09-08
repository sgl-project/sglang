"""Real file-store backups must drain after the last GPU batch finishes."""

import os
import tempfile
import unittest

import requests
from transformers import AutoTokenizer

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-large")


class TestBufferOnlyDrain(CustomTestCase):
    def test_file_backups_drain_on_both_tree_cores(self):
        model = os.environ.get("SGLANG_TEST_MODEL_PATH", "openai/gpt-oss-20b")
        tokenizer = AutoTokenizer.from_pretrained(model)

        def repeated_tokens(text, length):
            unit = tokenizer.encode(text, add_special_tokens=False)
            return (unit * (length // len(unit) + 1))[:length]

        def flush_cache():
            response = requests.post(
                DEFAULT_URL_FOR_TEST + "/flush_cache",
                params={"timeout": 30},
                timeout=45,
            )
            self.assertEqual(response.status_code, 200, response.text)

        for backend in ("python", "rust"):
            with (
                self.subTest(backend=backend),
                tempfile.TemporaryDirectory() as storage,
            ):
                process = popen_launch_server(
                    model,
                    DEFAULT_URL_FOR_TEST,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=[
                        "--mem-fraction-static",
                        "0.7",
                        "--max-total-tokens",
                        "262144",
                        "--chunked-prefill-size",
                        "16384",
                        "--swa-full-tokens-ratio",
                        "0.1",
                        "--page-size",
                        "64",
                        "--cuda-graph-backend-prefill",
                        "disabled",
                        "--cuda-graph-max-bs-decode",
                        "8",
                        "--max-running-requests",
                        "8",
                        "--enable-hierarchical-cache",
                        "--hicache-ratio",
                        "2",
                        "--hicache-host-memory-mode",
                        "buffer_only",
                        "--hicache-write-policy",
                        "write_through",
                        "--hicache-storage-backend",
                        "file",
                        "--hicache-storage-prefetch-policy",
                        "wait_complete",
                    ],
                    env={
                        "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": backend,
                        "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": storage,
                    },
                )
                try:
                    flush_cache()
                    for group in range(2):
                        prefix = repeated_tokens(
                            f"Experiment {group}. A library stores historical documents. "
                            "Readers carefully compare the dates and names in each document. ",
                            24576,
                        )
                        for question in ("Apple", "Banana", "Cherry", "Dragon"):
                            input_ids = prefix + repeated_tokens(
                                question
                                + " Explain the evidence and summarize the conclusion. ",
                                16384,
                            )
                            response = requests.post(
                                DEFAULT_URL_FOR_TEST + "/generate",
                                json={
                                    "input_ids": input_ids,
                                    "sampling_params": {
                                        "temperature": 0,
                                        "max_new_tokens": 1,
                                    },
                                },
                                timeout=180,
                            )
                            self.assertEqual(response.status_code, 200, response.text)
                            self.assertEqual(
                                response.json()["meta_info"]["completion_tokens"], 1
                            )
                        # All responses are complete, but CPU-to-storage writes
                        # can still hold staging slots and require the GIL.
                        flush_cache()
                    self.assertTrue(os.listdir(storage))
                finally:
                    terminate_and_kill_process_tree(process)


if __name__ == "__main__":
    unittest.main()
