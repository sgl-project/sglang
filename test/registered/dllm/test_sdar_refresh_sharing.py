import io
import tempfile
import unittest
from pathlib import Path

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-large")


class TestSDARRefreshSharing(CustomTestCase):
    """Compare full native-EOS outputs across first capture and graph reuse."""

    model = "JetLM/SDAR-8B-Chat"
    revision = "ac4528d2c076b04e02a03e6f430efa4385331ce8"
    base_url = DEFAULT_URL_FOR_TEST
    prompts = (
        "Question: Why is the sky blue? Explain in two sentences. Answer:",
        "Question: What is 12 times 13? Show the calculation. Answer:",
        "Question: Why is the sky blue? Explain in two sentences. Answer:",
    )

    def _collect_outputs(self, paired, config):
        stdout, stderr = io.StringIO(), io.StringIO()
        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--revision",
                self.revision,
                "--trust-remote-code",
                "--dtype",
                "bfloat16",
                "--tp-size",
                "1",
                "--random-seed",
                "0",
                "--attention-backend",
                "flashinfer",
                "--disable-overlap-schedule",
                "--disable-radix-cache",
                "--return-output-ids",
                "--max-running-requests",
                "1",
                "--max-total-tokens",
                "4096",
                "--context-length",
                "2048",
                "--mem-fraction-static",
                "0.8",
                "--cuda-graph-bs-decode",
                "1",
                "--dllm-algorithm",
                "LowConfidence",
                "--dllm-algorithm-config",
                str(config),
                "--dllm-fdfo",
            ],
            env={
                "SGLANG_SDAR_REFRESH_PAIR": str(int(paired)),
                "SGLANG_FLASHINFER_USE_PAGED": "0",
            },
            return_stdout_stderr=(stdout, stderr),
            device="cuda",
        )
        try:
            outputs = []
            for prompt in self.prompts:
                response = requests.post(
                    f"{self.base_url}/generate",
                    json={
                        "text": prompt,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 96,
                        },
                    },
                    timeout=180,
                )
                response.raise_for_status()
                result = response.json()
                ids = result["output_ids"]
                finish = result["meta_info"]["finish_reason"]
                self.assertTrue(ids)
                self.assertIn(finish["type"], ("stop", "length"))
                outputs.append((ids, finish))
            self.assertGreater(max(len(ids) for ids, _ in outputs), 8)
            if paired:
                log = stdout.getvalue() + stderr.getvalue()
                self.assertIn("SDAR_REFRESH_PAIR numerical_gate_passed", log)
                self.assertNotIn("Disable SDAR refresh sharing", log)
            return outputs
        finally:
            kill_process_tree(process.pid)

    def test_native_eos_and_graph_reuse(self):
        self.assertTrue(torch.cuda.is_available(), "This test requires a CUDA GPU")
        self.assertIsNone(torch.version.hip, "This test requires NVIDIA FlashInfer")
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "sdar.yaml"
            config.write_text("block_size: 4\nthreshold: 0.95\n")
            stock = self._collect_outputs(paired=False, config=config)
            paired = self._collect_outputs(paired=True, config=config)
        self.assertEqual(
            paired, stock, "Complete outputs and finish reasons must match"
        )
        self.assertEqual(
            paired[0], paired[2], "Request reuse must not retain old state"
        )


if __name__ == "__main__":
    unittest.main()
