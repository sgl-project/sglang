"""Four-Blackwell acceptance coverage for Kimi Linear TokenSpeed MLA DCP.

The captured-shape and eager-shape requests deliberately straddle
``--cuda-graph-max-bs-decode=64``.  This guards both the regular CUDA graph
decode path and the full-capacity eager DCP LSE scratch-buffer path.
"""

import unittest

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=900, stage="base-c", runner_config="4-gpu-b200")

KIMI_LINEAR_MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"


def _has_four_blackwell_gpus() -> bool:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        return False
    return all(
        torch.cuda.get_device_capability(device_index) >= (10, 0)
        for device_index in range(4)
    )


@unittest.skipUnless(
    _has_four_blackwell_gpus(),
    "TokenSpeed MLA DCP acceptance requires four Blackwell GPUs",
)
class TestKimiLinearDCP4(GSM8KMixin, CustomTestCase):
    model = KIMI_LINEAR_MODEL
    base_url = DEFAULT_URL_FOR_TEST
    gsm8k_score_threshold = 0.90
    gsm8k_num_examples = 200
    # Keep accuracy evaluation within the captured decode batch sizes so its
    # score is batch-invariant. The separate smoke test still exercises the
    # eager path with batch size 65.
    gsm8k_num_threads = 4
    gsm8k_num_shots = 5

    @classmethod
    def setUpClass(cls):
        cls.process = None
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=[
                "--tp-size",
                "4",
                "--dcp-size",
                "4",
                "--attention-backend",
                "tokenspeed_mla",
                "--kv-cache-dtype",
                "fp8_e4m3",
                "--dcp-comm-backend",
                "a2a",
                "--dcp-replicate-q-proj",
                "--trust-remote-code",
                "--random-seed",
                "0",
                "--dtype",
                "bfloat16",
                "--cuda-graph-max-bs-decode",
                "64",
                "--cuda-graph-backend-prefill",
                "disabled",
                "--mem-fraction-static",
                "0.80",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid, wait_timeout=60)

    def _assert_batch_completes(self, batch_size: int):
        prompts = [
            f"Reply with one short word for request {index}: the sky is"
            for index in range(batch_size)
        ]
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompts,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 8,
                    "ignore_eos": True,
                },
            },
            timeout=180,
        )
        response.raise_for_status()
        outputs = response.json()
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), batch_size)
        for output in outputs:
            self.assertTrue(output["text"].strip())
            self.assertGreater(output["meta_info"]["completion_tokens"], 0)

    def test_decode_cuda_graph_and_eager_batch(self):
        # Batch two replays a captured shape; batch 65 is above the configured
        # regular CUDA graph maximum and therefore exercises eager decode.
        self._assert_batch_completes(2)
        self._assert_batch_completes(2)
        self._assert_batch_completes(65)

    def test_physical_capacity_sanity(self):
        response = requests.get(self.base_url + "/server_info", timeout=30)
        response.raise_for_status()
        self.assertGreater(response.json()["max_total_num_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
