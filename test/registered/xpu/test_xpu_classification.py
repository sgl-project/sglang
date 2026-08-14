"""XPU classification parity test for Qwen2.5-1.5B-apeach via /v1/classify.

Usage:
python3 -m unittest test_xpu_classfication.TestXPUClassification
"""

import multiprocessing as mp
import unittest

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.runners import HFRunner
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_xpu_ci(est_time=120, suite="stage-b-test-1-gpu-xpu")

MODEL_PATH = "jason9693/Qwen2.5-1.5B-apeach"
TORCH_DTYPE = torch.bfloat16
# Softmax probabilities are far less sensitive to bf16 rounding than raw
# logits, so a modest probability tolerance is sufficient here.
PROB_TOLERANCE = 5e-2

PROMPTS = [
    "This movie has a tight plot and keeps me engaged.",
    "Shipping was late and the packaging arrived damaged.",
    "The features are fine, but the price feels too high.",
]


class TestXPUClassification(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.model_path = MODEL_PATH
        cls.process = popen_launch_server(
            cls.model_path,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--is-embedding",
                "--attention-backend",
                "intel_xpu",
                "--dtype",
                "bfloat16",
                "--trust-remote-code",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            kill_process_tree(cls.process.pid)

    def _hf_probs(self):
        """Reference probabilities from HuggingFace sequence classification."""
        with HFRunner(
            self.model_path,
            torch_dtype=TORCH_DTYPE,
            model_type="cross_encoder",
        ) as hf_runner:
            hf_scores = hf_runner.forward(PROMPTS).scores

        probs = []
        for row in hf_scores:
            tensor = row if torch.is_tensor(row) else torch.tensor(row)
            tensor = tensor.float().flatten()
            probs.append(torch.softmax(tensor, dim=-1))
        return probs

    def _srt_classify(self, prompt):
        """Call the /v1/classify endpoint for a single prompt."""
        response = requests.post(
            f"{self.base_url}/v1/classify",
            headers={"Content-Type": "application/json"},
            json={"model": self.model_path, "input": prompt},
            timeout=60,
        )
        self.assertEqual(
            response.status_code,
            200,
            f"/v1/classify failed: {response.status_code} {response.text}",
        )
        return response.json()["data"][0]

    def test_classification_logits(self):
        hf_probs = self._hf_probs()
        self.assertEqual(len(hf_probs), len(PROMPTS))

        for index, prompt in enumerate(PROMPTS):
            result = self._srt_classify(prompt)

            srt_probs = torch.tensor(result["probs"], dtype=torch.float32)
            hf_row = hf_probs[index]

            # Class count must match the model config.
            self.assertEqual(
                result["num_classes"],
                hf_row.numel(),
                f"num_classes mismatch at sample {index}",
            )
            self.assertEqual(
                srt_probs.shape,
                hf_row.shape,
                f"probability shape mismatch at sample {index}",
            )

            # Probabilities should be close (bf16-tolerant).
            max_abs_diff = torch.max(torch.abs(hf_row - srt_probs)).item()
            self.assertLess(
                max_abs_diff,
                PROB_TOLERANCE,
                f"classification probs diverged at sample {index}: {max_abs_diff}",
            )

            # Top-class agreement is the primary correctness signal.
            hf_pred = int(torch.argmax(hf_row).item())
            srt_pred = int(torch.argmax(srt_probs).item())
            self.assertEqual(
                hf_pred,
                srt_pred,
                f"top class mismatch at sample {index}",
            )


if __name__ == "__main__":
    unittest.main()
