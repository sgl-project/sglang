"""XPU classification parity test for Qwen2.5-1.5B-apeach.

Usage:
python3 -m unittest test_xpu_classification.TestXPUClassification
"""

import gc
import multiprocessing as mp
import unittest

import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.runners import HFRunner, SRTRunner
from sglang.test.test_utils import CustomTestCase, empty_gpu_cache

register_xpu_ci(est_time=120, suite="stage-b-test-1-gpu-xpu")

MODEL_PATH = "jason9693/Qwen2.5-1.5B-apeach"
TP_SIZE = 1
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

    def _hf_probs(self):
        """Reference probabilities from HuggingFace sequence classification."""
        with HFRunner(
            MODEL_PATH,
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

    def _srt_probs(self):
        """Reference probabilities from SRT sequence classification path."""
        with SRTRunner(
            MODEL_PATH,
            tp_size=TP_SIZE,
            torch_dtype=TORCH_DTYPE,
            # SRT classify uses embedding-mode encode outputs (class-logit vectors);
            model_type="embedding",
            attention_backend="intel_xpu",
            trust_remote_code=True,
            mem_fraction_static=0.55,
        ) as srt_runner:
            srt_logits = srt_runner.forward(PROMPTS).embed_logits

        probs = []
        for row in srt_logits:
            tensor = row if torch.is_tensor(row) else torch.tensor(row)
            tensor = tensor.float().flatten()
            probs.append(torch.softmax(tensor, dim=-1))
        return probs

    def test_classification_logits(self):
        hf_probs = self._hf_probs()
        gc.collect()
        empty_gpu_cache()
        srt_probs = self._srt_probs()

        self.assertEqual(len(hf_probs), len(PROMPTS))
        self.assertEqual(len(srt_probs), len(PROMPTS))

        for index, (hf_row, srt_row) in enumerate(zip(hf_probs, srt_probs)):
            self.assertEqual(
                srt_row.shape,
                hf_row.shape,
                f"probability shape mismatch at sample {index}",
            )

            # Probabilities should be close (bf16-tolerant).
            max_abs_diff = torch.max(torch.abs(hf_row - srt_row)).item()
            self.assertLess(
                max_abs_diff,
                PROB_TOLERANCE,
                f"classification probs diverged at sample {index}: {max_abs_diff}",
            )

            # Top-class agreement is the primary correctness signal.
            hf_pred = int(torch.argmax(hf_row).item())
            srt_pred = int(torch.argmax(srt_row).item())
            self.assertEqual(
                hf_pred,
                srt_pred,
                f"top class mismatch at sample {index}",
            )


if __name__ == "__main__":
    unittest.main()
