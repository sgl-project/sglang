"""Gemma-4 on `--attention-backend fa3`, which runs the 512-wide global layers
on FA4, against the fa4 backend, whose image-mask path this one reuses.

Text, then an image request, then text again through one server. The image
prefill is the one that needs `image_token_ranges`: without them the hybrid
would attend causally over the image while fa4 does not, and the prompt
logprobs of the image tokens would move by whole nats. Prompt logprobs
compare the prefill itself rather than a sampled sentence.

The triton backend is not the oracle here: on Gemma-4 it sits ~0.3 nats
mean from both FA backends even on text, while fa3 and fa4 agree to the
printed precision.
"""

import unittest

import requests
import torch
from transformers import AutoProcessor

from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_IMAGE_URL,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    build_vlm_image_prompt,
    popen_launch_server,
)

register_cuda_ci(est_time=420, stage="base-b", runner_config="1-gpu-large")

MODEL = "google/gemma-4-E2B-it"
TEXT_QUESTION = "Name three primary colors and one use for each."
IMAGE_QUESTION = "Describe this image in one sentence."
MAX_NEW_TOKENS = 16
# Same FA4 kernels on the global layers and on image batches; FA3 vs FA4 on
# the text-only SWA layers is the only kernel difference. A missing image
# mask moves the image tokens' prompt logprobs by whole nats.
MAX_MEAN_ABS_LOGPROB_DIFF = 0.02


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
@unittest.skipIf(get_device_sm() // 10 != 9, "fa3's FA4 layers need SM90")
class TestGemma4Fa3Fa4Hybrid(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        processor = AutoProcessor.from_pretrained(MODEL)
        cls.text_prompt = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": TEXT_QUESTION}]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        cls.image_prompt = build_vlm_image_prompt(processor, IMAGE_QUESTION)

    def _generate(self, prompt, image=False):
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": prompt,
                "image_data": DEFAULT_IMAGE_URL if image else None,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": MAX_NEW_TOKENS,
                },
                "return_logprob": True,
                "logprob_start_len": 0,
            },
            timeout=300,
        )
        response.raise_for_status()
        meta = response.json()["meta_info"]
        return {
            "output_ids": [tok for _, tok, *_ in meta["output_token_logprobs"]],
            "prompt_logprobs": [
                lp for lp, *_ in meta["input_token_logprobs"] if lp is not None
            ],
        }

    def _serve_and_generate(self, backend_args):
        process = popen_launch_server(
            MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--dtype", "bfloat16", "--random-seed", "0", *backend_args],
        )
        try:
            return {
                "text_before": self._generate(self.text_prompt),
                "image": self._generate(self.image_prompt, image=True),
                "text_after": self._generate(self.text_prompt),
            }
        finally:
            kill_process_tree(process.pid, wait_timeout=60)

    def test_hybrid_matches_fa4_around_an_image_prefill(self):
        hybrid = self._serve_and_generate(["--attention-backend", "fa3"])
        reference = self._serve_and_generate(["--attention-backend", "fa4"])

        for step in ("text_before", "image", "text_after"):
            with self.subTest(step=step):
                ours, theirs = hybrid[step], reference[step]
                self.assertEqual(
                    len(ours["prompt_logprobs"]), len(theirs["prompt_logprobs"])
                )
                self.assertGreater(len(ours["prompt_logprobs"]), 0)
                diff = torch.tensor(ours["prompt_logprobs"]) - torch.tensor(
                    theirs["prompt_logprobs"]
                )
                self.assertLess(diff.abs().mean().item(), MAX_MEAN_ABS_LOGPROB_DIFF)
                self.assertEqual(ours["output_ids"][:8], theirs["output_ids"][:8])

        # The image prompt is the long one: the image soft tokens were scored.
        self.assertGreater(
            len(hybrid["image"]["prompt_logprobs"]),
            2 * len(hybrid["text_before"]["prompt_logprobs"]),
        )


if __name__ == "__main__":
    unittest.main()
