"""Test original log probability alignment between SGLang and Hugging Face.

This test suite verifies the correctness of the `origin_logprobs` output (temperature=1)
and the `logprobs` output (temperature=0.5) in SGLang by comparing it against
raw logit-based probabilities computed directly from a reference Hugging Face model.

The test covers the following scenarios:
- Next-token prediction: Verifies that the log probability of the next token from
  SGLang matches the Hugging Face model.
- Top-k logprobs: Ensures that the top-k original logprobs returned by SGLang are
  consistent with Hugging Face outputs.
- Specified token IDs: Confirms that the original logprobs for specific token IDs
  match the values computed from Hugging Face logits.
- Multi-token decoding: Validates per-step log-probability accuracy across a
  complete generation sequence (max_new_tokens=8).

Two test classes run the same suite with SGLANG_RETURN_ORIGINAL_LOGPROB=True
and False respectively, each launching its own server instance.
"""

import os
import random
import unittest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.logprob_kit import (
    assert_position_logprobs_match,
    generate_with_logprobs,
    hf_logprobs_for_sequence,
)
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

register_cuda_ci(est_time=120, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=180, suite="stage-b-test-small-1-gpu-amd")

# ------------------------- Configurable via env ------------------------- #
MODEL_ID = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
PROMPTS = [
    "Hello, my name is",
    "The future of AI is",
    "The president of the United States is",
    "The capital of France is ",
]
SAMPLING_PARAMS = {
    "temperature": 0.5,
    "top_p": 1.0,
    "top_k": 10,
}
TOP_LOGPROBS_NUM = 50
NUM_RANDOM_TOKEN_IDS = 10
RTOL = 0.20
ATOL = 0.04
MULTI_TOKEN_COUNT = 8
# ----------------------------------------------------------------------- #

torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class _OriginalLogprobBase(unittest.TestCase):
    """Base class — subclasses set ``return_original_logprob`` to "True" or "False"."""

    return_original_logprob = None

    @classmethod
    def setUpClass(cls):
        if cls is _OriginalLogprobBase:
            raise unittest.SkipTest("Base class")

        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="right")
        cls.hf_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float32, device_map="auto"
        )

        env = os.environ.copy()
        env["SGLANG_RETURN_ORIGINAL_LOGPROB"] = cls.return_original_logprob
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            MODEL_ID,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--trust-remote-code", "--mem-fraction-static", "0.60"],
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        if cls is _OriginalLogprobBase:
            return
        kill_process_tree(cls.process.pid)

    @property
    def hf_temperature(self):
        """HF reference temperature: 1.0 for original logprobs, else request temp."""
        return (
            1.0
            if self.return_original_logprob == "True"
            else SAMPLING_PARAMS["temperature"]
        )

    # --------------------------------------------------------------------- #
    # Tests
    # --------------------------------------------------------------------- #

    def test_logprob_match(self):
        """Single-token: verify next-token logprobs against HF."""
        for prompt in PROMPTS:
            with self.subTest(prompt=prompt):
                random_token_ids = sorted(
                    random.sample(
                        range(self.tokenizer.vocab_size), NUM_RANDOM_TOKEN_IDS
                    )
                )
                input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][
                    0
                ].tolist()

                res = generate_with_logprobs(
                    self.base_url,
                    input_ids,
                    max_new_tokens=1,
                    top_logprobs_num=TOP_LOGPROBS_NUM,
                    token_ids_logprob=random_token_ids,
                    sampling_params=SAMPLING_PARAMS,
                )
                meta = res["meta_info"]

                hf_lp = hf_logprobs_for_sequence(
                    self.hf_model, input_ids, temperature=self.hf_temperature
                )

                assert_position_logprobs_match(
                    self,
                    ref_lp_vec=hf_lp[-1],
                    sgl_token_logprob=meta["output_token_logprobs"][0],
                    sgl_top_logprobs=meta["output_top_logprobs"][0],
                    sgl_ids_logprobs=meta["output_token_ids_logprobs"][0],
                    token_ids=random_token_ids,
                    top_k=TOP_LOGPROBS_NUM,
                    rtol=RTOL,
                    atol=ATOL,
                    tag=f"SGLang vs HF: {prompt} ({self.return_original_logprob})",
                )

    def test_output_logprob_multi_token(self):
        """Multi-token decode: validate per-step logprobs across an 8-token generation."""
        for prompt in PROMPTS:
            with self.subTest(prompt=prompt):
                random_token_ids = sorted(
                    random.sample(
                        range(self.tokenizer.vocab_size), NUM_RANDOM_TOKEN_IDS
                    )
                )
                input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][
                    0
                ].tolist()
                prompt_len = len(input_ids)

                res = generate_with_logprobs(
                    self.base_url,
                    input_ids,
                    max_new_tokens=MULTI_TOKEN_COUNT,
                    top_logprobs_num=TOP_LOGPROBS_NUM,
                    token_ids_logprob=random_token_ids,
                    sampling_params=SAMPLING_PARAMS,
                )
                meta = res["meta_info"]

                output_token_ids = [t[1] for t in meta["output_token_logprobs"]]
                self.assertEqual(len(output_token_ids), MULTI_TOKEN_COUNT)

                full_sequence = input_ids + output_token_ids
                hf_all_lp = hf_logprobs_for_sequence(
                    self.hf_model, full_sequence, temperature=self.hf_temperature
                )

                max_diff = 0.0
                for step in range(MULTI_TOKEN_COUNT):
                    hf_pos = prompt_len - 1 + step
                    assert_position_logprobs_match(
                        self,
                        ref_lp_vec=hf_all_lp[hf_pos],
                        sgl_token_logprob=meta["output_token_logprobs"][step],
                        sgl_top_logprobs=meta["output_top_logprobs"][step],
                        sgl_ids_logprobs=meta["output_token_ids_logprobs"][step],
                        token_ids=random_token_ids,
                        top_k=TOP_LOGPROBS_NUM,
                        rtol=RTOL,
                        atol=ATOL,
                        tag=f"Multi-token step {step}: {prompt} ({self.return_original_logprob})",
                    )
                    sgl_val = meta["output_token_logprobs"][step][0]
                    hf_val = hf_all_lp[hf_pos][
                        meta["output_token_logprobs"][step][1]
                    ].item()
                    max_diff = max(max_diff, abs(hf_val - sgl_val))

                print(
                    f"[Multi-token {prompt} ({self.return_original_logprob})] "
                    f"max|diff| = {max_diff:.4f}"
                )


class TestOriginalLogprobEnabled(_OriginalLogprobBase):
    """Tests with SGLANG_RETURN_ORIGINAL_LOGPROB=True."""

    return_original_logprob = "True"


class TestOriginalLogprobDisabled(_OriginalLogprobBase):
    """Tests with SGLANG_RETURN_ORIGINAL_LOGPROB=False."""

    return_original_logprob = "False"


if __name__ == "__main__":
    unittest.main()
