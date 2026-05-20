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
"""

import os
import random
import unittest

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import sglang as sgl
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci

# ------------------------- Configurable via env ------------------------- #
MODEL_ID = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

PROMPTS = [
    "Hello, my name is",
    "The future of AI is",
    "The president of the United States is",
    "The capital of France is ",
]
TOP_LOGPROBS_NUM = 50
NUM_RANDOM_TOKEN_IDS = 10
RTOL = 0.20
ATOL = 0.00
# ------------------------------------------------

torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestOriginalLogprob(unittest.TestCase):
    """Testcase: Verify the behavior and log probability alignment of SGLang under two configurations of the environment variable `SGLANG_RETURN_ORIGINAL_LOGPROB` (True/False),
        by comparing SGLang's output with reference values from Hugging Face.

    [Test Category] Parameter
    [Test Target] SGLANG_RETURN_ORIGINAL_LOGPROB
    """

    def setUp(self):
        # ----- HF side (float32 weights) -----
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="right")
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float32, device_map="auto"
        )

        # Shared sampling parameters
        self.sampling_params = {
            "temperature": 0.5,  # SGLang uses 0.5, but original logprobs are used 1.0
            "top_p": 1.0,
            "top_k": 10,
            "max_new_tokens": 1,
        }

    # ---------------------------------------------------------------------
    # Helper: compare one SGLang block (token_logprobs / top_logprobs / ids_logprobs)
    #         against a reference HF log‑prob vector.
    # ---------------------------------------------------------------------
    def assert_logprobs_block_equal(
        self,
        hf_log_probs: torch.Tensor,  # [V]
        token_log_probs: list,
        top_log_probs: list,
        ids_log_probs: list,
        random_token_ids: list,
        tag: str = "",
    ):
        vals, idxs, _ = zip(*token_log_probs)
        sgl_vals = torch.tensor(vals, device=self.hf_model.device, dtype=torch.float32)
        sgl_idxs = torch.tensor(idxs, device=self.hf_model.device, dtype=torch.long)
        hf_vals = hf_log_probs[sgl_idxs]

        self.assertTrue(
            torch.allclose(hf_vals, sgl_vals, rtol=RTOL, atol=ATOL),
            msg=f"[{tag}] token‑level mismatch at indices {sgl_idxs.tolist()}",
        )

        hf_topk, _ = torch.topk(hf_log_probs, k=TOP_LOGPROBS_NUM, dim=-1)

        sgl_topk = torch.tensor(
            [float(t[0]) for t in top_log_probs[0] if t and t[0] is not None][
                :TOP_LOGPROBS_NUM
            ],
            dtype=torch.float32,
            device=self.hf_model.device,
        )

        k = min(hf_topk.numel(), sgl_topk.numel())
        self.assertTrue(
            torch.allclose(hf_topk[:k], sgl_topk[:k], rtol=RTOL, atol=ATOL),
            msg=f"[{tag}] top‑k mismatch",
        )

        indices = torch.tensor(
            random_token_ids, dtype=torch.long, device=hf_log_probs.device
        )

        hf_token_ids = hf_log_probs[indices]

        sgl_token_ids = torch.tensor(
            [v for v, _, _ in ids_log_probs[0]],
            device=self.hf_model.device,
            dtype=torch.float32,
        )
        self.assertTrue(
            torch.allclose(hf_token_ids, sgl_token_ids, rtol=RTOL, atol=ATOL),
            msg=f"[{tag}] token‑IDs mismatch",
        )

        # Optional: print max abs diff for quick diagnostics
        max_diff = torch.max(torch.abs(hf_vals - sgl_vals)).item()
        print(f"[{tag}] max|diff| token‑level = {max_diff:.4f}")

    def test_logprob_match(self):
        vocab_size = self.tokenizer.vocab_size

        for env_val in ["True", "False"]:
            with self.subTest(return_original_logprob=env_val):
                os.environ["SGLANG_RETURN_ORIGINAL_LOGPROB"] = env_val

                # ----- SGLang side -----
                sgl_engine = sgl.Engine(
                    model_path=MODEL_ID,
                    skip_tokenizer_init=True,
                    trust_remote_code=True,
                    mem_fraction_static=0.60,
                    attention_backend="ascend",
                    disable_cuda_graph=True,
                )

                for prompt in PROMPTS:
                    random_token_ids = sorted(
                        random.sample(range(vocab_size), NUM_RANDOM_TOKEN_IDS)
                    )

                    enc = self.tokenizer(prompt, return_tensors="pt")
                    input_ids = enc["input_ids"].to(self.hf_model.device)
                    attn_mask = enc["attention_mask"].to(self.hf_model.device)

                    with torch.inference_mode():
                        hf_out = self.hf_model(
                            input_ids=input_ids,
                            attention_mask=attn_mask,
                            return_dict=True,
                        )
                    logits = hf_out.logits[:, -1, :]  # [1, V]
                    hf_log_probs = F.log_softmax(
                        logits.float() / self.sampling_params["temperature"], dim=-1
                    )[0]
                    hf_original_log_probs = F.log_softmax(logits.float(), dim=-1)[0]

                    outputs = sgl_engine.generate(
                        input_ids=input_ids[0].tolist(),
                        sampling_params=self.sampling_params,
                        return_logprob=True,
                        top_logprobs_num=TOP_LOGPROBS_NUM,
                        token_ids_logprob=random_token_ids,
                    )

                    if isinstance(outputs, list):
                        outputs = outputs[0]
                    meta = outputs["meta_info"]

                    # Check original logprobs only if enabled
                    if env_val.lower() == "true":
                        self.assert_logprobs_block_equal(
                            hf_log_probs=hf_original_log_probs,
                            token_log_probs=meta["output_token_logprobs"],
                            top_log_probs=meta["output_top_logprobs"],
                            ids_log_probs=meta["output_token_ids_logprobs"],
                            random_token_ids=random_token_ids,
                            tag=f"Original logprobs SGLang vs HF: {prompt} ({env_val})",
                        )
                    else:
                        # Always check regular logprobs
                        self.assert_logprobs_block_equal(
                            hf_log_probs=hf_log_probs,
                            token_log_probs=meta["output_token_logprobs"],
                            top_log_probs=meta["output_top_logprobs"],
                            ids_log_probs=meta["output_token_ids_logprobs"],
                            random_token_ids=random_token_ids,
                            tag=f"logprobs SGLang vs HF: {prompt} ({env_val})",
                        )
                sgl_engine.shutdown()


if __name__ == "__main__":
    unittest.main()
