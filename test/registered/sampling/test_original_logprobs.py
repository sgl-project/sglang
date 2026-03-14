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
- Input logprobs: Verifies input_token_logprobs and input_top_logprobs against
  the Hugging Face reference for input positions.
- logprob_start_len: Ensures correct boundary behavior for the starting index
  of input log-probability computation.
"""

import os
import random
import unittest

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import sglang as sgl
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST

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
TOP_LOGPROBS_NUM = 50
NUM_RANDOM_TOKEN_IDS = 10
RTOL = 0.20
ATOL = 0.00
MULTI_TOKEN_COUNT = 8
# ------------------------------------------------

torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class TestOriginalLogprob(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # ----- HF side (float32 weights, loaded once for all tests) -----
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="right")
        cls.hf_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float32, device_map="auto"
        )

    def setUp(self):
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
            with self.subTest(SGLANG_RETURN_ORIGINAL_LOGPROB=env_val):
                os.environ["SGLANG_RETURN_ORIGINAL_LOGPROB"] = env_val

                # ----- SGLang side -----
                sgl_engine = sgl.Engine(
                    model_path=MODEL_ID,
                    skip_tokenizer_init=True,
                    trust_remote_code=True,
                    mem_fraction_static=0.60,
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

    # ---------------------------------------------------------------------
    # Helper: compute HF per-position log-prob vectors for a token sequence.
    # logprobs[i] = log_softmax(logits[i] / temperature) predicts token[i+1].
    # ---------------------------------------------------------------------
    def hf_logprobs_for_sequence(self, token_ids, temperature=1.0):
        input_tensor = torch.tensor(
            [token_ids], dtype=torch.long, device=self.hf_model.device
        )
        with torch.inference_mode():
            logits = self.hf_model(input_ids=input_tensor, return_dict=True).logits[0]
        if temperature != 1.0:
            logits = logits / temperature
        return F.log_softmax(logits.float(), dim=-1)

    # ---------------------------------------------------------------------
    # Helper: compare SGLang logprob outputs against HF at a single position.
    # ---------------------------------------------------------------------
    def assert_position_logprobs_match(
        self,
        hf_lp_vec,
        sgl_token_logprob,
        sgl_top_logprobs=None,
        sgl_ids_logprobs=None,
        random_token_ids=None,
        tag="",
    ):
        sgl_val, sgl_idx = sgl_token_logprob[0], sgl_token_logprob[1]
        if sgl_val is None:
            return

        hf_val = hf_lp_vec[sgl_idx].item()
        self.assertTrue(
            torch.allclose(
                torch.tensor([hf_val]),
                torch.tensor([float(sgl_val)]),
                rtol=RTOL,
                atol=ATOL,
            ),
            msg=f"[{tag}] token logprob mismatch: HF={hf_val:.6f} vs SGL={sgl_val:.6f}",
        )

        if sgl_top_logprobs is not None:
            hf_topk_vals, _ = torch.topk(hf_lp_vec, k=TOP_LOGPROBS_NUM, dim=-1)
            sgl_vals = [float(t[0]) for t in sgl_top_logprobs if t and t[0] is not None]
            if sgl_vals:
                sgl_topk = torch.tensor(
                    sgl_vals[:TOP_LOGPROBS_NUM],
                    dtype=torch.float32,
                    device=self.hf_model.device,
                )
                k = min(hf_topk_vals.numel(), sgl_topk.numel())
                self.assertTrue(
                    torch.allclose(
                        hf_topk_vals[:k], sgl_topk[:k], rtol=RTOL, atol=ATOL
                    ),
                    msg=f"[{tag}] top-k mismatch",
                )

        if sgl_ids_logprobs is not None and random_token_ids:
            ids_tensor = torch.tensor(
                random_token_ids, dtype=torch.long, device=hf_lp_vec.device
            )
            hf_ids_vals = hf_lp_vec[ids_tensor]
            sgl_ids_vals = torch.tensor(
                [float(v) for v, _, _ in sgl_ids_logprobs],
                dtype=torch.float32,
                device=self.hf_model.device,
            )
            self.assertTrue(
                torch.allclose(hf_ids_vals, sgl_ids_vals, rtol=RTOL, atol=ATOL),
                msg=f"[{tag}] token-IDs mismatch",
            )

    def test_output_logprob_multi_token(self):
        """Multi-token decode: validate per-step logprobs across an 8-token generation."""
        vocab_size = self.tokenizer.vocab_size
        sampling_params = {
            "temperature": 0.5,
            "top_p": 1.0,
            "top_k": 10,
            "max_new_tokens": MULTI_TOKEN_COUNT,
        }

        for env_val in ["True", "False"]:
            with self.subTest(SGLANG_RETURN_ORIGINAL_LOGPROB=env_val):
                os.environ["SGLANG_RETURN_ORIGINAL_LOGPROB"] = env_val

                sgl_engine = sgl.Engine(
                    model_path=MODEL_ID,
                    skip_tokenizer_init=True,
                    trust_remote_code=True,
                    mem_fraction_static=0.60,
                )

                for prompt in PROMPTS:
                    random_token_ids = sorted(
                        random.sample(range(vocab_size), NUM_RANDOM_TOKEN_IDS)
                    )

                    enc = self.tokenizer(prompt, return_tensors="pt")
                    input_ids = enc["input_ids"][0].tolist()
                    prompt_len = len(input_ids)

                    outputs = sgl_engine.generate(
                        input_ids=input_ids,
                        sampling_params=sampling_params,
                        return_logprob=True,
                        top_logprobs_num=TOP_LOGPROBS_NUM,
                        token_ids_logprob=random_token_ids,
                    )

                    if isinstance(outputs, list):
                        outputs = outputs[0]
                    meta = outputs["meta_info"]

                    output_token_ids = [t[1] for t in meta["output_token_logprobs"]]
                    self.assertEqual(len(output_token_ids), MULTI_TOKEN_COUNT)

                    full_sequence = input_ids + output_token_ids
                    hf_temp = (
                        1.0 if env_val == "True" else sampling_params["temperature"]
                    )
                    hf_all_lp = self.hf_logprobs_for_sequence(
                        full_sequence, temperature=hf_temp
                    )

                    max_diff = 0.0
                    for step in range(MULTI_TOKEN_COUNT):
                        hf_pos = prompt_len - 1 + step
                        self.assert_position_logprobs_match(
                            hf_lp_vec=hf_all_lp[hf_pos],
                            sgl_token_logprob=meta["output_token_logprobs"][step],
                            sgl_top_logprobs=meta["output_top_logprobs"][step],
                            sgl_ids_logprobs=meta["output_token_ids_logprobs"][step],
                            random_token_ids=random_token_ids,
                            tag=f"Multi-token step {step}: {prompt} ({env_val})",
                        )
                        sgl_val = meta["output_token_logprobs"][step][0]
                        hf_val = hf_all_lp[hf_pos][
                            meta["output_token_logprobs"][step][1]
                        ].item()
                        max_diff = max(max_diff, abs(hf_val - sgl_val))

                    print(
                        f"[Multi-token {prompt} ({env_val})] max|diff| = {max_diff:.4f}"
                    )

                sgl_engine.shutdown()

    def test_input_logprobs(self):
        """Input logprobs: verify input_token_logprobs and input_top_logprobs against HF."""
        vocab_size = self.tokenizer.vocab_size
        sampling_params = {
            "temperature": 0.5,
            "top_p": 1.0,
            "top_k": 10,
            "max_new_tokens": 1,
        }

        for env_val in ["True", "False"]:
            with self.subTest(SGLANG_RETURN_ORIGINAL_LOGPROB=env_val):
                os.environ["SGLANG_RETURN_ORIGINAL_LOGPROB"] = env_val

                sgl_engine = sgl.Engine(
                    model_path=MODEL_ID,
                    skip_tokenizer_init=True,
                    trust_remote_code=True,
                    mem_fraction_static=0.60,
                )

                for prompt in PROMPTS:
                    random_token_ids = sorted(
                        random.sample(range(vocab_size), NUM_RANDOM_TOKEN_IDS)
                    )

                    enc = self.tokenizer(prompt, return_tensors="pt")
                    input_ids = enc["input_ids"][0].tolist()
                    prompt_len = len(input_ids)

                    outputs = sgl_engine.generate(
                        input_ids=input_ids,
                        sampling_params=sampling_params,
                        return_logprob=True,
                        logprob_start_len=0,
                        top_logprobs_num=TOP_LOGPROBS_NUM,
                        token_ids_logprob=random_token_ids,
                    )

                    if isinstance(outputs, list):
                        outputs = outputs[0]
                    meta = outputs["meta_info"]

                    input_token_logprobs = meta["input_token_logprobs"]
                    input_top_logprobs = meta["input_top_logprobs"]
                    input_ids_logprobs = meta.get("input_token_ids_logprobs")

                    self.assertEqual(len(input_token_logprobs), prompt_len)

                    hf_temp = (
                        1.0 if env_val == "True" else sampling_params["temperature"]
                    )
                    hf_all_lp = self.hf_logprobs_for_sequence(
                        input_ids, temperature=hf_temp
                    )

                    # Skip position 0 (no preceding logits for the first token)
                    max_diff = 0.0
                    for pos in range(1, prompt_len):
                        hf_lp_vec = hf_all_lp[pos - 1]

                        sgl_top = (
                            input_top_logprobs[pos] if input_top_logprobs else None
                        )
                        sgl_ids = (
                            input_ids_logprobs[pos] if input_ids_logprobs else None
                        )

                        self.assert_position_logprobs_match(
                            hf_lp_vec=hf_lp_vec,
                            sgl_token_logprob=input_token_logprobs[pos],
                            sgl_top_logprobs=sgl_top,
                            sgl_ids_logprobs=sgl_ids,
                            random_token_ids=random_token_ids,
                            tag=f"Input pos {pos}: {prompt} ({env_val})",
                        )

                        sgl_val = input_token_logprobs[pos][0]
                        if sgl_val is not None:
                            hf_val = hf_lp_vec[input_token_logprobs[pos][1]].item()
                            max_diff = max(max_diff, abs(hf_val - sgl_val))

                    print(
                        f"[Input logprobs {prompt} ({env_val})] "
                        f"max|diff| = {max_diff:.4f}"
                    )

                sgl_engine.shutdown()

    def test_logprob_start_len(self):
        """Verify logprob_start_len correctly controls the starting index."""
        sampling_params = {
            "temperature": 0.5,
            "max_new_tokens": 4,
        }

        os.environ["SGLANG_RETURN_ORIGINAL_LOGPROB"] = "True"
        sgl_engine = sgl.Engine(
            model_path=MODEL_ID,
            skip_tokenizer_init=True,
            trust_remote_code=True,
            mem_fraction_static=0.60,
        )

        for prompt in PROMPTS:
            enc = self.tokenizer(prompt, return_tensors="pt")
            input_ids = enc["input_ids"][0].tolist()
            prompt_len = len(input_ids)

            hf_all_lp = self.hf_logprobs_for_sequence(input_ids, temperature=1.0)

            for start_len in [0, 1, prompt_len // 2, prompt_len - 1]:
                with self.subTest(prompt=prompt, logprob_start_len=start_len):
                    outputs = sgl_engine.generate(
                        input_ids=input_ids,
                        sampling_params=sampling_params,
                        return_logprob=True,
                        logprob_start_len=start_len,
                        top_logprobs_num=5,
                    )

                    if isinstance(outputs, list):
                        outputs = outputs[0]
                    meta = outputs["meta_info"]

                    expected_input_len = prompt_len - start_len
                    self.assertEqual(
                        len(meta["input_token_logprobs"]),
                        expected_input_len,
                        msg=f"input_token_logprobs length mismatch "
                        f"for start_len={start_len}",
                    )
                    self.assertEqual(
                        len(meta["input_top_logprobs"]),
                        expected_input_len,
                        msg=f"input_top_logprobs length mismatch "
                        f"for start_len={start_len}",
                    )
                    self.assertEqual(
                        len(meta["output_token_logprobs"]),
                        sampling_params["max_new_tokens"],
                    )
                    self.assertEqual(
                        meta["prompt_tokens"],
                        start_len + len(meta["input_token_logprobs"]),
                    )

                    # Spot-check: verify the first returned position against HF
                    if start_len > 0 and expected_input_len > 0:
                        first_lp = meta["input_token_logprobs"][0]
                        hf_lp = hf_all_lp[start_len - 1]
                        sgl_val = first_lp[0]
                        if sgl_val is not None:
                            hf_val = hf_lp[first_lp[1]].item()
                            self.assertTrue(
                                torch.allclose(
                                    torch.tensor([hf_val]),
                                    torch.tensor([float(sgl_val)]),
                                    rtol=RTOL,
                                    atol=ATOL,
                                ),
                                msg=f"First position (start_len={start_len}) "
                                f"HF={hf_val:.6f} vs SGL={sgl_val:.6f}",
                            )

                    print(
                        f"[logprob_start_len={start_len} {prompt}] "
                        f"input_logprobs_len="
                        f"{len(meta['input_token_logprobs'])}, "
                        f"output_logprobs_len="
                        f"{len(meta['output_token_logprobs'])}"
                    )

        sgl_engine.shutdown()


if __name__ == "__main__":
    unittest.main()
