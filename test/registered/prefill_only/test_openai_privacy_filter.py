"""Compare all OPF token scores with HF, including the sliding-window boundary."""

import gc
import unittest

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

import sglang as sgl
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="nightly", runner_config="1-gpu-small")

MODEL = "openai/privacy-filter"
PROMPTS = [
    "Nothing private here.",
    "Contact Jane Smith at jane@example.com or +1 415 555 0123.",
    "My account number is 123456789; I live at 42 Maple Street, Boston.",
    "Born on January 2, 1980. Private URL: https://example.com/account/abc. "
    "Password: correct-horse-battery-staple.",
    "Ordinary context. " * 150 + " Contact Alice at alice@example.com.",
]


class TestOpenAIPrivacyFilter(unittest.TestCase):
    def test_token_scores_match_hf(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        input_ids = [tokenizer.encode(text) for text in PROMPTS]
        self.assertGreater(len(input_ids[-1]), 257)
        reference = (
            AutoModelForTokenClassification.from_pretrained(
                MODEL, dtype=torch.bfloat16, attn_implementation="eager"
            )
            .cuda()
            .eval()
        )
        with torch.inference_mode():
            expected = [
                reference(torch.tensor([ids], device="cuda")).logits[0].float().cpu()
                for ids in input_ids
            ]
        del reference
        gc.collect()
        torch.cuda.empty_cache()

        engine = sgl.Engine(
            model_path=MODEL,
            dtype="bfloat16",
            attention_backend="fa3",
            moe_runner_backend="triton",
            tp_size=1,
            context_length=2048,
        )
        try:
            batched = engine.encode(input_ids=input_ids)
            for ids, want, output in zip(input_ids, expected, batched, strict=True):
                actual = torch.tensor(output["embedding"])
                self.assertEqual(actual.shape, (len(ids), 33))
                # Match the public vLLM OPF test's BF16 probability tolerance.
                torch.testing.assert_close(
                    actual.softmax(-1), want.softmax(-1), atol=0.1, rtol=0.01
                )
                single = torch.tensor(engine.encode(input_ids=ids)["embedding"])
                torch.testing.assert_close(
                    actual.softmax(-1), single.softmax(-1), atol=0.01, rtol=0.01
                )
        finally:
            engine.shutdown()


if __name__ == "__main__":
    unittest.main()
