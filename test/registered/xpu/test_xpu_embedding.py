"""
XPU embedding parity test: compares HF and SRT embedding outputs on Intel XPU.

Usage:
python3 -m unittest test_xpu_embedding.TestXPUEmbedding
"""

import multiprocessing as mp
import unittest
from typing import Optional

import torch
from transformers import AutoConfig, AutoTokenizer

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.runners import DEFAULT_PROMPTS, HFRunner, SRTRunner
from sglang.test.test_utils import CustomTestCase, get_similarities

register_xpu_ci(est_time=180, suite="stage-b-test-1-gpu-xpu")

MODEL_PATH = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
TP_SIZE = 1
PREFILL_TOLERANCE = 1e-3
TORCH_DTYPE = torch.bfloat16


class TestXPUEmbedding(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def _truncate_prompts(self, prompts, model_path):
        config = AutoConfig.from_pretrained(model_path)
        max_length = config.to_dict().get("max_position_embeddings", 2048)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        truncated_prompts = []
        for prompt in prompts:
            tokens = tokenizer(prompt, return_tensors="pt", truncation=False)
            if len(tokens.input_ids[0]) > max_length:
                truncated_text = tokenizer.decode(
                    tokens.input_ids[0][: max_length - 1], skip_special_tokens=True
                )
                truncated_prompts.append(truncated_text)
            else:
                truncated_prompts.append(prompt)
        return truncated_prompts

    def assert_close_prefill_logits(
        self,
        prompts,
        model_path,
        tp_size,
        torch_dtype,
        prefill_tolerance,
        matryoshka_dim: Optional[int] = None,
    ) -> None:
        truncated_prompts = self._truncate_prompts(prompts, model_path)

        with HFRunner(
            model_path,
            torch_dtype=torch_dtype,
            model_type="embedding",
            matryoshka_dim=matryoshka_dim,
        ) as hf_runner:
            hf_outputs = hf_runner.forward(truncated_prompts)

        with SRTRunner(
            model_path,
            tp_size=tp_size,
            torch_dtype=torch_dtype,
            model_type="embedding",
            attention_backend="intel_xpu",
            json_model_override_args=(
                {"matryoshka_dimensions": [matryoshka_dim]}
                if matryoshka_dim is not None
                else None
            ),
        ) as srt_runner:
            srt_outputs = srt_runner.forward(
                truncated_prompts,
                dimensions=matryoshka_dim,
            )

        for prompt, hf_output, srt_output in zip(
            prompts,
            hf_outputs.embed_logits,
            srt_outputs.embed_logits,
        ):
            hf_logits = torch.Tensor(hf_output)
            srt_logits = torch.Tensor(srt_output)

            similarity = torch.tensor(get_similarities(hf_logits, srt_logits))
            if len(prompt) <= 1000:
                self.assertTrue(
                    torch.all(torch.abs(similarity - 1) < prefill_tolerance),
                    "embeddings are not all close",
                )

    def test_prefill_logits(self):
        self.assert_close_prefill_logits(
            DEFAULT_PROMPTS,
            MODEL_PATH,
            TP_SIZE,
            TORCH_DTYPE,
            PREFILL_TOLERANCE,
        )

    def test_matryoshka_embedding(self):
        self.assert_close_prefill_logits(
            DEFAULT_PROMPTS,
            MODEL_PATH,
            TP_SIZE,
            TORCH_DTYPE,
            PREFILL_TOLERANCE,
            matryoshka_dim=128,
        )


if __name__ == "__main__":
    unittest.main()
