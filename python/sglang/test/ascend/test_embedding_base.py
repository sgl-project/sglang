import multiprocessing as mp
from abc import ABC
from typing import List, Optional, Tuple

import torch
from transformers import AutoConfig, AutoTokenizer

from sglang.test.runners import DEFAULT_PROMPTS, HFRunner, SRTRunner
from sglang.test.test_utils import get_similarities


class BaseEmbeddingTest(ABC):
    """Base test class for embedding model tests"""

    MODELS: List[
        Tuple[str, int, float]
    ]  # [(model_path, tp_size, prefill_tolerance), ...]
    TORCH_DTYPES: List[torch.dtype] = [torch.float16]
    DEFAULT_PROMPTS: List[str] = DEFAULT_PROMPTS
    DEFAULT_MAX_LENGTH: int = 2048

    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def _truncate_prompts(self, prompts, model_path):
        """Truncate prompts to model's max length"""
        config = AutoConfig.from_pretrained(model_path)
        max_length = getattr(config, "max_position_embeddings", self.DEFAULT_MAX_LENGTH)

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
        """Assert embeddings from HF and SRT are within tolerance"""
        truncated_prompts = self._truncate_prompts(prompts, model_path)

        with HFRunner(
            model_path,
            torch_dtype=torch_dtype,
            model_type="embedding",
            matryoshka_dim=matryoshka_dim,
        ) as hf_runner:
            hf_outputs = hf_runner.forward(truncated_prompts)

        attention_backend = "ascend"
        json_model_override_args = (
            {"matryoshka_dimensions": [matryoshka_dim]} if matryoshka_dim else None
        )

        with SRTRunner(
            model_path,
            tp_size=tp_size,
            torch_dtype=torch_dtype,
            model_type="embedding",
            attention_backend=attention_backend,
            json_model_override_args=json_model_override_args,
        ) as srt_runner:
            srt_outputs = srt_runner.forward(
                truncated_prompts, dimensions=matryoshka_dim
            )

        for i in range(len(prompts)):
            hf_logits = torch.Tensor(hf_outputs.embed_logits[i])
            srt_logits = torch.Tensor(srt_outputs.embed_logits[i])

            similarity = torch.tensor(get_similarities(hf_logits, srt_logits))
            print("similarity diff", abs(similarity - 1))

            if len(prompts[i]) <= 1000:
                assert torch.all(
                    abs(similarity - 1) < prefill_tolerance
                ), "embeddings are not all close"

    def test_prefill_logits(self):
        """Main test method to run for all models and dtypes"""
        models_to_test = self.MODELS

        for model, tp_size, prefill_tolerance in models_to_test:
            for torch_dtype in self.TORCH_DTYPES:
                self.assert_close_prefill_logits(
                    self.DEFAULT_PROMPTS, model, tp_size, torch_dtype, prefill_tolerance
                )
