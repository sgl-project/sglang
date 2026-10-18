import unittest

import torch
from transformers import AutoModel, AutoTokenizer

from sglang.srt.entrypoints.engine import Engine
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

MODEL = "perplexity-ai/pplx-embed-v1-0.6b"
PROMPTS = [
    "Scientists explore the universe driven by curiosity.",
    "Children learn through curious exploration.",
    "What is the capital of France?",
    "Paris is the capital and most populous city of France, with an estimated population of 2 million residents.",
]


def hf_mean_pooled(prompts):
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL, trust_remote_code=True, dtype=torch.bfloat16
    ).cuda()
    embeddings = []
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            hidden = model(**inputs).last_hidden_state[0].float()
            embeddings.append(hidden.mean(dim=0).cpu())
    del model
    torch.cuda.empty_cache()
    return torch.stack(embeddings)


class TestPPLXEmbedding(CustomTestCase):
    def test_matches_hf_bidirectional_mean_pooling(self):
        """SGLang embeddings must match the checkpoint's bidirectional mean-pooled output."""
        expected = hf_mean_pooled(PROMPTS)
        engine = Engine(
            model_path=MODEL,
            is_embedding=True,
            trust_remote_code=True,
            dtype="bfloat16",
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            mem_fraction_static=0.3,
        )
        try:
            actual = torch.tensor([out["embedding"] for out in engine.encode(PROMPTS)])
        finally:
            engine.shutdown()
        similarity = torch.nn.functional.cosine_similarity(actual, expected, dim=-1)
        self.assertGreater(similarity.min().item(), 0.999, similarity.tolist())


if __name__ == "__main__":
    unittest.main()
