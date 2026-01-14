import multiprocessing as mp
import unittest

import torch

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.runners import TEST_RERANK_QUERY_DOCS, HFRunner, SRTRunner
from sglang.test.test_utils import CustomTestCase

register_npu_ci(
    est_time=400,
    suite="nightly-1-npu-a3",
    nightly=True,
    disabled="cross encoder scores are not all close",
)

MODELS = [
    ("/root/.cache/modelscope/hub/models/BAAI/bge-reranker-v2-m3", 1, 1e-2),
]
ATTENTION_BACKEND = ["ascend"]
TORCH_DTYPES = [torch.bfloat16]


class TestCrossEncoderModels(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def assert_close_prefill_logits(
        self,
        prompts,
        model_path,
        tp_size,
        torch_dtype,
        score_tolerance,
        attention_backend,
    ) -> None:
        with HFRunner(
            model_path,
            torch_dtype=torch_dtype,
            model_type="cross_encoder",
        ) as hf_runner:
            hf_scores = hf_runner.forward(prompts).scores

        with SRTRunner(
            model_path,
            tp_size=tp_size,
            torch_dtype=torch_dtype,
            model_type="cross_encoder",
            attention_backend=attention_backend,
            chunked_prefill_size=-1,
            disable_radix_cache=True,
        ) as srt_runner:
            srt_scores = srt_runner.forward(prompts).scores

        for i in range(len(srt_scores)):
            score_difference = abs(hf_scores[i] - srt_scores[i])

            assert (
                score_difference < score_tolerance
            ), "cross encoder scores are not all close"

    def preprocess_prompts(self, prompt):
        processed_prompts = []
        query = prompt["query"]
        documents = prompt["documents"]
        for document in documents:
            processed_prompts.append([query, document])

        return processed_prompts

    def test_prefill_logits(self):
        models_to_test = MODELS

        for model, tp_size, prefill_tolerance in models_to_test:
            for attention_backend in ATTENTION_BACKEND:
                for queryDocs in TEST_RERANK_QUERY_DOCS:
                    prompts = self.preprocess_prompts(queryDocs)
                    for torch_dtype in TORCH_DTYPES:
                        self.assert_close_prefill_logits(
                            prompts,
                            model,
                            tp_size,
                            torch_dtype,
                            prefill_tolerance,
                            attention_backend,
                        )


if __name__ == "__main__":
    unittest.main()
