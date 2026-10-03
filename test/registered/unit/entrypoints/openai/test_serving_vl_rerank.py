"""Regression tests for exact label probabilities in VL reranking."""

import asyncio
import math
import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.openai.protocol import V1RerankReqInput
from sglang.srt.entrypoints.openai.serving_rerank import OpenAIServingRerank

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

YES_ID = 10
NO_ID = 20
TEMPLATE = 'Answer can only be "yes" or "no". {{ document[0].text }}'


class _Tokenizer:
    chat_template = TEMPLATE

    def encode(self, text, **kwargs):
        return [{"yes": YES_ID, "no": NO_ID}[text]]


def _distribution(p_yes, p_no):
    # Fifty other tokens outrank "yes", excluding it from the top-50 list.
    return [
        (math.log(p_yes), YES_ID, "yes"),
        (math.log(p_no), NO_ID, "no"),
        *[(math.log((1 - p_yes - p_no) / 50), 100 + i, "other") for i in range(50)],
    ]


class _TokenizerManager:
    server_args = object()
    tokenizer = _Tokenizer()
    model_config = SimpleNamespace(
        is_generation=True, model_path="Qwen/Qwen3-VL-Reranker-2B"
    )

    def __init__(self):
        self.requests = []

    async def generate_request(self, request, raw_request):
        self.requests.append(request)
        p_yes, p_no = (0.002, 0.3) if "better" in request.text else (0.001, 0.2)
        distribution = _distribution(p_yes, p_no)
        meta_info = {}
        if request.top_logprobs_num:
            meta_info["output_top_logprobs"] = [
                sorted(distribution, reverse=True)[: request.top_logprobs_num]
            ]
        if request.token_ids_logprob:
            meta_info["output_token_ids_logprobs"] = [
                [row for row in distribution if row[1] in request.token_ids_logprob]
            ]
        yield {"meta_info": meta_info}


class TestServingVLRerank(CustomTestCase):
    def setUp(self):
        self.manager = _TokenizerManager()
        self.handler = OpenAIServingRerank(self.manager)

    def test_score_includes_label_outside_top_logprobs(self):
        distribution = _distribution(0.001, 0.2)
        top_logprobs = sorted(distribution, reverse=True)[:50]
        self.assertNotIn(YES_ID, [row[1] for row in top_logprobs])
        result = {
            "meta_info": {
                "output_top_logprobs": [top_logprobs],
                "output_token_ids_logprobs": [distribution[:2]],
            }
        }

        self.assertAlmostEqual(
            self.handler._extract_score_from_logprobs(result), 0.001 / 0.201
        )

    def test_handler_ranks_documents_using_both_labels(self):
        request = V1RerankReqInput(
            query="query",
            documents=["worse document", "better document"],
            return_documents=True,
            top_n=1,
        )
        result = asyncio.run(
            self.handler._handle_non_streaming_request(request, request, None)
        )

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].index, 1)
        self.assertEqual(result[0].document, "better document")
        self.assertAlmostEqual(result[0].score, 0.002 / 0.302)
        for internal_request in self.manager.requests:
            self.assertEqual(internal_request.token_ids_logprob, [YES_ID, NO_ID])
            self.assertFalse(internal_request.top_logprobs_num)
            self.assertEqual(internal_request.logprob_start_len, -1)

    def test_label_rows_are_matched_by_token_id(self):
        for p_yes, p_no in ((0.2, 0.8), (0.8, 0.2)):
            with self.subTest(p_yes=p_yes):
                result = {
                    "meta_info": {
                        "output_token_ids_logprobs": [
                            [
                                (math.log(p_no), NO_ID, None),
                                (math.log(p_yes), YES_ID, None),
                            ]
                        ]
                    }
                }
                self.assertAlmostEqual(
                    self.handler._extract_score_from_logprobs(result), p_yes
                )


if __name__ == "__main__":
    unittest.main()
