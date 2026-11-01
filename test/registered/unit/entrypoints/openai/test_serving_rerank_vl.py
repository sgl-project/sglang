"""Unit tests for the Qwen3-VL reranker path in serving_rerank.py.

The VL reranker scores each document with its own generate request. These
tests check that all documents of a rerank request are in flight at the
same time (so the scheduler can batch them), that scores still line up with
the input order when requests finish out of order, and that prompt
logprobs are not requested.
"""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import asyncio
import math
import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.openai.protocol import V1RerankReqInput
from sglang.srt.entrypoints.openai.serving_rerank import OpenAIServingRerank
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

YES_ID = 10
NO_ID = 20
# Detected as a Qwen3-VL reranker template: reranker phrase + vision token.
TEMPLATE = (
    'Answer can only be "yes" or "no". <|vision_start|>'
    "{% for x in document %}{{ x.text }}{% endfor %}"
)


class _Tokenizer:
    chat_template = TEMPLATE

    def encode(self, text, **kwargs):
        return [{"yes": YES_ID, "no": NO_ID}[text]]


class _TokenizerManager:
    server_args = object()
    tokenizer = _Tokenizer()
    model_config = SimpleNamespace(
        is_generation=True, model_path="Qwen/Qwen3-VL-Reranker-2B"
    )

    def __init__(self, p_yes_by_doc, delay_by_doc=None, wait_for=None):
        self.p_yes_by_doc = p_yes_by_doc
        self.delay_by_doc = delay_by_doc or {}
        # When set, each request waits until this many requests are in flight.
        self.wait_for = wait_for
        self.requests = []
        self.in_flight = 0
        self.max_in_flight = 0
        self._all_started = asyncio.Event()

    def _document(self, request):
        # The rendered prompt ends with the document text.
        return next(d for d in self.p_yes_by_doc if request.text.endswith(d))

    async def generate_request(self, request, raw_request):
        doc = self._document(request)
        self.requests.append(request)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        if self.wait_for is not None:
            if len(self.requests) == self.wait_for:
                self._all_started.set()
            # A serial implementation never starts the next request, so bound
            # the wait instead of hanging the test.
            await asyncio.wait_for(self._all_started.wait(), timeout=1.0)
        await asyncio.sleep(self.delay_by_doc.get(doc, 0))
        self.in_flight -= 1
        p_yes = self.p_yes_by_doc[doc]
        labels = [
            (math.log(p_yes), YES_ID, "yes"),
            (math.log(1 - p_yes), NO_ID, "no"),
        ]
        # Answer whichever logprob form the handler asked for.
        meta_info = {}
        if request.top_logprobs_num:
            meta_info["output_top_logprobs"] = [labels]
        if request.token_ids_logprob:
            meta_info["output_token_ids_logprobs"] = [labels]
        yield {"meta_info": meta_info}


def _rerank(manager, documents):
    handler = OpenAIServingRerank(manager)
    request = V1RerankReqInput(
        query="query", documents=documents, return_documents=True
    )

    async def run():
        return await handler._handle_non_streaming_request(request, request, None)

    return asyncio.run(run())


class TestServingRerankVLConcurrency(CustomTestCase):
    def test_documents_are_in_flight_together(self):
        docs = [f"doc{i}" for i in range(4)]
        manager = _TokenizerManager({doc: 0.5 for doc in docs}, wait_for=len(docs))

        result = _rerank(manager, docs)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(docs))
        self.assertEqual(manager.max_in_flight, len(docs))

    def test_scores_follow_input_order_when_finished_out_of_order(self):
        p_yes = {"doc0": 0.9, "doc1": 0.6, "doc2": 0.3, "doc3": 0.1}
        # The first document finishes last.
        delay = {"doc0": 0.03, "doc1": 0.02, "doc2": 0.01, "doc3": 0.0}
        manager = _TokenizerManager(p_yes, delay_by_doc=delay)

        result = _rerank(manager, list(p_yes))

        self.assertEqual([r.index for r in result], [0, 1, 2, 3])
        for r in result:
            self.assertEqual(r.document, f"doc{r.index}")
            self.assertAlmostEqual(r.score, p_yes[r.document])

    def test_requests_skip_prompt_logprobs(self):
        docs = ["doc0", "doc1"]
        manager = _TokenizerManager({doc: 0.5 for doc in docs})

        _rerank(manager, docs)

        self.assertEqual(len(manager.requests), len(docs))
        for request in manager.requests:
            self.assertTrue(request.return_logprob)
            self.assertEqual(request.logprob_start_len, -1)


if __name__ == "__main__":
    unittest.main()
