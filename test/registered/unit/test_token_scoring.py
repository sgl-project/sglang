import asyncio
import json
import unittest
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from sglang.srt.entrypoints.engine_score_mixin import EngineScoreMixin
from sglang.srt.entrypoints.openai.protocol import ScoringRequest
from sglang.srt.entrypoints.openai.serving_score import OpenAIServingScore
from sglang.srt.managers.tokenizer_manager_score_mixin import TokenizerManagerScoreMixin
from sglang.srt.runtime_context import publish, restore_context, snapshot_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class ScoringManager(TokenizerManagerScoreMixin):
    """Replace only model execution; keep request construction and score extraction real."""

    def __init__(self, enable_mis=False, generation=True):
        self.server_args = ServerArgs(model_path="dummy", enable_mis=enable_mis)
        publish(self.server_args, role="test")
        self.is_generation = generation
        self.tokenizer = SimpleNamespace(vocab_size=8)
        self.logits = torch.tensor([-1000.0, -999.0, -997.0, -996.0, 0, 1, 2, 3])
        self.requests = []

    async def generate_request(self, request, raw_request):
        self.requests.append(request)
        request.normalize_batch_and_arguments()
        results = []
        for index, ids in enumerate(request.input_ids):
            meta = {"prompt_tokens": len(ids)}
            if self.is_generation:
                assert request.sampling_params[index]["max_new_tokens"] == 0
                labels = request.token_ids_logprob[index]
                values = torch.log_softmax(self.logits, dim=0)
                logprobs = [(values[token].item(), token, None) for token in labels]
                if self.server_args.enable_mis:
                    count = len(request.multi_item_delimiter_indices[index])
                    meta["input_token_ids_logprobs"] = [logprobs] * count
                else:
                    meta["output_token_ids_logprobs"] = [logprobs]
                results.append({"meta_info": meta})
            else:
                embedding = [1.0, 3.0]
                if self.server_args.enable_mis:
                    count = len(request.multi_item_delimiter_indices[index])
                    embedding = [embedding] * count
                results.append({"meta_info": meta, "embedding": embedding})
        yield results


class TestTokenScoring(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.addCleanup(restore_context, snapshot_context())

    async def test_http_to_manager_ragged_candidates(self):
        for enable_mis in (False, True):
            with self.subTest(enable_mis=enable_mis):
                manager = ScoringManager(enable_mis=enable_mis)
                request = ScoringRequest(
                    query=[4],
                    items=[[5], [6, 7]],
                    label_token_ids=[[3, 1], [0, 2, 1]],
                    apply_softmax=True,
                    temperature=2.0,
                    return_token_logprobs=True,
                )
                handler = OpenAIServingScore(manager)
                response = await handler._handle_non_streaming_request(
                    request, request, None
                )
                body = json.loads(response.body)
                for labels, scores, logprobs in zip(
                    request.label_token_ids, body["scores"], body["token_logprobs"]
                ):
                    expected = torch.softmax(manager.logits[labels] / 2, dim=0)
                    torch.testing.assert_close(torch.tensor(scores), expected)
                    torch.testing.assert_close(
                        torch.tensor(logprobs),
                        torch.log_softmax(manager.logits, dim=0)[labels],
                    )
                self.assertEqual(body["usage"]["completion_tokens"], 0)

    async def test_default_scores_are_vocabulary_probabilities(self):
        manager = ScoringManager()
        result = await manager.score_request(
            query=[], items=[[4], [5]], label_token_ids=[7, 4]
        )
        expected = torch.softmax(manager.logits, dim=0)[[7, 4]]
        for scores in result.scores:
            torch.testing.assert_close(torch.tensor(scores), expected)
        self.assertIsNone(result.token_logprobs)

    async def test_mis_text_items_do_not_add_special_tokens(self):
        vocab = {"[UNK]": 0, "[BOS]": 1, "Rate": 2, ":": 3, "Option": 4, "A": 5, "B": 6}
        tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A", special_tokens=[("[BOS]", vocab["[BOS]"])]
        )
        manager = ScoringManager(enable_mis=True)
        manager.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, bos_token="[BOS]", unk_token="[UNK]"
        )
        items = [" Option A", " Option B"]
        for query in ("Rate:", ""):
            with self.subTest(query=query):
                await manager.score_request(
                    query=query, items=items, label_token_ids=[5, 6]
                )
                request = manager.requests[-1]
                packed = request.input_ids[0]
                delimiters = request.multi_item_delimiter_indices[0]
                prefix = packed[: delimiters[0]]
                for item, start, end in zip(items, delimiters, delimiters[1:]):
                    self.assertEqual(
                        prefix + packed[start + 1 : end],
                        manager.tokenizer.encode(query + item),
                    )

    async def test_async_engine_and_full_prompts(self):
        manager = ScoringManager()
        engine = EngineScoreMixin()
        engine.tokenizer_manager = manager
        kwargs = dict(
            label_token_ids=[[1, 3], [2]],
            apply_softmax=True,
            temperature=0.5,
            return_token_logprobs=True,
        )
        actual = await engine.async_score(query=[], items=[[4], [5]], **kwargs)
        expected = await manager.score_prompts([[4], [5]], **kwargs)
        self.assertEqual(actual, expected)
        self.assertEqual(actual.scores[1], [1.0])

    async def test_classification_temperature(self):
        for enable_mis in (False, True):
            manager = ScoringManager(enable_mis=enable_mis, generation=False)
            result = await manager.score_request(
                query=[4], items=[[5], [6]], apply_softmax=True, temperature=2.0
            )
            for scores in result.scores:
                torch.testing.assert_close(
                    torch.tensor(scores), torch.softmax(torch.tensor([0.5, 1.5]), 0)
                )
            with self.assertRaisesRegex(ValueError, "only supported for CausalLM"):
                await manager.score_request(
                    query=[], items=[[4]], return_token_logprobs=True
                )

    async def test_invalid_candidates_and_temperature(self):
        manager = ScoringManager()
        for labels in ([], [[]], [[1]], [[1], []], [1, 1], [-1], [8], [True], "bad"):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                await manager.score_request(
                    query=[], items=[[4], [5]], label_token_ids=labels
                )
        for temperature in (0, -1, float("nan"), float("inf")):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                await manager.score_request(
                    query=[],
                    items=[[4]],
                    label_token_ids=[1],
                    temperature=temperature,
                    apply_softmax=True,
                )
        with self.assertRaisesRegex(ValueError, "requires apply_softmax"):
            await manager.score_request(
                query=[], items=[[4]], label_token_ids=[1], temperature=2.0
            )
        self.assertEqual(manager.requests, [])

    async def test_http_out_of_vocabulary_error(self):
        for enable_mis in (False, True):
            for token_id in (8, 999999):
                for labels in ([7, token_id], [[7], [1, token_id]]):
                    with self.subTest(enable_mis=enable_mis, labels=labels):
                        manager = ScoringManager(enable_mis=enable_mis)
                        request = ScoringRequest(
                            query=[],
                            items=[[4], [5]],
                            label_token_ids=labels,
                            apply_softmax=True,
                        )
                        response = await OpenAIServingScore(
                            manager
                        )._handle_non_streaming_request(request, request, None)
                        self.assertEqual(response.status_code, 400)
                        body = json.loads(response.body)
                        self.assertEqual(body["type"], "BadRequestError")
                        self.assertEqual(
                            body["message"],
                            f"Token ID {token_id} is out of vocabulary (vocab size: 8)",
                        )
                        self.assertEqual(manager.requests, [])

    async def test_small_temperature_is_finite(self):
        for enable_mis in (False, True):
            for generation in (False, True):
                manager = ScoringManager(enable_mis=enable_mis, generation=generation)
                result = await manager.score_request(
                    query=[4],
                    items=[[5]],
                    label_token_ids=[1, 3],
                    apply_softmax=True,
                    temperature=1e-300,
                )
                self.assertEqual(result.scores, [[0.0, 1.0]])

    async def test_empty_batch(self):
        manager = ScoringManager()
        result = await manager.score_request(
            query=[], items=[], label_token_ids=[], return_token_logprobs=True
        )
        self.assertEqual(result.scores, [])
        self.assertEqual(result.token_logprobs, [])
        self.assertEqual(manager.requests, [])


class TestSyncTokenScoring(unittest.TestCase):
    def test_engine_score(self):
        self.addCleanup(restore_context, snapshot_context())
        engine = EngineScoreMixin()
        engine.tokenizer_manager = ScoringManager()
        engine.loop = asyncio.new_event_loop()
        try:
            result = engine.score(
                query=[],
                items=[[4]],
                label_token_ids=[[1, 3]],
                apply_softmax=True,
                temperature=2.0,
                return_token_logprobs=True,
            )
            torch.testing.assert_close(
                torch.tensor(result.scores[0]),
                torch.softmax(engine.tokenizer_manager.logits[[1, 3]] / 2, 0),
            )
        finally:
            engine.loop.run_until_complete(engine.loop.shutdown_asyncgens())
            engine.loop.close()


if __name__ == "__main__":
    unittest.main()
