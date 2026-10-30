import asyncio
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import Mistral3Config, PreTrainedTokenizerFast

from sglang.srt.configs.embedding_model_spec import (
    RerankInputFormat,
    resolve_embedding_model_spec,
)
from sglang.srt.entrypoints.openai.protocol import V1RerankReqInput
from sglang.srt.managers.tokenizer_manager_score_mixin import ScoreResult
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

# Keep consistent with other openai_server/basic unit tests.
register_cuda_ci(est_time=11, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")

try:
    from sglang.srt.entrypoints.openai.serving_rerank import (
        OpenAIServingRerank,
        _is_qwen3_reranker_template,
        _qwen3_rerank_score,
        _render_jinja_chat_template,
    )
except ModuleNotFoundError as e:
    # Some minimal environments used for unit tests may not have FastAPI/torch installed.
    # Skip this test in that case.
    if e.name in ("fastapi", "torch"):
        OpenAIServingRerank = None  # type: ignore[assignment]
    else:
        raise


class _DummyModelConfig:
    embedding_model_spec = resolve_embedding_model_spec(
        [], is_embedding_requested=False, is_embedding_gemma=False
    )
    # Keep consistent with TokenizerManager.model_config usage
    is_generation = False


class _DummyTokenizer:
    chat_template = ""


class _DummyTokenizerManager:
    # Minimal surface required by OpenAIServingBase/OpenAIServingRerank
    server_args = object()
    model_config = _DummyModelConfig()
    tokenizer = _DummyTokenizer()

    async def generate_request(self, *_args, **_kwargs):
        raise AssertionError("generate_request should not be called in this unit test")


@unittest.skipIf(OpenAIServingRerank is None, "fastapi/torch is not installed")
class TestOpenAIServingRerankUnit(unittest.TestCase):
    def setUp(self):
        self.handler = OpenAIServingRerank(_DummyTokenizerManager())

    def test_query_document_chat_format_preserves_roles_and_images(self):
        backend = Tokenizer(
            WordLevel(
                {"query": 0, "document": 1, "q": 2, "d": 3, "[IMG]": 4, "[UNK]": 5},
                unk_token="[UNK]",
            )
        )
        backend.pre_tokenizer = WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
        tokenizer.chat_template = """{%- for message in messages -%}
{{ message.role }} {{ ' ' }}
{%- for part in message.content -%}
{{ part.text if part.type == 'text' else '[IMG]' }} {{ ' ' }}
{%- endfor -%}
{%- endfor -%}"""
        tm = _DummyTokenizerManager()
        tm.tokenizer = tokenizer
        config = Mistral3Config(
            architectures=["Mistral3ForSequenceClassification"],
            text_config={
                "model_type": "ministral3",
                "is_causal": False,
                "pooling": "avg",
            },
            vision_config={"model_type": "pixtral"},
            is_causal=False,
            pooling="avg",
            num_labels=1,
        )
        tm.model_config = SimpleNamespace(
            is_generation=False,
            embedding_model_spec=resolve_embedding_model_spec(
                config.architectures,
                is_embedding_requested=False,
                is_embedding_gemma=False,
                hf_config=config,
            ),
        )
        handler = OpenAIServingRerank(tm)
        document = [
            {"type": "text", "text": "d"},
            {"type": "image_url", "image_url": {"url": "https://example.com/doc.png"}},
        ]
        request = V1RerankReqInput(query="q", documents=[document, "d"])
        adapted, _ = handler._convert_to_internal_request(request)
        self.assertEqual(adapted.input_ids, [[0, 2, 1, 3, 4], [0, 2, 1, 3]])
        self.assertEqual(adapted.image_data, [["https://example.com/doc.png"], None])
        adapted.normalize_batch_and_arguments()
        self.assertEqual(adapted.batch_size, 2)
        result = handler._build_rerank_response([2.0, 1.0], request)
        returned = result[0].model_dump()["document"]
        self.assertEqual(len(returned), 2)
        self.assertEqual(returned[0]["text"], "d")
        self.assertEqual(returned[1]["image_url"]["url"], "https://example.com/doc.png")

        tm.model_config.embedding_model_spec = replace(
            tm.model_config.embedding_model_spec,
            rerank_input_format=RerankInputFormat.DEFAULT,
        )
        other_handler = OpenAIServingRerank(tm)
        adapted, _ = other_handler._convert_to_internal_request(
            V1RerankReqInput(query="q", documents=["d"])
        )
        self.assertTrue(adapted.is_cross_encoder_request)
        self.assertEqual(adapted.text, [["q", "d"]])

    def test_convert_to_internal_request_cross_encoder_pairs(self):
        req = V1RerankReqInput(
            query="q",
            documents=["doc-a", "doc-b"],
            instruct="Retrieve semantically similar text.",
        )

        adapted, processed = self.handler._convert_to_internal_request(req)

        # Avoid importing EmbeddingReqInput (requires torch). Use duck-typing checks instead.
        self.assertTrue(hasattr(adapted, "is_cross_encoder_request"))
        self.assertTrue(adapted.is_cross_encoder_request)
        self.assertEqual(getattr(adapted, "text"), [["q", "doc-a"], ["q", "doc-b"]])
        self.assertEqual(processed, req)

    def test_convert_to_internal_request_qwen3_template_returns_request(self):
        tm = _DummyTokenizerManager()
        tm.tokenizer.chat_template = (
            '... Note that the answer can only be "yes" or "no". ...'
        )
        handler = OpenAIServingRerank(tm)
        req = V1RerankReqInput(query="q", documents=["d1"])
        adapted, processed = handler._convert_to_internal_request(req)
        self.assertIs(adapted, req)
        self.assertIs(processed, req)

    def test_build_rerank_response_embedding_list_uses_first_scalar(self):
        req = V1RerankReqInput(
            query="q",
            documents=["doc-a", "doc-b"],
            return_documents=True,
        )
        # Two results with embedding as list, should coerce embedding[0] to float.
        # Also verifies sorting (doc-b > doc-a).
        ret = [
            {"embedding": [0.1, 0.2], "meta_info": {"id": "a"}},
            {"embedding": [0.9, -1.0], "meta_info": {"id": "b"}},
        ]

        res = self.handler._build_rerank_response(ret, req)

        self.assertEqual(len(res), 2)

        # Sorted descending by score, so doc-b first.
        self.assertEqual(res[0].document, "doc-b")
        self.assertEqual(res[0].index, 1)
        self.assertAlmostEqual(res[0].score, 0.9)
        self.assertEqual(res[0].meta_info, {"id": "b"})

        self.assertEqual(res[1].document, "doc-a")
        self.assertEqual(res[1].index, 0)
        self.assertAlmostEqual(res[1].score, 0.1)
        self.assertEqual(res[1].meta_info, {"id": "a"})

    def test_build_rerank_response_float_list(self):
        req = V1RerankReqInput(
            query="q", documents=["a", "b", "c"], return_documents=True
        )
        scores = [0.2, 0.9, 0.1]
        res = self.handler._build_rerank_response(scores, req)
        self.assertEqual([r.document for r in res], ["b", "a", "c"])
        self.assertEqual([r.index for r in res], [1, 0, 2])
        self.assertAlmostEqual(res[0].score, 0.9)
        self.assertAlmostEqual(res[1].score, 0.2)
        self.assertAlmostEqual(res[2].score, 0.1)

    def test_helper_is_qwen3_reranker_template(self):
        self.assertTrue(
            _is_qwen3_reranker_template(
                'Note that the answer can only be "yes" or "no".'
            )
        )
        self.assertFalse(_is_qwen3_reranker_template("plain template"))

    def test_helper_qwen3_rerank_score(self):
        self.assertAlmostEqual(_qwen3_rerank_score(0.9, 0.1), 0.9)
        self.assertAlmostEqual(_qwen3_rerank_score(0.0, 0.0), 0.0)

    def test_helper_render_jinja_chat_template(self):
        # Skip if jinja2 isn't installed in this environment.
        try:
            import jinja2  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("jinja2 is not installed")

        tpl = "{{ instruct | default('DEF') }}|{{ messages[0]['content'] }}|{{ messages[1]['content'] }}"
        self.assertEqual(
            _render_jinja_chat_template(tpl, query="Q", document="D", instruct=None),
            "DEF|Q|D",
        )
        self.assertEqual(
            _render_jinja_chat_template(tpl, query="Q", document="D", instruct="I"),
            "I|Q|D",
        )

    def test_handle_non_streaming_request_qwen3_path_uses_score_prompts(self):
        class _TM(_DummyTokenizerManager):
            def __init__(self):
                self.server_args = object()
                self.model_config = Mock()
                self.model_config.is_generation = True
                self.model_config.model_path = "qwen/qwen3"
                self.tokenizer = Mock()
                self.tokenizer.chat_template = (
                    'Note that the answer can only be "yes" or "no". '
                    "{{ messages[0]['content'] }} {{ messages[1]['content'] }}"
                )

            async def score_prompts(
                self, prompts, label_token_ids, apply_softmax, request
            ):
                # Return [p_yes, p_no] for each prompt
                assert len(prompts) == 2
                assert label_token_ids and len(label_token_ids) == 2
                return ScoreResult(scores=[[0.9, 0.1], [0.2, 0.8]], prompt_tokens=42)

        handler = OpenAIServingRerank(_TM())
        req = V1RerankReqInput(query="q", documents=["d1", "d2"], return_documents=True)
        adapted, _ = handler._convert_to_internal_request(req)
        raw_request = Mock()

        res = asyncio.run(
            handler._handle_non_streaming_request(adapted, req, raw_request)
        )
        self.assertEqual([r.document for r in res], ["d1", "d2"])
        self.assertAlmostEqual(res[0].score, 0.9 / (0.9 + 0.1))
        self.assertAlmostEqual(res[1].score, 0.2 / (0.2 + 0.8))

    def test_build_rerank_response_return_documents_false(self):
        """Test that document field is None when return_documents=False"""
        req = V1RerankReqInput(
            query="q", documents=["a", "b", "c"], return_documents=False
        )
        scores = [0.2, 0.9, 0.1]
        res = self.handler._build_rerank_response(scores, req)
        # All documents should be None
        self.assertEqual([r.document for r in res], [None, None, None])
        # But scores and indices should still be correct
        self.assertEqual([r.index for r in res], [1, 0, 2])
        self.assertAlmostEqual(res[0].score, 0.9)

    def test_build_rerank_response_top_n(self):
        """Test that top_n limits the number of returned results"""
        req = V1RerankReqInput(
            query="q", documents=["a", "b", "c"], return_documents=True, top_n=2
        )
        scores = [0.2, 0.9, 0.1]
        res = self.handler._build_rerank_response(scores, req)
        # Should only return top 2 results
        self.assertEqual(len(res), 2)
        self.assertEqual([r.document for r in res], ["b", "a"])
        self.assertEqual([r.index for r in res], [1, 0])
        self.assertAlmostEqual(res[0].score, 0.9)
        self.assertAlmostEqual(res[1].score, 0.2)

    def test_build_rerank_response_top_n_greater_than_total(self):
        """Test that top_n greater than total documents returns all documents"""
        req = V1RerankReqInput(
            query="q", documents=["a", "b"], return_documents=True, top_n=10
        )
        scores = [0.2, 0.9]
        res = self.handler._build_rerank_response(scores, req)
        # Should return all 2 documents even though top_n=10
        self.assertEqual(len(res), 2)
        self.assertEqual([r.document for r in res], ["b", "a"])

    def test_build_rerank_response_top_n_with_return_documents_false(self):
        """Test top_n works correctly with return_documents=False"""
        req = V1RerankReqInput(
            query="q", documents=["a", "b", "c"], return_documents=False, top_n=1
        )
        scores = [0.2, 0.9, 0.1]
        res = self.handler._build_rerank_response(scores, req)
        # Should only return top 1 result, and document should be None
        self.assertEqual(len(res), 1)
        self.assertIsNone(res[0].document)
        self.assertEqual(res[0].index, 1)
        self.assertAlmostEqual(res[0].score, 0.9)

    def test_handle_vl_reranker_request(self):
        """Test the Qwen3-VL reranker path with mocked logprobs."""
        import math

        # Mock tokenizer manager that supports generate_request
        class _AsyncGen:
            def __init__(self, val):
                self.val = val

            def __aiter__(self):
                return self

            async def __anext__(self):
                return self.val

        class _TM(_DummyTokenizerManager):
            def __init__(self):
                self.server_args = object()
                self.model_config = Mock()
                self.model_config.is_generation = True
                self.model_config.model_path = "qwen/qwen3-vl"
                self.tokenizer = Mock()
                # Mock VL template detection
                self.tokenizer.chat_template = (
                    "{% for x in query %}{{ x.text }}{% endfor %}"
                    "{% for x in document %}{{ x.text }}{% endfor %}"
                    'answer can only be "yes" or "no" <|vision_start|>'
                )

            async def generate_request(self, req, _raw):
                # Return logprobs for yes/no
                # Mock logprobs: P(yes) > P(no) for first doc, P(no) > P(yes) for second

                if not hasattr(self, "call_count"):
                    self.call_count = 0

                if self.call_count == 0:
                    # First doc: yes is likely
                    yes_logprob = math.log(0.8)
                    no_logprob = math.log(0.2)
                else:
                    # Second doc: no is likely
                    yes_logprob = math.log(0.3)
                    no_logprob = math.log(0.7)

                self.call_count += 1

                # Qwen3 token IDs: YES=9693, NO=2152
                top_logprobs = [
                    (yes_logprob, 9693, "yes"),
                    (no_logprob, 2152, "no"),
                ]

                # The rerank handler checks output_top_logprobs[0] for the first generated token
                meta_info = {"output_top_logprobs": [top_logprobs]}

                yield {"meta_info": meta_info, "embedding": None}

        handler = OpenAIServingRerank(_TM())
        req = V1RerankReqInput(
            query="query", documents=["doc1", "doc2"], return_documents=True
        )
        # Force VL path is handled by detection logic inside handler
        # We mocked chat_template to satisfy _is_qwen3_vl_reranker_template

        raw_request = Mock()
        res = asyncio.run(handler._handle_non_streaming_request(req, req, raw_request))

        self.assertEqual(len(res), 2)
        # First doc should have higher score
        self.assertEqual(res[0].document, "doc1")
        self.assertAlmostEqual(res[0].score, 0.8)  # 0.8 / (0.8+0.2) = 0.8

        self.assertEqual(res[1].document, "doc2")
        self.assertAlmostEqual(res[1].score, 0.3)  # 0.3 / (0.3+0.7) = 0.3


if __name__ == "__main__":
    unittest.main(verbosity=2)
