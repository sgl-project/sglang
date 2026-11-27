import asyncio
import copy
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from sglang.srt.entrypoints.openai.protocol import V1RerankReqInput
from sglang.srt.managers.tokenizer_manager_score_mixin import ScoreResult
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

# Keep consistent with other openai_server/basic unit tests.
register_cuda_ci(est_time=11, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="base-b-test-1-gpu-small-amd")

try:
    from sglang.srt.entrypoints.openai.serving_rerank import (
        OpenAIServingRerank,
        _is_qwen3_reranker_template,
        _qwen3_rerank_score,
        _render_jinja_chat_template,
        _render_vl_jinja_template,
    )
except ModuleNotFoundError as e:
    # Some minimal environments used for unit tests may not have FastAPI/torch installed.
    # Skip this test in that case.
    if e.name in ("fastapi", "torch"):
        OpenAIServingRerank = None  # type: ignore[assignment]
    else:
        raise


def _text_tokenizer():
    backend = Tokenizer(
        models.WordLevel({"[UNK]": 0, "yes": 1, "no": 2}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")


def _vl_template():
    path = (
        Path(__file__).resolve().parents[3]
        / "examples/chat_template/qwen3_vl_reranker.jinja"
    )
    return path.read_text().strip("\n").replace("\\n", "\n")


class _DummyModelConfig:
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

        tokenizer = _text_tokenizer()
        tpl = "{{ instruct | default('DEF') }}|{{ messages[0]['content'] }}|{{ messages[1]['content'] }}"
        self.assertEqual(
            _render_jinja_chat_template(
                tpl, tokenizer=tokenizer, query="Q", document="D", instruct=None
            ),
            "DEF|Q|D",
        )
        self.assertEqual(
            _render_jinja_chat_template(
                tpl, tokenizer=tokenizer, query="Q", document="D", instruct="I"
            ),
            "I|Q|D",
        )

    def test_helper_render_vl_template_preserves_media_and_defaults(self):
        """Changing renderers must preserve prompt bytes and media alignment."""
        from jinja2 import BaseLoader, Undefined
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        tokenizer = _text_tokenizer()
        tokenizer.chat_template = "unrelated template"
        template = _vl_template()
        query = [
            {"type": "text", "text": "Query\n问题"},
            {"type": "image"},
            {"type": "video"},
        ]
        document = [
            {"type": "video"},
            {"type": "text", "text": "Document <|im_end|>"},
            {"type": "image"},
        ]
        before = copy.deepcopy((query, document))
        legacy = ImmutableSandboxedEnvironment(
            loader=BaseLoader(), autoescape=False, undefined=Undefined
        ).from_string(template)
        for instruct in (None, "", "Find relevant candidates."):
            with self.subTest(instruct=instruct):
                kwargs = {"query": query, "document": document}
                if instruct:
                    kwargs["instruct"] = instruct
                actual = _render_vl_jinja_template(
                    template,
                    tokenizer=tokenizer,
                    query=query,
                    document=document,
                    instruct=instruct,
                )
                self.assertEqual(actual.encode(), legacy.render(**kwargs).encode())
                self.assertEqual(actual.count("<|image_pad|>"), 2)
                self.assertEqual(actual.count("<|video_pad|>"), 2)
                self.assertEqual((query, document), before)
                self.assertEqual(tokenizer.chat_template, "unrelated template")

    def test_helper_render_vl_template_reuses_compilation(self):
        """Compile once without caching request-specific content or instructions."""
        from jinja2 import Environment

        tokenizer = _text_tokenizer()
        template = _vl_template() + "{# " + uuid.uuid4().hex + " #}"
        original = Environment.from_string
        compiled = []

        def observe(env, source, *args, **kwargs):
            if source == template:
                compiled.append(source)
            return original(env, source, *args, **kwargs)

        with patch.object(Environment, "from_string", new=observe):
            prompts = [
                _render_vl_jinja_template(
                    template,
                    tokenizer=tokenizer,
                    query=[{"type": "text", "text": f"q{i}"}],
                    document=[{"type": "text", "text": f"d{i}"}],
                    instruct=f"instruction{i}",
                )
                for i in range(3)
            ]
        self.assertEqual(len(compiled), 1)
        self.assertEqual(len(set(prompts)), 3)
        for i, prompt in enumerate(prompts):
            self.assertIn(f"<Instruct>: instruction{i}", prompt)
            self.assertIn(f"<Query>:q{i}", prompt)
            self.assertIn(f"<Document>:d{i}", prompt)

    def test_handle_non_streaming_request_qwen3_path_uses_score_prompts(self):
        class _TM(_DummyTokenizerManager):
            def __init__(self):
                self.server_args = object()
                self.model_config = Mock()
                self.model_config.is_generation = True
                self.model_config.model_path = "qwen/qwen3"
                self.tokenizer = _text_tokenizer()
                self.tokenizer.chat_template = (
                    'Note that the answer can only be "yes" or "no". '
                    "{{ messages[0]['content'] }} {{ messages[1]['content'] }}"
                )

            async def score_prompts(
                self, prompts, label_token_ids, apply_softmax, request
            ):
                # Return [p_yes, p_no] for each prompt
                assert prompts == [
                    'Note that the answer can only be "yes" or "no". q d1',
                    'Note that the answer can only be "yes" or "no". q d2',
                ]
                assert label_token_ids == [1, 2]
                assert apply_softmax is False
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

    def test_build_rerank_response_returns_multimodal_documents(self):
        """A document sent as content parts is returned with return_documents on
        (the default), the same as a text document."""
        parts = [
            {"type": "text", "text": "a cat"},
            {"type": "image_url", "image_url": {"url": "http://host/cat.png"}},
        ]
        req = V1RerankReqInput(query="q", documents=[parts, "plain"])
        res = self.handler._build_rerank_response([0.9, 0.2], req)
        self.assertEqual(res[1].document, "plain")
        returned = res[0].model_dump()["document"]
        self.assertEqual(returned[0], {"type": "text", "text": "a cat"})
        self.assertEqual(returned[1]["image_url"]["url"], "http://host/cat.png")

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
                self.tokenizer = _text_tokenizer()
                self.tokenizer.chat_template = _vl_template()
                self.call_count = 0

            async def generate_request(self, req, _raw):
                # Return logprobs for yes/no
                # Mock logprobs: P(yes) > P(no) for first doc, P(no) > P(yes) for second

                image = "<|vision_start|><|image_pad|><|vision_end|>"
                video = "<|vision_start|><|video_pad|><|vision_end|>"
                assert f"<Query>:query{image}{video}" in req.text
                if self.call_count == 0:
                    assert f"<Document>:doc1{video}{image}<|im_end|>" in req.text
                    assert req.image_data == ["query.png", "doc1.png"]
                    assert req.video_data == ["query.mp4", "doc1.mp4"]
                else:
                    assert "<Document>:doc2<|im_end|>" in req.text
                    assert req.image_data == ["query.png"]
                    assert req.video_data == ["query.mp4"]

                if self.call_count == 0:
                    # First doc: yes is likely
                    yes_logprob = math.log(0.8)
                    no_logprob = math.log(0.2)
                else:
                    # Second doc: no is likely
                    yes_logprob = math.log(0.3)
                    no_logprob = math.log(0.7)

                self.call_count += 1

                top_logprobs = [
                    (yes_logprob, 1, "yes"),
                    (no_logprob, 2, "no"),
                ]

                # The rerank handler checks output_top_logprobs[0] for the first generated token
                meta_info = {"output_top_logprobs": [top_logprobs]}

                yield {"meta_info": meta_info, "embedding": None}

        tm = _TM()
        handler = OpenAIServingRerank(tm)
        query = [
            {"type": "text", "text": "query"},
            {"type": "image_url", "image_url": {"url": "query.png"}},
            {"type": "video_url", "video_url": {"url": "query.mp4"}},
        ]
        doc1 = [
            {"type": "text", "text": "doc1"},
            {"type": "video_url", "video_url": {"url": "doc1.mp4"}},
            {"type": "image_url", "image_url": {"url": "doc1.png"}},
        ]
        req = V1RerankReqInput(
            query=query, documents=[doc1, "doc2"], return_documents=True
        )

        raw_request = Mock()
        res = asyncio.run(handler._handle_non_streaming_request(req, req, raw_request))

        self.assertEqual(tm.call_count, 2)
        self.assertEqual(len(res), 2)
        # First doc should have higher score
        self.assertEqual(res[0].document, req.documents[0])
        self.assertAlmostEqual(res[0].score, 0.8)  # 0.8 / (0.8+0.2) = 0.8

        self.assertEqual(res[1].document, "doc2")
        self.assertAlmostEqual(res[1].score, 0.3)  # 0.3 / (0.3+0.7) = 0.3


if __name__ == "__main__":
    unittest.main(verbosity=2)
