"""XPU rerank test suite.

This file validates score parity between HuggingFace and SRT for two rerank
serving styles:
- Decoder-only reranker scoring (Qwen3-Reranker style).
- Cross-encoder scoring (BAAI/bge-reranker-v2-m3).

Usage:
python3 -m unittest test_xpu_rerank.TestXPUDecoderRerank
python3 -m unittest test_xpu_rerank.TestXpuCrossEncoderReank
"""

import math
import multiprocessing as mp
import unittest

import torch
from jinja2.sandbox import ImmutableSandboxedEnvironment

from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.runners import TEST_RERANK_QUERY_DOCS, HFRunner, SRTRunner
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=180, suite="stage-b-test-1-gpu-xpu")

MODEL_PATH = "Qwen/Qwen3-Reranker-0.6B"
TP_SIZE = 1
SCORE_TOLERANCE = 1e-2
ATTENTION_BACKEND = "intel_xpu"
TORCH_DTYPE = torch.bfloat16
# Prompt template mirrored from examples/chat_template/qwen3_reranker.jinja.
QWEN3_RERANKER_TEMPLATE = r"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {{ instruct | default("Given a web search query, retrieve relevant passages that answer the query.") }}
<Query>: {{ messages[0]["content"] }}
<Document>: {{ messages[1]["content"] }}<|im_end|>
<|im_start|>assistant{{ '\\n' }}
"""

JINJA_ENV = ImmutableSandboxedEnvironment(autoescape=False)
QWEN3_RERANKER_JINJA = JINJA_ENV.from_string(QWEN3_RERANKER_TEMPLATE)

# Small decoder-reranker dataset (from the Qwen3-Reranker cookbook style).
# The documents intentionally include clear relevant/irrelevant contrast.

RERANK_QUERY_DOCS = [
    {
        "query": "法国首都是哪里？",
        "instruct": "Given a web search query, retrieve relevant passages that answer the query.",
        "documents": [
            "法国的首都是巴黎。",
            "德国的首都是柏林。",
            "香蕉是黄色的水果。",
        ],
    },
]


def format_prompt(query: str, document: str, instruct: str) -> str:
    """Render the canonical Qwen3 reranker Jinja template used by serving."""
    render_kwargs = {
        "messages": [
            {"role": "user", "content": query},
            {"role": "user", "content": document},
        ]
    }
    if instruct:
        render_kwargs["instruct"] = instruct
    return QWEN3_RERANKER_JINJA.render(**render_kwargs)


def yes_no_token_ids(tokenizer) -> tuple[int, int]:
    yes = tokenizer.encode("yes", add_special_tokens=False)
    no = tokenizer.encode("no", add_special_tokens=False)
    assert len(yes) == 1 and len(no) == 1, "yes/no must be single tokens"
    return yes[0], no[0]


def score_from_token_logprobs(logprob_yes: float, logprob_no: float) -> float:
    """score = P(yes) / (P(yes) + P(no))."""
    p_yes = math.exp(logprob_yes)
    p_no = math.exp(logprob_no)
    denom = p_yes + p_no
    return p_yes / denom if denom > 0.0 else 0.0


class TestXPUDecoderRerank(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)
        cls.tokenizer = get_tokenizer(MODEL_PATH)
        cls.yes_id, cls.no_id = yes_no_token_ids(cls.tokenizer)

    def _extract_scores(self, token_ids_output_logprobs) -> list[float]:
        """token_ids_output_logprobs shape: [num_prompts][num_gen_tokens][num_token_ids].

        We only generate 1 token and request exactly [yes_id, no_id], so we read
        index [0] (first/only generated token) -> [yes_lp, no_lp].
        """
        scores = []
        for per_prompt in token_ids_output_logprobs:
            first_token_lps = per_prompt[0]  # logprobs for [yes_id, no_id]
            yes_lp, no_lp = first_token_lps[0], first_token_lps[1]
            scores.append(score_from_token_logprobs(yes_lp, no_lp))
        return scores

    def _assert_close_scores(self, prompts) -> None:
        token_ids_logprob = [self.yes_id, self.no_id]

        # --- HuggingFace reference (generation) ---
        with HFRunner(
            MODEL_PATH,
            torch_dtype=TORCH_DTYPE,
            model_type="generation",
            output_str_only=False,
        ) as hf_runner:
            hf_out = hf_runner.forward(
                prompts,
                max_new_tokens=1,
                token_ids_logprob=token_ids_logprob,
            )
        hf_scores = self._extract_scores(hf_out.token_ids_output_logprobs)

        with SRTRunner(
            MODEL_PATH,
            tp_size=TP_SIZE,
            torch_dtype=TORCH_DTYPE,
            model_type="generation",
            attention_backend=ATTENTION_BACKEND,
        ) as srt_runner:
            srt_out = srt_runner.forward(
                prompts,
                max_new_tokens=1,
                token_ids_logprob=token_ids_logprob,
            )
        srt_scores = self._extract_scores(srt_out.token_ids_output_logprobs)

        self.assertEqual(len(hf_scores), len(srt_scores))
        for hf_score, srt_score in zip(hf_scores, srt_scores):
            self.assertLess(
                abs(hf_score - srt_score),
                SCORE_TOLERANCE,
                "decoder rerank scores are not all close",
            )

    def _preprocess_prompts(self, query_doc) -> list[str]:
        query = query_doc["query"]
        instruct = query_doc["instruct"]
        return [format_prompt(query, doc, instruct) for doc in query_doc["documents"]]

    def test_prefill_logits(self):
        for query_doc in RERANK_QUERY_DOCS:
            prompts = self._preprocess_prompts(query_doc)
            self._assert_close_scores(prompts)


# This cross-encoder test is ported from `test/manual/prefill_only/test_cross_encoder_models.py`,
# which uses float32 with the triton backend. The `intel_xpu` attention backend currently only
# supports the bfloat16 dtype, so we keep the triton backend here to preserve float32 parity.
CROSS_ENCODER_MODEL_PATH = "BAAI/bge-reranker-v2-m3"
CROSS_ENCODER_TP_SIZE = 1
CROSS_ENCODER_SCORE_TOLERANCE = 1e-2
CROSS_ENCODER_ATTENTION_BACKEND = "triton"
CROSS_ENCODER_TORCH_DTYPE = torch.float32


class TestXPUCrossEncoderRerank(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def _assert_close_scores(
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

        self.assertEqual(len(hf_scores), len(srt_scores))
        for hf_score, srt_score in zip(hf_scores, srt_scores):
            self.assertLess(
                abs(hf_score - srt_score),
                score_tolerance,
                "cross encoder scores are not all close",
            )

    def _preprocess_prompts(self, query_doc):
        query = query_doc["query"]
        return [[query, document] for document in query_doc["documents"]]

    def test_prefill_logits(self):
        for query_doc in TEST_RERANK_QUERY_DOCS:
            prompts = self._preprocess_prompts(query_doc)
            self._assert_close_scores(
                prompts,
                CROSS_ENCODER_MODEL_PATH,
                CROSS_ENCODER_TP_SIZE,
                CROSS_ENCODER_TORCH_DTYPE,
                CROSS_ENCODER_SCORE_TOLERANCE,
                CROSS_ENCODER_ATTENTION_BACKEND,
            )


if __name__ == "__main__":
    unittest.main()
