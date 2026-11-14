"""ReasonerGrammarBackend over the built-in llguidance and outlines backends.

These tests are in their own file and not in test_reasoner_grammar_backend.py.
That file also runs on the Intel CPU suite, and its image installs an outlines
release without the `outlines.fsm` module that outlines_backend.py imports.
"""

import inspect
import json
import os
import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from sglang.srt.constrained.base_grammar_backend import (
    GrammarMask,
    GrammarRow,
    create_grammar_backend,
)
from sglang.srt.constrained.llguidance_backend import GuidanceBackend
from sglang.srt.constrained.outlines_backend import OutlinesGrammarBackend
from sglang.srt.constrained.reasoner_grammar_backend import (
    ReasonerGrammarBackend,
    ReasonerGrammarObject,
)
from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend
from sglang.srt.parser.reasoning_parser import DeepSeekR1Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_SCHEMA = json.dumps(
    {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
)


def _byte_tokenizer() -> PreTrainedTokenizerFast:
    """256 byte tokens plus <think>, </think> and <eos> (259 ids); built inline."""
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tk = Tokenizer(
        models.BPE(vocab={ch: i for i, ch in enumerate(alphabet)}, merges=[])
    )
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tk.decoder = decoders.ByteLevel()
    tk.add_special_tokens(["<think>", "</think>", "<eos>"])
    return PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<eos>")


class _ReasonerOverBuiltinBackend:
    """ReasonerGrammarBackend over a real llguidance or outlines backend.

    Bug: ReasonerGrammarBackend reads allocate/move/apply_vocab_mask from the
    wrapped backend for every grammar it builds. GuidanceBackend and
    OutlinesGrammarBackend defined none of them, so with --reasoning-parser
    every json/regex/ebnf/structural_tag request was aborted with an
    AttributeError, and with --enable-strict-thinking and a parser without
    excluded tokens the scheduler raised for every request.

    Subclasses set self.tokenizer and self.backend and the keys to test. The
    model vocab is larger than the tokenizer, as for a real model whose embedding
    matrix is padded.
    """

    keys = ()
    padding = 61

    def setUp(self):
        super().setUp()
        # A think budget makes strict thinking filter tokens, which these
        # backends refuse. -1 is the default: no budget.
        patcher = patch.dict(os.environ, {"SGLANG_MAX_THINK_TOKENS": "-1"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tokenizer = _byte_tokenizer()
        self.model_vocab = len(self.tokenizer) + self.padding
        self.everything = set(range(self.model_vocab))
        self.think_end = self.tokenizer.encode("</think>", add_special_tokens=False)
        self.backend = self._make_backend()

    def _make_backend(self):
        raise NotImplementedError

    def _wrap(self, detector=None, enable_strict_thinking=False):
        return ReasonerGrammarBackend(
            self.backend,
            SimpleNamespace(detector=detector or DeepSeekR1Detector()),
            self.tokenizer,
            enable_strict_thinking=enable_strict_thinking,
        )

    def _token(self, text):
        (token_id,) = self.tokenizer.encode(text, add_special_tokens=False)
        return token_id

    def _expected(self, first_texts):
        return {self._token(text) for text in first_texts}

    def _build(self, reasoner, key, answering=False):
        grammar = reasoner._init_value_dispatch(key, True)
        self.assertIsInstance(grammar, ReasonerGrammarObject)
        if answering:
            for token_id in self.think_end:
                grammar.accept_token(token_id)
        return grammar

    def _allowed_rows(self, grammars):
        """Allowed ids per row, built as SamplingBatchInfo does for a batch."""
        first = next(grammar for grammar in grammars if grammar)
        mask = first.allocate_vocab_mask(
            vocab_size=self.model_vocab, batch_size=len(grammars), device="cpu"
        )
        entries = [
            GrammarRow(row=row, grammar=grammar)
            for row, grammar in enumerate(grammars)
            if grammar and not grammar.finished and not grammar.is_terminated()
        ]
        first.fill_vocab_mask_batched(entries, mask)
        mask = first.move_vocab_mask(mask, "cpu")
        logits = torch.zeros(len(grammars), self.model_vocab)
        GrammarMask(first, mask).apply(logits)
        return [
            set(torch.isfinite(logits[row]).nonzero().flatten().tolist())
            for row in range(len(grammars))
        ]

    def test_grammars_build_through_the_reasoner(self):
        reasoner = self._wrap()
        for key, _ in self.keys:
            with self.subTest(key=key[0]):
                future, cache_hit = reasoner.get_cached_or_future_value(key, True)
                self.assertFalse(cache_hit)
                grammar = future.result(timeout=60)
                self.assertIsInstance(grammar, ReasonerGrammarObject)
                self.assertIsNotNone(grammar.grammar)

    def test_mask_allows_everything_while_thinking_then_the_grammar(self):
        reasoner = self._wrap()
        for key, first_texts in self.keys:
            with self.subTest(key=key[0]):
                grammar = self._build(reasoner, key)
                self.assertEqual(self._allowed_rows([grammar]), [self.everything])
                for token_id in self.think_end:
                    grammar.accept_token(token_id)
                self.assertEqual(
                    self._allowed_rows([grammar]), [self._expected(first_texts)]
                )

    def test_strict_thinking_without_filter_uses_backend_mask_hooks(self):
        # DeepSeek-R1 has no excluded tokens, so strict thinking builds a
        # wrapper without an inner grammar for every request, also for one
        # without a grammar. Only the backend can allocate its mask.
        reasoner = self._wrap(enable_strict_thinking=True)
        self.assertFalse(reasoner.enable_token_filter)
        grammar = reasoner.init_strict_reasoning_grammar(reasoning=True)
        self.assertIsInstance(grammar, ReasonerGrammarObject)
        self.assertIsNone(grammar.grammar)
        self.assertEqual(self._allowed_rows([grammar]), [self.everything])

    def test_mixed_batch_over_a_padded_vocab_in_either_order(self):
        # The first grammar of a batch allocates the mask for all rows. A
        # strict-only wrapper then allocates with the backend hook and fills no
        # tokens, while the structured rows fill the same mask with their own
        # grammar. Both must agree on the mask format, also when the model vocab
        # is larger than the tokenizer.
        reasoner = self._wrap(enable_strict_thinking=True)
        key, first_texts = self.keys[0]
        strict_only = reasoner.init_strict_reasoning_grammar(reasoning=True)
        strict_only_answering = reasoner.init_strict_reasoning_grammar(reasoning=True)
        for token_id in self.think_end:
            strict_only_answering.accept_token(token_id)
        thinking = self._build(reasoner, key)
        answering = self._build(reasoner, key, answering=True)
        orders = {
            "strict-only first": [
                strict_only,
                answering,
                thinking,
                strict_only_answering,
                None,
            ],
            "answering first": [
                answering,
                strict_only,
                thinking,
                strict_only_answering,
                None,
            ],
        }
        for name, grammars in orders.items():
            with self.subTest(order=name):
                allowed = self._allowed_rows(grammars)
                for grammar, row in zip(grammars, allowed):
                    expected = (
                        self._expected(first_texts)
                        if grammar is answering
                        else self.everything
                    )
                    self.assertEqual(row, expected)


class TestReasonerOverGuidanceBackend(_ReasonerOverBuiltinBackend, CustomTestCase):
    keys = (
        (("json", _SCHEMA), ["{"]),
        (("regex", "[0-9]+"), list("0123456789")),
        (("ebnf", 'root ::= "yes" | "no"'), ["y", "n"]),
    )

    def _make_backend(self):
        return GuidanceBackend(
            self.tokenizer,
            n_vocab=self.model_vocab,
            eos_token_ids={self.tokenizer.eos_token_id},
        )


class TestReasonerOverOutlinesBackend(_ReasonerOverBuiltinBackend, CustomTestCase):
    # outlines has no EBNF support.
    keys = (
        (("json", _SCHEMA), ["{"]),
        (("regex", "[0-9]+"), list("0123456789")),
    )

    def _make_backend(self):
        return OutlinesGrammarBackend(self.tokenizer, whitespace_pattern=None)


class TestBuiltinBackendsDefineMaskHooks(CustomTestCase):
    """create_grammar_backend wraps every built-in backend in a
    ReasonerGrammarBackend when a reasoning parser is set, and that wrapper reads
    the three mask hooks from the backend (bug: llguidance and outlines lacked
    them). A new built-in backend without them fails every structured request."""

    BUILTIN_BACKENDS = (
        XGrammarGrammarBackend,
        GuidanceBackend,
        OutlinesGrammarBackend,
    )
    BUILTIN_NAMES = {"xgrammar", "llguidance", "outlines"}

    def test_this_list_covers_every_name_create_grammar_backend_handles(self):
        source = inspect.getsource(create_grammar_backend)
        handled = set(re.findall(r'name == "(\w+)"', source)) - {"none"}
        self.assertEqual(handled, self.BUILTIN_NAMES)

    def test_every_builtin_backend_defines_the_mask_hooks(self):
        for backend_cls in self.BUILTIN_BACKENDS:
            for hook in ("allocate_vocab_mask", "move_vocab_mask", "apply_vocab_mask"):
                with self.subTest(backend=backend_cls.__name__, hook=hook):
                    self.assertIn(hook, vars(backend_cls))


if __name__ == "__main__":
    unittest.main()
