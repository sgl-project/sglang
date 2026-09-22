import sys
from array import array
from threading import Lock

import pytest
import torch
import xgrammar as xgr

from sglang.srt.constrained.base_grammar_backend import GrammarRow
from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarBackend
from sglang.srt.constrained.xgrammar_backend import (
    XGrammarGrammarBackend,
    XGrammarThinkingGrammar,
)
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_utils import traverse_tree
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.fixture(scope="module")
def grammars():
    vocab = [bytes([i]) for i in range(256)] + [
        b"</think>",
        b"</think>Hello",
        b"\xc3\xa9",
        b"<eos>",
    ]
    backend = object.__new__(XGrammarGrammarBackend)
    backend.vocab_size = len(vocab)
    backend.override_stop_tokens = [259]
    backend.batch_matcher = xgr.BatchGrammarMatcher(max_threads=2)
    backend.tokenizer_info = xgr.TokenizerInfo(vocab, stop_token_ids=[259])
    backend.grammar_compiler = xgr.GrammarCompiler(
        backend.tokenizer_info, max_threads=1
    )
    backend._thinking_metadata = None
    backend._thinking_metadata_lock = Lock()
    key = FunctionCallParser([], "glm47").get_structure_constraint(
        "none", thinking_mode=True
    )
    original = backend.dispatch_ebnf(key[1])
    reasoner = object.__new__(ReasonerGrammarBackend)
    reasoner.grammar_backend = backend
    optimized = reasoner._init_value_dispatch(key, reasoning=False)
    assert isinstance(optimized, XGrammarThinkingGrammar)
    return backend, original, optimized


def assert_same_mask(original, optimized):
    assert original.is_terminated() == optimized.is_terminated()
    if original.is_terminated():
        return
    a = original.allocate_vocab_mask(original.vocab_size, 1, "cpu")
    b = a.clone()
    original.fill_vocab_mask(a, 0)
    optimized.fill_vocab_mask(b, 0)
    assert torch.equal(a, b)


@pytest.mark.parametrize(
    "tokens",
    [
        [ord("a"), 258, 256, ord("H"), 259],
        [256, 259],
        list(b"abc<tool_call"),
        list(b"abc<arg_key"),
        list(b"abc</think>Hello") + [259],
        [ord("a"), 257, 259],
        [ord("a"), 0xE2, 0x82, 0xAC, 256, ord("H"), 259],
    ],
)
def test_fast_thinking_matches_full_grammar_across_boundaries_and_rollback(
    grammars, tokens
):
    _, template, fast_template = grammars
    original, optimized = template.copy(), fast_template.copy()
    assert_same_mask(original, optimized)
    for token in tokens:
        original.accept_token(token)
        optimized.accept_token(token)
        assert_same_mask(original, optimized)
    for rollback in range(len(tokens) + 1):
        restored = fast_template.copy()
        for token in tokens:
            restored.accept_token(token)
        restored.rollback(rollback)
        expected = template.copy()
        for token in tokens[: len(tokens) - rollback]:
            expected.accept_token(token)
        assert_same_mask(expected, restored)


def test_fast_thinking_retokenization_crosses_both_boundaries(grammars):
    _, template, fast_template = grammars
    optimized = fast_template.copy()
    sequences = [
        [ord("a"), 256, ord("H")],
        list(b"a</think>Hello"),
        [ord("a"), 258, 256, ord("H")],
    ]
    previous = []
    for tokens in sequences:
        optimized.jump_and_retokenize(previous, tokens, -1)
        expected = template.copy()
        for token in tokens:
            expected.accept_token(token)
        assert_same_mask(expected, optimized)
        previous = tokens


def test_fast_thinking_batch_preserves_mixed_phase_rows(grammars):
    _, template, fast_template = grammars
    prefixes = [[ord("a")], [256, ord("H")], list(b"<tool_call")]
    originals, optimized = [], []
    for i in range(32):
        a, b = template.copy(), fast_template.copy()
        for token in prefixes[i % len(prefixes)]:
            a.accept_token(token)
            b.accept_token(token)
        originals.append(a)
        optimized.append(b)
    a = template.allocate_vocab_mask(template.vocab_size, 34, "cpu")
    b = a.clone()
    for i, grammar in enumerate(originals):
        grammar.fill_vocab_mask(a, i + 1)
    entries = [GrammarRow(i + 1, grammar) for i, grammar in enumerate(optimized)]
    for dispatcher in (optimized[0], originals[0]):
        dispatcher.fill_vocab_mask_batched(entries, b)
        assert torch.equal(a, b)


def test_noncanonical_full_assistant_grammar_keeps_original_matcher(grammars):
    backend, _, _ = grammars
    key = FunctionCallParser([], "glm47").get_structure_constraint(
        "none", thinking_mode=True
    )
    altered = key[1].replace(
        "thinking_block ::= thinking_block_content",
        'thinking_block ::= "x" thinking_block_content',
    )
    grammar = backend.dispatch_ebnf(altered)
    assert backend.wrap_full_assistant_grammar(grammar, altered) is grammar


def test_thinking_metadata_shared_without_sharing_generation_constraints(grammars):
    backend, template, _ = grammars
    originals, optimized = [], []
    for suffix in ("Hello", "Bye"):
        key = "\n".join(
            f'text_block ::= "{suffix}"' if line.startswith("text_block ::=") else line
            for line in template.key_string.splitlines()
        )
        original = backend.dispatch_ebnf(key)
        originals.append(original)
        optimized.append(backend.wrap_full_assistant_grammar(original.copy(), key))

    a, b = optimized
    assert a.thinking_mask is b.thinking_mask
    assert a.safe_tokens is b.safe_tokens
    for original, grammar in zip(originals, optimized):
        assert_same_mask(original, grammar.copy())
        original.accept_token(256)
        grammar.accept_token(256)
        assert_same_mask(original, grammar)
    a.accept_token(ord("H"))
    b.accept_token(ord("B"))
    for original, grammar, token in zip(originals, optimized, (ord("H"), ord("B"))):
        original.accept_token(token)
        assert_same_mask(original, grammar)


@pytest.mark.parametrize(
    "prefix, paths, phase",
    [
        ([], [[ord("<"), ord("x")], [ord("a")]], "thinking"),
        ([], [[256, ord("H"), 259], [257, 259], [ord("a")]], "thinking"),
        ([], [[0xE2, ord("A")], [0xE2, 0x82, 0xAC], [ord("a")]], "thinking"),
        ([0xE2], [[ord("A")], [0x82, ord("A")], [0x82, 0xAC]], "fallback"),
    ],
    ids=["fallback-sibling", "atomic-boundary", "utf8-rejection", "committed-utf8"],
)
def test_speculative_tree_restores_thinking_state(grammars, prefix, paths, phase):
    _, template, fast_template = grammars
    original, optimized = template.copy(), fast_template.copy()
    for token in prefix:
        original.accept_token(token)
        optimized.accept_token(token)

    # The root represents the already committed token; traversal accepts its children.
    tokens, children = [ord("a")], [[]]
    for path in paths:
        parent = 0
        for token in path:
            node = len(tokens)
            tokens.append(token)
            children.append([])
            children[parent].append(node)
            parent = node
    next_child, next_sibling = [-1] * len(tokens), [-1] * len(tokens)
    for node, nodes in enumerate(children):
        if nodes:
            next_child[node] = nodes[0]
        for left, right in zip(nodes, nodes[1:]):
            next_sibling[left] = right
    tree = [torch.tensor(x) for x in (next_child, next_sibling, tokens)]
    a = original.allocate_vocab_mask(original.vocab_size, len(tokens), "cpu")
    b = a.clone()
    traverse_tree(*tree, original, a, vocab_size=original.vocab_size)
    traverse_tree(*tree, optimized, b, vocab_size=optimized.vocab_size)
    assert torch.equal(a, b)
    assert optimized.accepted_tokens == prefix
    assert optimized.phase == phase
    assert_same_mask(original, optimized)


def test_committed_fallback_survives_request_retraction(grammars):
    _, template, fast_template = grammars
    req = Req("thinking-retraction", "", array("q", [1]), SamplingParams())
    req.grammar = fast_template.copy()
    req.output_ids.extend(b"<x")
    original = template.copy()
    for token in req.output_ids:
        original.accept_token(token)
        req.grammar.accept_token(token)
    req.reset_for_retract()
    assert req.is_retracted
    assert list(req.output_ids) == list(b"<x")
    assert req.grammar.accepted_tokens == list(b"<x")
    assert req.grammar.phase == "fallback"
    assert_same_mask(original, req.grammar)
    req.grammar.accept_token(ord("a"))
    req.grammar.rollback(1)
    assert req.grammar.phase == "fallback"
    assert_same_mask(original, req.grammar)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
