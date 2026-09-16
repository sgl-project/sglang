import sys

import pytest
import torch
import xgrammar as xgr

from sglang.srt.constrained.base_grammar_backend import GrammarRow
from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarObject
from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("prefix", ["", "a", "a12"])
@pytest.mark.parametrize("extra_rows", [0, 16])
def test_batched_mask_preserves_sparse_rows_and_reasoner_states(prefix, extra_rows):
    backend = object.__new__(XGrammarGrammarBackend)
    backend.vocab_size = 257
    backend.override_stop_tokens = [256]
    backend.batch_matcher = xgr.BatchGrammarMatcher(max_threads=2)
    backend.grammar_compiler = xgr.GrammarCompiler(
        xgr.TokenizerInfo(
            [bytes([i]) for i in range(256)] + [b"<eos>"],
            vocab_type=xgr.VocabType.RAW,
            stop_token_ids=[256],
        ),
        max_threads=1,
    )
    grammar = backend.dispatch_regex(r"[ab][0-9]+")
    copied = grammar.copy()
    for token in prefix.encode():
        copied.accept_token(token)
    copied.accept_token(ord("a") if not prefix else ord("3"))
    copied.rollback(1)

    thinking = ReasonerGrammarObject(
        grammar.copy(),
        think_end_ids=[255],
        think_excluded_token_ids=[3, 4],
        enable_token_filter=True,
        token_filter_fn=backend.set_token_filter,
    )
    thinking.maybe_init_reasoning(True)
    generation = thinking.copy()
    generation.accept_token(255)
    generation.accept_token(ord("b"))
    entries = [
        GrammarRow(5, copied),
        GrammarRow(1, thinking),
        GrammarRow(3, generation),
        GrammarRow(0, backend.dispatch_regex(r"true|false")),
    ]
    entries.extend(GrammarRow(7 + i, grammar.copy()) for i in range(extra_rows))
    serial = torch.full(
        xgr.get_bitmask_shape(7 + extra_rows, 257), 123, dtype=torch.int32
    )
    batched = serial.clone()
    for entry in entries:
        entry.grammar.fill_vocab_mask(serial, entry.row)
    copied.fill_vocab_mask_batched(entries, batched)
    assert torch.equal(serial, batched)
    assert torch.all(batched[[2, 4, 6]] == 123)
    copied.fill_vocab_mask_batched([], batched)
    assert torch.equal(serial, batched)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
