"""The grammar sampling plugin: masks each grammar request's logits row to the
tokens its grammar allows.

Its step state is `SamplingBatchInfo.grammars`, set at each launch from the
step's requests. The mask is built when the grammar slot runs, from the grammar
objects' state at that point. The scheduler owns everything around it:

- Host-side matching: it advances each grammar over the sampled tokens.
- With overlap scheduling, `sampling_info.grammars is not None` makes
  `TpModelWorker.forward_batch_generation` defer the sample until the previous
  step's tokens were accepted, so the mask sees the advanced grammars.
- `Sampler` syncs the sampled tokens across TP ranks when `grammars` is set.

Speculative verify builds its own mask over the draft tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarObject,
    GrammarMask,
    GrammarRow,
)

if TYPE_CHECKING:
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


def build_grammar_mask(
    grammars: Optional[List[Optional[BaseGrammarObject]]],
    *,
    vocab_size: int,
    device: str,
) -> Optional[GrammarMask]:
    """The vocab mask for the grammars' current state, one row per request."""
    if not grammars:
        return None

    first_grammar = next(grammar for grammar in grammars if grammar)

    vocab_mask = first_grammar.allocate_vocab_mask(
        vocab_size=vocab_size,
        batch_size=len(grammars),
        device=device,
    )

    # Rows omitted here (finished / terminated / non-grammar requests) retain
    # the freshly allocated buffer's unconstrained value.
    entries = [
        GrammarRow(row=row, grammar=grammar)
        for row, grammar in enumerate(grammars)
        if grammar and not grammar.finished and not grammar.is_terminated()
    ]
    first_grammar.fill_vocab_mask_batched(entries, vocab_mask)

    vocab_mask = first_grammar.move_vocab_mask(vocab_mask, device)
    return GrammarMask(first_grammar, vocab_mask)


def apply_grammar_mask(logits: torch.Tensor, sampling_info: SamplingBatchInfo) -> None:
    """Build the step's grammar mask and apply it to the logits, in place."""
    grammar_mask = build_grammar_mask(
        sampling_info.grammars,
        vocab_size=sampling_info.vocab_size,
        device=sampling_info.device,
    )
    if grammar_mask is not None:
        grammar_mask.apply(logits)
