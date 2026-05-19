"""Deterministic prompt generator from (seed, length).

Tests need synthetic prompts whose exact token sequence is reproducible
under a given (seed, length) but does not collide with reserved tokens
(BOS / EOS / PAD / etc. — which the sampler-override path or the
oracle's EOS picker may treat specially).

The generator produces token ids in a "safe" sub-range of a generic
vocab; the caller is expected to use this for pseudo-mode tests where
the load format is ``dummy`` (no real tokenizer / no real model
weights) so token semantics are irrelevant.
"""

from __future__ import annotations

import hashlib
import logging
from typing import List

logger = logging.getLogger(__name__)

# Safe range that avoids the conventional special-token cluster at the
# low end of the vocab (BOS=0, EOS=1, PAD=2, UNK=3 on most tokenizers)
# and stays well below typical small-model vocab sizes (Qwen3 = 151_936
# but we cap at 16k so any tokenizer-free test still resolves the ids).
_VOCAB_LOW: int = 1024
_VOCAB_HIGH: int = 16_384


def fake_prompt(length: int, *, seed: int = 0xBEEF) -> List[int]:
    """Return ``length`` deterministic token ids derived from ``seed``.

    The output is stable across processes / Python versions because the
    backing PRNG is SHA-256 over a counter, not Python's hash-randomised
    ``random.Random``. Same (seed, length) always yields the same list.
    """
    if length < 1:
        raise ValueError(f"fake_prompt: length must be >= 1, got {length}")
    out: List[int] = []
    span = _VOCAB_HIGH - _VOCAB_LOW
    for i in range(length):
        digest = hashlib.sha256(f"{seed}:{i}".encode("utf-8")).digest()
        # First 8 bytes of digest, little-endian unsigned int.
        word = int.from_bytes(digest[:8], "little", signed=False)
        out.append(_VOCAB_LOW + (word % span))
    return out
