from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True, slots=True, kw_only=True)
class CorpusControllerConfig:
    speculative_algorithm: str
    max_external_corpus_tokens: int


@dataclass(frozen=True, slots=True, kw_only=True)
class CorpusController:
    """add / remove / list external corpus endpoints (n-gram speculative decoding)."""

    add_external_corpus_communicator: Any
    remove_external_corpus_communicator: Any
    list_external_corpora_communicator: Any
    tokenizer: Optional[Any]
    config: CorpusControllerConfig
    auto_create_handle_loop: Callable[[], None]
