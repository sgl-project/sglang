# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The base class of a backend for grammar-guided constrained decoding."""

import dataclasses
import functools
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch

from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.runtime_context import get_resources
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

GRAMMAR_BACKEND_REGISTRY = {}


@dataclass
class GrammarStats:
    compilation_time: Optional[float] = None
    schema_count: Optional[int] = None
    ebnf_size: Optional[int] = None
    is_cache_hit: bool = False
    is_grammar_aborted: bool = False
    tree_traversal_time: List[float] = field(default_factory=list)
    dispatch_type: Optional[str] = None
    num_timeout: int = 0
    cache_source: str = "compile"
    cache_resolution_time: Optional[float] = None
    cache_lock_wait_time: Optional[float] = None
    cache_phase_seconds: Dict[str, float] = field(default_factory=dict)


class GrammarRow(NamedTuple):
    """Grammar and destination row for a batched vocab-mask fill."""

    row: int
    grammar: "BaseGrammarObject"


class BaseGrammarObject:

    def __init__(self):
        self._finished = False
        self.grammar_stats = None
        self.current_token = None

    def maybe_init_reasoning(self, reasoning: bool):
        pass

    def accept_token(self, token: int) -> None:
        """
        Accept a token in the grammar.
        """
        raise NotImplementedError()

    def rollback(self, k: int):
        raise NotImplementedError()

    def is_terminated(self):
        return False

    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:
        raise NotImplementedError()

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        raise NotImplementedError()

    @staticmethod
    def fill_vocab_mask_batched(
        entries: List[GrammarRow], vocab_mask: torch.Tensor
    ) -> None:
        """Fill listed rows, leaving unlisted rows untouched."""
        for entry in entries:
            entry.grammar.fill_vocab_mask(vocab_mask, entry.row)

    @staticmethod
    def reset_vocab_mask(vocab_mask: torch.Tensor) -> None:
        """Restore a reusable mask to the backend's unconstrained state."""
        raise NotImplementedError()

    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device) -> torch.Tensor:
        raise NotImplementedError()

    @staticmethod
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        raise NotImplementedError()

    def copy(self) -> "BaseGrammarObject":
        return self

    def cache_memory_bytes(self) -> int:
        return 0

    @property
    def finished(self):
        return self._finished

    @finished.setter
    def finished(self, finished):
        self._finished = finished

    def try_jump_forward(self, tokenizer) -> Optional[Tuple[List[int], str]]:
        """
        Try to jump forward in the grammar.

        Returns:
            A jump forward helper which may be used in `jump_forward_str_state`.
            None if the jump forward is not possible.
        """
        raise NotImplementedError()

    def jump_forward_str_state(self, helper: Tuple[List[int], str]) -> Tuple[str, int]:
        """
        Jump forward for the grammar.

        Returns:
            A tuple of the jump forward string and the next state of the grammar
            (which can be used in `jump_and_retokenize` if needed).
        """
        raise NotImplementedError()

    def jump_and_retokenize(
        self, old_output_ids: List[int], new_output_ids: List[int], next_state: int
    ) -> None:
        """
        Jump forward occurs, and update the grammar state if needed.
        """
        raise NotImplementedError()


class GrammarMask(NamedTuple):
    """A filled vocab_mask plus the backend that applies it.

    The grammar is any one of the batch's -- a handle, not per-request state.
    """

    grammar: BaseGrammarObject
    vocab_mask: torch.Tensor

    def apply(self, logits: torch.Tensor) -> None:
        self.grammar.apply_vocab_mask(logits=logits, vocab_mask=self.vocab_mask)


class InvalidGrammarObject(BaseGrammarObject):
    """Represents a grammar that failed to compile, carrying the original error message."""

    def __init__(
        self,
        error_message: str = "Unknown grammar error",
        grammar_stats: Optional[GrammarStats] = None,
    ):
        super().__init__()
        self.error_message = error_message
        self.grammar_stats = grammar_stats

    def __repr__(self):
        return f"InvalidGrammarObject(error_message={self.error_message!r})"

    def copy(self) -> "InvalidGrammarObject":
        stats = (
            None
            if self.grammar_stats is None
            else dataclasses.replace(self.grammar_stats)
        )
        return InvalidGrammarObject(self.error_message, stats)


class BaseGrammarBackend:
    _enable_strict_thinking: bool = False

    def __init__(self):
        executor_threads = int(
            os.environ.get("SGLANG_GRAMMAR_COMPILATION_WORKERS", "8")
        )
        self.cache_limit_bytes = int(
            os.environ.get("SGLANG_GRAMMAR_CACHE_BYTES", str(10 * 1024**3))
        )
        self.cache_limit_entries = int(
            os.environ.get("SGLANG_GRAMMAR_CACHE_ENTRIES", "100000")
        )
        if (
            executor_threads <= 0
            or self.cache_limit_bytes <= 0
            or self.cache_limit_entries <= 0
        ):
            raise ValueError(
                "optimized grammar worker and cache limits must all be positive"
            )
        self.executor = ThreadPoolExecutor(
            max_workers=executor_threads,
            thread_name_prefix="sglang-grammar",
        )
        self.cache: OrderedDict[Tuple[str, str], BaseGrammarObject] = OrderedDict()
        self._cache_entry_bytes: Dict[Tuple[str, str], int] = {}
        self._cache_bytes = 0
        self._inflight: Dict[Tuple[str, str], Future[BaseGrammarObject]] = {}
        self._cache_lock = threading.Lock()

    @staticmethod
    @functools.lru_cache(maxsize=16384)
    def _normalize_cache_key(key: Tuple[str, str]) -> Tuple[str, str]:
        key_type, key_string = key
        if key_type not in {"json", "structural_tag"} or key_string == "$$ANY$$":
            return key
        try:
            parsed = json.loads(key_string)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return key
        return (
            key_type,
            json.dumps(
                parsed,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )

    def initialize_vocab_mask_buffer(
        self,
        name: str,
        vocab_size: int,
        max_rows: int,
        device,
    ) -> Optional[torch.Tensor]:
        """Initialize a reusable mask buffer when supported by the backend."""
        return None

    def _not_supported(self, key_type: str, key_string: str) -> BaseGrammarObject:
        logger.warning(f"Skip unsupported {key_type=}, {key_string=}")
        return InvalidGrammarObject()

    @property
    def enable_strict_thinking(self):
        return self._enable_strict_thinking

    @property
    def is_support_token_filter(self):
        return False

    def set_token_filter(
        self, vocab_mask, token_ids, batch_idx, is_allowed=True, reset_vocab_mask=True
    ):
        """Set or clear specific tokens in the vocab mask. No-op by default."""
        pass

    def init_strict_reasoning_grammar(self, reasoning: bool):
        """Create a grammar object for strict token filtering only. Returns None by default."""
        return None

    def dispatch_fallback(self, key_type: str, key_string: str) -> BaseGrammarObject:
        """
        This function should not be reached in any case.
        """
        raise ValueError(f"Invalid key_type: {key_type}={key_string}")

    def dispatch_json(self, key_string: str) -> BaseGrammarObject:
        return self._not_supported("json", key_string)

    def dispatch_regex(self, key_string: str) -> BaseGrammarObject:
        return self._not_supported("regex", key_string)

    def dispatch_ebnf(self, key_string: str) -> BaseGrammarObject:
        return self._not_supported("ebnf", key_string)

    def dispatch_structural_tag(self, key_string: str) -> BaseGrammarObject:
        return self._not_supported("structural_tag", key_string)

    def _init_value_dispatch(
        self, key: Tuple[str, str], require_reasoning: bool
    ) -> BaseGrammarObject:
        s = time.perf_counter()
        key_type, key_string = key
        if key_type == "json":
            grammar = self.dispatch_json(key_string)
        elif key_type == "regex":
            grammar = self.dispatch_regex(key_string)
        elif key_type == "ebnf":
            grammar = self.dispatch_ebnf(key_string)
        elif key_type == "structural_tag":
            grammar = self.dispatch_structural_tag(key_string)
        else:
            grammar = self.dispatch_fallback(key_type, key_string)

        if grammar is not None and grammar.grammar_stats is not None:
            grammar.grammar_stats.compilation_time = time.perf_counter() - s
        return grammar

    def get_cached_or_future_value(
        self, key: Tuple[str, str], require_reasoning: bool
    ) -> Tuple[BaseGrammarObject | Future[BaseGrammarObject], bool]:
        key = self._normalize_cache_key(key)
        with self._cache_lock:
            value = self.cache.get(key)
            if value is not None:
                self.cache.move_to_end(key)
                return (
                    self._copy_for_request(
                        value,
                        require_reasoning=require_reasoning,
                        cache_source="memory",
                        resolution_time=0.0,
                        preserve_compilation_time=False,
                    ),
                    True,
                )

            shared_future = self._inflight.get(key)
            is_owner = shared_future is None
            if shared_future is None:
                shared_future = self.executor.submit(
                    self._init_value_dispatch,
                    key,
                    False,
                )
                self._inflight[key] = shared_future

        # Future.add_done_callback() invokes the callback synchronously when
        # the future has already completed. Register outside _cache_lock so a
        # fast compile cannot deadlock in _finish_inflight while reacquiring it.
        if is_owner:
            shared_future.add_done_callback(
                lambda future, cache_key=key: self._finish_inflight(
                    cache_key,
                    future,
                )
            )

        request_future: Future[BaseGrammarObject] = Future()
        wait_started = time.perf_counter()

        def finish_request(future: Future[BaseGrammarObject]) -> None:
            if request_future.cancelled():
                return
            try:
                prototype = future.result()
                prototype_stats = prototype.grammar_stats
                prototype_source = (
                    prototype_stats.cache_source
                    if prototype_stats is not None
                    else "compile"
                )
                source = prototype_source if is_owner else "inflight"
                request_future.set_result(
                    self._copy_for_request(
                        prototype,
                        require_reasoning=require_reasoning,
                        cache_source=source,
                        resolution_time=time.perf_counter() - wait_started,
                        preserve_compilation_time=is_owner,
                    )
                )
            except BaseException as exc:
                request_future.set_exception(exc)

        shared_future.add_done_callback(finish_request)
        return request_future, False

    def _finish_inflight(
        self,
        key: Tuple[str, str],
        future: Future[BaseGrammarObject],
    ) -> None:
        try:
            prototype = future.result()
        except BaseException:
            with self._cache_lock:
                self._inflight.pop(key, None)
            return
        with self._cache_lock:
            self._inflight.pop(key, None)
            self._set_cache_locked(key, prototype)

    @staticmethod
    def _copy_for_request(
        prototype: BaseGrammarObject,
        *,
        require_reasoning: bool,
        cache_source: str,
        resolution_time: float,
        preserve_compilation_time: bool,
    ) -> BaseGrammarObject:
        prototype_stats = getattr(prototype, "grammar_stats", None)
        copied_value = prototype.copy()
        copied_value.maybe_init_reasoning(require_reasoning)
        copied_stats = getattr(copied_value, "grammar_stats", None)
        if copied_stats is not None:
            copied_stats.cache_source = cache_source
            copied_stats.is_cache_hit = cache_source in {
                "memory",
                "disk",
                "inflight",
            }
            copied_stats.cache_resolution_time = resolution_time
            if not preserve_compilation_time:
                copied_stats.compilation_time = None
                copied_stats.cache_phase_seconds = {}
            elif prototype_stats is not None:
                copied_stats.compilation_time = prototype_stats.compilation_time
                copied_stats.cache_lock_wait_time = prototype_stats.cache_lock_wait_time
                copied_stats.cache_phase_seconds = dict(
                    prototype_stats.cache_phase_seconds
                )
        return copied_value

    def set_cache(self, key: Tuple[str, str], value: BaseGrammarObject):
        key = self._normalize_cache_key(key)
        with self._cache_lock:
            self._set_cache_locked(key, value)

    def _set_cache_locked(
        self,
        key: Tuple[str, str],
        value: BaseGrammarObject,
    ) -> None:
        previous_size = self._cache_entry_bytes.pop(key, 0)
        if key in self.cache:
            self.cache.pop(key)
        size = max(0, int(value.cache_memory_bytes()))
        self.cache[key] = value
        self._cache_entry_bytes[key] = size
        self._cache_bytes += size - previous_size
        while (
            self._cache_bytes > self.cache_limit_bytes
            or len(self.cache) > self.cache_limit_entries
        ):
            evicted_key, _ = self.cache.popitem(last=False)
            self._cache_bytes -= self._cache_entry_bytes.pop(evicted_key, 0)

    def reset(self):
        with self._cache_lock:
            self.cache.clear()
            self._cache_entry_bytes.clear()
            self._cache_bytes = 0


def register_vocab_mask_buffer(
    name: str, vocab_mask: torch.Tensor, max_rows: int
) -> torch.Tensor:
    """Register a fixed-capacity mask buffer, preserving an equivalent one."""
    if max_rows <= 0:
        raise ValueError(f"Grammar mask max_rows must be positive, got {max_rows}")
    if vocab_mask.ndim == 0 or vocab_mask.shape[0] != max_rows:
        raise ValueError(
            f"Grammar mask buffer {name!r} must have {max_rows} rows, "
            f"got shape {tuple(vocab_mask.shape)}"
        )

    buffers = get_resources().buffers
    existing = buffers.get(name)
    if existing is not None:
        if (
            existing.shape != vocab_mask.shape
            or existing.dtype != vocab_mask.dtype
            or existing.device != vocab_mask.device
        ):
            raise RuntimeError(
                f"Grammar mask buffer {name!r} was already initialized as "
                f"{tuple(existing.shape)}, {existing.dtype}, {existing.device}; "
                f"new buffer is {tuple(vocab_mask.shape)}, {vocab_mask.dtype}, "
                f"{vocab_mask.device}"
            )
        return existing

    buffers[name] = vocab_mask
    return vocab_mask


def get_vocab_mask_buffer(name: str, rows: int) -> Optional[torch.Tensor]:
    """Return the active rows of a registered mask buffer, if available."""
    vocab_mask = get_resources().buffers.get(name)
    if vocab_mask is None:
        return None
    if rows > vocab_mask.shape[0]:
        raise ValueError(
            f"Grammar batch needs {rows} mask rows, exceeding initialized "
            f"capacity {vocab_mask.shape[0]} for {name!r}"
        )
    return vocab_mask[:rows]


def register_grammar_backend(name, init_func):
    GRAMMAR_BACKEND_REGISTRY[name] = init_func


def create_grammar_backend(
    server_args: ServerArgs,
    tokenizer,
    vocab_size: int,
    eos_token_ids: Optional[set] = None,
    think_end_id: Optional[int] = None,
) -> Optional[BaseGrammarBackend]:
    name = server_args.grammar_backend

    # Custom grammar backend has the highest priority
    if name in GRAMMAR_BACKEND_REGISTRY:
        return GRAMMAR_BACKEND_REGISTRY[name](
            server_args, tokenizer, vocab_size, eos_token_ids
        )

    # Default grammar backends
    if name == "outlines":
        from sglang.srt.constrained.outlines_backend import OutlinesGrammarBackend

        grammar_backend = OutlinesGrammarBackend(
            tokenizer,
            whitespace_pattern=server_args.constrained_json_whitespace_pattern,
        )
    elif name == "xgrammar":
        from sglang.srt.constrained.xgrammar_backend import (
            TokenizerNotSupportedError,
            XGrammarGrammarBackend,
        )

        # Convert Set[int] to List[int] if needed
        eos_list = list(eos_token_ids) if eos_token_ids else None

        try:
            grammar_backend = XGrammarGrammarBackend(
                tokenizer,
                vocab_size=vocab_size,
                model_eos_token_ids=eos_list,
                any_whitespace=not server_args.constrained_json_disable_any_whitespace,
            )
        except TokenizerNotSupportedError as e:
            if server_args.enable_strict_thinking:
                raise ValueError(
                    f"--enable-strict-thinking requires a grammar backend with "
                    f"token filtering support, but XGrammar failed to initialize: "
                    f"{e}. Cannot fall back to grammar_backend='none' with strict "
                    f"thinking enabled."
                ) from e
            logger.warning(
                f"Grammar backend disabled because tokenizer is not supported by XGrammar: {e}. "
                "Falling back to grammar_backend='none'. "
                "Structured outputs (JSON schema, regex, EBNF) will not be available."
            )
            server_args.override("grammar.import_fallback", grammar_backend="none")
            return None
    elif name == "llguidance":
        from sglang.srt.constrained.llguidance_backend import GuidanceBackend

        grammar_backend = GuidanceBackend(
            tokenizer=tokenizer,
            any_whitespace=not server_args.constrained_json_disable_any_whitespace,
            whitespace_pattern=server_args.constrained_json_whitespace_pattern,
            n_vocab=vocab_size,
            eos_token_ids=eos_token_ids,
        )
    elif name == "none":
        if server_args.enable_strict_thinking:
            raise ValueError(
                "--enable-strict-thinking requires a grammar backend that supports "
                "token filtering, but grammar_backend='none' was specified. Use "
                "--grammar-backend xgrammar or another backend that supports token "
                "filtering."
            )
        return None
    else:
        raise ValueError(f"Invalid grammar backend: {name}")

    if server_args.reasoning_parser and think_end_id is not None:
        from sglang.srt.constrained.reasoner_grammar_backend import (
            ReasonerGrammarBackend,
        )

        reasoning_parser = ReasoningParser(
            model_type=server_args.reasoning_parser,
            stream_reasoning=False,
            tokenizer=tokenizer,
        )

        grammar_backend = ReasonerGrammarBackend(
            grammar_backend,
            reasoning_parser,
            tokenizer,
            enable_strict_thinking=server_args.enable_strict_thinking,
        )

    return grammar_backend
