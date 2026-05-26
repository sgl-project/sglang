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
# Ported from vLLM (vllm/tokenizers/hf.py, Apache-2.0).
#
# HuggingFace fast tokenizers (Rust/PyO3 backend) are not thread-safe for
# concurrent calls to __call__(padding=True) or apply_chat_template because
# they mutate a RefCell on the Rust side via set_truncation_and_padding.
# Concurrent access produces "RuntimeError: Already borrowed".
#
# When process_and_combine_mm_data is offloaded via run_in_executor the
# mm_processor threads call the tokenizer concurrently with the TM event loop,
# triggering the race. This module wraps the tokenizer with a deepcopy pool
# that serializes per-copy rather than per-process.

import contextlib
import copy
import logging
import queue
from typing import TypeVar

from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class ThreadSafeHFTokenizerMixin:
    """Marker mixin — lets us detect already-wrapped tokenizers and skip re-wrapping."""

    pass


def maybe_make_thread_pool(tokenizer: _T, copies: int = 1) -> _T:
    """
    If ``tokenizer`` is a ``PreTrainedTokenizerFast``, modify it in-place so
    its public interface is thread-safe by routing calls through a pool of
    deep-copied tokenizers.

    The in-place ``__class__`` swap preserves object identity, so all existing
    references (``TM.tokenizer``, ``processor.tokenizer``,
    ``mm_processor._tokenizer``, …) transparently gain the pool behavior
    without any reference-swapping at call sites.

    Call this AFTER all Python attributes have been set on the tokenizer
    (e.g. after ``attach_additional_stop_token_ids``).  Attributes present on
    the original object at swap time are visible in the pool copies because the
    deepcopy captures them; attributes set *after* this call only land on the
    original ``__dict__`` and are not propagated to pool copies.
    """
    if not isinstance(tokenizer, PreTrainedTokenizerFast) or isinstance(
        tokenizer, ThreadSafeHFTokenizerMixin
    ):
        return tokenizer

    og_tokenizer = copy.copy(tokenizer)

    tokenizer_pool: "queue.Queue[PreTrainedTokenizerFast]" = queue.Queue()
    for _ in range(copies):
        tokenizer_pool.put(copy.deepcopy(og_tokenizer))

    @contextlib.contextmanager
    def _borrow_from_pool():
        try:
            tok = tokenizer_pool.get_nowait()
            yield tok
        except queue.Empty:
            tok = copy.deepcopy(og_tokenizer)
            yield tok
        finally:
            tokenizer_pool.put(tok)

    class TokenizerPool(tokenizer.__class__, ThreadSafeHFTokenizerMixin):  # type: ignore[misc, valid-type]
        def apply_chat_template(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.apply_chat_template(*args, **kwargs)

        def batch_decode(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.batch_decode(*args, **kwargs)

        def batch_encode(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.batch_encode(*args, **kwargs)

        def convert_tokens_to_ids(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.convert_tokens_to_ids(*args, **kwargs)

        def convert_ids_to_tokens(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.convert_ids_to_tokens(*args, **kwargs)

        def convert_tokens_to_string(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.convert_tokens_to_string(*args, **kwargs)

        def decode(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.decode(*args, **kwargs)

        def encode(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.encode(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok(*args, **kwargs)

        def __reduce__(self):
            return maybe_make_thread_pool, (og_tokenizer, copies)

    TokenizerPool.__name__ = f"TokenizerPool[{og_tokenizer.__class__.__name__}]"

    tokenizer.__class__ = TokenizerPool
    logger.debug(
        "Wrapped tokenizer %s with %d-copy thread pool",
        og_tokenizer.__class__.__name__,
        copies,
    )
    return tokenizer
