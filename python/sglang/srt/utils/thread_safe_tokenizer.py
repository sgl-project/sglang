# Ported from vLLM (vllm/tokenizers/hf.py:19-101, Apache-2.0).
#
# Phase 2.1 of SGLang multimodal offload work — see
# artifacts/PHASE2_BLOCKED.md for background. The shared HF fast tokenizer
# (a PyO3 RefCell under the hood) races between TokenizerManager's own
# apply_chat_template() coroutine and the offloaded mm_processor threads,
# producing `RuntimeError: Already borrowed`.
#
# This port mirrors vLLM verbatim:
#   * deepcopy `copies` tokenizers into a queue.Queue
#   * synthesize a subclass that wraps every public method to borrow / return
#   * in-place monkey-patch the tokenizer's __class__ so EVERY existing
#     reference (TM.tokenizer, processor.tokenizer, mm_processor._tokenizer,
#     mm_processor._processor.tokenizer — all aliases of one Rust object)
#     transparently gains the borrowing behavior. No reference swapping
#     anywhere in callers.
#
# Mechanical port. Deviations from upstream:
#   - logger: SGLang's stdlib `logging.getLogger(__name__)` instead of
#     `from vllm.logger import init_logger`.
#   - types: just `PreTrainedTokenizerFast` from transformers; SGLang has no
#     equivalent of vLLM's `TokenizerLike` protocol.
#   - no `get_cached_tokenizer` / `CachedHfTokenizer` (out of scope for the
#     race fix).

import contextlib
import copy
import logging
import queue
import threading
from typing import TypeVar

from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class ThreadSafeHFTokenizerMixin:
    """Marker mixin so we can detect (and skip) already-wrapped tokenizers."""

    pass


def maybe_make_thread_pool(tokenizer: _T, copies: int = 1) -> _T:
    """
    If ``tokenizer`` is a ``PreTrainedTokenizerFast``, modify the tokenizer
    in-place so its public interface is thread-safe by routing calls through
    a pool of deep-copied tokenizers.

    Returns the (now-wrapped, but same object identity) tokenizer.

    Notes:
        * Only the public interface is thread-safe. Direct access to the inner
          ``_tokenizer`` property or mutation methods like
          ``add_special_tokens`` / ``add_tokens`` is NOT routed through the
          pool and remains race-prone if called concurrently.
        * Adjacent method calls may execute on different deep copies of the
          tokenizer. As long as no caller mutates the tokenizer's state via
          a public-but-stateful method, this is transparent.
        * Python attributes set on the wrapper *before* this call (e.g.
          ``additional_stop_token_ids``) are preserved on the original object,
          because the ``__class__`` swap is in-place and Python attribute
          lookup hits the original ``__dict__`` first.
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
            # Allow pickling: rebuild from the un-wrapped tokenizer.
            return maybe_make_thread_pool, (og_tokenizer, copies)

    TokenizerPool.__name__ = f"TokenizerPool{og_tokenizer.__class__.__name__}"

    tokenizer.__class__ = TokenizerPool
    logger.debug(
        "Wrapped tokenizer %s with %d-copy thread pool",
        og_tokenizer.__class__.__name__,
        copies,
    )
    return tokenizer


def maybe_make_thread_safe_lock(tokenizer: _T) -> _T:
    """Thread-safe alternative to ``maybe_make_thread_pool`` with ~zero memory.

    Instead of deep-copying N tokenizers, serialize all public tokenizer calls
    through a single ``threading.Lock`` via the same in-place ``__class__`` swap.
    Prevents the PyO3 ``RefCell`` race (``RuntimeError: Already borrowed``) by
    ensuring only one thread is inside the Rust tokenizer at a time.

    Trade-off vs the pool: tokenizer calls cannot run in parallel. This is fine
    when tokenization is NOT the bottleneck (e.g. URL workloads bound by image
    fetch / GPU), where it matches the pool's throughput at no RAM cost. Avoid
    if concurrent CPU-bound tokenization of huge texts is the hot path.
    """
    if not isinstance(tokenizer, PreTrainedTokenizerFast) or isinstance(
        tokenizer, ThreadSafeHFTokenizerMixin
    ):
        return tokenizer

    # RLock (reentrant) — NOT a plain Lock: apply_chat_template(tokenize=True)
    # internally calls self.encode/__call__ which re-enter on the SAME thread; a
    # non-reentrant lock would self-deadlock. RLock still blocks OTHER threads, so
    # cross-thread safety (the actual PyO3 race) is preserved.
    _lock = threading.RLock()
    # Shallow copy (shares the Rust backend, ~0 mem) kept only for pickling.
    og_tokenizer = copy.copy(tokenizer)

    class TokenizerLock(tokenizer.__class__, ThreadSafeHFTokenizerMixin):  # type: ignore[misc, valid-type]
        def __reduce__(self):
            return maybe_make_thread_safe_lock, (og_tokenizer,)

        def apply_chat_template(self, *args, **kwargs):
            with _lock:
                return super().apply_chat_template(*args, **kwargs)

        def batch_decode(self, *args, **kwargs):
            with _lock:
                return super().batch_decode(*args, **kwargs)

        def batch_encode(self, *args, **kwargs):
            with _lock:
                return super().batch_encode(*args, **kwargs)

        def convert_tokens_to_ids(self, *args, **kwargs):
            with _lock:
                return super().convert_tokens_to_ids(*args, **kwargs)

        def convert_ids_to_tokens(self, *args, **kwargs):
            with _lock:
                return super().convert_ids_to_tokens(*args, **kwargs)

        def convert_tokens_to_string(self, *args, **kwargs):
            with _lock:
                return super().convert_tokens_to_string(*args, **kwargs)

        def decode(self, *args, **kwargs):
            with _lock:
                return super().decode(*args, **kwargs)

        def encode(self, *args, **kwargs):
            with _lock:
                return super().encode(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            with _lock:
                return super().__call__(*args, **kwargs)

    TokenizerLock.__name__ = f"TokenizerLock{tokenizer.__class__.__name__}"
    tokenizer.__class__ = TokenizerLock
    logger.debug("Wrapped tokenizer with a shared lock (zero-copy thread safety)")
    return tokenizer
