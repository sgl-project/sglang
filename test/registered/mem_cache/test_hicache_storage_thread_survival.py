"""A storage backend that raises must not kill HiCache's storage threads.

``prefetch_thread_func``, ``prefetch_io_aux_func`` and ``backup_thread_func``
each wrap their body in ``except Empty: continue`` and nothing else, so any
other exception escapes the loop and ends the thread. They are daemon threads
with no supervisor, so nothing restarts them: one exception disables L2/L3 for
the life of the process.

Worse, it does not merely stop caching -- it leaks the resources those loops
were responsible for returning:

- ``prefetch_io_aux_func`` is the only caller of ``append_host_mem_release`` on
  the prefetch path. Without it the host pages a prefetch reserved are never
  freed and ``prefetch_tokens_occupied`` never falls, so once the total crosses
  ``prefetch_capacity_limit`` the rate limiter blocks every future prefetch --
  the leak ``prefetch_rate_limited`` already warns about.
- ``backup_thread_func`` is the only producer for ``ack_backup_queue``, which is
  what drives ``entry.release_host()`` in ``HiRadixCache``'s drain. Without it
  every backed-up node holds its host reference forever.

None of this raises anywhere visible. The engine keeps serving, every request
reports an ordinary cache miss, and the host pool quietly drains.

These tests drive the real loop functions against a backend that raises, so
they assert the *framework's* behaviour rather than a description of it. That is
the point: the conclusion "a backend must never raise" is what obliges every
backend -- including KVCR's -- to guard its own surface, and it is worth pinning
down rather than restating in a comment.

What turns this red: adding a general ``except Exception`` to those loops
upstream. That would be a fix, and these tests should then be replaced by ones
asserting the loop survives -- they are not a claim that the current behaviour
is desirable, only that it is what backends must be written against.

    python -m pytest test/registered/mem_cache/test_hicache_storage_thread_survival.py -v
"""

from __future__ import annotations

import threading
import unittest
from queue import Queue
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.cache_controller import HiCacheController

try:
    from sglang.test.ci.ci_register import register_cpu_ci

    register_cpu_ci(est_time=10, suite="base-a-test-cpu")
except Exception:  # pragma: no cover - registration is CI-only
    pass


class _RaisingBackend:
    """A storage backend whose every entry point raises after N calls.

    Modelled on a transient core fault rather than a permanently broken
    backend: the first ``ok_calls`` succeed, so a test can show the loop was
    alive and working right up to the exception.
    """

    def __init__(self, ok_calls: int = 0) -> None:
        self.remaining_ok = ok_calls
        self.calls = 0

    def _maybe_raise(self):
        self.calls += 1
        if self.remaining_ok > 0:
            self.remaining_ok -= 1
            return
        raise RuntimeError("backend fault")

    def batch_exists(self, keys, extra_info=None):
        self._maybe_raise()
        return len(keys)

    def batch_set_v1(self, keys, host_indices, extra_info=None):
        self._maybe_raise()
        return [True] * len(keys)

    def batch_get_v1(self, keys, host_indices, extra_info=None):
        self._maybe_raise()
        return [True] * len(keys)


def _controller(backend) -> HiCacheController:
    """The narrowest controller the three loop functions touch.

    Built with ``__new__``: the real constructor allocates device memory and
    starts several threads, none of which these loops use.
    """
    cc = HiCacheController.__new__(HiCacheController)
    cc.storage_backend = backend
    cc.storage_stop_event = threading.Event()
    cc.prefetch_queue = Queue()
    cc.prefetch_buffer = Queue()
    cc.backup_queue = Queue()
    cc.ack_backup_queue = Queue()
    cc.prefetch_hit_queue = Queue()
    cc.prefetch_revoke_queue = Queue()
    cc.host_mem_release_queue = Queue()
    cc.page_size = 1
    cc.prefetch_threshold = 1
    cc.backup_skip = False
    cc.has_draft = False
    cc.prefetch_sync_groups = []
    cc.mem_pool_host = MagicMock()
    cc.get_hash_str = lambda tokens, last_hash, page_size: [
        f"h{t}" for t in tokens
    ]
    cc.page_set_func = cc._page_set_zero_copy
    cc.page_get_func = cc._page_get_zero_copy
    return cc


def _operation(n_pages: int = 2):
    """A StorageOperation stand-in carrying what the loops read off it."""
    return MagicMock(
        request_id="req-1",
        token_ids=list(range(n_pages)),
        last_hash=None,
        prefix_keys=None,
        hash_value=[f"h{i}" for i in range(n_pages)],
        host_indices=torch.arange(n_pages),
        completed_tokens=0,
        storage_hit_count=0,
        is_terminated=lambda: False,
    )


def _run_until_idle(target, stop_event: threading.Event, settle_s: float = 0.5):
    """Run one loop function in a thread and report whether it survived.

    The loops block on a 1 s queue timeout, so ``settle_s`` has to be shorter
    than that: a thread that is merely *waiting* is still alive at 0.5 s, while
    one that took an exception has already exited. Returns the thread so the
    caller can assert on ``is_alive()``.
    """
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=settle_s)
    stop_event.set()
    return thread


class BackupThreadSurvivalTest(unittest.TestCase):
    def test_a_raising_backend_kills_the_backup_thread(self):
        """One exception out of ``batch_set_v1`` ends ``backup_thread_func``.

        Two operations are queued and the backend is allowed one good call, so
        the assertion is not "the thread never ran": it acked the first
        operation, then died on the second and left it unacked forever.
        """
        cc = _controller(_RaisingBackend(ok_calls=1))
        cc.backup_queue.put(_operation())
        cc.backup_queue.put(_operation())

        thread = _run_until_idle(cc.backup_thread_func, cc.storage_stop_event)

        self.assertFalse(thread.is_alive(), "backup thread survived the exception")
        self.assertEqual(cc.ack_backup_queue.qsize(), 1)
        self.assertEqual(cc.backup_queue.qsize(), 0, "the second op was consumed")

    def test_the_dead_backup_thread_never_acks_again(self):
        """The op that killed the thread is lost, and so is every later one.

        This is the leak, not just a missed cache write: ``HiRadixCache`` calls
        ``entry.release_host()`` only for operations it drains off
        ``ack_backup_queue``, so an op that never arrives holds its host pages
        for the life of the process.
        """
        cc = _controller(_RaisingBackend(ok_calls=0))
        cc.backup_queue.put(_operation())
        thread = _run_until_idle(cc.backup_thread_func, cc.storage_stop_event)
        self.assertFalse(thread.is_alive())

        # A later, healthy-looking operation. Nothing is consuming the queue.
        cc.backup_queue.put(_operation())
        self.assertEqual(cc.ack_backup_queue.qsize(), 0)
        self.assertEqual(cc.backup_queue.qsize(), 1)


class PrefetchThreadSurvivalTest(unittest.TestCase):
    def test_a_raising_batch_exists_kills_the_prefetch_thread(self):
        """``_storage_hit_query`` runs unguarded inside ``prefetch_thread_func``.

        The operation is neither put on the hit queue nor revoked, so the
        scheduler side is left with a prefetch it will never hear about again.
        """
        cc = _controller(_RaisingBackend(ok_calls=0))
        cc.prefetch_queue.put(_operation())

        thread = _run_until_idle(cc.prefetch_thread_func, cc.storage_stop_event)

        self.assertFalse(thread.is_alive(), "prefetch thread survived the exception")
        self.assertEqual(cc.prefetch_hit_queue.qsize(), 0)
        self.assertEqual(cc.prefetch_revoke_queue.qsize(), 0)

    def test_a_raising_get_kills_the_io_aux_thread_and_leaks_host_pages(self):
        """``prefetch_io_aux_func`` is the only caller that frees reserved pages.

        ``append_host_mem_release`` sits *after* ``_page_transfer`` in the loop
        body, so an exception from the backend skips it. The reservation those
        pages represent is what ``prefetch_capacity_limit`` counts against, and
        nothing else ever gives it back.
        """
        cc = _controller(_RaisingBackend(ok_calls=0))
        cc.prefetch_buffer.put(_operation())

        thread = _run_until_idle(cc.prefetch_io_aux_func, cc.storage_stop_event)

        self.assertFalse(thread.is_alive(), "io aux thread survived the exception")
        self.assertEqual(
            cc.host_mem_release_queue.qsize(),
            0,
            "pages reserved for the failed prefetch were never released",
        )


if __name__ == "__main__":
    unittest.main()
