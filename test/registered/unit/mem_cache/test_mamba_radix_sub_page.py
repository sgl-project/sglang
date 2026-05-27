"""
Unit tests for fix(mamba_radix_cache): graceful handling of sub-page and
mismatched-length inserts in MambaRadixCache.cache_finished_req.

Before this fix, both cases hit a hard assert that fired SIGQUIT and killed
the scheduler with all in-flight requests. After this fix:

  Case 1 — sub-page prefix (page_aligned_len == 0): silent early return.
  Case 2 — mamba/KV commit-length drift: logger.warning + graceful free.
"""
import logging
import unittest
from unittest.mock import MagicMock, call, patch

import torch

from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache, TreeNode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


def _make_cache(page_size: int = 4) -> MambaRadixCache:
    """Return a MambaRadixCache with only the attributes needed by cache_finished_req."""
    cache = object.__new__(MambaRadixCache)
    cache.page_size = page_size
    cache.disable = False
    cache.enable_mamba_extra_buffer = False
    cache.token_to_kv_pool_allocator = MagicMock()
    cache.req_to_token_pool = MagicMock()
    return cache


def _make_req(n_tokens: int, cache_protected_len: int = 0) -> MagicMock:
    """Return a minimal Req mock for cache_finished_req."""
    req = MagicMock()
    req.pop_committed_kv_cache.return_value = n_tokens
    req.origin_input_ids = list(range(n_tokens))
    req.output_ids = []
    req.req_pool_idx = 0
    req.cache_protected_len = cache_protected_len
    req.mamba_last_track_seqlen = None
    req.last_node = TreeNode()
    return req


class TestSubPageSkip(unittest.TestCase):
    """Case 1: prompt shorter than page_size — page_aligned_len == 0."""

    def _call(self, cache, req):
        with patch.object(cache, "dec_lock_ref"):
            cache.cache_finished_req(req)

    def test_no_crash_on_sub_page_prompt(self):
        """cache_finished_req must not raise when kv_committed_len < page_size."""
        cache = _make_cache(page_size=4)
        req = _make_req(n_tokens=2)  # 2 < 4 → page_aligned_len = 0
        kv = torch.zeros(2, dtype=torch.int32)
        cache.req_to_token_pool.req_to_token.__getitem__.return_value = kv
        self._call(cache, req)  # must not raise

    def test_kv_indices_freed_on_sub_page_prompt(self):
        """All KV indices must be freed when the prompt is sub-page."""
        cache = _make_cache(page_size=4)
        req = _make_req(n_tokens=2, cache_protected_len=0)
        kv = torch.zeros(2, dtype=torch.int32)
        cache.req_to_token_pool.req_to_token.__getitem__.return_value = kv

        with patch.object(cache, "dec_lock_ref"):
            cache.cache_finished_req(req)

        cache.token_to_kv_pool_allocator.free.assert_called_once()
        freed = cache.token_to_kv_pool_allocator.free.call_args[0][0]
        self.assertEqual(len(freed), 2)

    def test_mamba_cache_freed_on_sub_page_prompt(self):
        cache = _make_cache(page_size=4)
        req = _make_req(n_tokens=1)
        kv = torch.zeros(1, dtype=torch.int32)
        cache.req_to_token_pool.req_to_token.__getitem__.return_value = kv

        with patch.object(cache, "dec_lock_ref"):
            cache.cache_finished_req(req)

        cache.req_to_token_pool.free_mamba_cache.assert_called_once_with(req)

    def test_no_insert_into_radix_tree_on_sub_page_prompt(self):
        """The radix tree insert path must be bypassed — insert method not called."""
        cache = _make_cache(page_size=4)
        cache.insert = MagicMock()
        req = _make_req(n_tokens=3)
        kv = torch.zeros(3, dtype=torch.int32)
        cache.req_to_token_pool.req_to_token.__getitem__.return_value = kv

        with patch.object(cache, "dec_lock_ref"):
            cache.cache_finished_req(req)

        cache.insert.assert_not_called()

    def test_page_aligned_prompt_does_not_early_return(self):
        """A prompt that exactly fills one page must NOT early-return (normal path)."""
        cache = _make_cache(page_size=4)
        req = _make_req(n_tokens=4)  # 4 == page_size → page_aligned_len = 4
        kv = torch.zeros(4, dtype=torch.int32)
        cache.req_to_token_pool.req_to_token.__getitem__.return_value = kv

        # normal insert path needs more cache internals; just verify no early-return
        # by checking free is NOT called immediately (it may be called during insert)
        with patch.object(cache, "dec_lock_ref"):
            try:
                cache.cache_finished_req(req)
            except Exception:
                pass  # insert path may fail without full tree — that is expected

        # free may be called but not with the full kv tensor from the early-return path
        for c in cache.token_to_kv_pool_allocator.free.call_args_list:
            freed = c[0][0]
            # early-return frees ALL n_tokens; a normal path only frees un-cached tail
            self.assertNotEqual(
                len(freed),
                4,
                "early-return should not fire for a page-aligned prompt",
            )


class TestDriftWarning(unittest.TestCase):
    """Case 2: commit-length drift (cache_len != page_aligned_len > 0)."""

    def test_warning_logged_on_drift(self):
        """A logger.warning must be emitted when cache_len != page_aligned_len > 0."""
        cache = _make_cache(page_size=4)
        # 5 tokens → page_aligned_len = 4, cache_len = 5 (drift)
        req = _make_req(n_tokens=5)
        kv = torch.zeros(5, dtype=torch.int32)
        cache.req_to_token_pool.req_to_token.__getitem__.return_value = kv

        with patch.object(cache, "dec_lock_ref"), self.assertLogs(
            "sglang.srt.mem_cache.mamba_radix_cache", level=logging.WARNING
        ) as log_ctx:
            cache.cache_finished_req(req)

        self.assertTrue(
            any("mamba_radix_cache" in m for m in log_ctx.output),
            "Expected a warning from mamba_radix_cache logger",
        )

    def test_no_raise_on_drift(self):
        """Scheduler must survive a drift event — no exception."""
        cache = _make_cache(page_size=4)
        req = _make_req(n_tokens=5)
        kv = torch.zeros(5, dtype=torch.int32)
        cache.req_to_token_pool.req_to_token.__getitem__.return_value = kv

        with patch.object(cache, "dec_lock_ref"), self.assertLogs(
            "sglang.srt.mem_cache.mamba_radix_cache", level=logging.WARNING
        ):
            cache.cache_finished_req(req)  # must not raise

    def test_kv_freed_on_drift(self):
        cache = _make_cache(page_size=4)
        req = _make_req(n_tokens=5, cache_protected_len=0)
        kv = torch.zeros(5, dtype=torch.int32)
        cache.req_to_token_pool.req_to_token.__getitem__.return_value = kv

        with patch.object(cache, "dec_lock_ref"), self.assertLogs(
            "sglang.srt.mem_cache.mamba_radix_cache", level=logging.WARNING
        ):
            cache.cache_finished_req(req)

        cache.token_to_kv_pool_allocator.free.assert_called_once()
        cache.req_to_token_pool.free_mamba_cache.assert_called_once_with(req)

    def test_sub_page_does_not_log_drift_warning(self):
        """page_aligned_len == 0 must NOT log a drift warning — it is a silent skip."""
        cache = _make_cache(page_size=4)
        req = _make_req(n_tokens=2)  # sub-page, not drift
        kv = torch.zeros(2, dtype=torch.int32)
        cache.req_to_token_pool.req_to_token.__getitem__.return_value = kv

        with patch.object(cache, "dec_lock_ref"):
            with self.assertNoLogs(
                "sglang.srt.mem_cache.mamba_radix_cache", level=logging.WARNING
            ):
                cache.cache_finished_req(req)


if __name__ == "__main__":
    unittest.main()
