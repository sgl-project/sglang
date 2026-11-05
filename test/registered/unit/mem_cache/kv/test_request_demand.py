"""A request's demand: the window a chunked request keeps beside its next
chunk, whatever its cached prefix already holds, the rows the scheduler's
window retirement keeps live, and the chunk grid admission searches."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ReqKvInfo, ScheduleBatch
from sglang.srt.mem_cache.kv.alloc.demand import (
    DemandPurpose,
    chunk_granule,
    demand_inputs,
    largest_fitting_chunk,
    request_demand,
    smallest_admissible_chunk,
)
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.static_kv_storage import kv_storage_of
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE = 128
WINDOW = 4096
CHUNK = 1024
PROMPT = 8192


def _window_demand(cached_tokens, *, continuation=False):
    """The window tokens admission prices for the rest of a PROMPT-token
    prompt behind a ``cached_tokens`` prefix, in chunks of CHUNK."""
    return (
        demand_inputs(
            DemandPurpose.ADMIT,
            PROMPT,
            1,
            windows={"swa": WINDOW},
            page_size=PAGE,
            chunk_limit=CHUNK,
            prefill_retire_lag=0,
            eviction_interval=1,
            cached_tokens=cached_tokens,
            continuation=continuation,
        )
        .demand()
        .window_tokens["swa"]
    )


class TestChunkedWindowDemand(CustomTestCase):
    """Before its last chunk a request holds the window ending at its last
    row and takes the next chunk beside it; of that window, its cached
    prefix already holds up to a window of rows."""

    def test_no_prefix_takes_the_whole_window_beside_the_chunk(self):
        self.assertEqual(_window_demand(0), WINDOW + CHUNK + PAGE)

    def test_a_short_prefix_takes_the_rest_of_the_window(self):
        # A one-page prefix holds one page of the window; the later chunks
        # still keep a whole window beside the next chunk.
        self.assertEqual(_window_demand(PAGE), WINDOW - PAGE + CHUNK + PAGE)

    def test_a_whole_window_prefix_leaves_the_chunk(self):
        # The window is held already: the per-pass estimate, which a later
        # chunk is priced at too, covers the chunk beside it.
        for cached in (WINDOW, 2 * WINDOW):
            with self.subTest(cached=cached):
                self.assertEqual(_window_demand(cached), WINDOW + PAGE)
                self.assertEqual(
                    _window_demand(cached),
                    _window_demand(cached, continuation=True),
                )

    def test_the_demand_covers_the_rows_a_later_chunk_takes(self):
        for cached in range(0, 2 * WINDOW + 1, PAGE):
            with self.subTest(cached=cached):
                held = min(WINDOW, cached)
                retained = min(WINDOW, cached + PROMPT - CHUNK)
                self.assertGreaterEqual(_window_demand(cached), retained - held + CHUNK)
                if cached:
                    self.assertLessEqual(
                        _window_demand(cached), _window_demand(cached - PAGE)
                    )


class _WindowCache(RadixCache):
    """One window whose rows the scheduler's own retirement frees, counting
    every row it frees."""

    def __init__(self, *, window, page_size, rows, shares_prefixes, checkpoints=False):
        self.disable = False
        self.checkpoints = checkpoints
        self.sliding_window_size = window
        self.page_size = page_size
        self.shares_prefixes = shares_prefixes
        self.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(rows).reshape(1, -1)
        )
        allocator = Mock(
            spec=[
                "free_swa_segment",
                "free_group_begin",
                "free_group_end",
                "page_size",
                "storage",
                "window_pool_names",
                "target_window_pool",
                "get_kvcache",
            ]
        )
        allocator.page_size = page_size
        allocator.storage = None
        allocator.window_pool_names = lambda: ("swa",)
        allocator.target_window_pool = lambda: "swa"
        allocator.get_kvcache = lambda: SimpleNamespace(request_window=None)
        self.token_to_kv_pool_allocator = allocator
        kv_storage_of(self.req_to_token_pool, allocator).bind_window_retention(
            window, None
        )

    def supports_swa(self):
        return True

    def supports_prefix_sharing(self):
        return self.shares_prefixes

    def swa_retain_floor(self, req):
        # As `UnifiedRadixCache.swa_retain_floor`: keep the window behind the
        # latest state checkpoint.
        checkpoint = req.kv.mamba_last_track_seqlen
        if not self.checkpoints or checkpoint is None:
            return None
        return checkpoint - self.sliding_window_size

    def freed(self) -> int:
        calls = self.token_to_kv_pool_allocator.free_swa_segment.call_args_list
        return sum(len(call.args[0]) for call in calls)


def _retired_peak(
    prompt,
    new_tokens,
    *,
    window,
    page_size,
    chunk,
    overlap,
    shares,
    interval,
    checkpoint_grid=0,
    pages=False,
):
    """The most window rows one request holds while the scheduler runs its
    chunks and decode steps (`ScheduleBatch.maybe_evict_swa` before each
    allocation), as live rows: allocated (whole pages with ``pages``) less
    retired. With ``checkpoint_grid``, a state checkpoint lands on each
    multiple of it the decode reaches, and retirement keeps the window behind
    the latest one."""
    cache = _WindowCache(
        window=window,
        page_size=page_size,
        rows=prompt + new_tokens + page_size + 1,
        shares_prefixes=shares,
        checkpoints=bool(checkpoint_grid),
    )

    def live(allocated):
        if pages:
            allocated = -(-allocated // page_size) * page_size
        return allocated - cache.freed()

    req = SimpleNamespace(
        seqlen=0,
        extend_batch_idx=0,
        decode_batch_idx=0,
        kv=ReqKvInfo(req_pool_idx=0),
    )
    batch = ScheduleBatch(
        reqs=[req],
        tree_cache=cache,
        req_to_token_pool=cache.req_to_token_pool,
        token_to_kv_pool_allocator=cache.token_to_kv_pool_allocator,
        forward_mode=ForwardMode.EXTEND,
        enable_overlap=overlap,
    )
    allocated = peak = 0
    with (
        envs.SGLANG_SWA_EVICTION_INTERVAL.override(interval),
        envs.SGLANG_OPT_RELEASE_PREFILL_SWA.override(False),
        patch(
            "sglang.srt.managers.schedule_batch.get_schedule",
            return_value=SimpleNamespace(chunked_prefill_size=chunk),
        ),
    ):
        for idx, start in enumerate(range(0, prompt, chunk)):
            req.extend_batch_idx = idx
            batch.prefix_lens = [start]
            batch.maybe_evict_swa()
            allocated = min(prompt, start + chunk)
            peak = max(peak, live(allocated))
        # The prefill emits the first token; each decode step writes one row.
        batch.forward_mode = ForwardMode.DECODE
        for step in range(new_tokens - 1):
            req.decode_batch_idx = step
            req.seqlen = allocated + 1
            if checkpoint_grid:
                req.kv.mamba_last_track_seqlen = (
                    allocated // checkpoint_grid * checkpoint_grid
                )
            batch.maybe_evict_swa()
            allocated += 1
            peak = max(peak, live(allocated))
    return peak


class TestWindowDemandUnderRetirement(CustomTestCase):
    """The window demand bounds the rows the scheduler keeps live: retiring
    a chunk behind the frontier the overlap scheduler keeps for the prefill
    still running, and decode retiring once an interval slid out."""

    def _check(self, prompt, new_tokens, *, page_size, chunk, lag, interval, **kw):
        peak = _retired_peak(
            prompt,
            new_tokens,
            window=WINDOW,
            page_size=page_size,
            chunk=chunk,
            interval=interval,
            **kw,
        )
        demand = request_demand(
            prompt,
            new_tokens,
            windows={"swa": WINDOW},
            page_size=page_size,
            chunk_size=chunk,
            new_window=True,
            prefill_retire_lag=lag,
            eviction_interval=interval,
        ).window_tokens["swa"]
        self.assertGreaterEqual(demand, peak)
        return peak

    def test_an_overlapped_chunked_prefill_keeps_a_chunk_more(self):
        # No prefix sharing: the overlap scheduler retires a chunk behind.
        peak = self._check(
            PROMPT,
            1,
            page_size=PAGE,
            chunk=CHUNK,
            lag=CHUNK,
            interval=128,
            overlap=True,
            shares=False,
        )
        self.assertEqual(peak, WINDOW + 2 * CHUNK)

    def test_decode_keeps_an_eviction_interval_more(self):
        for shares in (False, True):
            with self.subTest(shares=shares):
                peak = self._check(
                    1,
                    2 * WINDOW,
                    page_size=1,
                    chunk=CHUNK,
                    lag=0,
                    interval=128,
                    overlap=False,
                    shares=shares,
                )
                self.assertEqual(peak, WINDOW + 128)


class TestWindowDemandUnderCheckpointRetention(CustomTestCase):
    """A cache that keeps state checkpoints keeps the window behind the latest
    one: decode holds up to the checkpoint interval past the window, which
    the demand must cover, page rounding included."""

    def test_the_window_behind_the_latest_checkpoint(self):
        window, page, interval, grid = 4096, 128, 128, 512
        prompt, new_tokens = 1024, 8192
        peak = _retired_peak(
            prompt,
            new_tokens,
            window=window,
            page_size=page,
            chunk=1024,
            overlap=False,
            shares=True,
            interval=interval,
            checkpoint_grid=grid,
            pages=True,
        )

        def demand(checkpoint_interval):
            return request_demand(
                prompt,
                new_tokens,
                windows={"swa": window},
                page_size=page,
                prefill_retire_lag=0,
                eviction_interval=interval,
                checkpoint_interval=checkpoint_interval,
                chunk_size=1024,
                new_window=True,
            ).window_tokens["swa"]

        self.assertGreater(peak, window + interval)
        self.assertLess(demand(0), peak)  # what the interval alone gives
        self.assertGreaterEqual(demand(grid), peak)

    def test_without_checkpoints_the_eviction_interval_bounds(self):
        peak = _retired_peak(
            1024,
            8192,
            window=4096,
            page_size=128,
            chunk=1024,
            overlap=False,
            shares=True,
            interval=128,
            pages=True,
        )
        self.assertGreaterEqual(
            request_demand(
                1024,
                8192,
                windows={"swa": 4096},
                page_size=128,
                prefill_retire_lag=0,
                eviction_interval=128,
                chunk_size=1024,
                new_window=True,
            ).window_tokens["swa"],
            peak,
        )


class TestChunkGrid(CustomTestCase):
    def test_the_smallest_chunk_is_on_the_grid(self):
        self.assertIsNone(smallest_admissible_chunk(None, 4))
        self.assertEqual(smallest_admissible_chunk(16, 4), 4)
        self.assertEqual(smallest_admissible_chunk(64, 4, 6), 12)
        self.assertEqual(smallest_admissible_chunk(384, 128, 192), 384)

    def test_a_limit_below_the_grid_admits_no_chunk(self):
        # lcm(128, 192) = 384: no positive chunk of the grid fits in 256,
        # as the search over that grid finds none either.
        granule = chunk_granule(128, 192)
        self.assertEqual(smallest_admissible_chunk(256, 128, 192), 0)
        self.assertEqual(largest_fitting_chunk(lambda _: True, 256, granule), 0)
        self.assertEqual(smallest_admissible_chunk(8, 4, 64), 0)


if __name__ == "__main__":
    unittest.main()
