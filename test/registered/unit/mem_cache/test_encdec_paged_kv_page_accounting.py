"""Encoder-decoder KV rows stay page-consistent, so paged teardown frees each page once.

Regression tests for the encoder-decoder KV page double-free / pool over-count at
``page_size > 1``.

Background
----------
An encoder-decoder request's KV row is ``[encoder region | decoder region]``: the
encoder is prepended as ``multimodal_inputs.num_image_tokens`` slots, and
``prepare_encoder_info_extend`` then subtracts that length from ``seq_lens``, so
from prefill onwards ``seq_lens`` counts decoder tokens only and decoder position
``L`` lives at row position ``encoder_len + L``.

``alloc_for_decode`` must therefore do its paged arithmetic on the *row* length:
``last_loc`` is read out of the row, and ``alloc_decode_kernel`` decides whether a
step needs a fresh page from ``row_len % page_size == 1``. Deriving either from the
encoder-stripped ``seq_lens`` makes ``last_loc + 1`` resolve to ``row[L]`` -- the
slot already held by encoder position ``L`` -- and the row stops satisfying

    row[j] == page[j // page_size] * page_size + (j % page_size)

That invariant is what the paged free path relies on: ``free_segment`` samples one
page representative every ``page_size`` row entries (``free_index[::page_size]``)
and hands them to ``free_page_ids``, which by contract does not dedup. An aliased
row therefore names the same physical page twice, ``available_size()`` grows by a
page per request, and the idle pool-leak invariant eventually aborts the scheduler.

These tests drive the real ``PagedTokenToKVPoolAllocator`` and the real
``alloc_for_decode`` on CPU. The two triton allocation kernels are replaced by the
pure-torch reference already in the tree (``alloc_extend_naive``) and a faithful
port of ``alloc_decode_kernel``; everything else -- free lists, ``free_segment``,
``available_size()``, the ``req_to_token`` writes -- is production code. The
allocator's ``debug_mode`` is enabled, so the same asserts that catch this on
hardware (``free_segment``'s strided-vs-unique check and
``_debug_check_no_duplicate_pages``) are armed here.
"""

import types
import unittest
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.mem_cache.allocation import alloc_for_decode  # noqa: E402
from sglang.srt.mem_cache.allocator.paged import (  # noqa: E402
    PagedTokenToKVPoolAllocator,
    alloc_extend_naive,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool  # noqa: E402

register_cpu_ci(est_time=50, suite="base-a-test-cpu")

MAX_CONTEXT_LEN = 4096


# --------------------------------------------------------------------------- #
# triton-kernel stand-ins (CPU)
# --------------------------------------------------------------------------- #
class _KernelShim:
    """Mimics a triton kernel's ``kernel[grid](*args)`` call shape."""

    def __init__(self, fn):
        self._fn = fn

    def __getitem__(self, _grid):
        return self._fn


def _extend_shim(
    prefix_lens, seq_lens, last_loc, free_pages, out_indices, _bs_upper, page_size
):
    alloc_extend_naive(
        prefix_lens,
        seq_lens,
        last_loc,
        free_pages,
        out_indices,
        page_size,
        out_indices.device,
    )


def _decode_shim(seq_lens, last_loc, free_pages, out_indices, _bs_upper, page_size):
    """Faithful port of kernels/ops/memory/allocator.py::alloc_decode_kernel."""
    ps = page_size
    pre_lens = seq_lens - 1
    pages_after = torch.div(seq_lens + ps - 1, ps, rounding_mode="floor")
    pages_before = torch.div(pre_lens + ps - 1, ps, rounding_mode="floor")
    num_new_pages = pages_after - pages_before
    # The kernel's `sum(masked to j <= pid) - own` is an exclusive cumsum.
    new_page_start = torch.cumsum(num_new_pages, 0) - num_new_pages
    for i in range(seq_lens.numel()):
        if int(num_new_pages[i]) == 0:
            out_indices[i] = int(last_loc[i]) + 1
        else:
            out_indices[i] = int(free_pages[int(new_page_start[i])]) * ps


def _patch_kernels():
    return (
        patch(
            "sglang.srt.mem_cache.allocator.paged.alloc_extend_kernel",
            _KernelShim(_extend_shim),
        ),
        patch(
            "sglang.srt.mem_cache.allocator.paged.alloc_decode_kernel",
            _KernelShim(_decode_shim),
        ),
    )


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _make_req():
    return types.SimpleNamespace(
        kv=types.SimpleNamespace(kv_allocated_len=0, kv_committed_len=0)
    )


class _Harness:
    """One paged KV pool + req_to_token pool + a fake enc-dec decode batch.

    ``encoder_lens[i] == 0`` models a plain decoder-only request.
    """

    def __init__(
        self,
        *,
        page_size,
        num_pages,
        encoder_lens,
        prompt_lens,
        debug=True,
        allocator=None,
    ):
        assert len(encoder_lens) == len(prompt_lens)
        self.ps = page_size
        self.bs = len(encoder_lens)
        self.encoder_lens = list(encoder_lens)
        self.allocator = allocator or PagedTokenToKVPoolAllocator(
            size=num_pages * page_size,
            page_size=page_size,
            dtype=torch.float16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        self.allocator.debug_mode = debug
        self.pool = ReqToTokenPool(
            size=self.bs + 1,
            max_context_len=MAX_CONTEXT_LEN,
            device="cpu",
            enable_memory_saver=False,
        )
        self.req_pool_indices = torch.arange(1, self.bs + 1, dtype=torch.int64)
        # tree_cache stub: is_chunk_cache() short-circuits eviction, and both
        # page_size lookups in _alloc_page_size agree so the DCP branch is inert.
        self.tree_cache = types.SimpleNamespace(
            page_size=page_size,
            token_to_kv_pool_allocator=self.allocator,
            is_chunk_cache=lambda: True,
        )
        self.batch = types.SimpleNamespace(
            maybe_evict_swa=lambda: None,
            model_config=types.SimpleNamespace(is_encoder_decoder=True),
            tree_cache=self.tree_cache,
            req_to_token_pool=self.pool,
            req_pool_indices=self.req_pool_indices,
            req_pool_indices_cpu=self.req_pool_indices.clone(),
            reqs=[_make_req() for _ in range(self.bs)],
            encoder_lens=torch.tensor(encoder_lens, dtype=torch.int64),
            encoder_lens_cpu=list(encoder_lens),
            seq_lens=torch.tensor(prompt_lens, dtype=torch.int64),
            seq_lens_cpu=torch.tensor(prompt_lens, dtype=torch.int64),
        )
        self._prefill(prompt_lens)

    # -- lifecycle -------------------------------------------------------- #
    def _prefill(self, prompt_lens):
        """One-shot prefill of the whole row (encoder block + decoder prompt).

        Mirrors alloc_for_extend: seq_lens still includes the encoder here,
        because prepare_encoder_info_extend strips it only afterwards.
        """
        row_lens = [e + p for e, p in zip(self.encoder_lens, prompt_lens)]
        row_lens_t = torch.tensor(row_lens, dtype=torch.int64)
        prefix = torch.zeros(self.bs, dtype=torch.int64)
        out = self.allocator.alloc_extend(
            prefix_lens=prefix,
            prefix_lens_cpu=prefix,
            seq_lens=row_lens_t,
            seq_lens_cpu=row_lens_t,
            last_loc=torch.full((self.bs,), -1, dtype=torch.int64),
            extend_num_tokens=int(row_lens_t.sum()),
        )
        assert out is not None, "prefill ran out of pages; size the pool up"
        off = 0
        for i, n in enumerate(row_lens):
            self.pool.req_to_token[self.req_pool_indices[i], :n] = out[
                off : off + n
            ].to(torch.int32)
            off += n
        for req, n in zip(self.batch.reqs, row_lens):
            req.kv.kv_allocated_len = n
            req.kv.kv_committed_len = n

    def decode_step(self, token_per_req=1):
        out = alloc_for_decode(self.batch, token_per_req=token_per_req)
        self.batch.seq_lens = self.batch.seq_lens + token_per_req
        self.batch.seq_lens_cpu = self.batch.seq_lens_cpu + token_per_req
        return out

    def row(self, i=0):
        return self.pool.req_to_token[self.req_pool_indices[i]]

    def row_len(self, i=0):
        return int(self.batch.reqs[i].kv.kv_committed_len)

    def teardown(self):
        """The production teardown for a radix-disabled request: one contiguous
        page-disjoint release of the whole row (mem_cache/common.py
        release_kv_cache -> cache_finished_req -> free_segment)."""
        for i in range(self.bs):
            self.allocator.free_segment(
                self.row(i)[: self.row_len(i)].to(torch.int64), start_pos=0
            )

    # -- assertions ------------------------------------------------------- #
    def assert_page_consistent(self, testcase, i=0):
        ps, row, n = self.ps, self.row(i), self.row_len(i)
        for j in range(n):
            slot = int(row[j])
            testcase.assertEqual(
                slot % ps,
                j % ps,
                f"req {i} row[{j}]={slot} sits at page offset {slot % ps}, "
                f"expected {j % ps}",
            )
            head = int(row[j - (j % ps)])
            testcase.assertEqual(
                slot // ps,
                head // ps,
                f"req {i} row[{j}]={slot} is on page {slot // ps}, but the page "
                f"starting at row[{j - (j % ps)}] is page {head // ps}",
            )

    def assert_release_shortcut_exact(self, testcase, i=0):
        """free_segment's strided sampling must equal the true page set."""
        ps = self.ps
        idx = self.row(i)[: self.row_len(i)].to(torch.int64)
        strided = torch.sort(idx[::ps] // ps)[0]
        exact = torch.unique(idx // ps)
        testcase.assertTrue(
            torch.equal(strided, exact),
            f"req {i}: strided reps {strided.tolist()} != unique pages "
            f"{exact.tolist()} -- the row is not page-consistent",
        )

    def assert_pool_balanced(self, testcase):
        pages = self.allocator.get_all_free_pages()
        testcase.assertEqual(
            len(torch.unique(pages)),
            pages.numel(),
            "a page id appears twice in the free pool (double free)",
        )
        testcase.assertEqual(
            self.allocator.available_size(),
            self.allocator.size,
            "pool did not return to its full size at idle "
            "(this is what SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE aborts on)",
        )

    def encoder_slots(self, i=0):
        return {int(x) for x in self.row(i)[: self.encoder_lens[i]]}

    def decoder_slots(self, i=0):
        return {int(x) for x in self.row(i)[self.encoder_lens[i] : self.row_len(i)]}


class _PagedAllocOnCpu(CustomTestCase):
    """Replaces the two triton allocation kernels with CPU reference ports.

    Both the prefill (``alloc_extend``) and decode (``alloc_decode``) paths of the
    real allocator are exercised, so the patches must be live for every test that
    builds a ``_Harness``.
    """

    def setUp(self):
        super().setUp()
        self._patches = _patch_kernels()
        for p in self._patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self._patches])


# --------------------------------------------------------------------------- #
# 1. arithmetic contract of alloc_for_decode
# --------------------------------------------------------------------------- #
class TestEncDecDecodeAllocArithmetic(_PagedAllocOnCpu):
    """alloc_for_decode must measure and index the row, not the stripped seq_len."""

    def _capture(self, *, is_encoder_decoder, page_size, token_per_req=1):
        h = _Harness(
            page_size=page_size,
            num_pages=64,
            encoder_lens=[24, 16],
            prompt_lens=[3, 5],
        )
        h.batch.model_config.is_encoder_decoder = is_encoder_decoder
        if not is_encoder_decoder:
            # A decoder-only batch has no encoder block: keep the row honest so
            # the captured last_loc is meaningful.
            h.batch.encoder_lens = torch.zeros(h.bs, dtype=torch.int64)
            h.batch.encoder_lens_cpu = [0, 0]
        seen = {}

        def _fake_alloc(**kwargs):
            seen.update(kwargs)
            return torch.zeros(h.bs, dtype=torch.int64)

        with patch(
            "sglang.srt.mem_cache.allocation.alloc_paged_token_slots_decode",
            side_effect=_fake_alloc,
        ):
            h.decode_step(token_per_req=token_per_req)
        return h, seen

    def test_encoder_decoder_uses_row_length_and_row_last_loc(self):
        h, seen = self._capture(is_encoder_decoder=True, page_size=8)
        enc = torch.tensor([24, 16])
        prompt = torch.tensor([3, 5])
        self.assertTrue(torch.equal(seen["seq_lens"], enc + prompt + 1))
        self.assertTrue(torch.equal(seen["seq_lens_cpu"], enc + prompt + 1))
        expected_last_loc = torch.stack(
            [h.row(i)[int(enc[i] + prompt[i]) - 1] for i in range(2)]
        )
        self.assertTrue(torch.equal(seen["last_loc"], expected_last_loc))

    def test_encoder_decoder_last_loc_is_the_decoder_tail_not_the_encoder(self):
        """The pre-fix bug: last_loc landed inside the encoder block."""
        h, seen = self._capture(is_encoder_decoder=True, page_size=8)
        enc = [24, 16]
        prompt = [3, 5]
        for i in range(2):
            stripped_slot = int(h.row(i)[prompt[i] - 1])
            self.assertNotEqual(
                int(seen["last_loc"][i]),
                stripped_slot,
                "last_loc was read at row[seq_len-1], i.e. inside the encoder region",
            )
            self.assertEqual(
                int(seen["last_loc"][i]), int(h.row(i)[enc[i] + prompt[i] - 1])
            )

    def test_decoder_only_arithmetic_is_unchanged(self):
        """Non-encoder-decoder models must see exactly the legacy values."""
        h, seen = self._capture(is_encoder_decoder=False, page_size=8)
        prompt = torch.tensor([3, 5])
        self.assertTrue(torch.equal(seen["seq_lens"], prompt + 1))
        self.assertTrue(torch.equal(seen["seq_lens_cpu"], prompt + 1))
        expected_last_loc = torch.stack(
            [h.row(i)[int(prompt[i]) - 1] for i in range(2)]
        )
        self.assertTrue(torch.equal(seen["last_loc"], expected_last_loc))

    def test_token_per_req_greater_than_one(self):
        _, seen = self._capture(is_encoder_decoder=True, page_size=8, token_per_req=3)
        self.assertTrue(
            torch.equal(seen["seq_lens"], torch.tensor([24 + 3 + 3, 16 + 5 + 3]))
        )
        self.assertTrue(
            torch.equal(seen["seq_lens_cpu"], torch.tensor([24 + 3 + 3, 16 + 5 + 3]))
        )

    def test_heterogeneous_encoder_lengths_are_per_request(self):
        """encoder_lens is per-request; a batch-wide scalar would be wrong."""
        h = _Harness(
            page_size=8, num_pages=64, encoder_lens=[24, 9, 40], prompt_lens=[2, 7, 1]
        )
        seen = {}

        def _fake_alloc(**kwargs):
            seen.update(kwargs)
            return torch.zeros(3, dtype=torch.int64)

        with patch(
            "sglang.srt.mem_cache.allocation.alloc_paged_token_slots_decode",
            side_effect=_fake_alloc,
        ):
            h.decode_step()
        self.assertTrue(
            torch.equal(
                seen["seq_lens"], torch.tensor([24 + 2 + 1, 9 + 7 + 1, 40 + 1 + 1])
            )
        )

    def test_row_write_position_includes_the_encoder_offset(self):
        h = _Harness(page_size=8, num_pages=64, encoder_lens=[24], prompt_lens=[3])
        sentinel = torch.tensor([4242], dtype=torch.int64)
        with patch(
            "sglang.srt.mem_cache.allocation.alloc_paged_token_slots_decode",
            return_value=sentinel,
        ):
            h.decode_step()
        self.assertEqual(int(h.row(0)[24 + 3]), 4242)

    def test_page_size_one_skips_the_row_arithmetic_entirely(self):
        """At page_size == 1 the non-paged branch runs: no last_loc, no kernel."""
        h = _Harness(page_size=1, num_pages=256, encoder_lens=[24], prompt_lens=[3])
        sentinel = torch.tensor([777], dtype=torch.int64)
        with (
            patch(
                "sglang.srt.mem_cache.allocation.alloc_token_slots",
                return_value=sentinel,
            ) as tokens,
            patch(
                "sglang.srt.mem_cache.allocation.alloc_paged_token_slots_decode"
            ) as paged,
        ):
            h.decode_step()
        tokens.assert_called_once()
        paged.assert_not_called()
        # ...and the write still lands at the encoder-offset row position.
        self.assertEqual(int(h.row(0)[24 + 3]), 777)


# --------------------------------------------------------------------------- #
# 2. end-to-end: row layout and pool accounting through a full request
# --------------------------------------------------------------------------- #
class TestEncDecRowPageConsistency(_PagedAllocOnCpu):
    def _run_lifecycle(
        self, *, page_size, encoder_len, prompt_len, steps, num_pages=64
    ):
        h = _Harness(
            page_size=page_size,
            num_pages=num_pages,
            encoder_lens=[encoder_len],
            prompt_lens=[prompt_len],
        )
        for _ in range(steps):
            h.decode_step()
            h.assert_page_consistent(self)
        h.assert_release_shortcut_exact(self)
        h.teardown()
        h.assert_pool_balanced(self)
        return h

    def test_small_scale_sweep_over_page_boundaries(self):
        """Exhaustive small-scale sweep: every page-boundary crossing pattern."""
        for page_size in (2, 4):
            for encoder_len in (5, 6, 9):
                for prompt_len in (1, 3):
                    for steps in range(0, 2 * page_size + 3):
                        with self.subTest(
                            ps=page_size,
                            enc=encoder_len,
                            prompt=prompt_len,
                            steps=steps,
                        ):
                            self._run_lifecycle(
                                page_size=page_size,
                                encoder_len=encoder_len,
                                prompt_len=prompt_len,
                                steps=steps,
                            )

    def test_whisper_scale_thresholds(self):
        """Whisper on XPU: encoder_len=1500, 4 forced prompt tokens.

        steps=32 is the last output length that stayed benign pre-fix; 33 is the
        first that tripped the idle invariant; 128/129 straddle the fresh-page
        step that pre-fix leaked a page outright.
        """
        for page_size in (64, 128):
            for steps in (1, 32, 33, 129, 300):
                with self.subTest(ps=page_size, steps=steps):
                    self._run_lifecycle(
                        page_size=page_size,
                        encoder_len=1500,
                        prompt_len=4,
                        steps=steps,
                        num_pages=(1500 + 4 + 300) // page_size + 8,
                    )

    def test_page_aligned_encoder_length_also_holds(self):
        """A page-aligned encoder does not make the bug go away pre-fix, and must
        stay correct post-fix."""
        for page_size in (64, 128):
            for steps in (1, 33, 200):
                with self.subTest(ps=page_size, steps=steps):
                    self._run_lifecycle(
                        page_size=page_size,
                        encoder_len=page_size * 12,
                        prompt_len=4,
                        steps=steps,
                        num_pages=page_size * 12 // page_size + 12,
                    )

    def test_decode_slots_never_alias_the_encoder_region(self):
        h = self._run_lifecycle(
            page_size=8, num_pages=64, encoder_len=20, prompt_len=3, steps=40
        )
        overlap = h.encoder_slots(0) & h.decoder_slots(0)
        self.assertEqual(
            overlap,
            set(),
            f"decode reused encoder KV slots {sorted(overlap)}",
        )

    def test_pool_returns_to_full_size_after_many_sequential_requests(self):
        """The idle invariant: repeated request lifecycles must not drift.

        Pre-fix this grew by one page per request until
        SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE raised.
        """
        shared = PagedTokenToKVPoolAllocator(
            size=32 * 128,
            page_size=128,
            dtype=torch.float16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        for i in range(5):
            h = _Harness(
                page_size=128,
                num_pages=32,
                encoder_lens=[1500],
                prompt_lens=[4],
                allocator=shared,
            )
            for _ in range(40):
                h.decode_step()
            h.teardown()
            with self.subTest(request=i):
                h.assert_pool_balanced(self)

    def test_multi_request_batch_with_mixed_encoder_lengths(self):
        h = _Harness(
            page_size=8,
            num_pages=128,
            encoder_lens=[20, 8, 33],
            prompt_lens=[3, 1, 5],
        )
        for _ in range(30):
            h.decode_step()
            for i in range(h.bs):
                h.assert_page_consistent(self, i)
        for i in range(h.bs):
            h.assert_release_shortcut_exact(self, i)
        h.teardown()
        h.assert_pool_balanced(self)

    def test_decoder_only_request_in_an_encoder_decoder_batch(self):
        """encoder_lens[i] == 0 (no multimodal input) must behave like plain decode."""
        h = _Harness(
            page_size=8, num_pages=64, encoder_lens=[0, 17], prompt_lens=[4, 4]
        )
        for _ in range(25):
            h.decode_step()
            h.assert_page_consistent(self, 0)
            h.assert_page_consistent(self, 1)
        h.teardown()
        h.assert_pool_balanced(self)


# --------------------------------------------------------------------------- #
# 3. the tests above are not vacuous: the pre-fix arithmetic still fails them
# --------------------------------------------------------------------------- #
class TestPreFixArithmeticIsStillCaught(_PagedAllocOnCpu):
    """Drive the *old* arithmetic and assert every guard above fires.

    Without this, a future refactor that quietly reintroduces the stripped-seq_len
    arithmetic could pass the suite if a fixture drifted.
    """

    @staticmethod
    def _legacy_decode_step(h, token_per_req=1):
        """alloc_for_decode as it was before the fix: paged math on seq_lens."""
        from sglang.srt.mem_cache.allocation import alloc_paged_token_slots_decode

        seq_lens = h.batch.seq_lens
        last_loc = h.pool.req_to_token[h.req_pool_indices, seq_lens - 1]
        out = alloc_paged_token_slots_decode(
            tree_cache=h.tree_cache,
            seq_lens=seq_lens + token_per_req,
            seq_lens_cpu=h.batch.seq_lens_cpu + token_per_req,
            last_loc=last_loc,
            token_per_req=token_per_req,
            req_pool_indices=h.req_pool_indices,
            batch=h.batch,
        )
        locs = h.batch.encoder_lens + seq_lens
        h.pool.write((h.req_pool_indices, locs), out.to(torch.int32))
        for req in h.batch.reqs:
            req.kv.kv_allocated_len += token_per_req
            req.kv.kv_committed_len += token_per_req
        h.batch.seq_lens = seq_lens + token_per_req
        h.batch.seq_lens_cpu = h.batch.seq_lens_cpu + token_per_req
        return out

    def test_legacy_arithmetic_breaks_page_consistency(self):
        h = _Harness(page_size=128, num_pages=32, encoder_lens=[1500], prompt_lens=[4])
        for _ in range(40):
            self._legacy_decode_step(h)
        with self.assertRaises(AssertionError):
            h.assert_page_consistent(self)

    def test_legacy_arithmetic_aliases_encoder_slots(self):
        h = _Harness(page_size=8, num_pages=64, encoder_lens=[20], prompt_lens=[3])
        for _ in range(20):
            self._legacy_decode_step(h)
        self.assertNotEqual(h.encoder_slots(0) & h.decoder_slots(0), set())

    def test_legacy_arithmetic_makes_the_release_shortcut_wrong(self):
        """This is the assert that fires on hardware at paged.py free_segment."""
        h = _Harness(page_size=128, num_pages=32, encoder_lens=[1500], prompt_lens=[4])
        for _ in range(40):
            self._legacy_decode_step(h)
        with self.assertRaises(AssertionError):
            h.assert_release_shortcut_exact(self)
        # ...and the allocator's own debug_mode assert catches it too.
        with self.assertRaises(AssertionError):
            h.teardown()

    def test_legacy_arithmetic_over_counts_the_pool_by_one_page(self):
        h = _Harness(
            page_size=128,
            num_pages=32,
            encoder_lens=[1500],
            prompt_lens=[4],
            debug=False,
        )
        total = h.allocator.size
        for _ in range(40):
            self._legacy_decode_step(h)
        h.teardown()
        self.assertEqual(
            h.allocator.available_size(),
            total + h.ps,
            "expected the historical +1 page (128 token) over-count",
        )

    def test_legacy_arithmetic_stays_benign_below_the_threshold(self):
        """Reproduces why short Whisper outputs never tripped the invariant."""
        h = _Harness(
            page_size=128,
            num_pages=32,
            encoder_lens=[1500],
            prompt_lens=[4],
            debug=False,
        )
        total = h.allocator.size
        for _ in range(32):  # row reaches 1536 == 12 pages exactly
            self._legacy_decode_step(h)
        h.teardown()
        self.assertEqual(h.allocator.available_size(), total)


if __name__ == "__main__":
    unittest.main()
