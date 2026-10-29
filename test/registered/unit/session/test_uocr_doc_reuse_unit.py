"""CPU tests for Unlimited-OCR document copy-splice KV reuse helpers."""

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.unlimited_dash_ops.page2d_rope import (
    MIN_PAGE_TOKENS,
    PageCoordState,
    page_capacity,
)
from sglang.srt.layers.attention.unlimited_dash_ops.page_state import (
    PageIsolationState,
)
from sglang.srt.layers.attention.unlimited_ocr_v2_backend import (
    UnlimitedOCRV2AttnBackend,
)
from sglang.srt.session import uocr_doc_reuse as r
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE, IMG, BOS, EOS = 7, 9, 1, 2
CFG = {"image_token_id": IMG, "page_token_id": PAGE, "per_page_image_tokens": 3}


def img(page, n):
    return [r.MM_PAD_SHIFT_VALUE + page * 100 + i for i in range(n)]


def prompt(pages, instr=(50, 51)):
    ids = [BOS]
    for p, n in pages:
        ids += [PAGE] + img(p, n)
    return ids + list(instr)


def layout(start, n, gen, gen_start=None):
    return SimpleNamespace(
        num_tokens=n,
        num_reqs=1,
        q_start=torch.tensor([0, n]),
        start_pos=torch.tensor([start]),
        block_table=torch.arange(64).view(1, -1),
        req_of_token=torch.zeros(n, dtype=torch.int64),
        num_decode_tokens=n if gen else 0,
        gen_start=None if gen_start is None else torch.tensor([gen_start]),
    )


class TestPromptPlanning(CustomTestCase):
    def test_parse_prompt(self):
        ids = prompt([(0, 2), (1, 3)])
        pages = r.parse_prompt(ids, PAGE, IMG)
        self.assertEqual(pages.prefix_len, 1)
        self.assertEqual(pages.page_starts, (1, 4))
        self.assertEqual(pages.page_ends, (4, 8))
        self.assertEqual(pages.instr_start, 8)

    def test_tail_copy_rejects_a_page_shorter_than_the_tail(self):
        old = prompt([(0, 6), (1, 6), (2, 6)])
        new = prompt([(2, 6), (3, 6)])
        plan = r.plan_tail_copy(old, new, 1, PAGE, IMG, 4)
        # Tail start = E - 128 - E % 4 falls before the last page's marker, so a
        # short window is rejected rather than copied.
        self.assertIsNone(plan)

    def test_tail_copy_keeps_block_alignment_and_marker(self):
        old = prompt([(p, 200) for p in range(3)])
        new = prompt([(2, 200), (3, 200)])
        plan = r.plan_tail_copy(old, new, 1, PAGE, IMG, 32)
        end = r.parse_prompt(old, PAGE, IMG).page_ends[-1]
        start = end - r.TAIL_TOKENS - end % 32
        self.assertEqual(start % 32, 0, "the tail must start on a KV block")
        # [<PAGE> | image tail] then the new window's own pages; no BOS block.
        self.assertEqual(plan.new_ids[0], PAGE)
        self.assertEqual(plan.new_ids[1 : plan.copy_len], old[start:end])
        self.assertEqual(plan.copy_len, 1 + end - start)
        instr_start = r.parse_prompt(new, PAGE, IMG).instr_start
        self.assertEqual(
            plan.new_ids[plan.copy_len :], new[plan.kept_from : instr_start]
        )
        # Every copied slot needs a source token, including the marker.
        self.assertEqual(len(plan.src_positions), plan.copy_len)
        self.assertEqual(
            [old[p] for p in plan.src_positions], plan.new_ids[: plan.copy_len]
        )

    def test_tail_copy_drops_the_instruction(self):
        """The document's first window prefilled the instruction; a reuse window
        repeating it costs one prefill of it per window for nothing."""
        old = prompt([(p, 200) for p in range(3)])
        new = prompt([(2, 200), (3, 200)], instr=tuple(range(50, 550)))
        plan = r.plan_tail_copy(old, new, 1, PAGE, IMG, 32)
        self.assertTrue(r.is_image_token(plan.new_ids[-1], IMG))
        self.assertFalse(set(range(50, 550)) & set(plan.new_ids))

    def test_cut_at_extra_marker_keeps_the_last_page_text(self):
        """The stop marker must go, or text_tail starts after it and the next
        window has no text to splice; overlap can add a token after it."""
        out = [PAGE, 30, 31, PAGE, 32, 33, PAGE, 99]
        self.assertEqual(r._cut_at_extra_marker(out, PAGE, 2), out[:6])
        self.assertEqual(
            r.text_tail(r._cut_at_extra_marker(out, PAGE, 2), 99, PAGE), (4, [32, 33])
        )
        # A window that has not passed its last page is left whole.
        self.assertEqual(r._cut_at_extra_marker(out[:6], PAGE, 2), out[:6])

    def test_tail_copy_rejects_a_different_page(self):
        old = prompt([(p, 200) for p in range(3)])
        new = prompt([(5, 200), (6, 200)])
        self.assertIsNone(r.plan_tail_copy(old, new, 1, PAGE, IMG, 32))

    def test_text_tail_stops_at_the_last_page_marker(self):
        """A copied marker would be replayed against the rewritten prompt's
        smaller page count and land in the wrong page bucket."""
        out = [PAGE, 30, 31, PAGE, 32, 33]
        self.assertEqual(r.text_tail(out, 99, PAGE), (4, [32, 33]))
        self.assertEqual(r.text_tail(out, 1, PAGE), (5, [33]))
        self.assertIsNone(r.text_tail([], 5, PAGE))
        self.assertIsNone(r.text_tail(out, 0, PAGE))
        # A window that ended on its marker has no text to copy; the follow-up
        # still continues from that marker, with the image tail as context.
        self.assertEqual(r.text_tail([30, PAGE], 99, PAGE), (2, []))


class TestStateReplay(CustomTestCase):
    def test_coords_match_streaming_state(self):
        ids = prompt([(0, 2), (1, 3)]) + [PAGE, 40, 41, PAGE, 42]
        gen = len(prompt([(0, 2), (1, 3)]))
        state = PageCoordState(CFG, max_reqs=2, max_pages=16)
        _, page = state.step(torch.tensor(ids[:gen]), layout(0, gen, False))
        pages = page.tolist()
        for pos in range(gen, len(ids)):
            _, page = state.step(torch.tensor([ids[pos]]), layout(pos, 1, True))
            pages += page.tolist()
        replay, (mc, pp, cnt) = r.replay_coords(ids, gen, PAGE, IMG, 16)
        self.assertEqual(replay, pages)
        self.assertEqual((mc, pp), (int(state.st_marker[0]), int(state.st_pp[0])))
        self.assertEqual(cnt, state.cnt[0].tolist())

    def test_default_page_capacity_keeps_long_documents_distinct(self):
        """Pages past the coordinate capacity are clamped onto one page; a
        298-page document must keep every page distinct at the default capacity."""
        n_pages = 298
        ids = prompt([(p, 1) for p in range(n_pages)])
        state = PageCoordState(CFG, max_reqs=2, max_pages=page_capacity(786432))
        tok, page = state.step(torch.tensor(ids), layout(0, len(ids), False))
        image_pages = [p for t, p in zip(ids, page.tolist()) if t != PAGE][
            1 : 1 + n_pages
        ]
        self.assertEqual(image_pages, list(range(1, n_pages + 1)))

    def test_capacity_covers_every_page_the_context_admits(self):
        """Pages past the capacity share the last page's coordinates; a prompt
        of the cheapest pages that fills the context must not reach it."""
        context = 4096
        n_pages = (context - 2) // MIN_PAGE_TOKENS
        self.assertLess(n_pages + 1, page_capacity(context))

    def test_step_memory_does_not_scale_with_tokens_times_pages(self):
        """A dense [tokens, pages] count table made one prefill chunk allocate
        tens of GiB once the page capacity was raised (100000 pages OOMed)."""
        ids = prompt([(0, 2), (1, 3)]) + [PAGE, 40, 41, PAGE, 42]
        gen = len(prompt([(0, 2), (1, 3)]))
        small = PageCoordState(CFG, max_reqs=2, max_pages=16)
        big = PageCoordState(CFG, max_reqs=2, max_pages=1 << 20)
        lay = layout(0, len(ids), False, gen_start=gen)
        want = small.step(torch.tensor(ids), lay)
        real_zeros = torch.zeros
        sizes = []

        def tracking_zeros(*shape, **kw):
            dims = (
                shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
            )
            sizes.append(int(torch.Size(dims).numel()))
            return real_zeros(*shape, **kw)

        big._ensure(torch.device("cpu"))
        torch.zeros = tracking_zeros
        try:
            got = big.step(torch.tensor(ids), lay)
        finally:
            torch.zeros = real_zeros
        self.assertEqual(got[0].tolist(), want[0].tolist())
        self.assertEqual(got[1].tolist(), want[1].tolist())
        # One per-request carry row is fine; one per token is the bug.
        self.assertLessEqual(max(sizes), 1 << 20)

    def test_page_isolation_matches_slot_state(self):
        ids = prompt([(0, 2), (1, 3)]) + [PAGE, 40]
        n = len(ids)
        state = PageIsolationState(CFG, block_size=4)
        lay = layout(0, n, False)
        state.step(torch.tensor(ids), lay, torch.arange(n), 256)
        got = r.replay_page_isolation(ids, PAGE, IMG, 3)
        self.assertEqual(list(got[0]), state._slot_img_page[:n].tolist())
        self.assertEqual(list(got[1]), state._slot_marker_cum[:n].tolist())
        self.assertEqual(list(got[2]), state._slot_img_run[:n].tolist())
        self.assertEqual(list(got[3]), state._slot_prompt_pages[:n].tolist())


class TestRetractedRePrefill(CustomTestCase):
    """A retracted request re-prefills prompt + output in one extend; the
    output must keep generated page numbers, as when it was decoded."""

    IDS = prompt([(0, 2), (1, 3)]) + [PAGE, 40, 41, PAGE, 42]
    GEN = len(prompt([(0, 2), (1, 3)]))

    def test_coords_match_original_decode(self):
        ids, gen = self.IDS, self.GEN
        decoded = PageCoordState(CFG, max_reqs=2, max_pages=16)
        tok, page = decoded.step(torch.tensor(ids[:gen]), layout(0, gen, False))
        toks, pages = tok.tolist(), page.tolist()
        for pos in range(gen, len(ids)):
            tok, page = decoded.step(torch.tensor([ids[pos]]), layout(pos, 1, True))
            toks += tok.tolist()
            pages += page.tolist()
        reprefill = PageCoordState(CFG, max_reqs=2, max_pages=16)
        tok, page = reprefill.step(
            torch.tensor(ids), layout(0, len(ids), False, gen_start=gen)
        )
        self.assertEqual(page.tolist(), pages)
        self.assertEqual(tok.tolist(), toks)
        # Without the generation start the output takes prompt page numbers.
        legacy = PageCoordState(CFG, max_reqs=2, max_pages=16)
        _, page = legacy.step(torch.tensor(ids), layout(0, len(ids), False))
        self.assertNotEqual(page.tolist(), pages)

    def test_page_isolation_matches_original_decode(self):
        ids, gen = self.IDS, self.GEN
        n = len(ids)
        decoded = PageIsolationState(CFG, block_size=4)
        q = decoded.step(
            torch.tensor(ids[:gen]), layout(0, gen, False), torch.arange(gen), 256
        )
        q_pages = q.tolist()
        for pos in range(gen, n):
            q = decoded.step(
                torch.tensor([ids[pos]]), layout(pos, 1, True), torch.tensor([pos]), 256
            )
            q_pages += q.tolist()
        reprefill = PageIsolationState(CFG, block_size=4)
        q = reprefill.step(
            torch.tensor(ids), layout(0, n, False, gen_start=gen), torch.arange(n), 256
        )
        self.assertEqual(q.tolist(), q_pages)
        for name in ("_slot_img_page", "_slot_marker_cum", "_slot_prompt_pages"):
            self.assertEqual(
                getattr(reprefill, name)[:n].tolist(),
                getattr(decoded, name)[:n].tolist(),
            )

    def test_prompt_only_extend_is_unchanged(self):
        ids, gen = self.IDS[: self.GEN], self.GEN
        a = PageCoordState(CFG, max_reqs=2, max_pages=16)
        b = PageCoordState(CFG, max_reqs=2, max_pages=16)
        ta, pa = a.step(torch.tensor(ids), layout(0, gen, False))
        tb, pb = b.step(torch.tensor(ids), layout(0, gen, False, gen_start=gen))
        self.assertEqual((ta.tolist(), pa.tolist()), (tb.tolist(), pb.tolist()))


class TestGlobalPageParity(CustomTestCase):
    """Copied KV is not re-rotated, so a page must keep one RoPE page encoding in
    every window that contains it."""

    def _backend(self, page_mod=2, rows=4):
        b = object.__new__(UnlimitedOCRV2AttnBackend)
        b.rope_main = SimpleNamespace(page_mod=page_mod)
        b._gen_start_row = torch.zeros(rows, dtype=torch.int64)
        b._page_base_row = torch.zeros(rows, dtype=torch.int64)
        return b

    def _doc_page_coords(self, lo, hi, doc_page, base):
        """RoPE page coordinates of ``doc_page``'s tokens in window [lo, hi)."""
        ids = prompt([(p, 2) for p in range(lo, hi)])
        state = PageCoordState(CFG, max_reqs=2, max_pages=16)
        _, page = state.step(
            torch.tensor(ids), layout(0, len(ids), False, gen_start=len(ids))
        )
        b = self._backend()
        b.note_prompt_lens(
            torch.tensor([1]), torch.tensor([len(ids)]), torch.tensor([base])
        )
        md = SimpleNamespace(
            coords=(torch.zeros_like(page), page.clone()),
            num_reqs=1,
            is_decode=False,
            layout=SimpleNamespace(
                req_of_token=torch.zeros(len(ids), dtype=torch.int64)
            ),
        )
        b._shift_page_coords(md, SimpleNamespace(req_pool_indices=torch.tensor([1])))
        # <PAGE> + 2 image tokens per page, after the 1-token prefix.
        start = 1 + (doc_page - lo) * 3
        return md.coords[1][start : start + 3].tolist()

    def test_overlap_page_keeps_parity_across_windows(self):
        # Page 3 is the last page of [0, 4) and the first of [3, 6): an odd shift.
        first = self._doc_page_coords(0, 4, 3, base=0)
        second = self._doc_page_coords(3, 6, 3, base=3)
        self.assertEqual([p % 2 for p in first], [p % 2 for p in second])
        # Window-local numbering gives the page opposite parities.
        local = self._doc_page_coords(3, 6, 3, base=0)
        self.assertNotEqual([p % 2 for p in first], [p % 2 for p in local])

    def test_prefix_and_instruction_stay_on_page_zero(self):
        ids = prompt([(p, 2) for p in range(3, 5)])
        state = PageCoordState(CFG, max_reqs=2, max_pages=16)
        _, page = state.step(
            torch.tensor(ids), layout(0, len(ids), False, gen_start=len(ids))
        )
        b = self._backend()
        b.note_prompt_lens(
            torch.tensor([1]), torch.tensor([len(ids)]), torch.tensor([3])
        )
        md = SimpleNamespace(
            coords=(torch.zeros_like(page), page.clone()),
            num_reqs=1,
            is_decode=False,
            layout=SimpleNamespace(
                req_of_token=torch.zeros(len(ids), dtype=torch.int64)
            ),
        )
        b._shift_page_coords(md, SimpleNamespace(req_pool_indices=torch.tensor([1])))
        self.assertEqual(md.coords[1][0].item(), 0)
        self.assertEqual(md.coords[1][-2:].tolist(), [0, 0])

    def test_unfolded_page_axis_keeps_window_local_pages(self):
        b = self._backend(page_mod=0)
        b.note_prompt_lens(torch.tensor([1]), torch.tensor([5]), torch.tensor([7]))
        self.assertEqual(int(b._page_base_row[1]), 0)


class TestCoordRowPin(CustomTestCase):
    def test_a_seeded_row_whose_request_never_steps_is_unpinned(self):
        """A coordinate row seeded for a follow-up dropped before its prefill
        must not outrank every LRU victim for the rest of the process."""
        b = object.__new__(UnlimitedOCRV2AttnBackend)
        b.coord_state = PageCoordState(CFG, max_reqs=4, max_pages=8)
        cs = b.coord_state
        for block in (5, 6, 7):
            b.seed_coord_row(
                first_block=torch.tensor([block]),
                marker=1,
                prompt_pages=1,
                counts=[0] * 8,
            )
        rows = {k: int((cs.key == k).nonzero()) for k in (5, 6, 7)}
        # Row 7's request stepped: its stamp is a step count, not the pin.
        cs.stamp[rows[7]] = 3
        b.unpin_coord_rows(torch.tensor([5]))
        self.assertEqual(int(cs.stamp[rows[5]]), 0)
        self.assertGreater(int(cs.stamp[rows[6]]), 1 << 40)
        b.unpin_coord_rows(None)
        self.assertEqual(int(cs.stamp[rows[6]]), 0)
        self.assertEqual(int(cs.stamp[rows[7]]), 3)


class _FakeAllocator:
    def __init__(self):
        self.freed = []

    def free(self, idx):
        self.freed.append(list(idx.tolist()))


class _FakeHolder:
    """Mirrors DocumentSessionSlot's idempotent release contract."""

    def __init__(self, allocated=8, holds=True):
        self.kv = SimpleNamespace(
            holds_kv=holds, kv_allocated_len=allocated, req_pool_idx=3
        )
        self._released = False

    def mark_released(self):
        if self._released:
            return False
        self._released = True
        return True


def _backend_stub(**extra):
    """The backend surface the reuse planner reads; its queued edits are inspected."""
    return SimpleNamespace(
        page_state=SimpleNamespace(
            page_token_id=PAGE, image_token_id=IMG, per_page_image_tokens=200
        ),
        block_size=32,
        coord_state=SimpleNamespace(max_pages=64),
        **extra,
    )


class _FakeCache:
    def __init__(self, backend=None):
        self.token_to_kv_pool_allocator = _FakeAllocator()
        self.freed_rows = []
        self.freed_req_slots = []
        self.page_size = 32
        self.finished = []
        self.reuse = r.DocReuse(
            cache=self, backend=backend, on_window_finished=self.finished.append
        )

    def free_kv_row(self, kv, ranges):
        self.freed_rows.append((kv.req_pool_idx, tuple(ranges)))

    @property
    def req_to_token_pool(self):
        cache = self
        return SimpleNamespace(free=lambda holder: cache.freed_req_slots.append(holder))


class TestDroppedRequest(CustomTestCase):
    """A window the adder rejects is dropped before alloc_for_extend runs."""

    def _req(self, holds_kv=False, pending=None, rid="w1"):
        return SimpleNamespace(
            rid=rid,
            kv=SimpleNamespace(holds_kv=holds_kv),
            uocr_doc=r.UocrReqState(pending=pending),
        )

    def test_unadmitted_slots_are_freed(self):
        cache = _FakeCache()
        req = self._req(pending=torch.arange(6))
        cache.reuse.on_req_dropped(req)
        self.assertEqual(cache.token_to_kv_pool_allocator.freed, [[0, 1, 2, 3, 4, 5]])
        self.assertIsNone(req.uocr_doc.pending)

    def test_admitted_slots_are_left_to_the_row(self):
        cache = _FakeCache()
        req = self._req(holds_kv=True, pending=torch.arange(6))
        cache.reuse.on_req_dropped(req)
        self.assertEqual(cache.token_to_kv_pool_allocator.freed, [])
        self.assertIsNone(req.uocr_doc.pending)

    def test_plain_request_is_untouched(self):
        cache = _FakeCache()
        req = SimpleNamespace(
            rid="plain", kv=SimpleNamespace(holds_kv=False), uocr_doc=None
        )
        cache.reuse.on_req_dropped(req)
        self.assertEqual(cache.token_to_kv_pool_allocator.freed, [])
        self.assertEqual(cache.freed_rows, [])
        self.assertEqual(cache.reuse.sources, {})

    def test_drop_releases_the_source_window(self):
        cache = _FakeCache()
        entry = r.DocEntry(
            holder=_FakeHolder(allocated=8),
            input_ids=[1],
            output_ids=[],
            text_ok=True,
        )
        req = self._req(pending=torch.arange(3))
        req.uocr_doc.source = entry
        cache.reuse.sources[id(entry)] = entry
        cache.reuse.on_req_dropped(req)
        self.assertEqual(cache.freed_rows, [(3, ((0, 8),))])
        self.assertEqual(cache.freed_req_slots, [entry.holder])
        self.assertEqual(cache.reuse.sources, {})
        self.assertIsNone(req.uocr_doc.source)
        self.assertEqual(cache.token_to_kv_pool_allocator.freed, [[0, 1, 2]])

    def test_release_doc_frees_only_the_named_unconsumed_window(self):
        """A split request that ends early (client disconnect, abort) must free
        its retained window and leave other documents' windows alone."""
        cache = _FakeCache()
        mine, other = _FakeHolder(allocated=8), _FakeHolder(allocated=8)
        for doc_id, holder in (("mine", mine), ("other", other)):
            cache.reuse.docs[doc_id] = r.DocEntry(
                holder=holder, input_ids=[1], output_ids=[], text_ok=True
            )
        cache.reuse.release_doc("mine")
        cache.reuse.release_doc("mine")
        cache.reuse.release_doc("never-retained")
        self.assertEqual(list(cache.reuse.docs), ["other"])
        self.assertEqual(cache.freed_req_slots, [mine])

    def test_drop_is_idempotent(self):
        cache = _FakeCache()
        req = self._req(pending=torch.arange(4))
        cache.reuse.on_req_dropped(req)
        cache.reuse.on_req_dropped(req)
        self.assertEqual(len(cache.token_to_kv_pool_allocator.freed), 1)


class TestMemoryPressure(CustomTestCase):
    """A rejected request reclaims retained windows instead of stalling."""

    def _blocked(self, prompt_len=1000, pending=None):
        return SimpleNamespace(
            rid="blocked",
            sampling_params=SimpleNamespace(max_new_tokens=100, custom_params=None),
            output_ids=[],
            full_untruncated_fill_ids=list(range(prompt_len)),
            prefix_len=64 if pending is not None else 0,
            kv=SimpleNamespace(holds_kv=False, cache_protected_len=0),
            retraction_count=0,
            uocr_doc=r.UocrReqState(pending=pending),
        )

    def _retain(self, cache, doc_id, tokens):
        cache.reuse.docs[doc_id] = r.DocEntry(
            holder=_FakeHolder(allocated=tokens),
            input_ids=[1],
            output_ids=[],
            text_ok=True,
        )

    def test_frees_oldest_windows_until_request_fits(self):
        cache = _FakeCache()
        for i, tokens in enumerate([640, 640, 640]):
            self._retain(cache, f"doc{i}", tokens)
        # need = 1000 + 100 + 32 = 1132; free = 0 -> deficit 1164 -> two windows.
        adder = SimpleNamespace(rem_total_tokens=0, can_run_list=[])
        self.assertTrue(
            cache.reuse.on_no_token(
                req=self._blocked(), adder=adder, nothing_running=True
            )
        )
        self.assertEqual(list(cache.reuse.docs), ["doc2"])
        self.assertEqual(len(cache.freed_rows), 2)

    def test_retained_windows_survive_when_eviction_cannot_admit(self):
        # Running requests hold the memory: 640 retained < 1164 needed.
        cache = _FakeCache()
        self._retain(cache, "doc0", 640)
        adder = SimpleNamespace(rem_total_tokens=0, can_run_list=[])
        self.assertFalse(
            cache.reuse.on_no_token(
                req=self._blocked(), adder=adder, nothing_running=False
            )
        )
        self.assertEqual(list(cache.reuse.docs), ["doc0"])

    def test_follow_up_evicts_only_what_admits_while_others_run(self):
        cache = _FakeCache()
        for i in range(4):
            self._retain(cache, f"doc{i}", 640)
        adder = SimpleNamespace(rem_total_tokens=0, can_run_list=[])
        # need = (1000 - 64) + 100 + 32 = 1068 -> deficit 1100 -> two windows.
        req = self._blocked(pending=torch.arange(64))
        self.assertTrue(
            cache.reuse.on_no_token(req=req, adder=adder, nothing_running=False)
        )
        self.assertEqual(list(cache.reuse.docs), ["doc2", "doc3"])

    def test_follow_up_that_lost_reuse_still_evicts(self):
        cache = _FakeCache()
        for i in range(4):
            self._retain(cache, f"doc{i}", 640)
        adder = SimpleNamespace(rem_total_tokens=0, can_run_list=[])
        req = self._blocked()
        req.sampling_params.custom_params = {r.KEY_DOC_ID: "x", r.KEY_REUSE: True}
        # Full recompute: need = 1000 + 100 + 32 -> deficit 1164 -> two windows.
        self.assertTrue(
            cache.reuse.on_no_token(req=req, adder=adder, nothing_running=False)
        )
        self.assertEqual(list(cache.reuse.docs), ["doc2", "doc3"])

    def test_new_document_waits_while_others_run(self):
        cache = _FakeCache()
        for i in range(4):
            self._retain(cache, f"doc{i}", 640)
        adder = SimpleNamespace(rem_total_tokens=0, can_run_list=[])
        self.assertFalse(
            cache.reuse.on_no_token(
                req=self._blocked(), adder=adder, nothing_running=False
            )
        )
        self.assertEqual(len(cache.reuse.docs), 4)

    def test_nothing_to_free_keeps_request_waiting(self):
        cache = _FakeCache()
        adder = SimpleNamespace(rem_total_tokens=0, can_run_list=[])
        self.assertFalse(
            cache.reuse.on_no_token(
                req=self._blocked(), adder=adder, nothing_running=True
            )
        )

    def test_blocked_follow_up_falls_back_to_recompute(self):
        cache = _FakeCache()
        req = self._blocked(pending=torch.arange(64))
        adder = SimpleNamespace(rem_total_tokens=0, can_run_list=[])
        self.assertTrue(
            cache.reuse.on_no_token(req=req, adder=adder, nothing_running=True)
        )
        self.assertEqual(cache.token_to_kv_pool_allocator.freed, [list(range(64))])
        self.assertEqual(req.prefix_len, 0)
        self.assertEqual(req.uocr_doc.match, (None, 0))

    def test_follow_up_keeps_reuse_while_others_run(self):
        cache = _FakeCache()
        req = self._blocked(pending=torch.arange(64))
        adder = SimpleNamespace(rem_total_tokens=0, can_run_list=[])
        self.assertFalse(
            cache.reuse.on_no_token(req=req, adder=adder, nothing_running=False)
        )
        self.assertEqual(cache.token_to_kv_pool_allocator.freed, [])
        self.assertEqual(req.prefix_len, 64)


class TestDecodePressure(CustomTestCase):
    def test_retained_windows_yield_to_running_decodes(self):
        cache = _FakeCache()
        for i in range(3):
            cache.reuse.docs[f"doc{i}"] = r.DocEntry(
                holder=_FakeHolder(allocated=64),
                input_ids=[1],
                output_ids=[],
                text_ok=True,
            )
        # Fits once two windows are gone.
        batch = SimpleNamespace(check_decode_mem=lambda: len(cache.freed_rows) >= 2)
        cache.reuse.free_for_decode(batch)
        self.assertEqual(list(cache.reuse.docs), ["doc2"])

    def test_nothing_evicted_when_decode_fits(self):
        cache = _FakeCache()
        cache.reuse.docs["doc0"] = r.DocEntry(
            holder=_FakeHolder(allocated=64),
            input_ids=[1],
            output_ids=[],
            text_ok=True,
        )
        cache.reuse.free_for_decode(SimpleNamespace(check_decode_mem=lambda: True))
        self.assertEqual(list(cache.reuse.docs), ["doc0"])


class TestRequestRowPressure(CustomTestCase):
    """Retained windows holding every request row must not stall admission."""

    def _retain(self, cache, *doc_ids):
        for doc_id in doc_ids:
            cache.reuse.docs[doc_id] = r.DocEntry(
                holder=_FakeHolder(allocated=64),
                input_ids=[1],
                output_ids=[],
                text_ok=True,
            )

    def _window(self, doc_id, reuse=True):
        params = {r.KEY_DOC_ID: doc_id, r.KEY_REUSE: reuse}
        return SimpleNamespace(sampling_params=SimpleNamespace(custom_params=params))

    def test_frees_the_newest_other_window_for_the_waiting_follow_up(self):
        cache = _FakeCache()
        self._retain(cache, "doc0", "doc1", "doc2")
        waiting = [self._window("doc0"), self._window("doc1")]
        self.assertTrue(cache.reuse.free_for_admission(waiting))
        # One row is enough: the follow-up releases its source after the copy.
        self.assertEqual(list(cache.reuse.docs), ["doc0", "doc1"])
        self.assertEqual(len(cache.freed_req_slots), 1)

    def test_never_frees_the_waiting_follow_ups_own_window(self):
        cache = _FakeCache()
        self._retain(cache, "doc0")
        self.assertFalse(cache.reuse.free_for_admission([self._window("doc0")]))
        self.assertEqual(list(cache.reuse.docs), ["doc0"])

    def test_nothing_freed_without_a_waiting_follow_up(self):
        cache = _FakeCache()
        self._retain(cache, "doc0", "doc1")
        waiting = [self._window("new", reuse=False), self._window("gone")]
        self.assertFalse(cache.reuse.free_for_admission(waiting))
        self.assertEqual(len(cache.reuse.docs), 2)

    def test_nothing_freed_without_retained_windows(self):
        cache = _FakeCache()
        self.assertFalse(cache.reuse.free_for_admission([self._window("doc0")]))


class TestRetraction(CustomTestCase):
    def test_retracted_window_is_not_retained(self):
        # Retraction frees KV through cache_finished_req with the request
        # unfinished; it must reach the stock free path.
        cache = _FakeCache()
        req = SimpleNamespace(
            sampling_params=SimpleNamespace(
                custom_params={r.KEY_DOC_ID: "doc", r.KEY_KEEP: True}
            ),
            finished=lambda: False,
            finished_reason=None,
            kv=SimpleNamespace(holds_kv=True),
            uocr_doc=None,
        )
        self.assertFalse(cache.reuse.on_cache_finished_req(req))
        self.assertEqual(cache.reuse.docs, {})


class TestDeferNewDocument(CustomTestCase):
    def _req(self, reuse=False, prompt_len=1000):
        cp = {r.KEY_DOC_ID: "new", r.KEY_REUSE: reuse}
        return SimpleNamespace(
            sampling_params=SimpleNamespace(custom_params=cp, max_new_tokens=100),
            output_ids=[],
            full_untruncated_fill_ids=list(range(prompt_len)),
            prefix_len=0,
            uocr_doc=None,
        )

    def _retain(self, cache, doc_id, tokens):
        cache.reuse.docs[doc_id] = r.DocEntry(
            holder=_FakeHolder(allocated=tokens),
            input_ids=[1],
            output_ids=[],
            text_ok=True,
        )

    def test_new_document_waits_for_expected_follow_ups(self):
        cache = _FakeCache()
        self._retain(cache, "a", 2048)
        # need 1132 + expected 2048 > 3000 free.
        adder = SimpleNamespace(rem_total_tokens=3000)
        self.assertTrue(
            cache.reuse.defer_new_document(
                req=self._req(), adder=adder, nothing_running=False
            )
        )
        roomy = SimpleNamespace(rem_total_tokens=4000)
        self.assertFalse(
            cache.reuse.defer_new_document(
                req=self._req(), adder=roomy, nothing_running=False
            )
        )

    def test_follow_ups_and_idle_scheduler_are_never_deferred(self):
        cache = _FakeCache()
        self._retain(cache, "a", 2048)
        adder = SimpleNamespace(rem_total_tokens=0)
        self.assertFalse(
            cache.reuse.defer_new_document(
                req=self._req(reuse=True), adder=adder, nothing_running=False
            )
        )
        self.assertFalse(
            cache.reuse.defer_new_document(
                req=self._req(), adder=adder, nothing_running=True
            )
        )


class TestTailOnlyRetention(CustomTestCase):
    """A finished window keeps only what its follow-up copies."""

    PROMPT = prompt([(p, 200) for p in range(3)])
    OUTPUT = [PAGE] + list(range(30, 230))

    def _ranges(self, text_ok=True, allocated=None):
        n = len(self.PROMPT) + len(self.OUTPUT)
        return r.tail_ranges(
            self.PROMPT,
            self.OUTPUT,
            text_ok,
            PAGE,
            IMG,
            32,
            n if allocated is None else allocated,
        )

    def test_keeps_marker_image_tail_and_text_tail(self):
        pages = r.parse_prompt(self.PROMPT, PAGE, IMG)
        end = pages.page_ends[-1]
        tail = end - r.TAIL_TOKENS - end % 32
        got = self._ranges()
        # Block 0 (BOS) is not read by the follow-up window, so it is freed.
        self.assertNotEqual(got[0][0], 0)
        # The last page's marker is copied too, so its block is kept.
        marker_block = pages.page_starts[-1] // 32 * 32
        self.assertTrue(any(lo <= marker_block < hi for lo, hi in got))
        self.assertTrue(any(lo == tail for lo, hi in got), got)
        n = len(self.PROMPT) + len(self.OUTPUT)
        self.assertEqual(got[-1][1], min(-(-n // 32) * 32, n))

    def test_untrusted_text_keeps_only_the_prompt_side(self):
        prompt_end = r.parse_prompt(self.PROMPT, PAGE, IMG).page_ends[-1]
        for lo, hi in self._ranges(text_ok=False):
            self.assertLessEqual(lo, -(-prompt_end // 32) * 32)

    def test_unreusable_layout_keeps_everything(self):
        self.assertIsNone(r.tail_ranges([BOS, 50, 51], [], True, PAGE, IMG, 32, 3))

    def _trim(self):
        cache = _FakeCache()
        cache.page_size = 32
        freed = []
        cache.free_kv_row = lambda kv, ranges: freed.append(list(ranges))
        n = len(self.PROMPT) + len(self.OUTPUT)
        entry = r.DocEntry(
            holder=_FakeHolder(allocated=n),
            input_ids=list(self.PROMPT),
            output_ids=list(self.OUTPUT),
            text_ok=True,
        )
        cache.reuse._backend = _backend_stub()
        cache.reuse._trim_to_tails(entry)
        return cache, entry, freed

    def test_trim_frees_the_rest_and_shrinks_the_memory_estimate(self):
        cache, entry, freed = self._trim()
        n = len(self.PROMPT) + len(self.OUTPUT)
        kept = cache.reuse._entry_tokens(entry)
        # Marker block + 128-token image tail + 128-token text tail,
        # far below the whole window; the idle leak check reads this number.
        self.assertLess(kept, 400)
        self.assertLess(kept, n // 2)
        cache.reuse.docs["d"] = entry
        self.assertEqual(cache.reuse.held_tokens(), kept)
        # Freeing releases exactly the kept ranges, never the whole row.
        cache.reuse._free_entry(entry)
        self.assertEqual(freed[-1], entry.kept)


class TestPromptRewrite(CustomTestCase):
    """Tail-only reuse rewrites the prompt; vision and RoPE depend on it."""

    OLD = prompt([(p, 200) for p in range(3)])
    NEW = prompt([(2, 200), (3, 200)])

    def _plan(self):
        return r.plan_tail_copy(self.OLD, self.NEW, 1, PAGE, IMG, 32)

    def _req(self, offsets):
        items = [SimpleNamespace(offsets=[o]) for o in offsets]
        ids = array("q", self.NEW)
        return SimpleNamespace(
            multimodal_inputs=SimpleNamespace(mm_items=items),
            origin_input_ids=ids,
            origin_input_ids_unpadded=ids,
            full_untruncated_fill_ids=array("q", self.NEW),
            surr_offset=3,
            read_offset=9,
            uocr_doc=None,
        )

    def _page_offsets(self):
        pages = r.parse_prompt(self.NEW, PAGE, IMG)
        return [(pages.page_starts[i] + 1, pages.page_ends[i] - 1) for i in range(2)]

    def test_discarded_pages_drop_and_the_rest_shift(self):
        plan = self._plan()
        req = self._req(self._page_offsets())
        self.assertTrue(r._rewrite_prompt(req, plan))
        self.assertEqual(len(req.multimodal_inputs.mm_items), 1)
        start, end = req.multimodal_inputs.mm_items[0].offsets[0]
        # The surviving offsets must still point at image placeholders, or
        # vision would re-encode the wrong span.
        self.assertTrue(
            all(t >= r.MM_PAD_SHIFT_VALUE for t in plan.new_ids[start : end + 1])
        )
        self.assertEqual(list(req.origin_input_ids), plan.new_ids)
        self.assertIs(req.origin_input_ids_unpadded, req.origin_input_ids)
        self.assertEqual(list(req.full_untruncated_fill_ids), plan.new_ids)
        # The detokenizer's read offset is derived from the prompt length.
        self.assertIsNone(req.read_offset)

    def test_item_straddling_the_boundary_is_refused(self):
        plan = self._plan()
        req = self._req([(2, plan.kept_from + 10)])
        self.assertFalse(r._rewrite_prompt(req, plan))

    def test_restore_undoes_the_rewrite(self):
        plan = self._plan()
        offsets = self._page_offsets()
        req = self._req(offsets)
        items = list(req.multimodal_inputs.mm_items)
        self.assertTrue(r._rewrite_prompt(req, plan))
        r._restore_prompt(req)
        self.assertEqual(list(req.origin_input_ids), self.NEW)
        self.assertEqual(req.multimodal_inputs.mm_items, items)
        self.assertEqual([i.offsets[0] for i in items], offsets)

    def test_rewritten_prompt_has_one_context_page_then_own_pages(self):
        plan = self._plan()
        pages, _ = r.replay_coords(plan.new_ids, len(plan.new_ids), PAGE, IMG, 64)
        # 1 = image tail (the context page), 2 = this window's own page.
        self.assertEqual(sorted({p for p in pages if p > 0}), [1, 2])


class TestReuseContinuationCoords(CustomTestCase):
    """A reuse window must decode on whole-document coordinates.

    The spliced text keeps the KV the source window baked, so the carry seeded
    for the first decode step has to land the next page where a single-pass run
    puts it -- on both axes, since token positions count inside a page bucket.
    """

    PAGES = [(p, 200) for p in range(4)]
    TEXT = [(60 + p, 61 + p) for p in range(4)]

    def _out(self, first, last):
        """Response for document pages ``[first, last)``: a marker, then text."""
        out = []
        for p in range(first, last):
            out += [PAGE] + list(self.TEXT[p])
        return out

    def _stream(self, ids, gen, base=0, seed=None):
        """Prefill ``ids[:gen]``, then decode the rest one token at a time."""
        state = PageCoordState(CFG, max_reqs=2, max_pages=64)
        tok, page = state.step(torch.tensor(ids[:gen]), layout(0, gen, False))
        got = list(zip(tok.tolist(), page.tolist()))
        if seed is not None:
            marker, pp, cnt = seed
            state.st_marker[0] = marker
            state.st_pp[0] = pp
            state.cnt[0] = torch.tensor(cnt, dtype=state.cnt.dtype)
        for pos in range(gen, len(ids)):
            tok, page = state.step(torch.tensor([ids[pos]]), layout(pos, 1, True))
            got += list(zip(tok.tolist(), page.tolist()))
        # The backend adds the window's page base to every real page.
        return [(t, p + base if p > 0 else p) for t, p in got]

    def _reference(self):
        ids = prompt(self.PAGES)
        return self._stream(ids + self._out(0, 4), len(ids))[len(ids) :]

    def test_window_decodes_on_whole_document_coordinates(self):
        old = prompt(self.PAGES[:3])
        new = prompt(self.PAGES[2:])
        plan = r.plan_tail_copy(old, new, 1, PAGE, IMG, 32)
        self.assertIsNotNone(plan)
        # Window [2, 4) with one overlap page emits from document page 3.
        emit_from, base = 3, 3 - r.CONTEXT_PAGES
        coord, (iso_marker, _, iso_pp) = r.continuation_carry(
            plan.new_ids, PAGE, IMG, CFG["per_page_image_tokens"], 64
        )
        decode = self._out(emit_from, 4)
        got = self._stream(plan.new_ids + decode, len(plan.new_ids), base, coord)
        want = self._reference()[-len(decode) :]
        self.assertEqual(got[-len(decode) :], want)
        # Page isolation derives the same page from its own carry.
        self.assertEqual(iso_marker + 1 - 1 - iso_pp, want[0][1] - base - 1)

    def test_source_window_text_matches_the_document(self):
        """The spliced KV is only valid if the source baked these coordinates."""
        old = prompt(self.PAGES[:3])
        got = self._stream(old + self._out(0, 3), len(old))[len(old) :]
        self.assertEqual(got, self._reference()[: len(got)])

    def test_decoding_without_the_carry_takes_the_context_pages(self):
        old = prompt(self.PAGES[:3])
        new = prompt(self.PAGES[2:])
        plan = r.plan_tail_copy(old, new, 1, PAGE, IMG, 32)
        decode = self._out(3, 4)
        got = self._stream(
            plan.new_ids + decode, len(plan.new_ids), 3 - r.CONTEXT_PAGES
        )
        # Without the seed the window transcribes the image tail first, so its
        # pages trail the document by CONTEXT_PAGES.
        self.assertEqual(
            got[-len(decode)][1] + r.CONTEXT_PAGES, self._reference()[-len(decode)][1]
        )


class TestReuseSignal(CustomTestCase):
    """serving drops overlap page bodies only for recomputed windows, so the
    reuse flag has to survive the meta_info carrier exactly as sent."""

    def _carried(self, forced_lens):
        """What the output streamer puts in meta_info for one request."""
        values = [[n] for n in forced_lens]
        return {r.META_FORCED_TOKENS: [v for value in values for v in value]}

    def test_a_spliced_window_reports_reuse(self):
        self.assertTrue(r.window_reused(self._carried([120])))
        # Streamed in several chunks: the count repeats, reuse still reads true.
        self.assertTrue(r.window_reused(self._carried([120, 120])))

    def test_a_recomputed_window_does_not(self):
        self.assertFalse(r.window_reused(self._carried([0])))
        self.assertFalse(r.window_reused({}))
        self.assertFalse(r.window_reused(None))
        # A window with no document reuse carries no key at all.
        self.assertFalse(r.window_reused({"cached_tokens": 161}))


class TestForcedTextIsHidden(CustomTestCase):
    """The spliced text must not reach the answer, and must not take the first
    generated token's text with it.

    The detokenizer subtracts the text of everything before ``read_offset``
    from what it decodes, so the chunk it receives has to carry that prefix:
    shipping only the tokens after the spliced run left the first generated
    token with no text, and when that token was a page marker the window came
    back one page short.
    """

    def _req(self, prompt_len=20, forced=8, generated=1):
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        req = Req("r0", "", array("q", range(1, prompt_len + 1)), SamplingParams())
        req.output_ids.extend(array("q", range(900, 900 + forced)))
        r.hide_forced_text(req, forced)
        req.output_ids.extend(array("q", range(800, 800 + generated)))
        return req

    def test_the_first_generated_token_still_has_text(self):
        req = self._req(forced=8, generated=2)
        ids, read_offset = req.init_incremental_detokenize()
        # What the streamer ships the detokenizer this step.
        self.assertEqual(list(ids[req.send_decode_id_offset :]), list(ids))
        # Text starts at the first generated token, not inside the spliced run.
        self.assertEqual(list(ids[read_offset:]), [800, 801])

    def test_the_spliced_tokens_stay_out_of_the_token_list(self):
        req = self._req(forced=8, generated=2)
        self.assertEqual(
            list(req.output_ids_through_stop[req.send_token_offset :]), [800, 801]
        )


class TestWindowAdjacency(CustomTestCase):
    """Contiguous windows share no page, so the page bookkeeping is the only
    thing that proves a retained tail is the page before the next window."""

    def test_next_page_after_counts_only_the_window_s_own_pages(self):
        # A recomputed window [10, 20) with one overlap page: 10 prompt pages,
        # the first of which page 10 already transcribed.
        self.assertEqual(
            r.next_page_after(prompt([(p, 2) for p in range(10)]), 11, 1, PAGE, IMG),
            20,
        )
        # Contiguous: all ten pages are its own.
        self.assertEqual(
            r.next_page_after(prompt([(p, 2) for p in range(10)]), 10, 0, PAGE, IMG),
            20,
        )
        # A reuse window's rewritten prompt opens with the image tail.
        self.assertEqual(
            r.next_page_after(
                prompt([(p, 2) for p in range(11)]), 10, r.CONTEXT_PAGES, PAGE, IMG
            ),
            20,
        )

    def test_a_layout_without_pages_has_no_next_page(self):
        self.assertIsNone(r.next_page_after([1, 2, 3], 0, 0, PAGE, IMG))

    def test_a_lookahead_page_is_read_but_not_owned(self):
        """A window reading the next window's first page must hand that page to
        the next window and keep the tail of its own last page, not of the lookahead."""
        # Window [0, 3) reads page 3 as lookahead.
        with_ahead = prompt([(p, 200) for p in range(4)])
        self.assertEqual(r.next_page_after(with_ahead, 0, 0, PAGE, IMG, lookahead=1), 3)
        own = r.parse_prompt(with_ahead, PAGE, IMG)
        kept = r.tail_ranges(
            with_ahead, [], False, PAGE, IMG, 32, len(with_ahead), lookahead=1
        )
        self.assertTrue(any(lo <= own.page_ends[2] - 1 < hi for lo, hi in kept), kept)
        # The follow-up copies the tail of page 2, not of the lookahead page 3;
        # the lookahead page itself follows that tail whole.
        new = prompt([(3, 200), (4, 200)])
        plan = r.plan_tail_copy(with_ahead, new, 0, PAGE, IMG, 32, old_lookahead=1)
        no_ahead = r.plan_tail_copy(
            prompt([(p, 200) for p in range(3)]), new, 0, PAGE, IMG, 32
        )
        tail = len(no_ahead.src_positions)
        self.assertEqual(plan.src_positions[:tail], no_ahead.src_positions)
        self.assertEqual(
            plan.src_positions[tail:],
            tuple(range(own.page_starts[3], own.page_ends[3])),
        )

    def test_the_lookahead_page_is_copied_not_recomputed(self):
        """The next window's first page was already encoded and prefilled as the
        previous window's lookahead; copying it must cover exactly that page."""
        old = prompt([(p, 200) for p in range(4)])  # window [0, 3) + lookahead 3
        new = prompt([(3, 200), (4, 200), (5, 200)])
        plan = r.plan_tail_copy(old, new, 0, PAGE, IMG, 32, old_lookahead=1)
        self.assertEqual(plan.copied_pages, 1)
        o, n = r.parse_prompt(old, PAGE, IMG), r.parse_prompt(new, PAGE, IMG)
        # Copied: <PAGE> + page-2 tail + lookahead page 3; prefilled: pages 4, 5.
        self.assertEqual(plan.kept_from, n.page_ends[0])
        self.assertEqual(
            plan.new_ids[plan.copy_len :], new[n.page_ends[0] : n.instr_start]
        )
        self.assertEqual(
            [old[p] for p in plan.src_positions], plan.new_ids[: plan.copy_len]
        )
        self.assertEqual(plan.src_positions[-1], o.page_ends[3] - 1)
        # The old window keeps the lookahead page for it.
        kept = r.tail_ranges(old, [], False, PAGE, IMG, 32, len(old), lookahead=1)
        self.assertTrue(
            any(lo <= o.page_starts[3] and o.page_ends[3] <= hi for lo, hi in kept)
        )

    def test_a_lookahead_that_is_not_the_next_page_is_not_copied(self):
        old = prompt([(p, 200) for p in range(4)])
        new = prompt([(7, 200), (8, 200)])
        plan = r.plan_tail_copy(old, new, 0, PAGE, IMG, 32, old_lookahead=1)
        self.assertEqual(plan.copied_pages, 0)

    def test_a_one_page_window_prefills_only_the_last_image_token(self):
        """A document's last window can be just the previous lookahead page;
        the extend still needs one token, so all but its last image token is
        copied and that token's item is allowed to straddle the prefix."""
        old = prompt([(p, 200) for p in range(4)])
        new = prompt([(3, 200)])
        plan = r.plan_tail_copy(old, new, 0, PAGE, IMG, 32, old_lookahead=1)
        n = r.parse_prompt(new, PAGE, IMG)
        self.assertEqual((plan.copied_pages, plan.allow_straddle), (1, True))
        self.assertEqual(plan.new_ids[plan.copy_len :], [new[n.page_ends[0] - 1]])
        self.assertEqual(
            [old[p] for p in plan.src_positions], plan.new_ids[: plan.copy_len]
        )
        # The page's item survives the rewrite and ends on the one prefilled token.
        item = SimpleNamespace(offsets=[(n.page_starts[0] + 1, n.page_ends[0] - 1)])
        ids = array("q", new)
        req = SimpleNamespace(
            multimodal_inputs=SimpleNamespace(mm_items=[item]),
            origin_input_ids=ids,
            origin_input_ids_unpadded=ids,
            full_untruncated_fill_ids=array("q", new),
            surr_offset=None,
            read_offset=None,
            uocr_doc=None,
        )
        self.assertTrue(r._rewrite_prompt(req, plan))
        self.assertEqual(req.multimodal_inputs.mm_items, [item])
        start, end = item.offsets[0]
        self.assertEqual(end, plan.copy_len)
        self.assertEqual(end - start + 1, n.page_ends[0] - n.page_starts[0] - 1)
        self.assertTrue(
            all(r.is_image_token(t, IMG) for t in plan.new_ids[start : end + 1])
        )

    def test_a_straddling_item_is_still_refused_without_the_plan_asking(self):
        old = prompt([(p, 200) for p in range(4)])
        new = prompt([(3, 200), (4, 200)])
        plan = r.plan_tail_copy(old, new, 0, PAGE, IMG, 32, old_lookahead=1)
        n = r.parse_prompt(new, PAGE, IMG)
        ids = array("q", new)
        item = SimpleNamespace(offsets=[(n.page_starts[0] + 1, n.page_ends[0] + 5)])
        req = SimpleNamespace(
            multimodal_inputs=SimpleNamespace(mm_items=[item]),
            origin_input_ids=ids,
            origin_input_ids_unpadded=ids,
            full_untruncated_fill_ids=array("q", new),
            surr_offset=None,
            read_offset=None,
            uocr_doc=None,
        )
        self.assertFalse(r._rewrite_prompt(req, plan))


class TestQueuePosition(CustomTestCase):
    def _req(self, doc_id=None, reuse=False):
        cp = {r.KEY_DOC_ID: doc_id, r.KEY_REUSE: reuse} if doc_id else None
        return SimpleNamespace(
            sampling_params=SimpleNamespace(custom_params=cp), uocr_doc=None
        )

    def test_follow_ups_go_ahead_of_new_documents_in_arrival_order(self):
        cache = _FakeCache()
        cache.reuse.docs.update({"a": object(), "b": object()})
        queue = [self._req(), self._req("c")]
        first = self._req("a", reuse=True)
        queue.insert(cache.reuse.queue_position(queue, first), first)
        second = self._req("b", reuse=True)
        queue.insert(cache.reuse.queue_position(queue, second), second)
        self.assertIs(queue[0], first)
        self.assertIs(queue[1], second)

    def test_other_requests_append(self):
        cache = _FakeCache()
        self.assertIsNone(cache.reuse.queue_position([], self._req()))
        self.assertIsNone(cache.reuse.queue_position([], self._req("a")))
        # Source window already gone (evicted or never kept): nothing to protect.
        self.assertIsNone(cache.reuse.queue_position([], self._req("a", reuse=True)))


class _AllocatingAllocator(_FakeAllocator):
    """Hands out consecutive slots so tests can tell copies from frees."""

    def __init__(self, start=1000):
        super().__init__()
        self.next = start
        self.allocated = []

    def alloc(self, n):
        out = torch.arange(self.next, self.next + n, dtype=torch.int64)
        self.next += n
        self.allocated.append(out.tolist())
        return out


class _MatchCache(_FakeCache):
    """A cache with the upstream match-side surface the reuse planner uses."""

    def __init__(self, backend=None):
        super().__init__(backend=backend)
        self.token_to_kv_pool_allocator = _AllocatingAllocator()
        self.req_to_token = torch.arange(4 * 2048, dtype=torch.int64).reshape(4, 2048)
        self.unlocked = []
        self.aligned_frees = []
        cache = self
        self.session = SimpleNamespace(
            _free_kv_aligned=lambda kv, target, end: cache.aligned_frees.append(
                (kv.req_pool_idx, target, end)
            )
        )

    def root_node_handle(self, extra_key=None):
        return ("root", extra_key)

    def unlock(self, lock):
        self.unlocked.append(lock)

    @property
    def req_to_token_pool(self):
        cache = self
        return SimpleNamespace(
            req_to_token=cache.req_to_token,
            free=lambda holder: cache.freed_req_slots.append(holder),
        )


class TestPrefixHandOff(CustomTestCase):
    """The match reports a prefix length; the slots reach alloc_for_extend
    through ``prefix_device_indices`` and the row owns them from then on."""

    OLD = prompt([(0, 200), (1, 200)])
    NEW = prompt([(2, 200)])
    OUT = [PAGE] + list(range(30, 60))

    def _follow_up(self):
        cp = {r.KEY_DOC_ID: "doc", r.KEY_REUSE: True, r.KEY_PAGE_BASE: 2}
        ids = array("q", self.NEW)
        return SimpleNamespace(
            rid="w2",
            origin_input_ids=ids,
            origin_input_ids_unpadded=ids,
            full_untruncated_fill_ids=array("q", self.NEW),
            multimodal_inputs=None,
            sampling_params=SimpleNamespace(max_new_tokens=1000, custom_params=cp),
            kv=SimpleNamespace(holds_kv=False, cache_protected_len=0),
            extra_key="ek",
            retraction_count=0,
            output_ids=[],
            surr_offset=None,
            read_offset=None,
            prefix_len=0,
            uocr_doc=None,
        )

    def _match(self):
        cache = _MatchCache(backend=_backend_stub())
        holder = _FakeHolder(allocated=len(self.OLD) + len(self.OUT))
        cache.reuse.docs["doc"] = r.DocEntry(
            holder=holder,
            input_ids=list(self.OLD),
            output_ids=list(self.OUT),
            text_ok=True,
            next_page=2,
        )
        req = self._follow_up()
        result = cache.reuse.try_match_prefix(SimpleNamespace(req=req))
        return cache, holder, req, result

    def test_match_reports_length_and_a_root_anchor(self):
        _, _, req, result = self._match()
        self.assertIsNotNone(result)
        plan = r.plan_tail_copy(self.OLD, self.NEW, 0, PAGE, IMG, 32)
        self.assertEqual(result.device_prefix_len, plan.copy_len)
        self.assertEqual(result.cache_protected_len, 0)
        for node in (
            result.last_device_node,
            result.last_host_node,
            result.best_match_node,
        ):
            self.assertEqual(node, ("root", "ek"))
        # The rewritten prompt starts with the copied tail.
        self.assertEqual(list(req.origin_input_ids), plan.new_ids)

    def test_alloc_for_extend_gets_the_copied_slots(self):
        cache, _, req, result = self._match()
        req.prefix_len = result.device_prefix_len  # as match_kv_cache adopts it
        (alloc,) = cache.token_to_kv_pool_allocator.allocated
        got = r.prefix_device_indices(req)
        self.assertEqual(got.tolist(), alloc[: result.device_prefix_len])
        # The copy into those slots is queued for the next forward.
        self.assertEqual(len(cache.reuse._ops), 1)
        # Once the request holds its row, the row is the only record.
        req.kv.holds_kv = True
        self.assertIsNone(r.prefix_device_indices(req))

    def test_plain_request_has_no_hand_off(self):
        req = SimpleNamespace(
            kv=SimpleNamespace(holds_kv=False), prefix_len=0, uocr_doc=None
        )
        self.assertIsNone(r.prefix_device_indices(req))

    def test_unified_cache_hands_the_copied_slots_to_alloc(self):
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.uocr_doc_reuse = r.DocReuse(
            cache=cache, backend=None, on_window_finished=lambda req: None
        )
        tree_path = torch.arange(3, dtype=torch.int64)
        cache.tree_core = SimpleNamespace(
            root_node_handle=lambda extra_key: 0,
            collect_full_device_indices=lambda node, root: tree_path,
        )
        cache.linker = None
        req = SimpleNamespace(
            rid="w",
            extra_key=None,
            last_node=0,
            kv=SimpleNamespace(holds_kv=False),
            prefix_len=4,
            uocr_doc=r.UocrReqState(pending=torch.arange(10, 20)),
        )
        self.assertEqual(cache.prefix_device_indices(req).tolist(), [10, 11, 12, 13])
        # Without a pending copy the tree path answers, as before.
        req.uocr_doc.pending = None
        req.prefix_len = 3
        self.assertEqual(cache.prefix_device_indices(req).tolist(), [0, 1, 2])

    def test_unadmitted_match_frees_every_slot_and_the_source_once(self):
        cache, holder, req, result = self._match()
        req.prefix_len = result.device_prefix_len
        (alloc,) = cache.token_to_kv_pool_allocator.allocated
        cache.reuse.on_req_dropped(req)
        cache.reuse.on_req_dropped(req)
        self.assertEqual(cache.token_to_kv_pool_allocator.freed, [alloc])
        self.assertEqual(cache.freed_req_slots, [holder])
        self.assertIsNone(r.prefix_device_indices(req))

    def test_a_dropped_match_runs_none_of_its_queued_copies(self):
        """A follow-up dropped before its first forward must not have its
        prefix copy write slots already handed back to the allocator, and the
        coordinate row it was going to seed must not stay pinned."""
        cache, _, req, result = self._match()
        req.prefix_len = result.device_prefix_len
        cache.reuse.on_req_dropped(req)
        backend = _RecordingBackend()
        cache.reuse.run_pending(backend)
        self.assertEqual(backend.moves, [])
        (first_block,) = backend.unpinned
        self.assertEqual(first_block.tolist(), [1000 // 32])

    def test_an_edit_that_raises_keeps_the_rest_queued(self):
        """Every queued edit backs state a later forward reads; one failure
        must not silently drop the ones behind it."""
        cache = _FakeCache()
        ran = []

        def boom(be):
            raise RuntimeError("edit failed")

        cache.reuse._enqueue("a", boom)
        cache.reuse._enqueue("b", lambda be: ran.append("b"))
        with self.assertRaises(RuntimeError):
            cache.reuse.run_pending(None)
        cache.reuse.run_pending(None)
        self.assertEqual(ran, ["b"])


class _RecordingBackend:
    """Runs queued edits by recording what they would write."""

    def __init__(self):
        self.moves = []
        self.unpinned = []
        self.kv_pool = SimpleNamespace(
            move_kv_cache=lambda dst, src: self.moves.append((dst, src))
        )

    def unpin_coord_rows(self, blocks):
        self.unpinned.append(blocks)


class TestRetractedRewrittenWindow(CustomTestCase):
    """A retracted window re-prefills its prompt from token ids; the rewritten
    prompt's image tail has no multimodal item, so it must get back the prompt
    and items it was sent."""

    OLD = prompt([(0, 200), (1, 200)])
    NEW = prompt([(1, 200), (2, 200)])
    OUT = [PAGE] + list(range(30, 60))

    def _window(self):
        pages = r.parse_prompt(self.NEW, PAGE, IMG)
        items = [
            SimpleNamespace(
                offsets=[(pages.page_starts[i] + 1, pages.page_ends[i] - 1)]
            )
            for i in range(2)
        ]
        ids = array("q", self.NEW)
        cp = {
            r.KEY_DOC_ID: "doc",
            r.KEY_REUSE: True,
            r.KEY_PAGE_BASE: 1,
            r.KEY_OVERLAP: 1,
        }
        return SimpleNamespace(
            rid="w2",
            origin_input_ids=ids,
            origin_input_ids_unpadded=ids,
            full_untruncated_fill_ids=array("q", self.NEW),
            multimodal_inputs=SimpleNamespace(mm_items=list(items)),
            sampling_params=SimpleNamespace(max_new_tokens=1000, custom_params=cp),
            kv=SimpleNamespace(holds_kv=False, cache_protected_len=0),
            extra_key=None,
            retraction_count=0,
            output_ids=[],
            surr_offset=None,
            read_offset=None,
            prefix_len=0,
            finished=lambda: False,
            finished_reason=None,
            page_guard_state=object(),
            uocr_doc=None,
        ), items

    def test_retract_restores_the_prompt_and_items_it_was_sent(self):
        cache = _MatchCache(backend=_backend_stub())
        cache.reuse.docs["doc"] = r.DocEntry(
            holder=_FakeHolder(allocated=len(self.OLD) + len(self.OUT)),
            input_ids=list(self.OLD),
            output_ids=list(self.OUT),
            text_ok=True,
            next_page=2,
        )
        req, items = self._window()
        offsets = [list(it.offsets) for it in items]
        self.assertIsNotNone(cache.reuse.try_match_prefix(SimpleNamespace(req=req)))
        self.assertNotEqual(list(req.origin_input_ids), self.NEW)
        self.assertEqual(len(req.multimodal_inputs.mm_items), 1)
        # The window runs its prefill, then is retracted while decoding.
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: True),
            reqs=[req],
            seq_lens_cpu=None,
        )
        cache.reuse.after_prefill_forward(batch, chunked_req=None)
        self.assertFalse(cache.reuse.on_cache_finished_req(req))
        cache.reuse.on_retracted(req)

        self.assertEqual(list(req.origin_input_ids), self.NEW)
        self.assertEqual(req.multimodal_inputs.mm_items, items)
        self.assertEqual([it.offsets for it in items], offsets)
        self.assertEqual(len(req.full_untruncated_fill_ids), 0)
        st = req.uocr_doc
        # Its outputs open on its own first page: the overlap page stays untranscribed.
        self.assertEqual(st.context_pages, 1)
        self.assertFalse(st.end_on_marker)
        self.assertIsNone(st.page_base)
        self.assertIsNone(req.page_guard_state)


class TestFlushReset(CustomTestCase):
    def test_reset_forgets_retained_windows_without_freeing_them(self):
        """A cache flush clears the pools itself; a retained window freed
        afterwards would return slots and a row the pools already reissued."""
        cache = _FakeCache(backend=_backend_stub())
        cache.reuse.docs["doc"] = r.DocEntry(
            holder=_FakeHolder(allocated=64), input_ids=[1], output_ids=[], text_ok=True
        )
        cache.reuse._input_overrides[3] = PAGE
        cache.reuse._enqueue("w", lambda be: be.moves.append("stale"))
        cache.reuse.reset()
        cache.reuse.release_doc("doc")
        cache.reuse.free_for_decode(SimpleNamespace(check_decode_mem=lambda: False))
        self.assertEqual(cache.freed_rows, [])
        self.assertEqual(cache.freed_req_slots, [])
        self.assertEqual(cache.reuse.held_tokens(), 0)
        buf = torch.zeros(8, dtype=torch.int64)
        cache.reuse.apply_input_overrides(output_tokens_buf=buf, relay=None)
        self.assertEqual(buf.tolist(), [0] * 8)
        backend = _RecordingBackend()
        cache.reuse.run_pending(backend)
        # Only the release of every pinned coordinate row is left.
        self.assertEqual(backend.moves, [])
        self.assertEqual(backend.unpinned, [None])


class TestRetainedWindowClaim(CustomTestCase):
    """Retention takes the finished request's row; release_kv_cache then
    returns early, so the planner drops the request's lock itself."""

    PROMPT = prompt([(0, 200), (1, 200)])

    def _finished(self, origin):
        from sglang.srt.managers.schedule_batch import (
            FINISH_MATCHED_TOKEN,
            ReqKvInfo,
        )

        out = [PAGE] + list(range(30, 60)) + [EOS]
        committed = len(origin) + len(out) - 1
        req = SimpleNamespace(
            rid="w1",
            sampling_params=SimpleNamespace(
                custom_params={r.KEY_DOC_ID: "doc", r.KEY_KEEP: True},
                stop_token_ids=None,
            ),
            tokenizer=None,
            finished=lambda: True,
            finished_reason=FINISH_MATCHED_TOKEN(matched=EOS),
            finished_len=None,
            eos_token_ids={EOS},
            origin_input_ids=array("q", origin),
            output_ids=array("q", out),
            kv=ReqKvInfo(
                req_pool_idx=2,
                kv_committed_len=committed,
                kv_allocated_len=committed + 5,
            ),
            lock="tree-lock",
            uocr_doc=None,
        )

        def detach_kv():
            kv, req.kv = req.kv, ReqKvInfo()
            return kv

        req.detach_kv = detach_kv
        return req, committed

    def test_retained_row_moves_to_the_holder_and_the_lock_is_dropped(self):
        cache = _MatchCache(backend=_backend_stub())
        req, committed = self._finished(self.PROMPT)
        self.assertTrue(cache.reuse.on_cache_finished_req(req))
        self.assertFalse(req.kv.holds_kv)
        self.assertEqual(cache.unlocked, ["tree-lock"])
        self.assertIsNone(req.lock)
        entry = cache.reuse.docs["doc"]
        self.assertEqual(entry.holder.kv.req_pool_idx, 2)
        # The uncommitted tail goes back now; the holder keeps the rest.
        self.assertEqual(cache.aligned_frees, [(2, committed, committed + 5)])
        self.assertEqual(entry.holder.kv.kv_allocated_len, committed)

    def test_unusable_layout_leaves_the_row_to_release_kv_cache(self):
        cache = _MatchCache(backend=_backend_stub())
        req, _ = self._finished([BOS, 50, 51])
        self.assertFalse(cache.reuse.on_cache_finished_req(req))
        self.assertTrue(req.kv.holds_kv)
        self.assertEqual(req.lock, "tree-lock")
        self.assertEqual(cache.unlocked, [])
        self.assertEqual(cache.aligned_frees, [])
        self.assertEqual(cache.reuse.docs, {})


if __name__ == "__main__":
    unittest.main()
