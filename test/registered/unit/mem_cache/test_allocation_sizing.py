from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest

from sglang.srt.mem_cache.allocation_sizing import (
    assert_alloc_within_row_width,
    get_alloc_page_size_upper_bound,
    get_alloc_reserve_per_decode,
    get_candidate_alloc_page_sizes,
    get_req_to_token_row_width,
)
from sglang.srt.utils.common import ceil_align
from sglang.test.test_utils import CustomTestCase


def _make_server_args(
    *,
    page_size: int = 1,
    dcp_size: int = 1,
    speculative_algorithm=None,
    max_speculative_num_draft_tokens=None,
    speculative_num_steps=None,
    speculative_eagle_topk=None,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        page_size=page_size,
        dcp_size=dcp_size,
        speculative_algorithm=speculative_algorithm,
        max_speculative_num_draft_tokens=max_speculative_num_draft_tokens,
        speculative_num_steps=speculative_num_steps,
        speculative_eagle_topk=speculative_eagle_topk,
    )


def _make_model_config(*, context_len: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(context_len=context_len)


class TestAllocPageSizeUpperBound(CustomTestCase):
    def test_upper_bound_is_the_server_page_size_without_dcp(self):
        """dcp_size is always >= 1, so the non-DCP case must multiply by an exact 1."""
        server_args = _make_server_args(page_size=64, dcp_size=1)

        self.assertEqual(get_alloc_page_size_upper_bound(server_args), 64)

    def test_upper_bound_scales_with_dcp_size(self):
        """DCP hands the plain paged allocator page_size * dcp_size, the largest page any branch takes."""
        server_args = _make_server_args(page_size=64, dcp_size=4)

        self.assertEqual(get_alloc_page_size_upper_bound(server_args), 256)


class TestReqToTokenRowWidth(CustomTestCase):
    def test_row_width_is_ceil_aligned_to_the_page_upper_bound(self):
        """A row ending mid-page cannot hold the final page-aligned allocation."""
        server_args = _make_server_args(page_size=64)
        model_config = _make_model_config(context_len=131074)

        row_width = get_req_to_token_row_width(
            server_args=server_args, model_config=model_config
        )

        self.assertEqual(row_width, 131136)
        self.assertEqual(row_width % 64, 0)

    def test_row_width_covers_the_aligned_context_len_on_a_grid(self):
        """Across pages and context lengths straddling a page edge, the row never truncates."""
        for page_size in (1, 16, 64, 128):
            for context_len in (
                page_size * 8 - 1,
                page_size * 8,
                page_size * 8 + 1,
            ):
                with self.subTest(page_size=page_size, context_len=context_len):
                    server_args = _make_server_args(page_size=page_size)
                    model_config = _make_model_config(context_len=context_len)

                    row_width = get_req_to_token_row_width(
                        server_args=server_args, model_config=model_config
                    )

                    self.assertGreaterEqual(
                        row_width, ceil_align(context_len, page_size)
                    )
                    self.assertEqual(row_width % page_size, 0)

    def test_row_width_adds_no_padding_when_page_size_is_one(self):
        """Unpaged non-speculative configs get exactly the 4-column headroom, with no ceil padding."""
        row_width = get_req_to_token_row_width(
            server_args=_make_server_args(page_size=1),
            model_config=_make_model_config(context_len=4096),
        )

        self.assertEqual(row_width, 4100)


class TestRowWidthCoversDecodeReserve(CustomTestCase):
    def _assert_row_headroom_is(self, server_args, *, expected_headroom: int) -> None:
        model_config = _make_model_config(context_len=4096)

        row_width = get_req_to_token_row_width(
            server_args=server_args, model_config=model_config
        )

        self.assertGreaterEqual(row_width, 4096 + expected_headroom)

    def test_ngram_row_covers_its_decode_reserve(self):
        """NGRAM with draft 12 on page 1 needs 2*12=24 columns; the old headroom was 4+12=16."""
        self._assert_row_headroom_is(
            _make_server_args(
                page_size=1,
                speculative_algorithm="NGRAM",
                max_speculative_num_draft_tokens=12,
            ),
            expected_headroom=24,
        )

    def test_dflash_row_covers_its_decode_reserve(self):
        """DFLASH with block 16 on page 1 needs 2*16=32 columns; the old headroom was 4+16=20."""
        self._assert_row_headroom_is(
            _make_server_args(
                page_size=1,
                speculative_algorithm="DFLASH",
                max_speculative_num_draft_tokens=16,
            ),
            expected_headroom=32,
        )

    def test_eagle_tree_row_covers_its_decode_reserve(self):
        """page 16, steps 3, topk 4: 2 pages per topk => 2*16*4=128, doubled to 256."""
        self._assert_row_headroom_is(
            _make_server_args(
                page_size=16,
                speculative_algorithm="EAGLE",
                max_speculative_num_draft_tokens=8,
                speculative_num_steps=3,
                speculative_eagle_topk=4,
            ),
            expected_headroom=256,
        )

    def test_eagle_tree_row_covers_the_dcp_paged_allocator_reserve(self):
        """A plain paged allocator under DCP runs at page 64, needing 2*(2*64*4)=1024, not the 256 the bare page_size implies."""
        self._assert_row_headroom_is(
            _make_server_args(
                page_size=16,
                dcp_size=4,
                speculative_algorithm="EAGLE",
                max_speculative_num_draft_tokens=8,
                speculative_num_steps=3,
                speculative_eagle_topk=4,
            ),
            expected_headroom=1024,
        )

    def test_page_one_with_dcp_row_covers_the_paged_allocator_reserve(self):
        """page 1 with dcp 4 still yields a paged allocator at page 4: 2 pages per topk => 2*(2*4*4)=64."""
        self._assert_row_headroom_is(
            _make_server_args(
                page_size=1,
                dcp_size=4,
                speculative_algorithm="EAGLE",
                max_speculative_num_draft_tokens=8,
                speculative_num_steps=3,
                speculative_eagle_topk=4,
            ),
            expected_headroom=64,
        )

    def test_row_width_handles_an_unset_draft_token_count(self):
        """A speculative plugin that leaves the draft token count unset must not crash sizing."""
        self._assert_row_headroom_is(
            _make_server_args(
                page_size=1,
                speculative_algorithm="NGRAM",
                max_speculative_num_draft_tokens=None,
            ),
            expected_headroom=2,
        )


class TestDecodeReserveTracksTheAllocatorPage(CustomTestCase):
    def _eagle_args(self, *, page_size: int, dcp_size: int):
        return _make_server_args(
            page_size=page_size,
            dcp_size=dcp_size,
            speculative_algorithm="EAGLE",
            max_speculative_num_draft_tokens=8,
            speculative_num_steps=3,
            speculative_eagle_topk=4,
        )

    def test_reserve_uses_the_page_it_is_given(self):
        """The reserve models the allocator's page-aligned footprint, so it must follow the real page."""
        server_args = self._eagle_args(page_size=16, dcp_size=4)

        self.assertEqual(get_alloc_reserve_per_decode(server_args, page_size=16), 256)
        self.assertEqual(get_alloc_reserve_per_decode(server_args, page_size=64), 1024)

    def test_reserve_defaults_to_the_bare_page_size(self):
        """SWA / HiSparse keep page_size under DCP, so the default must not inflate to the upper bound."""
        server_args = self._eagle_args(page_size=16, dcp_size=4)

        self.assertEqual(get_alloc_reserve_per_decode(server_args), 256)

    def test_candidate_pages_cover_both_allocator_shapes(self):
        """Row sizing must cover whichever allocator the configurator picks, without re-deriving its branches."""
        server_args = self._eagle_args(page_size=16, dcp_size=4)

        self.assertEqual(get_candidate_alloc_page_sizes(server_args), (16, 64))

    def test_candidate_pages_collapse_without_dcp(self):
        """Without DCP there is only one possible page, so covering both costs nothing."""
        server_args = self._eagle_args(page_size=16, dcp_size=1)

        self.assertEqual(get_candidate_alloc_page_sizes(server_args), (16, 16))


class TestAssertAllocWithinRowWidth(CustomTestCase):
    def test_allocation_exactly_filling_the_row_is_allowed(self):
        """The row width is the worst-case allocation endpoint, so equality is legal."""
        assert_alloc_within_row_width(max_alloc_len=128, row_width=128)

    def test_allocation_past_the_row_end_raises(self):
        """One column past the row is an out-of-bounds write into the next request."""
        with self.assertRaises(AssertionError):
            assert_alloc_within_row_width(max_alloc_len=129, row_width=128)


if __name__ == "__main__":
    unittest.main()
