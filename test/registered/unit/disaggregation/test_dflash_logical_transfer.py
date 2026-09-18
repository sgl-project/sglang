import sys
from types import SimpleNamespace

import pytest

from sglang.srt.disaggregation.dflash_kv import (
    draft_transfer_buffers,
    draft_transfer_start,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize("dcp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("wire_page", [16, 64])
def test_wire_page_geometry_is_independent_of_dcp(dcp_size, wire_page):
    bytes_per_token = 512
    allocation_page = wire_page * dcp_size
    pool = SimpleNamespace(
        page_size=allocation_page,
        get_contiguous_buf_infos=lambda: (
            [1000, 2000],
            [1048576, 1048576],
            [allocation_page * bytes_per_token] * 2,
        ),
    )
    ptrs, lengths, items = draft_transfer_buffers(pool, wire_page)
    assert ptrs == [1000, 2000]
    assert lengths == [1048576, 1048576]
    assert items == [wire_page * bytes_per_token] * 2
    # Raw logical locations, unlike DCP-local target slots, address every token.
    logical_page = 3
    assert ptrs[0] + logical_page * items[0] == 1000 + 3 * wire_page * bytes_per_token


@pytest.mark.parametrize(
    "seq_len,window,page,expected",
    [
        (10, None, 64, 0),
        (8192, None, 64, 0),
        (8192, 4096, 64, 4096),
        (8193, 4096, 64, 4096),
        (2048, 4096, 64, 0),
        (0, 4096, 64, 0),
    ],
)
def test_transfer_covers_full_prefix_or_page_aligned_draft_window(
    seq_len, window, page, expected
):
    assert draft_transfer_start(seq_len, window, page) == expected


@pytest.mark.parametrize("seq,window,page", [(-1, None, 64), (1, 0, 64), (1, None, 0)])
def test_invalid_extent_rejected(seq, window, page):
    with pytest.raises(ValueError):
        draft_transfer_start(seq, window, page)


def test_incompatible_page_geometry_rejected():
    pool = SimpleNamespace(
        page_size=64, get_contiguous_buf_infos=lambda: ([1], [512], [512])
    )
    with pytest.raises(ValueError):
        draft_transfer_buffers(pool, 128)
    with pytest.raises(ValueError):
        draft_transfer_buffers(pool, 0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
