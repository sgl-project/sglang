import sys
from types import SimpleNamespace

import pytest

from sglang.srt.disaggregation.dflash_kv import (
    draft_transfer_buffers,
    draft_transfer_start,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDraftTransferRegistration(CustomTestCase):
    def test_only_full_nixl_dflash_pools_use_logical_transfer(self):
        from sglang.srt.disaggregation.base.conn import StateType
        from sglang.srt.disaggregation.utils import setup_state_kv_args
        from sglang.srt.mem_cache.kv_cache_builder import get_draft_kv_pool
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        for backend, compact in (("nixl", False), ("nixl", True), ("mooncake", False)):
            with (
                self.subTest(backend=backend, compact=compact),
                published_topology(disaggregation_transfer_backend=backend) as args,
            ):
                pool = MHATokenToKVPool.__new__(MHATokenToKVPool)
                pool.page_size = 128
                pool.get_contiguous_buf_infos = lambda: ([1000], [8192], [1024])
                runner = SimpleNamespace(
                    token_to_kv_pool=pool,
                    model_config=SimpleNamespace(
                        hf_text_config=SimpleNamespace(sliding_window=4096)
                    ),
                )
                worker = SimpleNamespace(
                    use_compact_draft_cache=compact,
                    draft_worker=SimpleNamespace(draft_runner=runner),
                )
                selected = get_draft_kv_pool(
                    draft_worker=worker,
                    spec_algorithm=SpeculativeAlgorithm.DFLASH,
                    server_args=args,
                )
                kv_args = SimpleNamespace(page_size=64)
                setup_state_kv_args(kv_args, SimpleNamespace(), selected)
                enabled = backend == "nixl" and not compact
                self.assertEqual(
                    kv_args.state_types, [StateType.DFLASH_KV] if enabled else []
                )
                if enabled:
                    self.assertEqual(kv_args.state_item_lens, [[512]])
                    self.assertEqual(selected._pd_dflash_window, 4096)


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
