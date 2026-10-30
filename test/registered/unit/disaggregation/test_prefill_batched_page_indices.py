"""Batched send page ids must equal what each send would compute alone."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.common import kv_to_page_indices
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

PS = 4


class _IdentityAllocator(BaseTokenToKVPoolAllocator):
    def __init__(self):
        self.page_size = PS

    alloc = clear = free = None


class _VirtualAllocator(_IdentityAllocator):
    def translate_kv_indices_for_transfer(self, kv_indices):
        return kv_indices + 1


def _scheduler(allocator=None, staging=False, device="cpu"):
    s = object.__new__(SchedulerDisaggregationPrefillMixin)
    s.enable_staging = staging
    s.token_to_kv_pool_allocator = allocator or _IdentityAllocator()
    gen = torch.Generator().manual_seed(0)
    s.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.randperm(8 * 64, generator=gen)
        .reshape(8, 64)
        .to(torch.int32)
        .to(device)
    )
    return s


def _sender(defers=False, max_tokens=None):
    return SimpleNamespace(
        supports_host_tensor_indices=defers,
        get_max_transfer_tokens=lambda: max_tokens,
    )


def _req(pool_idx, start, end, input_len, sender=None):
    return SimpleNamespace(
        kv=SimpleNamespace(req_pool_idx=pool_idx),
        start_send_idx=start,
        extend_end=end,
        origin_input_ids=[0] * input_len,
        disagg_kv_sender=sender or _sender(),
    )


def _last_chunk_ends(reqs):
    return [min(r.extend_end, len(r.origin_input_ids)) for r in reqs]


def _expected(table, req, end):
    return kv_to_page_indices(
        table[
            req.kv.req_pool_idx, req.start_send_idx : max(end, req.start_send_idx)
        ].cpu(),
        PS,
    )


def test_batched_page_ids_match_per_request_reads():
    s = _scheduler()
    reqs = [
        _req(3, 0, 23, 23),  # partial last page
        _req(0, 8, 40, 40),  # resumes after an earlier send
        _req(5, 12, 12, 12),  # nothing left to send
        _req(7, 20, 30, 25),  # input shorter than the extend range
        _req(1, 16, 9, 9),  # already sent past the end
    ]
    ends = _last_chunk_ends(reqs)
    got = s._batched_page_indices(reqs, ends)
    table = s.req_to_token_pool.req_to_token
    for req, end, pages in zip(reqs, ends, got):
        expected = _expected(table, req, end)
        assert pages.dtype == expected.dtype
        np.testing.assert_array_equal(pages, expected)


def test_early_send_ends_select_the_cached_prefix_only():
    s = _scheduler()
    reqs = [_req(2, 0, 40, 40), _req(4, 8, 40, 40)]
    got = s._batched_page_indices(reqs, [16, 24])
    table = s.req_to_token_pool.req_to_token
    for req, end, pages in zip(reqs, [16, 24], got):
        np.testing.assert_array_equal(pages, _expected(table, req, end))


@pytest.mark.parametrize(
    "case",
    ["single", "staging", "virtual_ids", "cp_sharded", "transfer_cap", "mixed_defer"],
)
def test_unbatchable_sends_keep_the_per_request_path(case):
    allocator = _VirtualAllocator() if case == "virtual_ids" else None
    s = _scheduler(allocator, staging=case == "staging")
    reqs = [_req(0, 0, 8, 8), _req(1, 0, 8, 8)]
    if case == "single":
        reqs = reqs[:1]
    if case == "transfer_cap":
        reqs[1].disagg_kv_sender = _sender(max_tokens=4)
    if case == "mixed_defer":
        reqs[1].disagg_kv_sender = _sender(defers=True)
    with patch(
        "sglang.srt.disaggregation.prefill.is_cp_sharded_padded",
        return_value=case == "cp_sharded",
    ):
        assert s._batched_page_indices(reqs, [8] * len(reqs)) is None


def test_precomputed_page_ids_replace_the_per_request_gather():
    """send_kv_chunk ships the batched ids as is, with no row gather of its own."""
    s = _scheduler()
    sender = _sender()
    sender.should_send_kv_chunk = lambda num_pages, is_last: True
    sender.send = Mock()
    req = _req(0, 0, 8, 8, sender)
    req.disagg_decode_prefix_len = 0
    req.rid = "precomputed"
    s.disagg_prefill_pending_chunk_rids = set()
    s.req_to_token_pool.req_to_token = Mock(side_effect=AssertionError)
    precomputed = np.array([11, 12], dtype=np.int32)

    SchedulerDisaggregationPrefillMixin._send_kv_chunk(
        s, req, last_chunk=False, end_idx=8, page_indices=precomputed
    )

    assert sender.send.call_args.args[0] is precomputed
    assert req.start_send_idx == 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_deferred_batched_page_ids_land_behind_the_forward():
    """Deferring senders get the copy in flight; once the stream reaches it,
    the page ids equal the per-request reads."""
    s = _scheduler(device="cuda")
    table = s.req_to_token_pool.req_to_token
    reqs = [
        _req(i, start, end, end, _sender(defers=True))
        for i, (start, end) in enumerate(((0, 23), (8, 40), (20, 30)))
    ]
    ends = _last_chunk_ends(reqs)
    torch.cuda._sleep(50_000_000)  # the copy is still queued when this returns
    got = s._batched_page_indices(reqs, ends)
    torch.cuda.synchronize()
    for req, end, pages in zip(reqs, ends, got):
        np.testing.assert_array_equal(pages, _expected(table, req, end))
