"""A decode node sizes each window of a unified pool by that window's own length.

A unified pool may hold windows of several lengths (a target window and a
draft window of another length, each its own sub-pool). The prefill sends,
for each window entry, the tail its own window keeps. The decode must receive
exactly that tail per entry: the transfer pairs source and destination pages
by position, so a destination sized by another window's length pairs pages of
different token positions.
"""

import unittest
from array import array
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.checksum import state_indices_for_request
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    alloc_for_decode_prealloc,
)
from sglang.srt.disaggregation.utils import (
    check_state_wire,
    retained_window_lengths,
    state_wire,
)
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.kv.build import UnifiedBuildContext, build_kv_storage
from sglang.srt.mem_cache.kv.decl import MHAGeometry, WindowLayers, kv_layer_decls
from sglang.srt.mem_cache.kv.plan import plan_kv_storage
from sglang.srt.mem_cache.kv.sizing import KVSizingInputs, size_plan
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import setup_allocator_context

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_MHA = MHAGeometry(head_num=1, head_dim=8, v_head_dim=8, kv_cache_dtype=torch.bfloat16)
_PAGE = 4
# The target's window (the scheduler's sliding window) and two shorter windows
# of their own sub-pools, as draft windows of other lengths are.
_TARGET_WINDOW = 16
_WINDOWS = (_TARGET_WINDOW, 8, 4)


def _storage(windows=_WINDOWS, device="cpu"):
    setup_allocator_context()
    decls = kv_layer_decls(
        full_layer_ids=(0,),
        full_geometry=_MHA,
        windows=[
            WindowLayers(layer_ids=(i + 1,), geometry=_MHA, window=w)
            for i, w in enumerate(windows)
        ],
    )
    plan = plan_kv_storage(decls, page_size=_PAGE)
    sized = size_plan(
        plan,
        KVSizingInputs(
            full_tokens=256,
            window_tokens=256,
            state_slots=0,
            max_running_requests=3,
            context_len=128,
            alignment=1,
        ),
    )
    ctx = UnifiedBuildContext(
        device=device,
        start_layer=0,
        end_layer=len(windows) + 1,
        enable_memory_saver=False,
        need_sort=False,
        max_num_reqs=3,
        model_context_len=128,
        lazy_compaction=True,
    )
    return build_kv_storage(plan, sized, ctx)


def _window_sub_pools(storage):
    return [e.sub_pool for e in storage.wire_entries()]


def _prefill_pages(storage, seq_len, reaches=None):
    """The pages a prefill sends per window entry for a ``seq_len`` request
    that decode holds no prefix of."""
    kept = retained_window_lengths(storage.allocator, reaches, _TARGET_WINDOW)
    req = SimpleNamespace(kv=ReqKvInfo())
    row = storage.acquire([req])[0]
    tokens = storage.allocator.alloc(-(-seq_len // _PAGE) * _PAGE)
    storage.requests.req_to_token[row, : len(tokens)] = tokens.to(torch.int32)
    pages = {
        sub_pool: len(
            storage.resume_units(
                sub_pool,
                req,
                token_start=0,
                token_boundary=seq_len,
                retain=kept.get(sub_pool),
            ).unit_ids
        )
        for sub_pool in _window_sub_pools(storage)
    }
    storage.allocator.free(tokens)
    storage.requests.free(req)
    return pages


class _HostPool:
    def __init__(self, page_size=_PAGE):
        self.page_size = page_size
        self.taken = []

    def available_size(self):
        return 1 << 20

    def alloc(self, n):
        self.taken.append(n)
        return torch.arange(n, dtype=torch.int64)

    def free(self, indices):
        pass


def _host_receive_queue(storage, sliding_window_size, reaches=None):
    sub_pools = _window_sub_pools(storage)
    window_pools = {name: _HostPool() for name in sub_pools}
    queue = SimpleNamespace(
        host_pool=_HostPool(),
        host_reserved_tokens=0,
        tree_cache=SimpleNamespace(
            disable=True,
            sidecar_pool_specs=[],
            host_pool_group=SimpleNamespace(get_entry=lambda name: None),
        ),
        host_window_entries=[
            (name, window_pools[name], 0, window)
            for name, window in DecodePreallocQueue._window_lengths_of(
                storage, sliding_window_size, reaches
            ).items()
        ],
        scheduler=SimpleNamespace(
            sliding_window_size=sliding_window_size,
            metrics_reporter=SimpleNamespace(current_scheduler_metrics_enabled=False),
        ),
        token_to_kv_pool_allocator=SimpleNamespace(page_size=_PAGE),
        _send_kv_metadata=Mock(),
    )
    return queue, window_pools


class TestUnifiedWindowLengths(CustomTestCase):
    def test_each_window_entry_receives_its_own_tail_on_host(self):
        storage = _storage()
        for seq_len in (5, 23, 37, 64):
            with self.subTest(seq_len=seq_len):
                sent = _prefill_pages(storage, seq_len)
                queue, _ = _host_receive_queue(storage, _TARGET_WINDOW)
                req = SimpleNamespace(
                    origin_input_ids=array("q", [1] * seq_len), kv=ReqKvInfo()
                )
                decode_req = SimpleNamespace(req=req)
                with patch(
                    "sglang.srt.disaggregation.decode.get_disagg",
                    return_value=SimpleNamespace(
                        disaggregation_decode_enable_radix_cache=False
                    ),
                ):
                    self.assertTrue(
                        DecodePreallocQueue._pre_alloc_host(queue, decode_req)
                    )
                state_indices = queue._send_kv_metadata.call_args.args[3]
                received = {
                    name: len(pages)
                    for (name, *_), pages in zip(
                        queue.host_window_entries, state_indices
                    )
                }
                self.assertEqual(received, sent)

    def test_a_reader_reaching_further_back_widens_what_both_peers_carry(self):
        """A fused draft reading front rows before its window widens the tail
        a request keeps of that window. The prefill sends that tail and the
        decode receives all of it; peers keeping different tails refuse to
        pair."""
        storage = _storage()
        reaches = {"window_8": 13}
        self.assertEqual(
            retained_window_lengths(storage.allocator, reaches, _TARGET_WINDOW),
            {"window_16": 16, "window_8": 13, "window_4": 4},
        )
        for seq_len in (5, 23, 37, 64):
            with self.subTest(seq_len=seq_len):
                sent = _prefill_pages(storage, seq_len, reaches)
                self.assertEqual(
                    sent["window_8"],
                    -(-seq_len // _PAGE) - max(0, seq_len - 13) // _PAGE,
                )
                queue, _ = _host_receive_queue(storage, _TARGET_WINDOW, reaches)
                req = SimpleNamespace(
                    origin_input_ids=array("q", [1] * seq_len), kv=ReqKvInfo()
                )
                with patch(
                    "sglang.srt.disaggregation.decode.get_disagg",
                    return_value=SimpleNamespace(
                        disaggregation_decode_enable_radix_cache=False
                    ),
                ):
                    self.assertTrue(
                        DecodePreallocQueue._pre_alloc_host(
                            queue, SimpleNamespace(req=req)
                        )
                    )
                state_indices = queue._send_kv_metadata.call_args.args[3]
                received = {
                    name: len(pages)
                    for (name, *_), pages in zip(
                        queue.host_window_entries, state_indices
                    )
                }
                self.assertEqual(received, sent)

        def wire(tails):
            kv_args = SimpleNamespace(
                state_types=["swa"] * 3,
                state_sub_pools=list(tails),
                state_window_tails=list(tails.values()),
            )
            return state_wire(kv_args)

        widened = wire({"window_16": 16, "window_4": 4, "window_8": 13})
        narrow = wire({"window_16": 16, "window_4": 4, "window_8": 8})
        check_state_wire(widened, widened)
        # A prefill keeping more of a window sends its decode's tail; one
        # keeping less cannot.
        check_state_wire(widened, narrow)
        with self.assertRaisesRegex(RuntimeError, "shorter window tail"):
            check_state_wire(narrow, widened)

    def test_window_retirement_names_its_window(self):
        """A pool of several windows has no default window to retire."""
        storage = _storage()
        req = SimpleNamespace(kv=ReqKvInfo())
        storage.acquire([req])
        with self.assertRaisesRegex(ValueError, "tree component"):
            storage.advance(req, 20, window=_TARGET_WINDOW)
        storage.requests.free(req)

    def test_checksum_digests_each_window_entry_by_its_own_rows(self):
        storage = _storage()
        allocator = storage.allocator
        for reaches, seq_len in product(({}, {"window_8": 13}), (23, 37)):
            scheduler = SimpleNamespace(
                token_to_kv_pool_allocator=allocator,
                req_to_token_pool=storage.requests,
                sliding_window_size=_TARGET_WINDOW,
                tree_cache=SimpleNamespace(draft_window_reaches=reaches),
            )
            tails = retained_window_lengths(allocator, reaches, _TARGET_WINDOW)
            with self.subTest(reaches=reaches, seq_len=seq_len):
                req = SimpleNamespace(kv=ReqKvInfo())
                row = storage.acquire([req])[0]
                tokens = allocator.alloc(-(-seq_len // _PAGE) * _PAGE)
                storage.requests.req_to_token[row, : len(tokens)] = tokens.to(
                    torch.int32
                )
                rows = state_indices_for_request(scheduler, req, seq_len)
                self.assertEqual(len(rows), len(storage.wire_entries()))
                for entry, entry_rows in zip(storage.wire_entries(), rows):
                    sent = storage.resume_units(
                        entry.sub_pool,
                        req,
                        token_start=0,
                        token_boundary=seq_len,
                        retain=tails.get(entry.sub_pool),
                    )
                    torch.testing.assert_close(
                        entry_rows, sent.unit_ids.to(entry_rows.dtype)
                    )
                allocator.free(tokens)
                storage.requests.free(req)

    @unittest.skipUnless(torch.cuda.is_available(), "the paged take runs on CUDA")
    def test_prealloc_binds_each_window_its_own_tail(self):
        storage = _storage(device="cuda")
        allocator = storage.allocator
        for seq_len in (5, 23, 37, 64):
            with self.subTest(seq_len=seq_len):
                req = SimpleNamespace(kv=ReqKvInfo())
                row = storage.acquire([req])[0]
                tails = DecodePreallocQueue._window_tails_of(
                    storage, _TARGET_WINDOW, seq_len, keep_insert_boundary=False
                )
                kv_loc = alloc_for_decode_prealloc(
                    storage,
                    allocator,
                    req=req,
                    fill_len=seq_len,
                    delta_len=seq_len,
                    prefix_len=0,
                    total_prefix_len=0,
                    prefix_indices=None,
                    uses_swa_tail=True,
                    window_tails=tails,
                )
                self.assertIsNotNone(kv_loc)
                storage.requests.req_to_token[row, :seq_len] = kv_loc.to(torch.int32)
                sent = _prefill_pages(_storage(device="cuda"), seq_len)
                for band in allocator.window_allocators.values():
                    name = band.sub_pool_name
                    ids = allocator.id_spaces()[name].ids(kv_loc.to(torch.int64))
                    bound = int((ids > 0).sum())
                    tail = DecodePreallocQueue._window_tails_of(
                        storage, _TARGET_WINDOW, seq_len, keep_insert_boundary=False
                    )[name]
                    # Each window binds the tokens of its own tail, no more.
                    self.assertEqual(bound, tail, name)
                    # The pages the decode receives for the window are all
                    # bound, as many as the prefill sends.
                    resume = storage.resume_units(
                        name, req, token_start=0, token_boundary=seq_len
                    )
                    self.assertTrue(bool(torch.all(resume.unit_ids > 0)), name)
                    self.assertEqual(len(resume.unit_ids), sent[name], name)
                    self.assertEqual(
                        req.kv.get_evicted_seqlen(allocator.window_component(band)),
                        seq_len - tail,
                        name,
                    )
                allocator.free(kv_loc)
                storage.requests.free(req)


if __name__ == "__main__":
    unittest.main()
