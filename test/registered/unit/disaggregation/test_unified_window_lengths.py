"""A PD decode sizes each window of a unified pool by that window's own tail:
the transfer pairs pages by position, so the tails must match the prefill's."""

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
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.kv.build import UnifiedBuildContext, build_kv_storage
from sglang.srt.mem_cache.kv.decl import (
    MHAGeometry,
    StateGeometry,
    WindowLayers,
    kv_layer_decls,
)
from sglang.srt.mem_cache.kv.plan import plan_kv_storage
from sglang.srt.mem_cache.kv.sizing import KVSizingInputs, size_plan
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import setup_allocator_context

register_cuda_ci(est_time=20, stage="extra-a", runner_config="1-gpu-large")

_MHA = MHAGeometry(head_num=1, head_dim=8, v_head_dim=8, kv_cache_dtype=torch.bfloat16)
_PAGE = 4
# The target's window (the scheduler's sliding window) and two shorter windows
# of their own sub-pools, as draft windows of other lengths are.
_TARGET_WINDOW = 16
_WINDOWS = (_TARGET_WINDOW, 8, 4)


def _storage(windows=_WINDOWS, device="cpu", state=False):
    """One full layer, one layer per window, and with ``state`` one recurrent
    state layer after them."""
    setup_allocator_context()
    state_layer = len(windows) + 1
    decls = kv_layer_decls(
        full_layer_ids=(0,),
        full_geometry=_MHA,
        windows=[
            WindowLayers(layer_ids=(i + 1,), geometry=_MHA, window=w)
            for i, w in enumerate(windows)
        ],
        state_layer_ids=(state_layer,) if state else (),
        state_geometry=(
            StateGeometry(
                conv_state_shapes=((3, 8),),
                conv_dtype=torch.bfloat16,
                temporal_state_shape=(1, 4, 4),
                temporal_dtype=torch.float32,
            )
            if state
            else None
        ),
    )
    plan = plan_kv_storage(decls, page_size=_PAGE)
    sized = size_plan(
        plan,
        KVSizingInputs(
            full_tokens=256,
            window_tokens=256,
            state_slots=8 if state else 0,
            max_running_requests=3,
            context_len=128,
            alignment=1,
        ),
    )
    ctx = UnifiedBuildContext(
        device=device,
        start_layer=0,
        end_layer=len(windows) + 1 + int(state),
        enable_memory_saver=False,
        need_sort=False,
        max_num_reqs=3,
        model_context_len=128,
        lazy_compaction=True,
        state_cache_params=SimpleNamespace(layers=(state_layer,)) if state else None,
    )
    storage = build_kv_storage(plan, sized, ctx)
    storage.bind_window_retention(_TARGET_WINDOW, None)
    return storage


def _window_sub_pools(storage):
    return [e.sub_pool for e in storage.wire_entries()]


def _prefill_pages(storage, seq_len):
    """The pages a prefill sends per window entry for a ``seq_len`` request
    that decode holds no prefix of."""
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
            ).unit_ids
        )
        for sub_pool in _window_sub_pools(storage)
    }
    storage.allocator.free(tokens)
    storage.requests.free(req)
    return pages


class _HostPool:
    # Its own allocation: the receive is priced pool by pool.
    shared_allocation_domain = None

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

    def peer_write_indices(self, indices):
        # Not a shared compacting pool: the peer writes the slots as allocated.
        return indices


def _host_receive_queue(storage, sliding_window_size):
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
            for name, window in storage.retained_window_lengths().items()
        ],
        scheduler=SimpleNamespace(
            sliding_window_size=sliding_window_size,
            metrics_reporter=SimpleNamespace(current_scheduler_metrics_enabled=False),
        ),
        token_to_kv_pool_allocator=SimpleNamespace(page_size=_PAGE),
        _send_kv_metadata=Mock(),
    )
    queue._host_receive_keeps_reserve = lambda num_tokens, window_num_tokens: (
        DecodePreallocQueue._host_receive_keeps_reserve(
            queue, num_tokens, window_num_tokens
        )
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
        a request keeps; the prefill sends that tail and the decode receives it."""
        storage = _storage()
        storage.bind_window_retention(_TARGET_WINDOW, {"window_8": 13})
        self.assertEqual(
            storage.retained_window_lengths(),
            {"window_16": 16, "window_8": 13, "window_4": 4},
        )
        for seq_len in (5, 23, 37, 64):
            with self.subTest(seq_len=seq_len):
                sent = _prefill_pages(storage, seq_len)
                self.assertEqual(
                    sent["window_8"],
                    -(-seq_len // _PAGE) - max(0, seq_len - 13) // _PAGE,
                )
                queue, _ = _host_receive_queue(storage, _TARGET_WINDOW)
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

    def test_each_layer_routes_to_its_window(self):
        """A window layer's route carries its own window; a full layer's none."""
        storage = _storage()
        router = storage.allocator.get_kvcache().layer_router
        for i, window in enumerate(_WINDOWS):
            name = f"window_{window}"
            self.assertEqual(router.sub_pool(name).window, window)
            self.assertEqual(router.layer(i + 1).route.window, window)
        self.assertIsNone(router.layer(0).route.window)

    def test_draft_window_sidecar_follows_the_target_window_host_pool(self):
        """BUG REGRESSION. A non-fused window draft's HiCache sidecar must use
        the target window's own host pool, not ``swa``, or startup raises KeyError."""
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler

        storage = _storage()
        target = storage.allocator.target_window_pool()
        cache = SimpleNamespace(
            token_to_kv_pool_allocator=storage.allocator,
            req_to_token_pool=storage.requests,
        )
        host = assembler.target_window_host_pool(cache)
        self.assertEqual(host, storage.host_pool_name(target))
        self.assertNotEqual(host, "swa")
        target_host = SimpleNamespace(size=64, page_size=4, layout="layer_first")
        cache.cache_controller = SimpleNamespace(
            mem_pool_host=SimpleNamespace(
                entry_map={
                    storage.host_pool_name(storage.owner): SimpleNamespace(),
                    host: SimpleNamespace(host_pool=target_host),
                }
            )
        )
        draft_window = SimpleNamespace(
            layer_num=1,
            size=32,
            page_size=4,
            get_device_pool_infos=lambda: [SimpleNamespace(buffer_info=None)],
        )
        with (
            patch.object(assembler, "_get_allocator_type", return_value="default"),
            patch.object(
                assembler,
                "_build_mha_mla_host_pool",
                return_value=SimpleNamespace(layer_num=1),
            ) as build_host,
            patch.object(assembler, "build_pool_entry", return_value="entry"),
        ):
            specs, entries = assembler.build_swa_draft_pools(
                draft_kv_pool=SimpleNamespace(swa_kv_pool=draft_window),
                tree_cache=cache,
                target_window_host=host,
            )
        self.assertEqual(entries, ["entry"])
        self.assertEqual(specs[0].indices_from_pool, host)
        self.assertEqual(build_host.call_args.kwargs["page_size"], 4)

    def test_window_retirement_names_its_window(self):
        """A pool of several windows has no default window to retire."""
        storage = _storage()
        req = SimpleNamespace(kv=ReqKvInfo())
        storage.acquire([req])
        with self.assertRaisesRegex(ValueError, "window's sub-pool"):
            storage.advance(req, 20, window=_TARGET_WINDOW)
        storage.requests.free(req)

    def test_checksum_digests_each_window_entry_by_its_own_rows(self):
        storage = _storage()
        allocator = storage.allocator
        for reaches, seq_len in product(({}, {"window_8": 13}), (23, 37)):
            storage.bind_window_retention(_TARGET_WINDOW, reaches)
            scheduler = SimpleNamespace(
                token_to_kv_pool_allocator=allocator,
                req_to_token_pool=storage.requests,
                sliding_window_size=_TARGET_WINDOW,
            )
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
                    )
                    torch.testing.assert_close(
                        entry_rows, sent.unit_ids.to(entry_rows.dtype)
                    )
                allocator.free(tokens)
                storage.requests.free(req)

    def test_checksum_digests_recurrent_state_by_its_own_slots(self):
        """BUG REGRESSION. Beside a window, the recurrent-state entry must digest
        the request's state slots, not the window's rows."""
        storage = _storage(windows=(_TARGET_WINDOW,), state=True)
        allocator = storage.allocator
        scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=allocator,
            req_to_token_pool=storage.requests,
            sliding_window_size=_TARGET_WINDOW,
        )
        state_names = storage.state_sub_pools()
        self.assertEqual(len(state_names), 1)
        entries = [e.sub_pool for e in storage.wire_entries()]
        self.assertIn(state_names[0], entries)
        self.assertTrue(set(entries) - set(state_names))
        for seq_len in (23, 37):
            with self.subTest(seq_len=seq_len):
                req = SimpleNamespace(kv=ReqKvInfo())
                row = storage.acquire([req])[0]
                tokens = allocator.alloc(-(-seq_len // _PAGE) * _PAGE)
                storage.requests.req_to_token[row, : len(tokens)] = tokens.to(
                    torch.int32
                )
                rows = state_indices_for_request(scheduler, req, seq_len)
                self.assertEqual(len(rows), len(entries))
                for name, entry_rows in zip(entries, rows):
                    if name in state_names:
                        slots = allocator.state_ids(name).ids(
                            storage.requests.req_index_to_mamba_index_mapping[row]
                        )
                        sent = slots.reshape(-1)
                    else:
                        sent = storage.resume_units(
                            name, req, token_start=0, token_boundary=seq_len
                        ).unit_ids
                    torch.testing.assert_close(entry_rows, sent.to(entry_rows.dtype))
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
                tails = storage.window_tails(seq_len, keep_insert_boundary=False)
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
                    tail = tails[name]
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
                        req.kv.get_evicted_seqlen(name),
                        seq_len - tail,
                        name,
                    )
                allocator.free(kv_loc)
                storage.requests.free(req)


if __name__ == "__main__":
    unittest.main()
