"""Compact beam page addressing, ancestry, snapshots, and terminal ownership."""

import itertools
import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.beam_search.paged_kv import GRPagedKVState
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class PageAllocator:
    """Nonadjacent physical pages expose accidental contiguous-arena assumptions."""

    def __init__(self, page_size, page_count=192):
        self.page_size = page_size
        self.size = page_count * page_size
        self.device = "cpu"
        self.storage = None
        self.free_pages = torch.arange(page_count, dtype=torch.int64) * 3 + 2
        self.allocations = []
        self.frees = []

    def available_size(self):
        return self.free_pages.numel() * self.page_size

    def alloc(self, size):
        assert size % self.page_size == 0
        count = size // self.page_size
        if count > self.free_pages.numel():
            return None
        pages = self.free_pages[:count]
        self.free_pages = self.free_pages[count:]
        self.allocations.append(pages.clone())
        return (
            pages[:, None] * self.page_size + torch.arange(self.page_size)[None, :]
        ).flatten()

    def free(self, slots):
        self.free_page_ids((slots // self.page_size).unique())

    def free_page_ids(self, pages):
        assert not torch.isin(pages, self.free_pages).any(), "double free"
        assert pages.unique().numel() == pages.numel(), "duplicate page IDs"
        self.frees.append(pages.clone())
        self.free_pages = torch.cat((self.free_pages, pages))


def context_locations(length, page_size):
    pages = torch.arange((length + page_size - 1) // page_size) * 7 + 1000
    return (pages[:, None] * page_size + torch.arange(page_size)).flatten()[:length]


class TestGRPagedKV(unittest.TestCase):
    def test_layout_and_ancestry_with_distinct_values(self):
        # Exercise independent context/beam page tails, multiple pages per step,
        # noncontiguous physical pages, duplicate parents, and retained pruning.
        for page_size, width, depth in itertools.product(
            (1, 4, 16, 32), (1, 2, 4, 16, 17, 33), (2, 3, 5)
        ):
            lengths = {1, max(1, page_size - 1), page_size, page_size + 1}
            for context_len in lengths:
                with self.subTest(p=page_size, w=width, d=depth, c=context_len):
                    allocator = PageAllocator(page_size)
                    before = allocator.available_size()
                    state = GRPagedKVState(context_len, width, depth, page_size)
                    self.assertEqual(state.reserved_slots, 0)
                    state.reserve(allocator)
                    expected_slots = (
                        (depth - 1) * ((width + page_size - 1) // page_size) * page_size
                    )
                    self.assertEqual(state.reserved_slots, expected_slots)
                    self.assertEqual(
                        before - allocator.available_size(), expected_slots
                    )
                    context = context_locations(context_len, page_size)
                    pool = torch.full((int(context.max()) + page_size + 1,), -1)
                    pool[context] = 100000 + torch.arange(context_len)
                    paths = [[] for _ in range(width)]
                    written = set()
                    for step in range(depth - 1):
                        metadata = state.prepare_decode(context)
                        self.assertEqual(metadata.step, step)
                        self.assertEqual(metadata.topk_indices.shape, (width, step + 1))
                        current_slots = set(metadata.out_cache_loc.tolist())
                        self.assertEqual(len(current_slots), width)
                        self.assertFalse(current_slots & written)
                        self.assertFalse(current_slots & set(context.tolist()))
                        self.assertEqual(int(metadata.out_cache_loc[0]) % page_size, 0)
                        written.update(current_slots)
                        values = 1000 * (step + 1) + torch.arange(width)
                        pool[metadata.out_cache_loc] = values
                        current_paths = [
                            paths[b] + [int(values[b])] for b in range(width)
                        ]
                        torch.testing.assert_close(
                            pool[metadata.beam_locations()],
                            torch.tensor(current_paths),
                            rtol=0,
                            atol=0,
                        )
                        torch.testing.assert_close(
                            pool[metadata.context_locs],
                            100000 + torch.arange(context_len),
                            rtol=0,
                            atol=0,
                        )
                        # First selection duplicates parent zero; later selections
                        # permute/prune its children. Old metadata must remain valid.
                        parents = (
                            torch.zeros(width, dtype=torch.int64)
                            if step == 0
                            else (torch.arange(width) * 3 + step) % width
                        )
                        old_indices = metadata.topk_indices.clone()
                        state.advance(parents)
                        paths = [current_paths[parent] for parent in parents.tolist()]
                        torch.testing.assert_close(metadata.topk_indices, old_indices)
                        torch.testing.assert_close(
                            pool[metadata.beam_locations()], torch.tensor(current_paths)
                        )
                    state.release(allocator.free)
                    state.release(allocator.free)
                    self.assertEqual(allocator.available_size(), before)
                    self.assertEqual(len(allocator.frees), 1)
                    self.assertEqual(state.reserved_slots, 0)
                    self.assertTrue(state.released)

    def test_small_duplicate_parent_example(self):
        allocator = PageAllocator(4)
        allocator.free_pages = torch.tensor([2, 3])
        state = GRPagedKVState(3, 2, 3, 4)
        state.reserve(allocator)
        first = state.prepare_decode(torch.tensor([4, 5, 6]))
        self.assertEqual(first.out_cache_loc.tolist(), [8, 9])
        state.advance(torch.tensor([0, 0]))
        second = state.prepare_decode(torch.tensor([4, 5, 6]))
        self.assertEqual(second.out_cache_loc.tolist(), [12, 13])
        self.assertEqual(second.beam_locations().tolist(), [[8, 12], [8, 13]])
        self.assertEqual(first.beam_locations().tolist(), [[8], [9]])
        state.release(allocator.free)
        self.assertEqual(allocator.frees[0].tolist(), [2, 3])

    def test_context_mapping_is_forward_owned(self):
        allocator = PageAllocator(4)
        state = GRPagedKVState(5, 2, 3, 4)
        state.reserve(allocator)
        context = torch.tensor([400, 401, 402, 403, 800])
        first = state.prepare_decode(context)
        context.copy_(torch.tensor([1200, 1201, 1202, 1203, 1600]))
        self.assertEqual(first.context_locs.tolist(), [400, 401, 402, 403, 800])
        self.assertEqual(first.context_pages.tolist(), [100, 200])
        # A later checkpoint may remap the leader; a new forward sees that map.
        state.advance(torch.tensor([1, 0]))
        second = state.prepare_decode(context)
        self.assertEqual(second.context_pages.tolist(), [300, 400])
        state.release(allocator.free)

    def test_depth_one_has_no_beam_allocation(self):
        for page_size in (1, 4, 16, 32):
            allocator = PageAllocator(page_size)
            state = GRPagedKVState(7, 16, 1, page_size)
            state.reserve(allocator)
            self.assertTrue(state.has_reservation)
            self.assertEqual(state.arena_slots, 0)
            self.assertEqual(state.reserved_slots, 0)
            self.assertEqual(allocator.allocations, [])
            with self.assertRaisesRegex(RuntimeError, "no remaining decode"):
                state.prepare_decode(context_locations(7, page_size))
            state.release(allocator.free)
            state.release(allocator.free)
            self.assertFalse(state.has_reservation)
            self.assertTrue(state.released)
            self.assertEqual(allocator.frees, [])

    def test_exhaustion_has_no_partial_reservation(self):
        allocator = PageAllocator(4, page_count=1)
        state = GRPagedKVState(3, 5, 3, 4)
        before = allocator.available_size()
        with self.assertRaisesRegex(MemoryError, "16 slots"):
            state.reserve(allocator)
        self.assertEqual(allocator.available_size(), before)
        self.assertFalse(state.has_reservation)
        self.assertEqual(state.reserved_slots, 0)
        self.assertIsNone(state.beam_pages)
        state.release(allocator.free)
        self.assertEqual(allocator.frees, [])

    def test_metadata_preparation_failure_rolls_back_allocated_pages(self):
        allocator = PageAllocator(16)
        before = allocator.available_size()
        state = GRPagedKVState(17, 17, 3, 16)
        with (
            patch(
                "sglang.srt.beam_search.paged_kv.torch.empty",
                side_effect=RuntimeError("injected metadata allocation failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "injected metadata"),
        ):
            state.reserve(allocator)
        self.assertEqual(allocator.available_size(), before)
        self.assertEqual(len(allocator.frees), 1)
        self.assertFalse(state.has_reservation)
        self.assertIsNone(state.beam_pages)
        self.assertIsNone(state.history)
        state.release(allocator.free)
        self.assertEqual(len(allocator.frees), 1)

    def test_early_exit_and_repeated_cleanup(self):
        for stage in ("queued", "prefill", "decode", "selected"):
            allocator = PageAllocator(16)
            before = allocator.available_size()
            state = GRPagedKVState(17, 17, 5, 16)
            if stage != "queued":
                state.reserve(allocator)
            if stage in ("decode", "selected"):
                state.prepare_decode(context_locations(17, 16))
            if stage == "selected":
                state.advance(torch.arange(17).flip(0))
            state.release(allocator.free)
            state.release(allocator.free)
            self.assertEqual(allocator.available_size(), before)
            self.assertEqual(len(allocator.frees), int(stage != "queued"))
            self.assertEqual(state.reserved_slots, 0)
            with self.assertRaises(RuntimeError):
                state.reserve(allocator)

    def test_contract_errors(self):
        for values in ((0, 2, 3, 4), (3, 0, 3, 4), (3, 2, 0, 4), (3, 2, 3, 0)):
            with self.assertRaises(ValueError):
                GRPagedKVState(*values)
        state = GRPagedKVState(3, 2, 3, 4)
        with self.assertRaisesRegex(ValueError, "page sizes differ"):
            state.reserve(PageAllocator(16))
        allocator = PageAllocator(4)
        state.reserve(allocator)
        with self.assertRaises(RuntimeError):
            state.reserve(allocator)
        with self.assertRaisesRegex(RuntimeError, "before advancing"):
            state.advance(torch.tensor([0, 1]))
        with self.assertRaisesRegex(ValueError, "context locations"):
            state.prepare_decode(torch.tensor([4, 5]))
        state.prepare_decode(torch.tensor([4, 5, 6]))
        with self.assertRaisesRegex(ValueError, "fixed number"):
            state.advance(torch.tensor([0]))
        state.advance(torch.tensor([0, 1], dtype=torch.int32))
        state.prepare_decode(torch.tensor([4, 5, 6]))
        state.advance(torch.tensor([0, 1]))
        with self.assertRaisesRegex(RuntimeError, "no remaining decode"):
            state.prepare_decode(torch.tensor([4, 5, 6]))
        state.release(allocator.free)

    def test_group_accounting_and_cleanup_before_and_after_members(self):
        from sglang.srt.beam_search.beam_group import BeamGroup
        from sglang.srt.beam_search.fork import free_member_rows

        for spawn in (False, True):
            allocator = PageAllocator(4)
            before = allocator.available_size()
            group = BeamGroup(beam_width=2, max_new_tokens=3)
            group.prompt_len = 3
            group.leader = SimpleNamespace(
                kv=SimpleNamespace(kv_committed_len=3, kv_allocated_len=3)
            )
            group.gr_paged_kv = GRPagedKVState(3, 2, 3, 4)
            group.gr_paged_kv.reserve(allocator)
            self.assertEqual(group.extra_uncached_tokens(), 8)
            table = torch.arange(32).reshape(2, 16)
            old_table = table.clone()
            freed_rows = []
            pool = SimpleNamespace(req_to_token=table, free_rows=freed_rows.extend)
            if spawn:
                group.member_rows = torch.tensor([1])
                group.member_rows_cpu = torch.tensor([1])
                group.all_rows = torch.tensor([0, 1])
            storage = SimpleNamespace(
                requests=pool, free_tokens=allocator.free, release_rows=pool.free_rows
            )
            free_member_rows(group, storage)
            free_member_rows(group, storage)
            self.assertEqual(group.extra_uncached_tokens(), 0)
            self.assertEqual(allocator.available_size(), before)
            self.assertEqual(len(allocator.frees), 1)
            self.assertEqual(freed_rows, [1] if spawn else [])
            torch.testing.assert_close(table, old_table)
            self.assertEqual(group.leader.kv.kv_allocated_len, 3)

    def test_survivors_update_ancestry_without_request_suffix_mapping(self):
        from sglang.srt.beam_search.beam_group import BeamGroup
        from sglang.srt.beam_search.coordinator import BeamCoordinator

        allocator = PageAllocator(4)
        table = torch.arange(32).reshape(2, 16)
        old_table = table.clone()
        group = BeamGroup(beam_width=2, max_new_tokens=3)
        group.prompt_len = 3
        group.leader = SimpleNamespace(
            output_ids=[], kv=SimpleNamespace(kv_committed_len=3)
        )
        group.all_rows = torch.tensor([0, 1])
        group.gr_paged_kv = GRPagedKVState(3, 2, 3, 4)
        group.gr_paged_kv.reserve(allocator)
        group.gr_paged_kv.prepare_decode(torch.tensor([400, 401, 402]))
        coordinator = BeamCoordinator(
            model_config=None,
            spec_algorithm=None,
            dllm_enabled=False,
            max_req_len=8192,
            req_to_token_pool=SimpleNamespace(req_to_token=table),
            token_to_kv_pool_allocator=allocator,
            tree_cache=None,
            future_map=None,
        )
        with patch.object(BeamCoordinator, "_stash_next_tokens") as stash:
            coordinator._apply_survivors(
                group, torch.tensor([7, 8]), torch.tensor([0, 0]), 1
            )
        self.assertEqual(group.gr_paged_kv.history.tolist(), [[0], [0]])
        self.assertEqual(group.leader.output_ids, [0])
        self.assertEqual(group.pending_orphans, [])
        torch.testing.assert_close(table, old_table)
        stash.assert_called_once()
        coordinator._num_live_groups = 1
        coordinator._retire_group(group)
        coordinator._retire_group(group)
        self.assertTrue(group.gr_paged_kv.released)
        self.assertEqual(coordinator._num_live_groups, 0)
        self.assertEqual(len(allocator.frees), 1)

    def test_admission_gate_budget_and_context_only_prefix_policy(self):
        from sglang.srt.beam_search.coordinator import BeamCoordinator
        from sglang.srt.sampling.sampling_params import SamplingParams

        for enabled, page_size, page_count, max_prefill, expected in (
            (True, 16, 8, 128, None),
            (False, 16, 8, 128, "requires --page-size 1"),
            (True, 16, 5, 128, "needs 96 KV slots"),
            (True, 16, 8, 16, "full prompt"),
            (False, 1, 128, 128, None),
        ):
            with self.subTest(enabled=enabled, p=page_size, capacity=page_count):
                req = SimpleNamespace(
                    sampling_params=SamplingParams(
                        beam_width=17, n=17, max_new_tokens=3
                    ),
                    session_id=None,
                    lora_id=None,
                    origin_input_ids=[11] * 17,
                    eos_token_ids=[0],
                    tokenizer=None,
                )
                recv = SimpleNamespace(
                    session_params=None,
                    return_logprob=False,
                    return_hidden_states=False,
                    return_sampling_mask=False,
                    return_routed_experts=False,
                )
                allocator = PageAllocator(page_size, page_count)
                coordinator = BeamCoordinator(
                    model_config=SimpleNamespace(
                        vocab_size=256, is_encoder_decoder=False
                    ),
                    spec_algorithm=SimpleNamespace(is_none=lambda: True),
                    dllm_enabled=False,
                    max_req_len=8192,
                    req_to_token_pool=SimpleNamespace(size=64, device="cpu"),
                    token_to_kv_pool_allocator=allocator,
                    tree_cache=SimpleNamespace(
                        supports_swa=lambda: False, supports_mamba=lambda: False
                    ),
                    future_map=None,
                    gr_paged_enabled=enabled,
                )
                values = {
                    "get_serving": SimpleNamespace(enable_gr=True),
                    "get_disagg": SimpleNamespace(disaggregation_mode="null"),
                    "get_schedule": SimpleNamespace(
                        page_size=page_size, max_prefill_tokens=max_prefill
                    ),
                    "get_memory": SimpleNamespace(
                        enable_hisparse=False, enable_hierarchical_cache=False
                    ),
                    "get_parallel": SimpleNamespace(pp_size=1),
                    "is_dp_attention_enabled": False,
                }
                with ExitStack() as stack:
                    for name, value in values.items():
                        stack.enter_context(
                            patch(
                                "sglang.srt.beam_search.coordinator." + name,
                                return_value=value,
                            )
                        )
                    error = coordinator.validate_and_init(req, recv)
                if expected is not None:
                    self.assertIn(expected, error)
                    continue
                self.assertIsNone(error)
                self.assertEqual(req.skip_radix_cache_insert, not enabled)
                self.assertEqual(req.beam_group.gr_paged_kv is not None, enabled)
                if enabled:
                    self.assertEqual(req.beam_group.gr_paged_kv.arena_slots, 64)
                    self.assertFalse(req.beam_group.gr_paged_kv.has_reservation)
                    self.assertEqual(allocator.allocations, [])
                    # Queued cancellation retires metadata without allocating KV.
                    coordinator._retire_group(req.beam_group)
                    self.assertTrue(req.beam_group.gr_paged_kv.released)
                    self.assertEqual(allocator.frees, [])

    def test_tp_reservation_agreement_rolls_back_successful_peer(self):
        from sglang.srt.managers.schedule_batch import ScheduleBatch

        for local_failure, peer_failure in (
            (True, False),
            (False, True),
            (False, False),
        ):
            with self.subTest(local=local_failure, peer=peer_failure):
                allocator = PageAllocator(16)
                before = allocator.available_size()
                state = GRPagedKVState(17, 17, 3, 16)
                batch = SimpleNamespace(
                    reqs=[
                        SimpleNamespace(beam_group=SimpleNamespace(gr_paged_kv=state))
                    ],
                    tree_cache=object(),
                    token_to_kv_pool_allocator=allocator,
                )
                cpu_group = object()
                observed = []

                def reduce_status(status, *, op, group):
                    self.assertIs(group, cpu_group)
                    self.assertEqual(status.device.type, "cpu")
                    self.assertEqual(op, torch.distributed.ReduceOp.MIN)
                    observed.append(int(status.item()))
                    if peer_failure:
                        status.zero_()

                with ExitStack() as stack:
                    stack.enter_context(
                        patch(
                            "sglang.srt.managers.schedule_batch.evict_from_tree_cache"
                        )
                    )
                    stack.enter_context(
                        patch("torch.distributed.is_initialized", return_value=True)
                    )
                    stack.enter_context(
                        patch(
                            "sglang.srt.distributed.get_tp_group",
                            return_value=SimpleNamespace(
                                world_size=2, cpu_group=cpu_group
                            ),
                        )
                    )
                    stack.enter_context(
                        patch("torch.distributed.all_reduce", side_effect=reduce_status)
                    )
                    if local_failure:
                        stack.enter_context(
                            patch.object(
                                state,
                                "reserve",
                                side_effect=RuntimeError(
                                    "injected rank-local metadata failure"
                                ),
                            )
                        )
                    if local_failure or peer_failure:
                        with self.assertRaisesRegex(MemoryError, "every TP rank"):
                            ScheduleBatch._reserve_gr_paged_arena(batch)
                        self.assertTrue(state.released)
                        self.assertEqual(state.reserved_slots, 0)
                    else:
                        ScheduleBatch._reserve_gr_paged_arena(batch)
                        self.assertTrue(state.has_reservation)
                        state.release(allocator.free)
                self.assertEqual(observed, [int(not local_failure)])
                self.assertEqual(allocator.available_size(), before)
                self.assertEqual(len(allocator.frees), int(not local_failure))

    def test_reserved_arena_has_zero_incremental_decode_budget(self):
        from sglang.srt.managers.schedule_batch import ScheduleBatch

        paged = SimpleNamespace(
            beam_group=SimpleNamespace(gr_paged_kv=GRPagedKVState(32, 17, 3, 16)),
            kv=SimpleNamespace(kv_committed_len=32),
        )
        ordinary = SimpleNamespace(
            beam_group=None, kv=SimpleNamespace(kv_committed_len=32)
        )
        batch = SimpleNamespace(
            token_to_kv_pool_allocator=PageAllocator(16),
            reqs=[paged, ordinary],
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
        )
        self.assertEqual(ScheduleBatch.new_tokens_required_next_decode(batch, [0]), 0)
        self.assertEqual(ScheduleBatch.new_tokens_required_next_decode(batch, [1]), 16)
        self.assertEqual(ScheduleBatch.new_tokens_required_next_decode(batch), 16)

    def test_admission_counts_active_groups_not_queued_groups(self):
        from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder

        for prefix_len, depth in itertools.product((0, 16), (1, 3)):
            state = GRPagedKVState(17, 17, depth, 16)
            req = SimpleNamespace(
                beam_group=SimpleNamespace(gr_paged_kv=state),
                full_untruncated_fill_ids=[11] * 17,
                prefix_len=prefix_len,
                last_node=object(),
            )
            needed = ((17 - prefix_len + 15) // 16) * 16 + state.arena_slots
            for free_slots in (needed - 1, needed):
                adder = SimpleNamespace(
                    # The coordinator can already know many queued groups.
                    _num_live_groups=10,
                    can_run_list=[],
                    running_batch=SimpleNamespace(reqs=[]),
                    ceil_paged_tokens=lambda n: ((n + 15) // 16) * 16,
                    _lock_node=lambda node: nullcontext(),
                    rem_total_tokens=free_slots,
                    rem_input_tokens=128,
                )
                admitted = []

                def commit(candidate, admission, *, state_slots):
                    self.assertEqual(admission.prefix_len, prefix_len)
                    self.assertEqual(admission.extend_len, 17 - prefix_len)
                    self.assertEqual(admission.max_new_tokens, state.arena_slots)
                    self.assertFalse(admission.is_chunked)
                    self.assertEqual(state_slots, 0)
                    admitted.append(candidate)
                    adder.can_run_list.append(candidate)

                adder._commit_prefill_admission = commit
                result = PrefillAdder._add_gr_paged_req(adder, req)
                if free_slots < needed:
                    self.assertEqual(result, AddReqResult.NO_TOKEN)
                    self.assertEqual(admitted, [])
                else:
                    self.assertEqual(result, AddReqResult.OTHER)
                    self.assertEqual(admitted, [req])
                    # A second queued request cannot join this forward.
                    PrefillAdder._add_gr_paged_req(adder, req)
                    self.assertEqual(admitted, [req])
                self.assertFalse(state.has_reservation)
            adder.can_run_list = []
            admitted.clear()
            adder.running_batch.reqs = [SimpleNamespace(finished=lambda: False)]
            PrefillAdder._add_gr_paged_req(adder, req)
            self.assertEqual(admitted, [])
            adder.running_batch.reqs = [SimpleNamespace(finished=lambda: True)]
            PrefillAdder._add_gr_paged_req(adder, req)
            self.assertEqual(admitted, [req])

    def test_retraction_aborts_and_releases_arena(self):
        from sglang.srt.beam_search.beam_group import BeamGroup
        from sglang.srt.beam_search.coordinator import BeamCoordinator
        from sglang.srt.managers.schedule_batch import retract_all
        from sglang.srt.managers.scheduler_components.request_retirer import (
            RequestRetirer,
        )

        allocator = PageAllocator(16)
        before = allocator.available_size()
        req = Mock()
        req.rid = "paged-retract"
        req.output_ids = []
        req.weight_version_events = []
        req.prefill_weight_versions = None
        req.request_finish.side_effect = lambda reason: setattr(
            req, "to_finish", reason
        )
        req.session = None
        req.kv = SimpleNamespace(
            req_pool_idx=0, kv_allocated_len=17, kv_committed_len=17
        )
        group = BeamGroup(beam_width=2, max_new_tokens=3)
        group.leader = req
        group.prompt_len = 17
        group.member_rows = torch.tensor([1])
        group.member_rows_cpu = torch.tensor([1])
        group.all_rows = torch.tensor([0, 1])
        group.gr_paged_kv = GRPagedKVState(17, 2, 3, 16)
        group.gr_paged_kv.reserve(allocator)
        req.beam_group = group
        pool = SimpleNamespace(free_rows=Mock(), req_to_token=torch.zeros(2, 32))
        coordinator = BeamCoordinator(
            model_config=None,
            spec_algorithm=None,
            dllm_enabled=False,
            max_req_len=8192,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            tree_cache=None,
            future_map=None,
        )
        coordinator._num_live_groups = 1
        cache = Mock(req_to_token_pool=pool, token_to_kv_pool_allocator=allocator)
        send_abort = Mock()
        retirer = RequestRetirer(
            tree_cache=cache,
            beam_coordinator=coordinator,
            hisparse_coordinator=None,
            uocr_runtime=None,
            send_abort_output=send_abort,
            stream_output=Mock(),
        )

        def release_context(candidate, *, offload_kv):
            self.assertIs(candidate, req)
            self.assertFalse(offload_kv)
            self.assertTrue(group.gr_paged_kv.released)
            self.assertEqual(allocator.available_size(), before)
            pool.free_rows.assert_called_once_with([1])
            return False

        with (
            patch.object(retirer, "retract", side_effect=release_context),
            patch("sglang.srt.managers.schedule_batch.evict_from_tree_cache"),
        ):
            aborts = retract_all(
                reqs=[req], tree_cache=cache, request_retirer=retirer, offload_kv=False
            )
        self.assertEqual(aborts, [req])
        with patch(
            "sglang.srt.managers.utils.get_serving",
            return_value=SimpleNamespace(weight_version="default"),
        ):
            retirer.abort_retracted(req, reason=req.to_finish)
        self.assertTrue(group.retired)
        self.assertEqual(coordinator._num_live_groups, 0)
        self.assertEqual(len(allocator.frees), 1)
        send_abort.assert_called_once()
        req.reset_for_retract.assert_called_once()


if __name__ == "__main__":
    unittest.main()
