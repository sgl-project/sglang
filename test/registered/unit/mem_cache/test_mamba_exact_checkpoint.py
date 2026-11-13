"""CPU-only unit tests for --enable-mamba-exact-checkpoint.

A finished request's Mamba state is cached where its own state slot holds it,
at its exact end, instead of at the last tracked checkpoint. A state that ends
inside a page keeps that page in a leaf of its own, and a request resuming
from it copies the page into one of its own before its first forward.

A stand-in device runs requests through the real tree, allocator and pools on
CPU: a token's KV row holds its position and token in every full-attention
layer, and a state slot holds the tokens folded into it, in order. Before a
request computes anything, the state and the KV rows it resumed from must name
exactly the tokens in front of it.
"""

import random
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.req_types import FINISH_ABORT, FINISH_LENGTH
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.allocation import _prefix_kv_indices
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.kv.advance import TrackingAdvance
from sglang.srt.mem_cache.kv.alloc.demand import KVDemand
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, HybridLinearKVPool
from sglang.srt.mem_cache.static_kv_storage import (
    StaticKVStorage,
    acquire_rows,
    kv_storage_of,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.mamba import (
    exact_resume_alignment,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    build_unified_bundle,
    setup_allocator_context,
)

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

FULL, MAMBA = ComponentType.FULL, ComponentType.MAMBA
PAGE = 4
# Decode tracks a state every TRACK tokens (the extra buffer's ping-pong pair).
TRACK = 8


def _paged_allocator(size, dtype, device, kvcache, need_sort):
    return PagedTokenToKVPoolAllocator(
        size=size,
        page_size=PAGE,
        dtype=dtype,
        device=device,
        kvcache=kvcache,
        need_sort=need_sort,
    )


class _Device:
    """The model, on the real CPU pools: a token's KV row holds (position,
    token) in every full-attention layer; a state slot holds its length and
    the tokens folded into it, in order (tokens stay below 256, which the
    bfloat16 state holds exactly)."""

    def __init__(self, allocator):
        self.k_buffers = allocator.get_kvcache().full_kv_pool.k_buffer
        self.states = allocator.state.pool.mamba_cache.temporal

    def write_kv(self, slots: torch.Tensor, tokens, start: int) -> None:
        for k in self.k_buffers:
            k[slots, 0, 0] = torch.arange(start, start + len(tokens), dtype=k.dtype)
            k[slots, 0, 1] = torch.tensor(tokens, dtype=k.dtype)

    def kv(self, slots: torch.Tensor) -> list[tuple[int, int]]:
        rows = [
            [(int(p), int(t)) for p, t in zip(k[slots, 0, 0], k[slots, 0, 1])]
            for k in self.k_buffers
        ]
        assert all(r == rows[0] for r in rows), "layers disagree"
        return rows[0]

    def set_state(self, slot: int, tokens) -> None:
        flat = self.states[:, slot].view(self.states.shape[0], -1)
        flat.zero_()
        flat[:, 0] = len(tokens)
        flat[:, 1 : 1 + len(tokens)] = torch.tensor(list(tokens), dtype=flat.dtype)

    def state(self, slot: int) -> list[int]:
        flat = self.states[0, slot].view(-1)
        return [int(x) for x in flat[1 : 1 + int(flat[0])]]


class _Engine:
    """Requests through a FULL + MAMBA unified radix cache with the extra
    buffer, a paged allocator and the stand-in device, on CPU: match, lock,
    bind, lay out the prefix and the extend, run the deferred copies, check
    what the request resumed from, prefill, decode, finish."""

    def __init__(
        self,
        test,
        exact,
        *,
        kv_size=512,
        mamba_cache_size=48,
        **cfg_kwargs,
    ):
        self.test = test
        cfg = CacheConfig(
            components=(FULL, MAMBA),
            page_size=PAGE,
            dtype=torch.float32,
            enable_mamba_extra_buffer=True,
            enable_mamba_exact_checkpoint=exact,
            kv_size=kv_size,
            mamba_cache_size=mamba_cache_size,
            max_num_reqs=64,
            max_context_len=1024,
            **cfg_kwargs,
        )
        fixtures = "sglang.test.kits.unified_radix_cache.fixtures"
        with (
            patch(f"{fixtures}.get_device", return_value="cpu"),
            patch(f"{fixtures}.TokenToKVPoolAllocator", _paged_allocator),
            # The tiled copy kernel needs a GPU; the native move is the same copy.
            envs.SGLANG_NATIVE_MOVE_KV_CACHE.override(True),
        ):
            self.cache, self.allocator, self.pool = build_fixture(
                cfg, mamba_cache_chunk_size=PAGE
            )
        self.core = self.cache.tree_core
        self.mamba = self.cache.components[MAMBA]
        self.dev = _Device(self.allocator)
        self.storage = kv_storage_of(self.pool, self.allocator)
        self.kv_capacity = self.allocator.available_size()
        self.state_capacity = self.allocator.state.slots.available_size()
        # What ScheduleBatch's per-request extend preparation reads.
        self.batch = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(mamba_chunk_size=PAGE)
            ),
            tree_cache=self.cache,
            req_to_token_pool=self.pool,
            token_to_kv_pool_allocator=self.allocator,
        )
        # What ModelRunner's deferred page copy reads.
        self.runner = SimpleNamespace(
            req_to_token_pool=self.pool,
            token_to_kv_pool_allocator=self.allocator,
            token_to_kv_pool=self.allocator.get_kvcache(),
            is_draft_worker=False,
        )
        self.rid = 0

    # -- a request's life --

    def new_req(self, prompt) -> Req:
        self.rid += 1
        return Req(
            rid=f"exact-{self.rid}",
            origin_input_text="",
            origin_input_ids=array("q", prompt),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=256),
        )

    def admit(self, req: Req) -> None:
        req.init_next_round_input(self.cache, cow_mamba=True)
        req.lock = self.cache.lock(req.last_node)
        # As alloc_for_extend: the prefix is read before the row is bound.
        reading = SimpleNamespace(tree_cache=self.cache, req_to_token_pool=self.pool)
        prefix = _prefix_kv_indices(reading, req) if req.prefix_len else None
        state = self.allocator.state
        needed = state.pending_request_slots(self.pool, req)
        if not state.slots.slots_fit(needed):
            self.cache.evict_for_alloc(KVDemand(state_slots=needed))
        acquire_rows(self.pool, state, [req])
        tokens = list(req.full_untruncated_fill_ids)
        row = req.kv.req_pool_idx
        if prefix is not None:
            self.pool.write((row, slice(0, req.prefix_len)), prefix)
        self._take(req, req.prefix_len, len(tokens))
        self._run_deferred_copies(req)
        self.check_resumed(req, tokens)
        start = req.prefix_len
        slots = self.pool.req_to_token[row, start : len(tokens)].long()
        self.dev.write_kv(slots, tokens[start:], start)
        self.dev.set_state(int(req.kv.mamba_pool_idx), tokens)
        req.extend_end = len(tokens)
        entry = ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend(
            self.batch, req
        )
        if entry.track_mask:
            buf = req.kv.mamba_ping_pong_track_buffer
            self.dev.set_state(
                int(buf[entry.track_position]),
                tokens[: req.kv.mamba_last_track_seqlen],
            )
        req.kv.kv_committed_len = len(tokens)

    def decode(self, req: Req, output, *, overlap=False) -> None:
        """Sample ``output``; each step feeds the token sampled before it. An
        overlapped extra step also feeds the last one, as an EOS the
        scheduler has not seen yet does."""
        seq = list(req.origin_input_ids) + list(output)
        req.output_ids = array("q", output)
        end = len(seq) if overlap else len(seq) - 1
        for p in range(len(req.origin_input_ids), end):
            self._take(req, p, p + 1)
            slot = self.pool.req_to_token[req.kv.req_pool_idx, p : p + 1].long()
            self.dev.write_kv(slot, [seq[p]], p)
            self.dev.set_state(int(req.kv.mamba_pool_idx), seq[: p + 1])
            req.kv.kv_committed_len = p + 1
            if (p + 1) % TRACK == 0:
                track = req.kv.mamba_next_track_idx
                buf = req.kv.mamba_ping_pong_track_buffer
                self.dev.set_state(int(buf[track]), seq[: p + 1])
                self.storage.advance(
                    req,
                    tracking=TrackingAdvance(
                        boundary=p + 1, last_position=track, preserve_previous=True
                    ),
                )
                self.storage.advance(
                    req,
                    tracking=TrackingAdvance(
                        next_position=self.pool.get_mamba_ping_pong_other_idx(track)
                    ),
                )

    def finish(self, req: Req, reason=None, finished_len=None) -> None:
        req.finished_reason = reason or FINISH_LENGTH(length=len(req.output_ids))
        req.finished_len = finished_len
        release_kv_cache(req, self.cache, checkpoint=True)

    def serve(self, prompt, output, **kwargs) -> Req:
        req = self.new_req(prompt)
        self.admit(req)
        self.decode(req, output, overlap=kwargs.pop("overlap", False))
        self.finish(req, **kwargs)
        self.check()
        return req

    # -- plumbing --

    def _take(self, req: Req, start: int, end: int) -> None:
        """Slots for positions [start, end): the rest of the page the row
        ends in, then whole new pages."""
        row = req.kv.req_pool_idx
        slots = []
        pos = start
        if pos % PAGE and pos < end:
            last = int(self.pool.req_to_token[row, pos - 1])
            n = min(PAGE - pos % PAGE, end - pos)
            slots.extend(range(last + 1, last + 1 + n))
            pos += n
        if pos < end:
            need = -(-(end - pos) // PAGE) * PAGE
            pages = self.allocator.alloc(need)
            if pages is None:
                self.cache.evict(EvictParams(num_tokens=need))
                pages = self.allocator.alloc(need)
            assert pages is not None, "the test's KV pool is exhausted"
            slots.extend(pages[: end - pos].tolist())
        if slots:
            self.pool.write(
                (row, slice(start, end)), torch.tensor(slots, dtype=torch.int32)
            )
        req.kv.kv_allocated_len = max(req.kv.kv_allocated_len, end)

    def _run_deferred_copies(self, req: Req) -> None:
        """What the forward runs first (ModelRunner's deferred copies)."""
        ScheduleBatch._collect_deferred_mamba_cow_and_clear(self.batch, [req])
        ScheduleBatch._collect_deferred_kv_page_copies(self.batch, [req])
        if self.batch.mamba_cow_src_indices is not None:
            self.allocator.state.copy(
                self.batch.mamba_cow_src_indices, self.batch.mamba_cow_dst_indices
            )
        if self.batch.mamba_clear_indices is not None:
            for slot in self.batch.mamba_clear_indices.tolist():
                self.dev.set_state(slot, [])
        forward_batch = SimpleNamespace(
            kv_copy_src_locs=self.batch.kv_copy_src_locs,
            kv_copy_dst_locs=self.batch.kv_copy_dst_locs,
        )
        ModelRunner._maybe_copy_deferred_kv_pages(self.runner, forward_batch)
        # Copied once: a later forward of the batch must not copy again over
        # what the request has written since.
        self.test.assertIsNone(forward_batch.kv_copy_src_locs)

    # -- checks --

    def check_resumed(self, req: Req, tokens) -> None:
        n = req.prefix_len
        if n == 0:
            return
        row = req.kv.req_pool_idx
        self.test.assertEqual(
            self.dev.state(int(req.kv.mamba_pool_idx)),
            tokens[:n],
            f"{req.rid} resumed at {n} from a state of other tokens",
        )
        self.test.assertEqual(
            self.dev.kv(self.pool.req_to_token[row, :n].long()),
            [(p, tokens[p]) for p in range(n)],
            f"{req.rid} resumed at {n} on KV rows of other tokens",
        )

    def check(self) -> None:
        self.cache.sanity_check()
        tree_tokens = len(self.core.all_values_flatten())
        self.test.assertEqual(
            self.allocator.available_size() + tree_tokens,
            self.kv_capacity,
            "KV slots leaked or double-freed",
        )
        states = sum(
            1
            for node in self.core._collect_all_nodes()
            if node is not self.core.root_node
            and node.component_data[MAMBA].value is not None
        )
        self.test.assertEqual(
            self.allocator.state.slots.available_size() + states,
            self.state_capacity,
            "state slots leaked or double-freed",
        )

    # -- tree inspection --

    def node(self, tokens):
        """The node whose key ends right after ``tokens``, leaves ending
        inside a page included; None if there is none."""
        node, depth = self.core.root_node, 0
        while depth < len(tokens):
            # A leaf's tokens can begin a sibling's page: take the longest.
            matches = [
                child
                for child in node.children.values()
                if tokens[depth : depth + len(child.key)]
                == list(child.key.raw_token_ids())
            ]
            if not matches:
                return None
            node = max(matches, key=lambda child: len(child.key))
            depth += len(node.key)
        return node

    def state_of(self, tokens):
        node = self.node(tokens)
        if node is None or node.component_data[MAMBA].value is None:
            return None
        return self.dev.state(int(node.component_data[MAMBA].value))


def _tokens(start, n):
    return [start + i for i in range(n)]


class TestExactCheckpoint(CustomTestCase):
    def test_finished_request_caches_its_state_at_its_end(self):
        """A request that processed 13 tokens (decode tracked the state at 8)
        caches the state after all 13, with the page holding tokens 12-13."""
        prompt, output = _tokens(1, 6), _tokens(100, 8)
        for exact in (False, True):
            with self.subTest(exact=exact):
                eng = _Engine(self, exact)
                req = eng.serve(prompt, output)
                seq = prompt + output
                if exact:
                    leaf = eng.node(seq[:13])
                    self.assertEqual(len(leaf.key), 1)
                    self.assertEqual(len(leaf.parent.key) % PAGE, 0)
                    self.assertEqual(eng.state_of(seq[:13]), seq[:13])
                    self.assertIsNone(eng.state_of(seq[:12]))
                    self.assertEqual(len(leaf.component_data[FULL].value), PAGE)
                else:
                    # Upstream: the last tracked state, at 8.
                    self.assertEqual(eng.state_of(seq[:8]), seq[:8])
                    self.assertIsNone(eng.node(seq[:13]))
                self.assertFalse(req.kv.holds_kv)

    def test_next_turn_resumes_at_the_exact_end(self):
        """The next turn extends the first turn's tokens: it resumes after all
        13 of them on its own copy of the page, and the cached page stays as
        it was."""
        prompt, output = _tokens(1, 6), _tokens(100, 8)
        seq = prompt + output
        eng = _Engine(self, exact=True)
        eng.serve(prompt, output)
        leaf_page = eng.node(seq[:13]).component_data[FULL].value.clone()
        cached_rows = eng.dev.kv(leaf_page)
        turn2 = eng.new_req(seq + _tokens(200, 6))
        eng.admit(turn2)
        self.assertEqual(turn2.prefix_len, 13)
        self.assertEqual(turn2.kv.cache_protected_len, 12)
        own = eng.pool.req_to_token[turn2.kv.req_pool_idx, 12:16].long()
        self.assertTrue(set(own.tolist()).isdisjoint(leaf_page.tolist()))
        # Turn two wrote positions 13-15 into its own page only.
        self.assertEqual(eng.dev.kv(leaf_page), cached_rows)
        eng.decode(turn2, _tokens(150, 3))
        eng.finish(turn2)
        eng.check()
        # Turn two caches its own end (22 tokens); the leaf it resumed from
        # stays until eviction takes it, as a finished request's state does
        # upstream.
        seq2 = seq + _tokens(200, 6) + _tokens(150, 2)
        self.assertEqual(eng.state_of(seq2), seq2)
        self.assertEqual(eng.state_of(seq[:13]), seq[:13])

    def test_overlapped_extra_step_feeds_the_last_token(self):
        """With one more step in flight the last token is committed and
        processed too: the state is exact after all 14 tokens."""
        prompt, output = _tokens(1, 6), _tokens(100, 8)
        seq = prompt + output
        eng = _Engine(self, exact=True)
        eng.serve(prompt, output, overlap=True)
        self.assertEqual(eng.state_of(seq), seq)
        turn2 = eng.new_req(seq + _tokens(200, 3))
        eng.admit(turn2)
        self.assertEqual(turn2.prefix_len, 14)

    def test_states_not_known_to_be_exact_fall_back(self):
        prompt, output = _tokens(1, 6), _tokens(100, 8)
        seq = prompt + output
        cases = {
            # A stop string ended the output two tokens early.
            "cut": dict(finished_len=6),
            "aborted": dict(reason=FINISH_ABORT()),
        }
        for outcome, kwargs in cases.items():
            with self.subTest(outcome=outcome):
                eng = _Engine(self, exact=True)
                eng.serve(prompt, output, **kwargs)
                self.assertIsNone(eng.node(seq[:13]))
                if outcome == "cut":
                    self.assertEqual(eng.state_of(seq[:8]), seq[:8])

        with self.subTest(outcome="verify_in_flight"):
            eng = _Engine(self, exact=True)
            eng.mamba.speculative = True
            req = eng.new_req(prompt)
            eng.admit(req)
            eng.decode(req, output)
            req.inflight_steps = 2
            eng.finish(req)
            eng.check()
            self.assertIsNone(eng.node(seq[:13]))
            self.assertEqual(eng.state_of(seq[:8]), seq[:8])

    def test_longest_leaf_within_the_key(self):
        """Two requests that end inside the same page leave two leaves; a
        match takes the longest one its tokens continue, but never past the
        token it keeps for computing logits."""
        eng = _Engine(self, exact=True)
        base = _tokens(1, 8)
        eng.serve(base, [40, 41])  # processes 9 tokens: leaf [40]
        eng.serve(base, [40, 42, 43])  # 10: leaf [40, 42]
        node = eng.node(base)
        self.assertEqual(
            sorted(len(leaf.key) for leaf in eng.core._exact_leaves[node]), [1, 2]
        )
        for prompt, hit in (
            (base + [40, 42, 7], 10),
            (base + [40, 9, 7], 9),
            (base + [40, 42], 9),  # the last token is computed, not matched
        ):
            with self.subTest(prompt=prompt):
                req = eng.new_req(prompt)
                req.init_next_round_input(eng.cache, cow_mamba=False)
                self.assertEqual(req.prefix_len, hit)

    def test_eviction_takes_the_leaf_with_its_page(self):
        prompt, output = _tokens(1, 6), _tokens(100, 8)
        seq = prompt + output
        for unit_eviction in (False, True):
            with self.subTest(unit_eviction=unit_eviction):
                eng = _Engine(
                    self, exact=True, enable_mamba_unit_eviction=unit_eviction
                )
                eng.serve(prompt, output)
                free_before = eng.allocator.available_size()
                eng.cache.evict(EvictParams(mamba_num=1))
                eng.check()
                self.assertIsNone(eng.node(seq[:13]))
                self.assertFalse(eng.core._exact_leaves)
                freed = eng.allocator.available_size() - free_before
                # The page goes with the leaf; with unit eviction the KV that
                # only it served goes too.
                self.assertEqual(freed, 16 if unit_eviction else PAGE)

    def test_a_dsa_pool_copies_index_keys_with_their_pages(self):
        """A pooled index key mixes several tokens: the resume copy moves the
        keys with their pages, a token move still refuses them, and a request
        resumes only on a pooling group's boundary. Keys that are not pooled
        copy as tokens and resume anywhere."""
        for kpool in (4, 1):
            with self.subTest(index_kpool=kpool):
                page = 64 * kpool  # 64-token physical pages
                with get_parallel().override(attn_dcp_size=1):
                    pool = DSATokenToKVPool(
                        size=4 * page,
                        page_size=page,
                        kv_lora_rank=16,
                        dtype=torch.bfloat16,
                        qk_rope_head_dim=8,
                        layer_num=2,
                        device="cpu",
                        enable_memory_saver=False,
                        kv_cache_dim=24,
                        index_head_dim=128,
                        index_kpool=kpool,
                        index_kpool_compress=kpool > 1,
                        max_running_requests=4,
                    )
                src = torch.arange(page, 2 * page)
                dst = torch.arange(3 * page, 4 * page)
                keys = pool.index_key_cache.buffer[0]
                keys[1] = torch.arange(keys.shape[1]) % 251
                for rows in pool.kv_buffer:
                    rows[src] = 3
                StaticKVStorage.copy_token_pages(SimpleNamespace(), pool, dst, src)
                self.assertTrue(torch.equal(keys[3], keys[1]))
                self.assertTrue(
                    all(bool((rows[dst] == 3).all()) for rows in pool.kv_buffer)
                )
                if kpool > 1:
                    with self.assertRaisesRegex(AssertionError, "regrouping"):
                        pool.move_kv_cache(dst[:4], src[:4])
                # A hybrid model's pool hands the copy to its full-attention pool.
                HybridLinearKVPool.move_kv_pages(
                    SimpleNamespace(full_kv_pool=pool),
                    torch.arange(2 * page, 3 * page),
                    src,
                )
                self.assertTrue(torch.equal(keys[2], keys[1]))
                self.assertEqual(exact_resume_alignment(pool), kpool)
        mha = _Engine(self, exact=True).allocator.get_kvcache()
        self.assertEqual(exact_resume_alignment(mha), 1)

    def test_a_unified_pool_copies_the_page_through_the_arena(self):
        setup_allocator_context()
        bundle = build_unified_bundle(
            UnifiedGeometry(
                chain="mamba",
                full_layer_ids=(0,),
                state_layer_ids=(1,),
                page_size=PAGE,
                full_tokens=96,
                state_slots=16,
                context_len=256,
                max_num_reqs=8,
            )
        )
        allocator = bundle.token_to_kv_pool_allocator
        src, dst = allocator.alloc(PAGE), allocator.alloc(PAGE)
        kv = allocator.get_kvcache()
        k_buffer = kv.get_key_buffer(0)
        # The allocator hands out virtual ids; the rows sit at the physical ones.
        src_rows = allocator.translate_kv_loc(src).long()
        dst_rows = allocator.translate_kv_loc(dst).long()
        rows = torch.arange(1, 1 + src.numel() * k_buffer[0].numel())
        k_buffer[src_rows] = rows.view(src.numel(), *k_buffer.shape[1:]).to(
            k_buffer.dtype
        )

        def run_copy(is_draft_worker):
            runner = SimpleNamespace(
                req_to_token_pool=bundle.req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                token_to_kv_pool=kv,
                is_draft_worker=is_draft_worker,
            )
            ModelRunner._maybe_copy_deferred_kv_pages(
                runner, SimpleNamespace(kv_copy_src_locs=src, kv_copy_dst_locs=dst)
            )

        run_copy(is_draft_worker=False)
        self.assertTrue(torch.equal(k_buffer[dst_rows], k_buffer[src_rows]))
        self.assertFalse(torch.equal(dst_rows, src_rows))
        # The target's copy moved a fused draft's rows too, and the forward
        # has written the page since: the draft runner's copy does nothing.
        k_buffer[dst_rows] = 0
        run_copy(is_draft_worker=True)
        self.assertFalse(k_buffer[dst_rows].any())

    def test_server_args_refuse_what_it_does_not_support(self):
        from sglang.srt.arg_groups.mamba_hook import handle_mamba_backend

        for flag, kwargs in (
            ("--enable-hierarchical-cache", dict(enable_hierarchical_cache=True)),
            ("--enable-session-radix-cache", dict(enable_session_radix_cache=True)),
            ("--kv-events-config", dict(kv_events_config='{"publisher": "null"}')),
            ("--disable-radix-cache", dict(disable_radix_cache=True)),
            ("--enable-two-batch-overlap", dict(enable_two_batch_overlap=True)),
            (
                "--enable-dsa-cache-layer-split",
                dict(enable_dsa_cache_layer_split=True),
            ),
        ):
            with self.subTest(flag=flag):
                args = ServerArgs(
                    model_path="dummy", enable_mamba_exact_checkpoint=True, **kwargs
                )
                with self.assertRaisesRegex(ValueError, flag):
                    handle_mamba_backend(args)
        handle_mamba_backend(
            ServerArgs(model_path="dummy", enable_mamba_exact_checkpoint=True)
        )

    def test_resumed_state_and_kv_are_exact_under_pressure(self):
        """Sessions that grow turn by turn, fork, overlap, stop early and get
        evicted, under each eviction policy: every request resumes from a
        state and KV rows that name exactly the tokens before it, and nothing
        leaks."""
        gds = dict(
            eviction_policy="gds",
            eviction_policy_config={"prefill_cost": [1.862e-5, 3.33e-6]},
        )
        for policy, cfg in (
            ("lru", {}),
            ("unit", dict(enable_mamba_unit_eviction=True)),
            ("gds", gds),
        ):
            with self.subTest(policy=policy):
                rng = random.Random(0)
                eng = _Engine(self, True, kv_size=384, mamba_cache_size=20, **cfg)
                sessions = [[1 + s] for s in range(5)]
                exact_hits = 0
                for _ in range(70):
                    if rng.random() < 0.1:
                        eng.cache.evict(EvictParams(mamba_num=rng.randint(1, 3)))
                        eng.check()
                        continue
                    i = rng.randrange(len(sessions))
                    if rng.random() < 0.1:
                        j = rng.randrange(len(sessions))
                        sessions[i] = sessions[j][: rng.randint(1, len(sessions[j]))]
                    prompt = sessions[i] + [
                        rng.randrange(1, 250) for _ in range(rng.randint(1, 9))
                    ]
                    if len(prompt) > 120:
                        prompt = [1 + i, rng.randrange(1, 250)]
                    output = [rng.randrange(1, 250) for _ in range(rng.randint(1, 12))]
                    req = eng.new_req(prompt)
                    eng.admit(req)
                    exact_hits += req.prefix_len % PAGE != 0
                    eng.decode(req, output, overlap=rng.random() < 0.3)
                    # A stop string may end the output a token early.
                    cut = rng.random() < 0.1 and len(output) > 2
                    finished = len(output) - 1 if cut else len(output)
                    eng.finish(req, finished_len=finished if cut else None)
                    eng.check()
                    sessions[i] = prompt + output[:finished]
                self.assertGreater(exact_hits, 0)


if __name__ == "__main__":
    unittest.main()
