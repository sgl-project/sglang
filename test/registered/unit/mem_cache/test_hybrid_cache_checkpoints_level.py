"""CPU-only unit tests for --hybrid-cache-checkpoints-level."""

import argparse
import itertools
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.req_types import FINISH_LENGTH, FINISH_MATCHED_TOKEN
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import mark_shared_prompt_checkpoints
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.request_list import (
    KeyedReqList,
    ListKind,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.common import checkpoint_kv_cache
from sglang.srt.mem_cache.kv.advance import TrackingAdvance
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.static_kv_storage import acquire_rows, kv_storage_of
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import get_context, get_disagg, get_exec
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_utils import (
    _verify_commit_step_indices,
    prepare_mamba_track_for_verify,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture
from sglang.test.mem_cache_utils import finish_req
from sglang.test.scheduler_test_utils import release_only_retirer
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")

PROMPT = list(range(1, 17))


def _build(
    level,
    components=(ComponentType.FULL, ComponentType.MAMBA),
    page_size=4,
    enable_mamba_extra_buffer=True,
    mamba_cache_size=8,
    **kwargs,
):
    cfg = CacheConfig(
        page_size=page_size,
        components=components,
        enable_mamba_extra_buffer=enable_mamba_extra_buffer,
        kv_size=64,
        max_context_len=64,
        mamba_cache_size=mamba_cache_size,
        hybrid_cache_checkpoints_level=level,
        **kwargs,
    )
    with patch(
        "sglang.test.kits.unified_radix_cache.fixtures.get_device",
        return_value="cpu",
    ):
        return build_fixture(cfg, mamba_cache_chunk_size=4)


def _running_req(cache, allocator, pool, up_to, prompt=PROMPT, rid="running"):
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", prompt),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )
    acquire_rows(pool, pool.state, [req])
    req.output_ids = array("q")
    req.refresh_fill_ids()
    req.kv.kv_committed_len = len(prompt)
    req.kv.kv_allocated_len = len(prompt)
    req.kv.cache_protected_len = 0
    if cache.enable_mamba_extra_buffer:
        req.kv.mamba_last_track_seqlen = up_to
    pool.write(
        (req.kv.req_pool_idx, slice(0, len(prompt))),
        allocator.alloc(len(prompt)),
    )
    req.last_node = cache.root_node_handle()
    req.lock = None
    req.extend_end = up_to
    return req


def _cached_len(cache, tokens):
    return cache.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", tokens)))
    ).device_prefix_len


class TestCheckpointsLevelArgs(CustomTestCase):
    def _cap(self, **fields):
        with get_context().override_server_args(**fields):
            return get_exec().mamba.mamba_path_state_cap

    def test_path_state_cap(self):
        self.assertEqual(self._cap(), 1)
        self.assertEqual(self._cap(hybrid_cache_checkpoints_level=3), -1)
        self.assertEqual(self._cap(mamba_max_states_per_path=3), 3)
        self.assertEqual(self._cap(mamba_max_states_per_path=-1), -1)
        self.assertEqual(
            self._cap(hybrid_cache_checkpoints_level=2, mamba_max_states_per_path=2),
            2,
        )

    def test_deprecated_flag_still_sets_cap(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        with self.assertLogs(
            "sglang.srt.arg_groups.argparse_actions", level="WARNING"
        ) as logs:
            args = parser.parse_args(
                ["--model-path", "dummy", "--mamba-max-states-per-path", "2"]
            )
        self.assertIn("--hybrid-cache-checkpoints-level 2", logs.output[0])
        self.assertEqual(ServerArgs.from_cli_args(args).mamba_max_states_per_path, 2)
        default = parser.parse_args(["--model-path", "dummy"])
        self.assertIsNone(default.mamba_max_states_per_path)
        self.assertEqual(default.hybrid_cache_checkpoints_level, 2)


class TestRequestEndCheckpoints(CustomTestCase):
    def test_level3_publishes_running_checkpoint(self):
        cache, allocator, pool = _build(3)
        req = _running_req(cache, allocator, pool, up_to=8)
        cache.checkpoint(req, up_to=8)
        self.assertEqual(_cached_len(cache, PROMPT), 8)
        self.assertIsNone(req.kv.mamba_last_track_seqlen)
        cache.unlock(req.lock)
        cache.sanity_check()

    def test_level2_defers_running_checkpoint_until_finish(self):
        cache, allocator, pool = _build(2)
        req = _running_req(cache, allocator, pool, up_to=8)
        free_slots = pool.mamba_allocator.available_size()
        cache.checkpoint(req, up_to=8)
        self.assertEqual(_cached_len(cache, PROMPT), 0)
        self.assertEqual(req.kv.mamba_last_track_seqlen, 8)
        self.assertEqual(pool.mamba_allocator.available_size(), free_slots)
        self.assertIsNone(req.lock)
        cache.sanity_check()

        req.finished_reason = FINISH_LENGTH(length=0)
        cache.checkpoint(req, up_to=len(PROMPT))
        self.assertEqual(_cached_len(cache, PROMPT), 8)
        cache.unlock(req.lock)
        cache.sanity_check()

    def test_level2_publishes_branching_point(self):
        """Both strategies publish at the state boundary, not a later token depth."""
        for extra_buffer in (False, True):
            for branching_seqlen in (8, 12):
                with self.subTest(
                    extra_buffer=extra_buffer, branching_seqlen=branching_seqlen
                ):
                    cache, allocator, pool = _build(
                        2, page_size=1, enable_mamba_extra_buffer=extra_buffer
                    )
                    req = _running_req(cache, allocator, pool, up_to=8)
                    req.mamba_branching_seqlen = branching_seqlen
                    if not extra_buffer:
                        self.assertIsNone(req.kv.mamba_last_track_seqlen)
                    cache.checkpoint(req, up_to=16 if extra_buffer else 8)
                    self.assertEqual(
                        _cached_len(cache, PROMPT),
                        8 if branching_seqlen == 8 else 0,
                    )
                    cache.unlock(req.lock)
                    cache.sanity_check()

    def test_level2_publishes_complete_prompt_on_prefill_worker(self):
        cache, allocator, pool = _build(2)
        with get_disagg().override(disaggregation_mode="prefill"):
            partial = _running_req(cache, allocator, pool, up_to=8)
            cache.checkpoint(partial, up_to=8)
            self.assertEqual(_cached_len(cache, PROMPT), 0)

            req = _running_req(cache, allocator, pool, up_to=len(PROMPT))
            cache.checkpoint(req, up_to=len(PROMPT))
        self.assertEqual(_cached_len(cache, PROMPT), len(PROMPT))
        cache.unlock(req.lock)
        cache.sanity_check()

    def test_level2_keeps_every_chunk_with_sliding_window(self):
        cache, allocator, pool = _build(
            2,
            components=(ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA),
            page_size=1,
            sliding_window_size=4,
        )
        req = _running_req(cache, allocator, pool, up_to=8)
        cache.checkpoint(req, up_to=8)
        self.assertEqual(_cached_len(cache, PROMPT), 8)
        cache.unlock(req.lock)
        cache.sanity_check()


# A burst of requests over one shared system prompt: B and C share A's first 12
# tokens and differ after them.
SHARED = 12
B_PROMPT = PROMPT[:SHARED] + [101, 102, 103, 104]
C_PROMPT = PROMPT[:SHARED] + [201, 202, 203, 204]


def _prompt_req(prompt, rid, extend_end=None, cached=0, **fields):
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", prompt),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        **fields,
    )
    req.extend_end = len(prompt) if extend_end is None else extend_end
    req.kv.cache_protected_len = cached
    return req


def _marked(req, live, grid=4, **kwargs):
    mark_shared_prompt_checkpoints([req], [live], grid, **kwargs)
    return req.mamba_publish_prompt


class TestMarkSharedPrompts(CustomTestCase):
    """Which prompts the scheduler marks for publishing at their end."""

    def test_prompt_shared_by_a_waiting_request_is_marked(self):
        a = _prompt_req(PROMPT, "a")
        self.assertTrue(_marked(a, [a, _prompt_req(B_PROMPT, "b")]))

    def test_lone_prompt_is_not_marked(self):
        a = _prompt_req(PROMPT, "a")
        a.mamba_publish_prompt = True  # a stale mark is cleared
        unrelated = _prompt_req([900 + i for i in range(16)], "u")
        # Sharing less than one grid step is no share.
        template = _prompt_req(PROMPT[:3] + [300] * 13, "t")
        self.assertFalse(_marked(a, []))
        self.assertFalse(_marked(a, [a]))
        self.assertFalse(_marked(a, [unrelated, template]))

    def test_share_within_the_cached_prefix_is_not_marked(self):
        # A continuation already matched its first 12 tokens (a multi-turn
        # session); a request sharing only those gains nothing.
        a = _prompt_req(PROMPT, "a", cached=SHARED)
        self.assertFalse(_marked(a, [_prompt_req(B_PROMPT, "b")]))
        # Control: a request sharing the whole prompt does gain.
        self.assertTrue(_marked(a, [_prompt_req(PROMPT + [7], "fork")]))

    def test_branching_point_counts_as_cached(self):
        a = _prompt_req(PROMPT, "a")
        a.mamba_branching_seqlen = SHARED
        self.assertFalse(_marked(a, [_prompt_req(B_PROMPT, "b")]))
        a.mamba_branching_seqlen = 8
        self.assertTrue(_marked(a, [_prompt_req(B_PROMPT, "b")]))

    def test_chunk_short_of_the_prompt_end_is_not_marked(self):
        a = _prompt_req(PROMPT, "a", extend_end=8)
        self.assertFalse(_marked(a, [_prompt_req(B_PROMPT, "b")]))
        a.extend_end = len(PROMPT)
        self.assertTrue(_marked(a, [_prompt_req(B_PROMPT, "b")]))

    def test_other_cache_namespace_is_not_marked(self):
        a = _prompt_req(PROMPT, "a")
        salted = _prompt_req(B_PROMPT, "b", cache_salt="tenant")
        keyed = _prompt_req(B_PROMPT, "c", extra_key="lora")
        self.assertFalse(_marked(a, [salted, keyed]))
        self.assertTrue(_marked(a, [salted, keyed, _prompt_req(B_PROMPT, "d")]))

    def test_unpublishable_prompt_is_not_marked(self):
        a = _prompt_req(PROMPT, "a")
        a.skip_radix_cache_insert = True
        self.assertFalse(_marked(a, [_prompt_req(PROMPT, "b")]))

    def test_comparisons_are_bounded(self):
        a = _prompt_req(PROMPT, "a")
        live = [_prompt_req([900] * 16, "u"), _prompt_req(B_PROMPT, "b")]
        self.assertFalse(_marked(a, live, max_compared=1))
        self.assertTrue(_marked(a, live, max_compared=2))


class TestSharedPromptCheckpoints(CustomTestCase):
    """Level 2 publishes a prompt end that another live request shares."""

    def _burst(self, level=2, mark=True):
        """A finishes its prefill while B waits; then B prefills while C waits.
        Returns B's match after A's prefill and C's cached prefix after B's."""
        cache, allocator, pool = _build(level, mamba_cache_size=16)
        grid = cache.shared_prompt_checkpoint_grid()
        a = _running_req(cache, allocator, pool, len(PROMPT), rid="a")
        b_waiting = _prompt_req(B_PROMPT, "b")
        if mark and grid is not None:
            mark_shared_prompt_checkpoints([a], [[b_waiting]], grid)
        checkpoint_kv_cache(a, cache)
        self.assertFalse(a.mamba_publish_prompt)
        b_match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", B_PROMPT)))
        )

        # B is admitted with that match: it tracks its branching point, if any.
        b = _running_req(cache, allocator, pool, len(B_PROMPT), B_PROMPT, "b")
        b.mamba_branching_seqlen = b_match.mamba_branching_seqlen
        if b.mamba_branching_seqlen is not None:
            b.kv.mamba_last_track_seqlen = b.mamba_branching_seqlen
        if mark and grid is not None:
            mark_shared_prompt_checkpoints([b], [[_prompt_req(C_PROMPT, "c")]], grid)
            # B's own share with C lies within the branch it publishes anyway.
            self.assertFalse(b.mamba_publish_prompt)
        checkpoint_kv_cache(b, cache)
        c_cached = _cached_len(cache, C_PROMPT)
        for req in (a, b):
            if req.lock is not None:
                cache.unlock(req.lock)
        cache.sanity_check()
        return b_match, c_cached

    def test_level2_burst_reuses_the_shared_prefix(self):
        b_match, c_cached = self._burst()
        # A's published prompt gives B the shared KV and its branching point;
        # B publishes there, so C matches the shared prefix.
        self.assertEqual(b_match.full_kv_hit_length, SHARED)
        self.assertEqual(b_match.mamba_branching_seqlen, SHARED)
        self.assertEqual(c_cached, SHARED)

    def test_level2_burst_without_the_mark_shares_nothing(self):
        # Negative control: unmarked, A keeps its prompt until it finishes.
        b_match, c_cached = self._burst(mark=False)
        self.assertEqual(b_match.full_kv_hit_length, 0)
        self.assertIsNone(b_match.mamba_branching_seqlen)
        self.assertEqual(c_cached, 0)

    def test_escape_hatch_restores_deferral(self):
        with envs.SGLANG_MAMBA_PUBLISH_SHARED_PROMPTS.override(False):
            cache, allocator, pool = _build(2)
            self.assertIsNone(cache.shared_prompt_checkpoint_grid())
            req = _running_req(cache, allocator, pool, up_to=len(PROMPT))
            req.mamba_publish_prompt = True
            cache.checkpoint(req, up_to=len(PROMPT))
        self.assertEqual(_cached_len(cache, PROMPT), 0)
        cache.sanity_check()

    def test_level2_marked_prompt_publishes_its_end(self):
        for extra_buffer in (True, False):
            with self.subTest(extra_buffer=extra_buffer):
                cache, allocator, pool = _build(
                    2,
                    page_size=4 if extra_buffer else 1,
                    enable_mamba_extra_buffer=extra_buffer,
                )
                self.assertEqual(cache.shared_prompt_checkpoint_grid(), 4)
                req = _running_req(cache, allocator, pool, up_to=len(PROMPT))
                req.mamba_publish_prompt = True
                free_slots = pool.mamba_allocator.available_size()
                cache.checkpoint(req, up_to=len(PROMPT))
                # A request whose prompt extends this one (another sample of
                # it) matches all of it.
                self.assertEqual(_cached_len(cache, PROMPT + [7, 8]), len(PROMPT))
                # The tree takes one state; the request keeps decoding on its own.
                self.assertEqual(
                    pool.mamba_allocator.available_size(), free_slots - 1
                )
                self.assertIsNone(req.kv.mamba_last_track_seqlen)
                cache.unlock(req.lock)
                cache.sanity_check()

    def test_level2_mark_ignores_a_chunk_short_of_the_prompt_end(self):
        cache, allocator, pool = _build(2)
        req = _running_req(cache, allocator, pool, up_to=8)
        req.mamba_publish_prompt = True
        free_slots = pool.mamba_allocator.available_size()
        cache.checkpoint(req, up_to=8)
        self.assertEqual(_cached_len(cache, PROMPT), 0)
        self.assertEqual(pool.mamba_allocator.available_size(), free_slots)
        cache.sanity_check()

    def test_level2_lone_request_still_defers(self):
        cache, allocator, pool = _build(2)
        req = _running_req(cache, allocator, pool, up_to=len(PROMPT))
        mark_shared_prompt_checkpoints(
            [req], [[req]], cache.shared_prompt_checkpoint_grid()
        )
        self.assertFalse(req.mamba_publish_prompt)
        free_slots = pool.mamba_allocator.available_size()
        checkpoint_kv_cache(req, cache)
        self.assertEqual(_cached_len(cache, PROMPT), 0)
        self.assertEqual(pool.mamba_allocator.available_size(), free_slots)
        self.assertEqual(req.kv.mamba_last_track_seqlen, len(PROMPT))
        self.assertIsNone(req.lock)
        cache.sanity_check()

    def test_level2_multi_turn_continuation_still_defers(self):
        """Turn 2 extends turn 1's cached end; another session shares only the
        system prompt, so turn 2 keeps one state on its path until it ends."""
        cache, allocator, pool = _build(2, mamba_cache_size=16)
        turn1 = _running_req(cache, allocator, pool, len(PROMPT), rid="turn1")
        finish_req(cache, turn1, len(PROMPT))
        self.assertEqual(_cached_len(cache, PROMPT), len(PROMPT))

        turn2_prompt = PROMPT + list(range(17, 25))
        turn2 = _running_req(cache, allocator, pool, len(turn2_prompt), turn2_prompt)
        turn2.kv.cache_protected_len = len(PROMPT)
        other_session = _prompt_req(PROMPT[:8] + [300] * 8, "other")
        grid = cache.shared_prompt_checkpoint_grid()
        mark_shared_prompt_checkpoints([turn2], [[other_session]], grid)
        self.assertFalse(turn2.mamba_publish_prompt)
        # Control: a fork of turn 2's new tokens would have marked it.
        fork = _prompt_req(turn2_prompt + [5], "fork")
        mark_shared_prompt_checkpoints([turn2], [[fork]], grid)
        self.assertTrue(turn2.mamba_publish_prompt)

        turn2.mamba_publish_prompt = False
        free_slots = pool.mamba_allocator.available_size()
        cache.checkpoint(turn2, up_to=len(turn2_prompt))
        self.assertEqual(pool.mamba_allocator.available_size(), free_slots)
        self.assertEqual(_cached_len(cache, turn2_prompt), len(PROMPT))
        cache.sanity_check()

    def test_level3_and_sliding_window_need_no_marks(self):
        self.assertIsNone(_build(3)[0].shared_prompt_checkpoint_grid())
        swa = _build(
            2,
            components=(ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA),
            page_size=1,
            sliding_window_size=4,
        )[0]
        self.assertIsNone(swa.shared_prompt_checkpoint_grid())
        cache = _build(2)[0]
        with get_disagg().override(disaggregation_mode="prefill"):
            self.assertIsNone(cache.shared_prompt_checkpoint_grid())

    def test_level3_burst_unchanged(self):
        # Level 3 publishes every prompt end; it marks nothing.
        b_match, c_cached = self._burst(level=3)
        self.assertEqual(b_match.full_kv_hit_length, SHARED)
        self.assertEqual(b_match.mamba_branching_seqlen, SHARED)
        self.assertEqual(c_cached, SHARED)


# The decode checkpoint grid is lcm(chunk, page, interval) = the interval. It is
# twice the draft length so two verifies in a row never both cross a boundary,
# as with a served model, whose grid is at least the 64-token FLA chunk.
SPEC_PAGE = 4
SPEC_GRID = 8
SPEC_DRAFT = 4
# (level, strategy, overlap): lazy tracking runs with the overlap scheduler only.
SPEC_CONFIGS = tuple(
    (level, strategy, overlap)
    for level in (2, 3)
    for strategy, overlap in (
        ("extra_buffer", True),
        ("extra_buffer", False),
        ("extra_buffer_lazy", True),
    )
)


class _SpecRequest:
    """One request driven through the real tracking code: the prompt's checkpoint,
    spec verifies, and a stop inside the last accepted run. ``written`` models the
    device: how many tokens the state in each tracking slot has consumed."""

    def __init__(self, level, strategy, overlap, prompt_len, publish_prompt=False):
        self.publish_prompt = publish_prompt
        self.lazy = strategy == "extra_buffer_lazy"
        self.overlap = overlap
        cfg = CacheConfig(
            page_size=SPEC_PAGE,
            components=(ComponentType.FULL, ComponentType.MAMBA),
            enable_mamba_extra_buffer=True,
            enable_mamba_extra_buffer_lazy=self.lazy,
            enable_overlap_schedule=overlap,
            kv_size=256,
            max_context_len=128,
            mamba_cache_size=8,
            hybrid_cache_checkpoints_level=level,
        )
        with patch(
            "sglang.test.kits.unified_radix_cache.fixtures.get_device",
            return_value="cpu",
        ):
            self.cache, self.allocator, self.pool = build_fixture(
                cfg, mamba_cache_chunk_size=SPEC_PAGE
            )
        self.storage = kv_storage_of(self.pool, self.allocator)
        self.written = {}
        self.next_token = 1000
        self.prompt = list(range(1, prompt_len + 1))
        sampling_params = SamplingParams(temperature=0, max_new_tokens=1000)
        sampling_params.normalize(None)
        req = Req(
            rid="spec",
            origin_input_text="",
            origin_input_ids=array("q", self.prompt),
            sampling_params=sampling_params,
        )
        acquire_rows(self.pool, self.pool.state, [req])
        req.output_ids = array("q")
        self._take_kv(req, 0, prompt_len)
        req.kv.cache_protected_len = 0
        req.last_node = self.cache.root_node_handle()
        req.lock = None
        req.extend_end = prompt_len
        self.req = req
        self.processor = SchedulerBatchResultProcessor(
            is_generation=True,
            disaggregation_mode=None,
            model_config=SimpleNamespace(
                think_end_ids=None, self_opened_think_start_ids=None
            ),
            token_to_kv_pool_allocator=MagicMock(),
            tree_cache=self.cache,
            hisparse_coordinator=None,
            beam_coordinator=MagicMock(),
            request_retirer=release_only_retirer(tree_cache=self.cache),
            req_to_token_pool=None,
            kv_weight_version_tracker=None,
            decode_offload_manager=None,
            metrics_collector=None,
            metrics_reporter=MagicMock(),
            draft_worker=None,
            model_worker=SimpleNamespace(),
            logprob_result_processor=None,
            output_streamer=MagicMock(),
            abort_request=lambda *args, **kwargs: None,
            still_in_batch=lambda req: True,
            prefill_launched_reqs=KeyedReqList(ListKind.PREFILL_LAUNCHED),
        )

    def _take_kv(self, req, start, num_tokens):
        self.pool.write(
            (req.kv.req_pool_idx, slice(start, start + num_tokens)),
            self.allocator.alloc(num_tokens),
        )
        req.kv.kv_committed_len = start + num_tokens
        req.kv.kv_allocated_len = start + num_tokens

    def _slot(self, position):
        return int(self.req.kv.mamba_ping_pong_track_buffer[position])

    def prefill(self):
        """Track the prompt's aligned end as ScheduleBatch's extend does, then cache
        the request as the prefill result does: level 3 donates the state, level 2
        keeps it in the request."""
        req = self.req
        boundary = len(self.prompt) // SPEC_GRID * SPEC_GRID
        position = req.kv.mamba_next_track_idx
        self.storage.advance(
            req,
            tracking=TrackingAdvance(
                boundary=boundary,
                last_position=position,
                next_position=position
                if self.lazy
                else self.pool.get_mamba_ping_pong_other_idx(position),
            ),
        )
        self.written[self._slot(position)] = boundary
        # The token the prefill samples has no KV yet.
        req.output_ids.append(999)
        req.refresh_fill_ids()
        # Left set through decode: only the prompt's checkpoint reads it.
        req.mamba_publish_prompt = self.publish_prompt
        self.cache.checkpoint(req, up_to=len(self.prompt))

    def plan(self, device_seq_len):
        """Launch a verify: pick the slot its boundary crossing writes."""
        batch = ScheduleBatch(reqs=[self.req])
        batch.req_to_token_pool = self.pool
        batch.token_to_kv_pool_allocator = self.allocator
        batch.tree_cache = self.cache
        batch.req_pool_indices = torch.tensor([self.req.kv.req_pool_idx])
        batch.seq_lens = torch.tensor([device_seq_len])
        batch.mamba_track_mask_cpu = None
        batch.spec_algorithm = SimpleNamespace(
            is_none=lambda: False, is_dflash=lambda: False
        )
        if self.lazy:
            batch.mamba_lazy_spec_prepare(SPEC_GRID, SPEC_DRAFT)
        prepare_mamba_track_for_verify(batch)
        return batch

    def commit(self, batch, accept_len):
        """The verify's state commit, with the index math the forward uses."""
        accept_index = torch.full((1, SPEC_DRAFT), -1)
        accept_index[0, :accept_len] = torch.arange(accept_len)
        _, steps = _verify_commit_step_indices(
            batch=batch,
            accept_index=accept_index,
            accept_lens=torch.tensor([accept_len]),
            draft_token_num=SPEC_DRAFT,
        )
        slot = int(batch.mamba_track_indices[0])
        # Slot 0 is the reserved sink a no-slot plan writes.
        if int(steps[0]) >= 0 and slot > 0:
            self.written[slot] = int(batch.seq_lens[0]) + int(steps[0]) + 1

    def process(self, batch, accept_len, stop_at=None):
        """The result path: commit the whole accepted run, finish on a stop at
        ``stop_at`` inside it, and let the result handler track the boundary and
        release a finished request into the tree."""
        req = self.req
        tokens = list(range(self.next_token, self.next_token + accept_len))
        self.next_token += accept_len
        self._take_kv(req, req.kv.kv_committed_len, accept_len)
        req.output_ids.extend(tokens)
        if stop_at is not None:
            req.finished_reason = FINISH_MATCHED_TOKEN(matched=tokens[stop_at])
            req.finished_len = len(req.output_ids) - accept_len + stop_at + 1
        result = GenerationBatchResult(
            next_token_ids=torch.tensor(tokens),
            accept_lens=torch.tensor([accept_len]),
            speculative_num_draft_tokens=SPEC_DRAFT,
        )
        result.num_correct_drafts_per_req_cpu = [accept_len - 1]
        result.num_non_draft_tokens_per_req = 1
        result.grammar_retained_tokens = None
        self.processor._handle_finish_state_updated_req(
            req, batch, result, 0, None, accept_len
        )

    def run(self, accept_lens, stop_at):
        self.prefill()
        device_seq_len = len(self.prompt)
        batch = self.plan(device_seq_len)
        for step, accept_len in enumerate(accept_lens):
            self.commit(batch, accept_len)
            device_seq_len += accept_len
            last = step == len(accept_lens) - 1
            # The overlap scheduler launches the next verify before this result.
            in_flight = self.plan(device_seq_len) if self.overlap else None
            self.process(batch, accept_len, stop_at if last else None)
            if last:
                if in_flight is not None:
                    # It still runs for the finished request.
                    self.commit(in_flight, SPEC_DRAFT)
                return
            batch = in_flight or self.plan(device_seq_len)

    def tree_states(self):
        """(key length, tokens the slot's state consumed) for each state on the
        request's path, deepest first."""
        tokens = self.prompt + list(self.req.output_ids)
        states = []
        end = len(tokens)
        while end > 0:
            match = self.cache.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", tokens[:end])))
            )
            if match.device_prefix_len == 0:
                break
            slot = self.cache.tree_core.get_component_device_value(
                match.best_match_node, ComponentType.MAMBA
            )
            states.append(
                (match.device_prefix_len, self.written.get(int(slot.reshape(-1)[0])))
            )
            end = match.device_prefix_len - 1
        return states


def _spec_request_states(
    level, strategy, overlap, prompt_len, accept_lens, stop_at, publish_prompt=False
):
    req = _SpecRequest(level, strategy, overlap, prompt_len, publish_prompt)
    with get_context().override_server_args(
        mamba_radix_cache_strategy=strategy,
        mamba_track_interval=SPEC_GRID,
        _mamba_cache_chunk_size=SPEC_PAGE,
        speculative_algorithm="EAGLE",
        speculative_num_draft_tokens=SPEC_DRAFT,
        disable_overlap_schedule=not overlap,
        hybrid_cache_checkpoints_level=level,
    ):
        req.run(accept_lens, stop_at)
    return req.tree_states()


class TestSpecDecodeRequestEndState(CustomTestCase):
    """Level 2 keeps the prompt's checkpoint in the request through every verify
    until it finishes, and a stop inside a verify's accepted run finishes it
    mid-run. A state the tree keeps under one length but written at another
    resumes the next turn from the wrong state, with no error."""

    def test_every_kept_state_holds_its_key(self):
        for level, strategy, overlap in SPEC_CONFIGS:
            with self.subTest(level=level, strategy=strategy, overlap=overlap):
                for prompt_len, accept_lens in itertools.product(
                    (5, 8, 11), itertools.product((1, 3, 4), repeat=3)
                ):
                    for stop_at in range(accept_lens[-1]):
                        states = _spec_request_states(
                            level, strategy, overlap, prompt_len, accept_lens, stop_at
                        )
                        wrong = [(key, got) for key, got in states if key != got]
                        self.assertFalse(
                            wrong, f"{prompt_len=} {accept_lens=} {stop_at=}: {wrong}"
                        )

    def test_level2_short_answer_keeps_the_prompt_state(self):
        # Prompt end 8 is tracked; three verifies reach 14, short of 16.
        for strategy, overlap in (("extra_buffer", True), ("extra_buffer_lazy", True)):
            with self.subTest(strategy=strategy):
                states = _spec_request_states(2, strategy, overlap, 11, (1, 1, 1), 0)
                self.assertEqual(states, [(8, 8)])

    def test_level2_finish_keeps_the_latest_decode_state(self):
        # The second verify runs 15 -> 19 past 16; the finish keeps that state
        # and drops the prompt's at 8 (one state per path).
        for strategy in ("extra_buffer", "extra_buffer_lazy"):
            with self.subTest(strategy=strategy):
                states = _spec_request_states(2, strategy, True, 11, (4, 4, 1), 0)
                self.assertEqual(states, [(16, 16)])

    def test_level3_finish_adds_the_latest_decode_state(self):
        # The prefill donated the prompt's state at 8; the finish adds the state the
        # second verify tracked at 16 and keeps both on the path.
        for strategy, overlap in (
            ("extra_buffer", True),
            ("extra_buffer", False),
            ("extra_buffer_lazy", True),
        ):
            with self.subTest(strategy=strategy, overlap=overlap):
                states = _spec_request_states(3, strategy, overlap, 11, (4, 4, 1), 0)
                self.assertEqual(states, [(16, 16), (8, 8)])

    def test_level2_shared_prompt_states_hold_their_keys(self):
        # A marked prompt is donated at prefill as level 3 donates it; the
        # request then decodes on a fresh tracking slot to its finish.
        for level, strategy, overlap in SPEC_CONFIGS:
            if level != 2:
                continue
            with self.subTest(strategy=strategy, overlap=overlap):
                for prompt_len, accept_lens in itertools.product(
                    (5, 8, 11), itertools.product((1, 4), repeat=3)
                ):
                    for stop_at in range(accept_lens[-1]):
                        states = _spec_request_states(
                            2,
                            strategy,
                            overlap,
                            prompt_len,
                            accept_lens,
                            stop_at,
                            publish_prompt=True,
                        )
                        wrong = [(key, got) for key, got in states if key != got]
                        self.assertFalse(
                            wrong, f"{prompt_len=} {accept_lens=} {stop_at=}: {wrong}"
                        )

    def test_level2_shared_prompt_keeps_the_prompt_state_past_the_finish(self):
        # Unmarked, the finish keeps only (16, 16) (see above). Marked, the
        # prompt's state at 8 is in the tree from the prefill on, and the
        # request's lock on it keeps the cap from dropping it at the finish.
        for strategy in ("extra_buffer", "extra_buffer_lazy"):
            with self.subTest(strategy=strategy):
                states = _spec_request_states(
                    2, strategy, True, 11, (4, 4, 1), 0, publish_prompt=True
                )
                self.assertEqual(states, [(16, 16), (8, 8)])

    def test_level2_stop_before_the_crossing_falls_back_to_the_prompt_state(self):
        # The second verify runs 13 -> 17 past 16, but the stop at its first token
        # leaves a 15-token key: the finish keeps the prompt's state at 8.
        states = _spec_request_states(2, "extra_buffer", True, 11, (2, 4), 0)
        self.assertEqual(states, [(8, 8)])


if __name__ == "__main__":
    unittest.main()
