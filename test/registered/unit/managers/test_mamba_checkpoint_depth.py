"""The donated mamba checkpoint depth must land on the tree page.

DCP widens the tree page past the mamba chunk grid. A checkpoint picked on the
finer grid names a depth no radix node can carry, so it gets attached to the
preceding node and a later request resumes from a state that already covers
tokens past that node. The same happens when the depth is measured from a
prefix that a chunked prefill left on the scheduler page but off the widened
tree page: 64 + 512 = 576 is no 512-page boundary, and the finished request
then frees two kv-row segments that share the page holding 576.
"""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec, patch

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.schedule_batch import (
    Req,
    ScheduleBatch,
    _MambaRadixCacheV2TrackEntry,
)
from sglang.srt.managers.schedule_policy import (
    CacheAwarePolicy,
    PrefillAdder,
    mamba_prefill_chunk_grid,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
from sglang.srt.runtime_context import get_context, mamba_track_grid
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import (
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

CHUNK = 64


def _prepare_extend(
    *,
    tree_page: int,
    prefix_len: int,
    extend_len: int,
    dcp_enabled: bool = False,
    next_idx: int = 0,
    prompt_len: int | None = None,
    branching: int | None = None,
    checkpoint_pos: int | None = None,
) -> tuple[_MambaRadixCacheV2TrackEntry, Req]:
    """Run one extend through the tracker; return its track entry and the request."""
    server_args = ServerArgs(model_path="dummy", page_size=CHUNK)
    # The property would otherwise load the HF config for the dummy model.
    server_args._mamba_cache_chunk_size = CHUNK
    set_global_server_args_for_scheduler(server_args)

    sampling_params = SamplingParams(max_new_tokens=1)
    sampling_params.normalize(None)
    req = Req(
        rid="req",
        origin_input_text="",
        origin_input_ids=array(
            "q", [1] * (prefix_len + extend_len if prompt_len is None else prompt_len)
        ),
        sampling_params=sampling_params,
        vocab_size=128,
        mamba_checkpoint_pos=checkpoint_pos,
    )
    req.refresh_fill_ids()
    req.prefix_len = prefix_len
    req.extend_end = prefix_len + extend_len
    req.kv.mamba_next_track_idx = next_idx
    req.mamba_branching_seqlen = branching

    batch = ScheduleBatch(reqs=[req])
    batch.token_to_kv_pool_allocator = SimpleNamespace(storage=None)
    batch.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(mamba_chunk_size=CHUNK)
    )
    batch.tree_cache = SimpleNamespace(page_size=tree_page)
    batch.req_to_token_pool = MagicMock()
    batch.req_to_token_pool.get_mamba_ping_pong_other_idx.return_value = 1 - next_idx

    with patch(
        "sglang.srt.managers.schedule_batch.get_parallel",
        return_value=SimpleNamespace(dcp_enabled=dcp_enabled),
    ):
        entry = batch._mamba_radix_cache_v2_req_prepare_for_extend(req)
    return entry, req


def _track_seqlen(**kwargs) -> int | None:
    """The donated depth of one extend, or None when it donates no checkpoint."""
    entry, req = _prepare_extend(**kwargs)
    if not entry.track_mask:
        assert req.kv.mamba_last_track_seqlen is None
        return None
    return req.kv.mamba_last_track_seqlen


class TestMambaCheckpointDepth(unittest.TestCase):
    def test_widened_tree_page_moves_the_donated_depth_onto_it(self):
        # 4066 tokens past a 16384 prefix: the chunk grid would stop at 20416,
        # which a 256-token page cannot name.
        depth = _track_seqlen(
            tree_page=256,
            prefix_len=16384,
            extend_len=4066,
            dcp_enabled=True,
        )
        self.assertEqual(depth % 256, 0)
        self.assertEqual(depth, 20224)

    def test_unwidened_tree_page_keeps_the_chunk_grid(self):
        depth = _track_seqlen(tree_page=CHUNK, prefix_len=16384, extend_len=4066)
        self.assertEqual(depth, 20416)

    def test_dcp_depth_satisfies_tree_page_and_snapshot_grids(self):
        # (tree_page, prefix_len, extend_len) -> donated depth, or None when no
        # prefix-relative kernel snapshot lands on an absolute tree page.
        for tree_page, prefix_len, extend_len, expected in (
            (512, 64, 540, 512),
            (512, 448, 128, 512),
            (512, 64, 400, None),
            (512, 37, 540, None),
        ):
            with self.subTest(prefix_len=prefix_len, extend_len=extend_len):
                self.assertEqual(
                    _track_seqlen(
                        tree_page=tree_page,
                        prefix_len=prefix_len,
                        extend_len=extend_len,
                        dcp_enabled=True,
                    ),
                    expected,
                )

    def test_non_dcp_depth_remains_relative_to_chunked_prefill_prefix(self):
        # Active requests can stop off the absolute grid between prefill chunks.
        # Keep donating their prefix-relative snapshots when DCP is disabled.
        self.assertEqual(
            _track_seqlen(tree_page=16, prefix_len=4253, extend_len=16384),
            20637,
        )


PREFIX, EXTEND = 16384, 4066  # prompt end 20450


def _track(pos, **kwargs):
    kwargs = {
        "tree_page": CHUNK,
        "prefix_len": PREFIX,
        "extend_len": EXTEND,
        **kwargs,
    }
    entry, req = _prepare_extend(checkpoint_pos=pos, **kwargs)
    depth = req.kv.mamba_last_track_seqlen if entry.track_mask else None
    return depth, req.mamba_tail_seqlen, entry.track_seqlen


class TestMambaCheckpointPosDepth(CustomTestCase):
    def test_position_rounds_down_onto_the_grid_past_the_prefix_and_reads_h(self):
        for tree_page, pos, expected in (
            (CHUNK, 18500, 18496),
            (CHUNK, 16384 + CHUNK, 16384 + CHUNK),
            (256, 18500, 18432),
            (256, 20449, 20224),
        ):
            with self.subTest(tree_page=tree_page, pos=pos):
                depth, tail, track_seqlen = _track(pos, tree_page=tree_page)
                self.assertEqual((depth, tail), (expected, expected))
                # Intermediate h states use the following tracking position.
                self.assertEqual(track_seqlen, expected + 1)

    def test_position_replaces_the_branching_point_of_the_final_forward(self):
        self.assertEqual(_track(None, branching=17408)[0], 17408)
        self.assertEqual(_track(18500, branching=17408)[0], 18496)

    def test_invalid_position_keeps_the_native_depth(self):
        for label, pos in (
            ("at the cached prefix", 16384),
            ("a grid step short of the prefix", 16384 + CHUNK - 1),
            ("past the prompt end", 30000),
        ):
            with self.subTest(case=label):
                self.assertEqual(_track(pos)[:2], (20416, None))

    def test_position_at_the_prompt_end_keeps_the_forward_end_checkpoint(self):
        # Level 2 publishes the end checkpoint as a tail without changing its read.
        for extend_len in (EXTEND, 4096):
            with self.subTest(extend_len=extend_len):
                main = _track(None, extend_len=extend_len)
                depth, tail, track_seqlen = _track(
                    PREFIX + extend_len, extend_len=extend_len
                )
                self.assertEqual((depth, tail), (main[0], main[0]))
                self.assertEqual(track_seqlen, main[2])

    def test_forward_that_stops_short_of_the_prompt_end_ignores_the_position(self):
        # The requested position is in this chunk, but the prompt ends in a later one.
        depth, tail, _ = _track(8000, prefix_len=0, extend_len=16384, prompt_len=20000)
        self.assertEqual((depth, tail), (16384, None))

    def test_dcp_keeps_the_native_depth(self):
        self.assertEqual(
            _track(18500, tree_page=256, dcp_enabled=True)[:2], (20224, None)
        )


class TestMambaCheckpointPosReachesReq(CustomTestCase):
    """Checkpoint positions survive normalization, tokenization, and scheduler IPC."""

    def setUp(self):
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)

    def _tokenized(self, **fields):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.preferred_sampling_params = None
        manager.sampling_params_class = SamplingParams
        manager.tokenizer = None
        manager.fake_bootstrap_room_counter = 0
        manager.model_config = SimpleNamespace(vocab_size=128, token_id_vocab_size=128)
        fields.setdefault("sampling_params", {"max_new_tokens": 1})
        obj = GenerateReqInput(**fields)
        obj.normalize_batch_and_arguments()
        manager.rid_to_state = {
            rid: SimpleNamespace(time_stats=APIServerReqTimeStats())
            for rid in ([obj.rid] if obj.is_single else obj.rid)
        }
        items = [obj] if obj.is_single else [obj[i] for i in range(len(obj.rid))]
        out = []
        for item in items:
            tokenized = manager._create_tokenized_object(item, "", list(item.input_ids))
            tokenized.time_stats = None  # Time stats use a separate pickle hop.
            tokenized.wrap_pickle_fields()
            tokenized = msgpack_decode(msgpack_encode(tokenized))
            tokenized.unwrap_pickle_fields()
            out.append(tokenized)
        return out

    def _handle(self, recv_req) -> Req:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_session_radix_cache = False
        scheduler.model_config = SimpleNamespace(
            hf_eos_token_id={1}, vocab_size=128, token_id_vocab_size=128
        )
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
        scheduler.tokenizer = None
        scheduler.dllm_config = None
        built = []

        def build(*args, **kwargs):
            built.append(Req(*args, **kwargs))
            return built[-1]

        # Stop before queueing; only Req construction is under test.
        class _Stop(Exception):
            pass

        scheduler._maybe_namespace_elastic_radix_cache = MagicMock(side_effect=_Stop)
        with (
            patch("sglang.srt.managers.scheduler.Req", side_effect=build),
            patch(
                "sglang.srt.managers.scheduler.BeamCoordinator.request_beam_width",
                return_value=1,
            ),
            self.assertRaises(_Stop),
        ):
            scheduler.handle_generate_request(recv_req)
        (req,) = built
        return req

    def test_single_request_reaches_req(self):
        (tokenized,) = self._tokenized(input_ids=[1, 2, 3, 4], mamba_checkpoint_pos=2)
        self.assertEqual(tokenized.mamba_checkpoint_pos, 2)
        self.assertEqual(self._handle(tokenized).mamba_checkpoint_pos, 2)

    def test_unset_reaches_req_as_none(self):
        (tokenized,) = self._tokenized(input_ids=[1, 2, 3, 4])
        self.assertIsNone(tokenized.mamba_checkpoint_pos)
        self.assertIsNone(self._handle(tokenized).mamba_checkpoint_pos)

    def test_each_batch_item_keeps_its_own_position(self):
        tokenized = self._tokenized(
            input_ids=[[1, 2, 3], [4, 5, 6, 7], [8, 9]],
            mamba_checkpoint_pos=[2, None, 1],
        )
        self.assertEqual(
            [self._handle(t).mamba_checkpoint_pos for t in tokenized], [2, None, 1]
        )

    def test_parallel_samples_each_carry_the_position(self):
        tokenized = self._tokenized(
            input_ids=[[1, 2, 3], [4, 5, 6, 7]],
            mamba_checkpoint_pos=[2, 3],
            sampling_params={"n": 2, "max_new_tokens": 1},
        )
        self.assertEqual(len(tokenized), 4)
        self.assertEqual(
            [self._handle(t).mamba_checkpoint_pos for t in tokenized], [2, 3, 2, 3]
        )

    def test_parallel_samples_of_one_prompt_each_carry_the_position(self):
        tokenized = self._tokenized(
            input_ids=[1, 2, 3],
            mamba_checkpoint_pos=2,
            sampling_params={"n": 2, "max_new_tokens": 1},
        )
        self.assertEqual(
            [self._handle(t).mamba_checkpoint_pos for t in tokenized], [2, 2]
        )


class TestChunkedPrefillStaysOnTheChunkGrid(unittest.TestCase):
    """The scheduler side of the same constraint.

    `test_dcp_depth_satisfies_tree_page_and_snapshot_grids` shows a prefix of
    37 donates nothing. A prefix only reaches 37 because a prefill chunk was
    cut there, and every later chunk of that request inherits the offset, so
    one unaligned cut costs the request every checkpoint it would ever donate.
    """

    def _adder(self, grid: int = CHUNK, *, strict: bool = False) -> PrefillAdder:
        adder = object.__new__(PrefillAdder)
        adder.mamba_chunk_grid = grid
        adder.strict_chunk_grid = strict
        return adder

    def _floor(self, trunc_len: int, remaining: int, grid: int = CHUNK) -> int:
        return self._adder(grid).floor_chunked_tokens(trunc_len, remaining)

    def test_a_cut_is_rounded_down_onto_the_grid(self):
        self.assertEqual(self._floor(100, 8192), 64)
        self.assertEqual(self._floor(8010, 100000), 8000)

    def test_a_cut_already_on_the_grid_is_untouched(self):
        self.assertEqual(self._floor(8192, 100000), 8192)

    def test_the_chunk_that_finishes_the_request_may_stop_anywhere(self):
        self.assertEqual(self._floor(37, 37), 37)

    def test_a_cut_shorter_than_the_grid_still_makes_progress(self):
        # Rounding to zero would stall the request forever; the donation is
        # the cheaper thing to lose.
        self.assertEqual(self._floor(40, 8192), 40)

    def test_no_rounding_without_the_mamba_extra_buffer(self):
        self.assertEqual(self._floor(100, 8192, grid=1), 100)

    def _published_grid(self, *, dcp_size: int, strategy: str = "extra_buffer") -> int:
        with get_context().override_server_args(
            disable_radix_cache=False,
            mamba_radix_cache_strategy=strategy,
            dcp_size=dcp_size,
            _mamba_cache_chunk_size=CHUNK,
        ):
            return mamba_prefill_chunk_grid()

    def test_dcp_extra_buffer_uses_the_chunk_grid(self):
        self.assertEqual(self._published_grid(dcp_size=2), CHUNK)
        self.assertEqual(
            self._floor(100, 8192, grid=self._published_grid(dcp_size=2)), 64
        )

    def test_non_dcp_extra_buffer_leaves_the_cut(self):
        # #40907 already donates from a prefix-relative snapshot here, so the
        # cut must stay where the prefill budget put it.
        self.assertEqual(self._published_grid(dcp_size=1), 1)
        self.assertEqual(
            self._floor(100, 8192, grid=self._published_grid(dcp_size=1)), 100
        )

    def test_non_dcp_does_not_change_truncation_alignment(self):
        # The old gate would write the chunk size into a still-empty alignment.
        scheduler = object.__new__(Scheduler)
        scheduler.truncation_align_size = None
        with get_context().override_server_args(
            disable_radix_cache=False,
            mamba_radix_cache_strategy="extra_buffer",
            dcp_size=1,
            _mamba_cache_chunk_size=CHUNK,
        ):
            scheduler.init_mamba_truncation_align()
        self.assertIsNone(scheduler.truncation_align_size)

    def test_dcp_folds_the_chunk_into_truncation_alignment(self):
        scheduler = object.__new__(Scheduler)
        scheduler.truncation_align_size = 48
        with get_context().override_server_args(
            disable_radix_cache=False,
            mamba_radix_cache_strategy="extra_buffer",
            dcp_size=2,
            _mamba_cache_chunk_size=CHUNK,
        ):
            scheduler.init_mamba_truncation_align()
        self.assertEqual(scheduler.truncation_align_size, 192)

    def _window_tree(self, *, windows=(("swa", 512),), checkpoints=256):
        # Specced to the real cache: the grid asks only what it defines.
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        tree = create_autospec(UnifiedRadixCache, instance=True)
        tree.page_size = CHUNK
        tree.window_sub_pools.return_value = windows
        tree.window_checkpoint_interval.return_value = checkpoints
        return tree

    def _unified_grid(self, tree, *, unified=True):
        with get_context().override_server_args(
            disable_radix_cache=False,
            mamba_radix_cache_strategy="extra_buffer",
            dcp_size=1,
            enable_unified_memory=unified,
            _mamba_cache_chunk_size=CHUNK,
        ):
            return mamba_prefill_chunk_grid(tree)

    def test_unified_windows_behind_checkpoints_cut_on_the_grid(self):
        # A prefill retires its window rows behind its latest checkpoint, and
        # a chunk shorter than the grid takes none, so every cut lands on it.
        self.assertEqual(self._unified_grid(self._window_tree()), CHUNK)
        # No windows, no checkpoints (no prefix sharing), or no unified
        # memory: the cut stays where the budget put it.
        self.assertEqual(self._unified_grid(self._window_tree(windows=())), 1)
        self.assertEqual(self._unified_grid(self._window_tree(checkpoints=0)), 1)
        self.assertEqual(self._unified_grid(self._window_tree(), unified=False), 1)
        self.assertEqual(self._unified_grid(None), 1)

    def test_a_chunk_below_the_checkpoint_grid_is_refused_at_startup(self):
        scheduler = object.__new__(Scheduler)
        scheduler.truncation_align_size = None
        scheduler.tree_cache = self._window_tree()
        scheduler.chunked_prefill_size = CHUNK // 2
        scheduler.disaggregation_mode = None
        with get_context().override_server_args(
            disable_radix_cache=False,
            mamba_radix_cache_strategy="extra_buffer",
            dcp_size=1,
            enable_unified_memory=True,
            _mamba_cache_chunk_size=CHUNK,
        ):
            scheduler.init_mamba_truncation_align()
            self.assertEqual(scheduler.truncation_align_size, CHUNK)
            with self.assertRaisesRegex(ValueError, "checkpoint grid"):
                scheduler.check_truncation_align_fits_chunk()

    def test_a_continuation_never_runs_below_a_strict_grid(self):
        """REGRESSION: where windows retire only behind checkpoints, a pass's
        runtime limits (here a shortest-prefill reservation) must not cut a
        continuing request below the checkpoint grid, nor starve it there."""
        from sglang.srt.managers.schedule_batch import prefill_checkpoint_boundary
        from sglang.srt.managers.schedule_policy import SchedulePolicy

        # The chunk reservation reads the encoder replay flag.
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        policy = object.__new__(SchedulePolicy)
        policy.policy = CacheAwarePolicy.SHORTEST_PREFILL_FIRST
        policy.tree_cache = SimpleNamespace(staged_storage_prefix_end=lambda handle: 0)
        policy.shortest_prefill_aging_factor = 0.0
        policy.shortest_prefill_load_scaling = None
        adder = self._adder(strict=True)
        prompt, prefix, chunks = 4096, 0, []
        while prefix < prompt:
            chunked = SimpleNamespace(
                full_untruncated_fill_ids=[0] * prompt,
                prefix_len=prefix,
                arrival_processed_tokens=0,
            )
            arrival = SimpleNamespace(
                origin_input_ids=[0] * 32,
                output_ids=[],
                num_matched_prefix_tokens=0,
                cache_request_handle=None,
                arrival_processed_tokens=0,
            )
            limit = policy.shortest_prefill_chunk_limit(
                chunked, [arrival], 64, 16, processed_tokens=0
            )
            self.assertEqual(limit, 32)
            remaining = prompt - prefix
            limit = adder.grid_continuation_limit(limit, remaining)
            chunk = adder.floor_chunked_tokens(min(remaining, limit), remaining)
            chunks.append(chunk)
            prefix += chunk
        self.assertEqual(set(chunks), {64})
        self.assertTrue(
            all(
                prefill_checkpoint_boundary(i * 64, 64, CHUNK) is not None
                for i in range(len(chunks))
            )
        )
        # The last chunk may be shorter; without the strict grid the cut stays.
        self.assertEqual(adder.grid_continuation_limit(32, 40), 40)
        self.assertEqual(self._adder().grid_continuation_limit(32, 4096), 32)

    def test_every_chunk_of_a_cut_up_request_can_still_donate(self):
        # Walk a 100k prompt through chunks the scheduler cut to whatever the
        # shared prefill budget had left, and check each one donates.
        prefix_len, total = 0, 100_000
        for budget in (8192, 5000, 8192, 3333, 8192, 777):
            trunc_len = self._floor(budget, total - prefix_len)
            self.assertIsNotNone(
                _track_seqlen(
                    tree_page=256,
                    prefix_len=prefix_len,
                    extend_len=trunc_len,
                    dcp_enabled=True,
                ),
                f"no donation at {prefix_len=} {trunc_len=}",
            )
            prefix_len += trunc_len


class TestMambaTrackPosition(unittest.TestCase):
    def test_track_position_is_taken_before_the_ping_pong_swap(self):
        # prepare_for_extend gathers the slot id at this position; the read it
        # replaced took buffer[mamba_next_track_idx] before the boundary swap.
        for next_idx in (0, 1):
            with self.subTest(next_idx=next_idx):
                entry, req = _prepare_extend(
                    tree_page=CHUNK, prefix_len=0, extend_len=96, next_idx=next_idx
                )
                self.assertTrue(entry.track_mask)
                self.assertEqual(entry.track_position, next_idx)
                self.assertEqual(req.kv.mamba_next_track_idx, 1 - next_idx)


class TestMambaTrackGrid(unittest.TestCase):
    def _grid(self, *, interval: int, tree_page: int, chunk: int = CHUNK) -> int:
        with get_context().override_server_args(
            mamba_track_interval=interval, _mamba_cache_chunk_size=chunk
        ):
            return mamba_track_grid(tree_page)

    def test_widened_tree_page_rounds_the_interval_up(self):
        self.assertEqual(self._grid(interval=256, tree_page=512), 512)

    def test_grid_stays_on_the_chunk_size(self):
        self.assertEqual(self._grid(interval=192, tree_page=64, chunk=128) % 128, 0)


if __name__ == "__main__":
    unittest.main()
