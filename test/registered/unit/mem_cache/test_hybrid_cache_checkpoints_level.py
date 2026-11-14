"""CPU-only unit tests for --hybrid-cache-checkpoints-level."""

import argparse
import unittest
from array import array
from unittest.mock import patch

from sglang.srt.managers.req_types import FINISH_LENGTH
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.static_kv_storage import acquire_rows
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import get_context, get_disagg, get_exec
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PROMPT = list(range(1, 17))


def _build(
    level,
    components=(ComponentType.FULL, ComponentType.MAMBA),
    page_size=4,
    enable_mamba_extra_buffer=True,
    **kwargs,
):
    cfg = CacheConfig(
        page_size=page_size,
        components=components,
        enable_mamba_extra_buffer=enable_mamba_extra_buffer,
        kv_size=64,
        max_context_len=64,
        mamba_cache_size=8,
        hybrid_cache_checkpoints_level=level,
        **kwargs,
    )
    with patch(
        "sglang.test.kits.unified_radix_cache.fixtures.get_device",
        return_value="cpu",
    ):
        return build_fixture(cfg, mamba_cache_chunk_size=4)


def _running_req(cache, allocator, pool, up_to):
    req = Req(
        rid="running",
        origin_input_text="",
        origin_input_ids=array("q", PROMPT),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )
    acquire_rows(pool, pool.state, [req])
    req.output_ids = array("q")
    req.refresh_fill_ids()
    req.kv.kv_committed_len = len(PROMPT)
    req.kv.kv_allocated_len = len(PROMPT)
    req.kv.cache_protected_len = 0
    if cache.enable_mamba_extra_buffer:
        req.kv.mamba_last_track_seqlen = up_to
    pool.write(
        (req.kv.req_pool_idx, slice(0, len(PROMPT))),
        allocator.alloc(len(PROMPT)),
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


if __name__ == "__main__":
    unittest.main()
