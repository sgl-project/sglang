"""Keep matched states reusable when a later prompt stops before the tail."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.environ import envs
from sglang.srt.managers.req_types import FINISH_LENGTH
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.common import checkpoint_kv_cache, release_kv_cache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.static_kv_storage import acquire_rows
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

GRID = 4
PAGE = 2
DECODE_GRID = 8
OUT = 4
# Scaled from AgentX session 7ca99d: server prompt lengths / 32 - 2000.
TURNS = (20, 30, 44, 54, 52, 102, 144)


def _prompts(lens):
    """Reuse prompt prefixes but replace model replies with fresh token IDs."""
    out, fresh = [], 10_000
    for i, n in enumerate(lens):
        if i == 0:
            out.append(list(range(1, n + 1)))
            continue
        keep = min(n, len(out[-1]))
        out.append(out[-1][:keep] + list(range(fresh, fresh + n - keep)))
        fresh += 1000
    return out


class _Session:
    """Replay prefill, decode, and finish checkpoints for one chat session."""

    def __init__(self, resume, tail_tokens=0):
        cfg = CacheConfig(
            page_size=PAGE,
            components=(ComponentType.FULL, ComponentType.MAMBA),
            enable_mamba_extra_buffer=True,
            kv_size=4096,
            max_context_len=512,
            mamba_cache_size=60,
            hybrid_cache_checkpoints_level=2,
            max_num_reqs=16,
        )
        with patch(
            "sglang.test.kits.unified_radix_cache.fixtures.get_device",
            return_value="cpu",
        ):
            self.cache, self.allocator, self.pool = build_fixture(
                cfg, mamba_cache_chunk_size=GRID
            )
        self.resume = resume
        self.tail_tokens = tail_tokens
        self.rid = 0
        self.fresh = 50_000

    def cached(self, prompt):
        """Return the admission hit, excluding the final prompt token."""
        key = RadixKey(array("q", prompt[:-1]))
        return self.cache.match_prefix(MatchPrefixParams(key=key)).device_prefix_len

    def run(self, prompt):
        cache, pool = self.cache, self.pool
        self.rid += 1
        sampling_params = SamplingParams(temperature=0, max_new_tokens=OUT)
        sampling_params.normalize(None)
        req = Req(
            rid=f"r{self.rid}",
            origin_input_text="",
            origin_input_ids=array("q", prompt),
            sampling_params=sampling_params,
            mamba_checkpoint_pos=len(prompt) if self.resume else None,
        )
        acquire_rows(pool, pool.state, [req])
        req.output_ids = array("q")
        req.init_next_round_input(cache)
        prefix, end = req.prefix_len, len(prompt)
        req.extend_end = end
        req.lock = cache.lock(req.last_node)
        if prefix:
            matched = cache.path_device_indices(req.last_node)[:prefix]
            pool.write((req.kv.req_pool_idx, slice(0, prefix)), matched)
        fresh_rows = self.allocator.alloc(end + OUT - prefix)
        pool.write((req.kv.req_pool_idx, slice(prefix, end + OUT)), fresh_rows)
        req.kv.kv_committed_len = end
        req.kv.kv_allocated_len = end + OUT

        batch = ScheduleBatch(reqs=[req])
        batch.token_to_kv_pool_allocator = SimpleNamespace(storage=None)
        batch.model_config = SimpleNamespace(
            hf_text_config=SimpleNamespace(mamba_chunk_size=GRID)
        )
        batch.tree_cache = SimpleNamespace(page_size=PAGE)
        batch.req_to_token_pool = MagicMock()
        batch.req_to_token_pool.get_mamba_ping_pong_other_idx.return_value = (
            1 - req.kv.mamba_next_track_idx
        )
        req.refresh_fill_ids()
        with envs.SGLANG_OPT_MAMBA_TAIL_CHECKPOINT_TOKENS.override(self.tail_tokens):
            batch._mamba_radix_cache_v2_req_prepare_for_extend(req)
        checkpoint_kv_cache(req, cache)

        # The last sampled token has no forward, so it cannot produce a checkpoint.
        req.output_ids = array("q", range(self.fresh, self.fresh + OUT))
        self.fresh += OUT
        req.refresh_fill_ids()
        tracked = [t for t in range(end + 1, end + OUT) if t % DECODE_GRID == 0]
        if tracked:
            req.kv.mamba_prev_track_seqlen = req.kv.mamba_last_track_seqlen
            req.kv.mamba_last_track_seqlen = tracked[-1]
            req.kv.mamba_last_track_idx = req.kv.mamba_next_track_idx
            req.kv.mamba_next_track_idx = 1 - req.kv.mamba_last_track_idx
        req.kv.kv_committed_len = end + OUT
        req.finished_reason = FINISH_LENGTH(length=OUT)
        release_kv_cache(req, cache, checkpoint=True)
        cache.sanity_check()


class TestResumePointStrictPrefix(CustomTestCase):
    def _replay(self, **kwargs):
        session, hits = _Session(**kwargs), []
        for prompt in _prompts(TURNS):
            hits.append(session.cached(prompt))
            session.run(prompt)
        return hits

    def test_replay_matches_the_server_rows(self):
        self.assertEqual(self._replay(resume=False), [0, 20, 20, 28, 28, 48, 52])
        self.assertEqual(self._replay(resume=True), [0, 20, 28, 44, 44, 52, 100])

    def test_prompt_stopping_below_the_resume_point_still_hits(self):
        # Prefixes below 52 need the state turn 4 matched at 44.
        base = _prompts(TURNS[:4])
        for length, expected in ((53, 52), (52, 44), (49, 44), (45, 44)):
            with self.subTest(length=length):
                session = _Session(resume=True)
                for prompt in base:
                    session.run(prompt)
                self.assertEqual(session.cached(base[3][:length]), expected)

    def test_tail_env_has_the_same_hit(self):
        base = _prompts(TURNS[:4])
        session = _Session(resume=False, tail_tokens=8)
        for prompt in base:
            session.run(prompt)
        self.assertEqual(session.cached(base[3][:52]), 44)

    def test_no_resume_point_keeps_the_main_rows(self):
        self.assertEqual(
            self._replay(resume=False, tail_tokens=0), [0, 20, 20, 28, 28, 48, 52]
        )


if __name__ == "__main__":
    unittest.main()
