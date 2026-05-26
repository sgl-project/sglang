"""DP attention × chunked: naive ScriptedRuntime smoke.

DP attention shards attention across DP ranks; the chunked path runs
independently on each DP rank. The naive smoke just verifies a
chunked request completes when DP attention is on.

Requires 2 GPUs and ScriptedRuntime multi-rank support (wishlist §4
P2 (12)).
"""

import unittest

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until_all_finished,
    run_until_finished,
)
from sglang.test.test_utils import CustomTestCase


class TestScriptedDPAttention(CustomTestCase):
    def test_naive_dp_attention_chunked(self):
        """DP attention × chunked: naive ScriptedRuntime smoke."""
        execute_scripted_runtime(
            self._script_naive_dp_attention_chunked,
            **base_engine_kwargs(
                tp_size=2,
                dp_size=2,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_dp_attention=True,
            ),
        )

    @staticmethod
    def _script_naive_dp_attention_chunked(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2

    def test_dp_chunked_on_one_rank_other_idle(self):
        """DP rank 0 chunked, rank 1 fully idle — rank 1 must not block rank 0."""
        execute_scripted_runtime(
            self._script_dp_chunked_on_one_rank_other_idle,
            **base_engine_kwargs(
                tp_size=2,
                dp_size=2,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_dp_attention=True,
            ),
        )

    # [a-DP1] DP rank imbalance — rank 0 has a long chunked req while
    # rank 1 stays idle; rank 1 must not block the chunked progress on
    # rank 0 via the cross-rank barrier.
    @staticmethod
    def _script_dp_chunked_on_one_rank_other_idle(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4, dp_rank=0)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.chunks_done >= 2
        # Other rank must report idle the whole time.
        assert t.dp_rank_max_pending(1) == 0, (
            f"rank 1 must remain idle, observed peak pending "
            f"{t.dp_rank_max_pending(1)}"
        )

    def test_dp_two_chunked_one_per_rank(self):
        """DP one chunked req per rank with different chunk sizes — chunks_done tracked per rank."""
        execute_scripted_runtime(
            self._script_dp_two_chunked_one_per_rank,
            **base_engine_kwargs(
                tp_size=2,
                dp_size=2,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_dp_attention=True,
            ),
        )

    # [a-DP2] Per-rank chunked reqs with different sizes — each rank
    # tracks its own chunks_done; ranks must not cross-contaminate.
    @staticmethod
    def _script_dp_two_chunked_one_per_rank(t: ScriptedRuntime):
        r0 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, dp_rank=0)
        r1 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN * 2, max_new_tokens=2, dp_rank=1
        )
        yield from run_until_all_finished(handles=[r0, r1], max_steps=800)
        assert r0.finished and r1.finished
        # r1 has the longer prompt so should report strictly more chunks.
        assert r1.chunks_done > r0.chunks_done, (
            f"r1 (long prompt) should chunk more than r0, got "
            f"r0.chunks_done={r0.chunks_done}, r1.chunks_done={r1.chunks_done}"
        )

    def test_dp_chunked_completion_skew(self):
        """DP rank 0 finishes while rank 1 still chunking — broadcast stays consistent."""
        execute_scripted_runtime(
            self._script_dp_chunked_completion_skew,
            **base_engine_kwargs(
                tp_size=2,
                dp_size=2,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_dp_attention=True,
            ),
        )

    # [a-DP3] DP completion skew — rank 0 finishes early; rank 1 keeps
    # chunking. The cross-rank broadcast must stay consistent and rank 0
    # must report idle once done.
    @staticmethod
    def _script_dp_chunked_completion_skew(t: ScriptedRuntime):
        r0 = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE * 2, max_new_tokens=2, dp_rank=0)
        r1 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN * 2, max_new_tokens=2, dp_rank=1
        )
        yield from run_until_finished(r0, max_steps=400)
        assert r0.finished
        # While r1 keeps chunking, rank 0 must report idle.
        assert t.dp_rank_is_idle(0), "rank 0 should be idle while rank 1 chunks"
        yield from run_until_finished(r1, max_steps=1200)
        assert r1.finished


if __name__ == "__main__":
    unittest.main()
