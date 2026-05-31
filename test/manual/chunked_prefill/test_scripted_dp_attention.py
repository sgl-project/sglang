
import unittest

from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until_all_finished,
    run_until_finished,
)


class TestDPAttnBasic(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        tp_size=2,
        dp_size=2,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_dp_attention=True,
    )

    def test_naive_dp_attention_chunked(self):
        self.server.execute_script(self._script_naive_dp_attention_chunked)

    @staticmethod
    def _script_naive_dp_attention_chunked(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2

    def test_dp_chunked_on_one_rank_other_idle(self):
        self.server.execute_script(self._script_dp_chunked_on_one_rank_other_idle)

    @staticmethod
    def _script_dp_chunked_on_one_rank_other_idle(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4, dp_rank=0)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.chunks_done >= 2
        assert t.dp_rank_max_pending(1) == 0, (
            f"rank 1 must remain idle, observed peak pending "
            f"{t.dp_rank_max_pending(1)}"
        )

    def test_dp_two_chunked_one_per_rank(self):
        self.server.execute_script(self._script_dp_two_chunked_one_per_rank)

    @staticmethod
    def _script_dp_two_chunked_one_per_rank(t: ScriptedContext):
        r0 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, dp_rank=0)
        r1 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN * 2, max_new_tokens=2, dp_rank=1
        )
        yield from run_until_all_finished(handles=[r0, r1], max_steps=800)
        assert r0.finished and r1.finished
        assert r1.chunks_done > r0.chunks_done, (
            f"r1 (long prompt) should chunk more than r0, got "
            f"r0.chunks_done={r0.chunks_done}, r1.chunks_done={r1.chunks_done}"
        )

    def test_dp_chunked_completion_skew(self):
        self.server.execute_script(self._script_dp_chunked_completion_skew)

    @staticmethod
    def _script_dp_chunked_completion_skew(t: ScriptedContext):
        r0 = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE * 2, max_new_tokens=2, dp_rank=0)
        r1 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN * 2, max_new_tokens=2, dp_rank=1
        )
        yield from run_until_finished(r0, max_steps=400)
        assert r0.finished
        assert t.dp_rank_is_idle(0), "rank 0 should be idle while rank 1 chunks"
        yield from run_until_finished(r1, max_steps=1200)
        assert r1.finished


if __name__ == "__main__":
    unittest.main()
