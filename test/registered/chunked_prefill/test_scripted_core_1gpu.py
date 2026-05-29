import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime.testcase import ScriptedRuntimeTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    LIFECYCLE_STAGES,
    advance_to_lifecycle_stage,
    base_engine_kwargs,
    run_until_finished,
)

# Single-GPU scripted chunked-prefill core tests. The PP sweep, which
# needs 4 GPUs, lives in test_scripted_core_4gpu.py.
register_cuda_ci(est_time=300, stage="extra-a", runner_config="1-gpu-small")


_CHUNK_SIZE = 64
# Deliberately a few tokens short of a chunk-size multiple so the last
# chunk is partial — exercises the off-by-one path instead of clean
# chunk boundaries.
_PROMPT_LEN = 4 * _CHUNK_SIZE - 3

# Chunks that keep the req in the chunked_req slot (is_chunking == True):
# the final extend completes prefill in one shot and is not chunked, so the
# count is ceil(_PROMPT_LEN / _CHUNK_SIZE) - 1 == (_PROMPT_LEN - 1) // _CHUNK_SIZE.
_NUM_MIDDLE_CHUNKS = (_PROMPT_LEN - 1) // _CHUNK_SIZE
# Enough decode steps that first / middle / last decode are distinct points.
_LIFECYCLE_MAX_NEW_TOKENS = 4


def _advance_to_stage(r, stage: str):
    """Yield until the req reaches ``stage``, wired to this file's prompt config."""
    yield from advance_to_lifecycle_stage(
        r,
        stage,
        num_middle_chunks=_NUM_MIDDLE_CHUNKS,
        max_new_tokens=_LIFECYCLE_MAX_NEW_TOKENS,
    )


class TestScriptedCore(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=_CHUNK_SIZE)

    def test_chunked_prefill_smoke(self):
        """Engine boots with small chunk_size and a multi-chunk req finishes cleanly."""
        self.runtime.run(self._script_chunked_prefill_smoke)

    @staticmethod
    def _script_chunked_prefill_smoke(t: ScriptedRuntime):
        r = t.start_req(prompt_len=_PROMPT_LEN, max_new_tokens=3)
        yield from run_until_finished(r)
        assert r.finished, "req did not finish"

    def test_chunked_prefill_smoke_at_chunk_boundary_offsets(self):
        """Prompt lengths just off a chunk-size multiple (+/-1, +/-2) still finish cleanly."""
        for offset in (-2, -1, 1, 2):
            prompt_len = 2 * _CHUNK_SIZE + offset
            with self.subTest(offset=offset, prompt_len=prompt_len):
                self.runtime.run(
                    self._script_chunked_prefill_smoke_at_offset,
                    args=(prompt_len,),
                )

    @staticmethod
    def _script_chunked_prefill_smoke_at_offset(t: ScriptedRuntime, prompt_len: int):
        r = t.start_req(prompt_len=prompt_len, max_new_tokens=3)
        yield from run_until_finished(r)
        assert r.finished, f"req with prompt_len={prompt_len} did not finish"

    def test_pause_retract_at_lifecycle_points_then_resume(self):
        """Pause(retract) at each lifecycle stage, sit paused, continue, and the req still finishes."""
        for stage in LIFECYCLE_STAGES:
            with self.subTest(stage=stage):
                self.runtime.run(
                    self._script_pause_retract_at_stage,
                    args=(stage,),
                )

    @staticmethod
    def _script_pause_retract_at_stage(t: ScriptedRuntime, stage: str):
        r = t.start_req(
            prompt_len=_PROMPT_LEN, max_new_tokens=_LIFECYCLE_MAX_NEW_TOKENS
        )
        yield from _advance_to_stage(r, stage)

        t.pause_generation(mode="retract")
        # Retract state is reflected in the scheduler on the next event-loop iter.
        yield

        # _lookup_req_status is pending reimplementation, so check the retract
        # landing directly against the scheduler: the req must be parked back
        # in waiting_queue rather than still running or finished.
        req = r.req
        assert req is not None and req in t._scheduler.waiting_queue, (
            f"stage={stage}: pause(retract) should park the req back in "
            f"waiting_queue; found={req!r}"
        )

        # retract releases committed KV, so kv_committed_len drops here; a
        # correctly paused engine then makes no forward progress, so it must
        # stay put across the paused iters.
        committed_while_paused = req.kv_committed_len
        for _ in range(3):
            yield
            req = r.req
            assert req is not None and req.kv_committed_len == committed_while_paused, (
                f"stage={stage}: paused engine advanced the req "
                f"(kv_committed_len={req.kv_committed_len if req is not None else None}, "
                f"expected {committed_while_paused})"
            )

        t.continue_generation()
        yield from run_until_finished(r)
        assert r.finished, f"stage={stage}: req did not finish after pause/continue"

    def test_abort_all_at_lifecycle_points(self):
        """abort_all() at each lifecycle stage terminates the req within a few yields."""
        for stage in LIFECYCLE_STAGES:
            with self.subTest(stage=stage):
                self.runtime.run(self._script_abort_all_at_stage, args=(stage,))

    @staticmethod
    def _script_abort_all_at_stage(t: ScriptedRuntime, stage: str):
        r = t.start_req(
            prompt_len=_PROMPT_LEN, max_new_tokens=_LIFECYCLE_MAX_NEW_TOKENS
        )
        yield from _advance_to_stage(r, stage)

        t.abort_all()
        # abort_request marks the req FINISH_ABORT but the chunked_req slot
        # is only cleared on the next normal-cleanup iter; give a few yields.
        for _ in range(8):
            yield
            if r.finished:
                break

        assert r.finished, f"stage={stage}: req did not finish after abort_all"

    def test_chunked_req_single_decode_finishes(self):
        """A chunked-prefill req with max_new_tokens=1 finishes cleanly after its single decode step."""
        self.runtime.run(self._script_chunked_req_single_decode_finishes)

    @staticmethod
    def _script_chunked_req_single_decode_finishes(t: ScriptedRuntime):
        r = t.start_req(prompt_len=_PROMPT_LEN, max_new_tokens=1)
        yield from run_until_finished(r)
        assert r.finished, "single-decode chunked req did not finish"

    def test_chunked_prefill_does_not_inflate_radix_hit_count(self):
        """Chunked inserts skip hit_count, so every radix node is bumped exactly once (==1), never per-chunk inflated."""
        self.runtime.run(self._script_chunked_prefill_does_not_inflate_radix_hit_count)

    @staticmethod
    def _script_chunked_prefill_does_not_inflate_radix_hit_count(t: ScriptedRuntime):
        # runtime.run starts every script from a flushed cache (t.flush_cache),
        # so the radix tree holds only this request's nodes.
        r = t.start_req(prompt_len=_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished

        # Chunked cache_unfinished_req inserts skip _inc_hit_count; only the
        # single non-chunked cache_finished_req insert bumps, touching each node
        # on the committed path exactly once. Without the guard the early prefix
        # nodes would be re-bumped once per chunk and exceed 1.
        hit_counts = t.get_all_node_hit_counts()
        assert hit_counts, "expected radix nodes after a chunked prefill"
        assert all(count == 1 for count in hit_counts.values()), (
            f"chunked prefill must bump each radix node exactly once; "
            f"got {hit_counts}"
        )


if __name__ == "__main__":
    unittest.main()
