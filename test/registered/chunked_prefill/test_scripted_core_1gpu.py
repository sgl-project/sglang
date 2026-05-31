import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    LIFECYCLE_STAGES,
    advance_to_lifecycle_stage,
    base_engine_kwargs,
    run_until_finished,
)

register_cuda_ci(est_time=300, stage="extra-a", runner_config="1-gpu-small")


_CHUNK_SIZE = 64
_PROMPT_LEN = 4 * _CHUNK_SIZE - 3

_NUM_MIDDLE_CHUNKS = (_PROMPT_LEN - 1) // _CHUNK_SIZE
_LIFECYCLE_MAX_NEW_TOKENS = 4


def _advance_to_stage(r, stage: str):
    """Yield until the req reaches ``stage``, wired to this file's prompt config."""
    yield from advance_to_lifecycle_stage(
        r,
        stage,
        num_middle_chunks=_NUM_MIDDLE_CHUNKS,
        max_new_tokens=_LIFECYCLE_MAX_NEW_TOKENS,
    )


class TestScriptedCore(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=_CHUNK_SIZE)

    def test_chunked_prefill_smoke(self):
        """Engine boots with small chunk_size and a multi-chunk req finishes cleanly."""
        self.server.execute_script(self._script_chunked_prefill_smoke)

    @staticmethod
    def _script_chunked_prefill_smoke(t: ScriptedContext):
        r = t.start_req(prompt_len=_PROMPT_LEN, max_new_tokens=3)
        yield from run_until_finished(r)
        assert r.finished, "req did not finish"

    def test_chunked_prefill_smoke_at_chunk_boundary_offsets(self):
        """Prompt lengths just off a chunk-size multiple (+/-1, +/-2) still finish cleanly."""
        for offset in (-2, -1, 1, 2):
            prompt_len = 2 * _CHUNK_SIZE + offset
            with self.subTest(offset=offset, prompt_len=prompt_len):
                self.server.execute_script(
                    self._script_chunked_prefill_smoke_at_offset,
                    args=(prompt_len,),
                )

    @staticmethod
    def _script_chunked_prefill_smoke_at_offset(t: ScriptedContext, prompt_len: int):
        r = t.start_req(prompt_len=prompt_len, max_new_tokens=3)
        yield from run_until_finished(r)
        assert r.finished, f"req with prompt_len={prompt_len} did not finish"

    def test_pause_retract_at_lifecycle_points_then_resume(self):
        """Pause(retract) at each lifecycle stage, sit paused, continue, and the req still finishes."""
        for stage in LIFECYCLE_STAGES:
            with self.subTest(stage=stage):
                self.server.execute_script(
                    self._script_pause_retract_at_stage,
                    args=(stage,),
                )

    @staticmethod
    def _script_pause_retract_at_stage(t: ScriptedContext, stage: str):
        r = t.start_req(
            prompt_len=_PROMPT_LEN, max_new_tokens=_LIFECYCLE_MAX_NEW_TOKENS
        )
        yield from _advance_to_stage(r, stage)

        req = r.req
        assert req is not None, f"stage={stage}: req vanished before pause"
        output_tokens_before_pause = len(req.output_ids)

        t.pause_generation(mode="retract")
        yield

        req = r.req
        assert req is not None and req in t._scheduler.waiting_queue, (
            f"stage={stage}: pause(retract) should park the req back in "
            f"waiting_queue; found={req!r}"
        )

        for _ in range(3):
            yield
            req = r.req
            assert (
                req is not None and len(req.output_ids) == output_tokens_before_pause
            ), (
                f"stage={stage}: paused engine advanced the req "
                f"({len(req.output_ids) if req is not None else None} output tokens, "
                f"expected {output_tokens_before_pause})"
            )

        t.continue_generation()
        yield from run_until_finished(r)
        assert r.finished, f"stage={stage}: req did not finish after pause/continue"

    def test_abort_all_at_lifecycle_points(self):
        """abort_all() at each lifecycle stage terminates the req within a few yields."""
        for stage in LIFECYCLE_STAGES:
            with self.subTest(stage=stage):
                self.server.execute_script(
                    self._script_abort_all_at_stage, args=(stage,)
                )

    @staticmethod
    def _script_abort_all_at_stage(t: ScriptedContext, stage: str):
        r = t.start_req(
            prompt_len=_PROMPT_LEN, max_new_tokens=_LIFECYCLE_MAX_NEW_TOKENS
        )
        yield from _advance_to_stage(r, stage)

        t.abort_all()
        for _ in range(8):
            yield
            if r.finished:
                break

        assert r.finished, f"stage={stage}: req did not finish after abort_all"

    def test_chunked_req_single_decode_finishes(self):
        """A chunked-prefill req with max_new_tokens=1 finishes cleanly after its single decode step."""
        self.server.execute_script(self._script_chunked_req_single_decode_finishes)

    @staticmethod
    def _script_chunked_req_single_decode_finishes(t: ScriptedContext):
        r = t.start_req(prompt_len=_PROMPT_LEN, max_new_tokens=1)
        yield from run_until_finished(r)
        assert r.finished, "single-decode chunked req did not finish"

    def test_chunked_prefill_does_not_inflate_radix_hit_count(self):
        """Chunked inserts skip hit_count, so every radix node is bumped exactly once (==1), never per-chunk inflated."""
        self.server.execute_script(
            self._script_chunked_prefill_does_not_inflate_radix_hit_count
        )

    @staticmethod
    def _script_chunked_prefill_does_not_inflate_radix_hit_count(t: ScriptedContext):
        r = t.start_req(prompt_len=_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished

        hit_counts = t.get_all_node_hit_counts()
        assert hit_counts, "expected radix nodes after a chunked prefill"
        assert all(count == 1 for count in hit_counts.values()), (
            f"chunked prefill must bump each radix node exactly once; "
            f"got {hit_counts}"
        )


if __name__ == "__main__":
    unittest.main()
