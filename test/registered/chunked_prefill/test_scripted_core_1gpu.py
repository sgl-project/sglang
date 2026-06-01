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
    yield from advance_to_lifecycle_stage(
        r,
        stage,
        num_middle_chunks=_NUM_MIDDLE_CHUNKS,
        max_new_tokens=_LIFECYCLE_MAX_NEW_TOKENS,
    )


class TestScriptedCore(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=_CHUNK_SIZE)

    def test_chunked_prefill_smoke(self):
        self.server.execute_script(self._script_chunked_prefill_smoke)

    @staticmethod
    def _script_chunked_prefill_smoke(t: ScriptedContext):
        r = t.start_req(prompt_len=_PROMPT_LEN, max_new_tokens=3)
        yield from run_until_finished(r)
        assert r.finished, "req did not finish"

    def test_chunked_prefill_smoke_at_chunk_boundary_offsets(self):
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
        self.server.execute_script(self._script_chunked_req_single_decode_finishes)

    @staticmethod
    def _script_chunked_req_single_decode_finishes(t: ScriptedContext):
        r = t.start_req(prompt_len=_PROMPT_LEN, max_new_tokens=1)
        yield from run_until_finished(r)
        assert r.finished, "single-decode chunked req did not finish"

    def test_chunked_prefill_radix_hit_count(self):
        # prompt > chunk size -> prefilled across several chunks
        self.server.execute_script(self._script_chunked_prefill_radix_hit_count)

    @staticmethod
    def _script_chunked_prefill_radix_hit_count(t: ScriptedContext):
        r = t.start_req(prompt_len=_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        _assert_prefill_twice_decode_once(t.get_all_node_hit_counts())

    def test_nonchunked_prefill_radix_hit_count(self):
        # prompt < chunk size -> prefilled in a single forward (not chunked)
        self.server.execute_script(self._script_nonchunked_prefill_radix_hit_count)

    @staticmethod
    def _script_nonchunked_prefill_radix_hit_count(t: ScriptedContext):
        r = t.start_req(prompt_len=_CHUNK_SIZE - 20, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        _assert_prefill_twice_decode_once(t.get_all_node_hit_counts())


def _assert_prefill_twice_decode_once(hit_counts: dict) -> None:
    # A request inserts its prompt prefix into the radix cache twice -- once via
    # cache_unfinished_req when prefill completes, then again via
    # cache_finished_req on completion -- while decode-generated nodes are
    # inserted only once. So prompt/prefill nodes settle at hit_count 2 and
    # decode nodes at 1. Chunking only splits the prompt into more nodes; it must
    # not change the counts, so chunked and non-chunked prefill agree here.
    assert hit_counts, "expected radix nodes after the request finished"
    assert set(hit_counts.values()) == {1, 2}, (
        f"prefill nodes must be hit twice and decode nodes once "
        f"(no 0 = skipped, no >2 = inflated); got {hit_counts}"
    )


if __name__ == "__main__":
    unittest.main()
