import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    LIFECYCLE_STAGES,
    advance_to_lifecycle_stage,
    base_engine_kwargs,
    run_until_finished,
)

register_cuda_ci(est_time=100, stage="extra-a", runner_config="1-gpu-small")
register_amd_ci(est_time=198, stage="extra-a", runner_config="1-gpu-small-amd")


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

        assert r.req is not None, f"stage={stage}: req vanished before pause"

        t.pause_generation(mode="retract")
        yield

        # At the last_decode stage the final decode can complete during the
        # retract; a finished req is removed from the scheduler, so its
        # output_ids are no longer observable through the harness. That case's
        # only observable consequence — clean completion — is covered by the
        # run_until_finished tail below. When the req is not finished,
        # pause(retract) must park it back in the waiting_queue and the paused
        # engine must not advance it.
        if not r.finished:
            req = r.req
            assert req is not None and req in t.scheduler.waiting_queue, (
                f"stage={stage}: pause(retract) should park the req back in "
                f"waiting_queue; found={req!r}"
            )
            output_tokens_after_pause = len(req.output_ids)
            for _ in range(3):
                yield
                req = r.req
                assert (
                    req is not None and len(req.output_ids) == output_tokens_after_pause
                ), (
                    f"stage={stage}: paused engine advanced the req "
                    f"({len(req.output_ids) if req is not None else None} output "
                    f"tokens, expected {output_tokens_after_pause})"
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
        self.server.execute_script(self._script_chunked_prefill_radix_hit_count)

    @staticmethod
    def _script_chunked_prefill_radix_hit_count(t: ScriptedContext):
        r = t.start_req(prompt_len=_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        _assert_prefill_twice_decode_once(t, prompt_len=_PROMPT_LEN)

    def test_nonchunked_prefill_radix_hit_count(self):
        self.server.execute_script(self._script_nonchunked_prefill_radix_hit_count)

    @staticmethod
    def _script_nonchunked_prefill_radix_hit_count(t: ScriptedContext):
        prompt_len = _CHUNK_SIZE - 20
        r = t.start_req(prompt_len=prompt_len, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        _assert_prefill_twice_decode_once(t, prompt_len=prompt_len)


def _assert_prefill_twice_decode_once(t: ScriptedContext, *, prompt_len: int) -> None:
    root = t.scheduler.tree_cache.root_node
    prefill_hits: list[int] = []
    decode_hits: list[int] = []
    stack = [(child, len(child.key)) for child in root.children.values()]
    while stack:
        node, end_index = stack.pop()
        bucket = prefill_hits if end_index <= prompt_len else decode_hits
        bucket.append(node.hit_count)
        for child in node.children.values():
            stack.append((child, end_index + len(child.key)))

    assert prefill_hits and decode_hits, (
        f"expected both prefill and decode radix nodes; "
        f"prefill={prefill_hits}, decode={decode_hits}, prompt_len={prompt_len}"
    )
    assert all(h == 2 for h in prefill_hits), (
        f"each prefill node must be hit exactly twice; "
        f"prefill={prefill_hits}, decode={decode_hits}"
    )
    assert all(h == 1 for h in decode_hits), (
        f"each decode node must be hit exactly once; "
        f"prefill={prefill_hits}, decode={decode_hits}"
    )


if __name__ == "__main__":
    unittest.main()
