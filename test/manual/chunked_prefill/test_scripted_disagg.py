import unittest

from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_STEPS,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_finished,
)
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST


class TestDisaggBasic(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=DEFAULT_MODEL_NAME_FOR_TEST,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        disaggregation_mode="prefill",
    )

    def test_naive_disagg_chunked(self):
        self.server.execute_script(self._script_naive_disagg_chunked)

    @staticmethod
    def _script_naive_disagg_chunked(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2
        assert r.kv_send_last_chunk_events == 1
        assert r.disagg_send_state in (None, "idle")

    def test_disagg_prefill_per_chunk_kv_send(self):
        self.server.execute_script(self._script_disagg_prefill_per_chunk_kv_send)

    @staticmethod
    def _script_disagg_prefill_per_chunk_kv_send(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.chunks_done >= 2
        assert r.kv_send_events == r.chunks_done, (
            f"expected kv_send_events == chunks_done, got "
            f"kv_send_events={r.kv_send_events}, chunks_done={r.chunks_done}"
        )
        assert (
            r.kv_send_last_chunk_events == 1
        ), f"expected exactly one last_chunk=True send, got {r.kv_send_last_chunk_events}"

    def test_disagg_retract_resets_send_state(self):
        self.server.execute_script(self._script_disagg_retract_resets_send_state)

    @staticmethod
    def _script_disagg_retract_resets_send_state(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        t.pause_generation(mode="retract")
        yield
        assert (
            r.req.start_send_idx == 0
        ), f"start_send_idx must reset on retract, got {r.req.start_send_idx}"
        assert (
            r.req.tmp_end_idx == -1
        ), f"tmp_end_idx must reset on retract, got {r.req.tmp_end_idx}"
        t.continue_generation()
        yield from run_until_finished(r, max_steps=800)
        assert r.finished

    def test_disagg_send_state_reset_on_retract_invariant(self):
        self.server.execute_script(
            self._script_disagg_send_state_reset_on_retract_invariant
        )

    @staticmethod
    def _script_disagg_send_state_reset_on_retract_invariant(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(
            r,
            lambda h: h.is_chunking and h.chunks_done >= 1,
            max_steps=DEFAULT_MAX_STEPS,
        )
        req = t.find_req_by_rid(r.rid)
        assert req is not None, "req must still be live mid-chunk"
        assert req.start_send_idx > 0 or req.tmp_end_idx >= 0, (
            f"setup expected disagg send-side state to have advanced "
            f"mid-chunk, got start_send_idx={req.start_send_idx}, "
            f"tmp_end_idx={req.tmp_end_idx}"
        )

        req.reset_for_retract()

        assert req.start_send_idx == 0, (
            f"D3 invariant violation: reset_for_retract did not clear "
            f"start_send_idx; got {req.start_send_idx}. Without this "
            f"reset, the next send_kv_chunk on the re-admitted req would "
            f"skip already-staged-but-not-yet-sent bytes."
        )
        assert req.tmp_end_idx == -1, (
            f"D3 invariant violation: reset_for_retract did not clear "
            f"tmp_end_idx; got {req.tmp_end_idx}. Stale tmp_end_idx "
            f"would index the wrong slice on the next overlap send."
        )


class TestDisaggOverlap(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=DEFAULT_MODEL_NAME_FOR_TEST,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        disaggregation_mode="prefill",
        disable_overlap_schedule=False,
    )

    def test_disagg_overlap_mid_chunk_tmp_end_idx(self):
        self.server.execute_script(self._script_disagg_overlap_mid_chunk_tmp_end_idx)

    @staticmethod
    def _script_disagg_overlap_mid_chunk_tmp_end_idx(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        first_tmp = r.req.tmp_end_idx
        yield from run_until(r, lambda h: h.chunks_done >= 2)
        second_tmp = r.req.tmp_end_idx
        assert second_tmp > first_tmp, (
            f"tmp_end_idx must advance across chunks, got "
            f"first_tmp={first_tmp}, second_tmp={second_tmp}"
        )
        yield from run_until_finished(r, max_steps=800)
        assert r.finished


if __name__ == "__main__":
    unittest.main()
