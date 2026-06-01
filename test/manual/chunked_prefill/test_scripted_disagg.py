import unittest

from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
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
        # VERY_LONG_PROMPT_LEN (2048) is an exact multiple of DEFAULT_CHUNK_SIZE
        # (256), so the scheduler chunks it into ceil(2048 / 256) = 8 partial
        # prefill iterations. The disagg prefill path slices KV sends but does
        # not change how the scheduler chunks the prompt, so the count matches
        # the non-disagg model.
        assert r.chunks_done == 8

    def test_disagg_retract_resets_send_state(self):
        self.server.execute_script(self._script_disagg_retract_resets_send_state)

    @staticmethod
    def _script_disagg_retract_resets_send_state(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        t.pause_generation(mode="retract")
        yield
        t.continue_generation()
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.chunks_done >= 2


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
        assert r.chunks_done >= 2
        second_tmp = r.req.tmp_end_idx
        assert second_tmp > first_tmp, (
            f"tmp_end_idx must advance across chunks, got "
            f"first_tmp={first_tmp}, second_tmp={second_tmp}"
        )
        yield from run_until_finished(r, max_steps=800)
        assert r.finished


if __name__ == "__main__":
    unittest.main()
