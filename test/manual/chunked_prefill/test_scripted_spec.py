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

_SPEC_MODEL = "Qwen/Qwen3-8B"
_SPEC_DRAFT = "Qwen/Qwen3-8B-EAGLE"


def _spec_engine_kwargs(**overrides):
    return base_engine_kwargs(
        model_path=_SPEC_MODEL,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        speculative_algorithm="EAGLE",
        speculative_draft_model_path=_SPEC_DRAFT,
        speculative_num_steps=3,
        speculative_eagle_topk=4,
        speculative_num_draft_tokens=8,
        **overrides,
    )


class TestSpecBasic(ScriptedTestCase):
    ENGINE_KWARGS = _spec_engine_kwargs()

    def test_naive_spec_chunked(self):
        self.server.execute_script(self._script_naive_spec_chunked)

    @staticmethod
    def _script_naive_spec_chunked(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=16)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2

    def test_spec_chunked_handoff_first_verify(self):
        self.server.execute_script(self._script_spec_chunked_handoff_first_verify)

    @staticmethod
    def _script_spec_chunked_handoff_first_verify(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=16)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.chunks_done >= 2
        assert r.spec_first_verify_prefix_len == VERY_LONG_PROMPT_LEN, (
            f"first spec verify must see prefix_len == prompt_len, got "
            f"{r.spec_first_verify_prefix_len}"
        )
        assert r.req.spec_verify_ct >= 1, (
            f"expected >=1 spec verify after chunked handoff, got "
            f"{r.req.spec_verify_ct}"
        )

    def test_spec_acceptance_chunked_matches_baseline(self):
        self.server.execute_script(
            self._script_spec_acceptance_chunked_matches_baseline
        )

    @staticmethod
    def _script_spec_acceptance_chunked_matches_baseline(t: ScriptedContext):
        r_chunked = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=32)
        yield from run_until_finished(r_chunked, max_steps=800)
        r_baseline = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE // 2, max_new_tokens=32)
        yield from run_until_finished(r_baseline, max_steps=400)
        assert r_chunked.finished and r_baseline.finished
        delta = abs(r_chunked.spec_accept_rate - r_baseline.spec_accept_rate)
        assert delta < 0.25, (
            f"chunked vs baseline spec accept rate diverge by {delta:.3f} "
            f"(chunked={r_chunked.spec_accept_rate:.3f}, "
            f"baseline={r_baseline.spec_accept_rate:.3f})"
        )

    def test_spec_abort_during_chunked_prepare(self):
        self.server.execute_script(self._script_spec_abort_during_chunked_prepare)

    @staticmethod
    def _script_spec_abort_during_chunked_prepare(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=16)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        t.abort(r)
        yield
        assert r.kv_pages == 0
        assert r.lock_refs == 0
        assert (
            r.spec_draft_state_cleared
        ), "spec draft state must be cleared after abort during chunked prefill"


class TestSpecDisagg(ScriptedTestCase):
    ENGINE_KWARGS = _spec_engine_kwargs(disaggregation_mode="prefill")

    def test_spec_eagle_disagg_chunked(self):
        self.server.execute_script(self._script_spec_eagle_disagg_chunked)

    @staticmethod
    def _script_spec_eagle_disagg_chunked(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=8)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.chunks_done >= 2
        assert r.eagle_topk_p_captured
        assert r.eagle_topk_index_captured
        assert r.eagle_hidden_states_captured


if __name__ == "__main__":
    unittest.main()
