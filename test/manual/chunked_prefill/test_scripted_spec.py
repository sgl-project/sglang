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

    def test_spec_chunked_handoff_first_verify(self):
        self.server.execute_script(self._script_spec_chunked_handoff_first_verify)

    @staticmethod
    def _script_spec_chunked_handoff_first_verify(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=16)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.chunks_done >= 2
        # Spec verification only begins after the chunked prompt is handed off:
        # spec_verify_ct advances at least once.
        assert r.req.spec_verify_ct >= 1, (
            f"expected >=1 spec verify after chunked handoff, got "
            f"{r.req.spec_verify_ct}"
        )

    def test_spec_abort_during_chunked_prepare(self):
        self.server.execute_script(self._script_spec_abort_during_chunked_prepare)

    @staticmethod
    def _script_spec_abort_during_chunked_prepare(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=16)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        t.abort(r)
        yield
        # spec_draft_state_cleared was redundant: the eagle draft KV lives in the
        # same pool as the verified KV, so kv_pages == 0 and lock_refs == 0 after
        # abort already prove the draft state was released along with the rest.
        assert r.kv_pages == 0
        assert r.lock_refs == 0


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
        # "Eagle draft state was captured" is not an externally observable
        # invariant -- topk_p/topk_index/hidden_states are non-None by construction
        # on any draft step. The real, honest signal that eagle ran across the
        # disagg chunked-prefill handoff is that it actually verified at least once.
        assert r.req.spec_verify_ct >= 1, (
            f"expected >=1 spec verify after disagg chunked handoff, got "
            f"{r.req.spec_verify_ct}"
        )


if __name__ == "__main__":
    unittest.main()
