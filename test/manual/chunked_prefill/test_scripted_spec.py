"""Speculative decoding × chunked: naive ScriptedRuntime smoke.

EAGLE-style spec decoding adds a verify pass between prefill chunks
and decode. We want a long chunked prompt to be admitted, complete
its chunk loop, and then transition to spec-decoded output without
any state-machine surprises.
"""

import unittest

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_finished,
)
from sglang.test.test_utils import CustomTestCase

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


class TestScriptedSpec(CustomTestCase):
    def test_naive_spec_chunked(self):
        """Speculative decoding × chunked: naive ScriptedRuntime smoke."""
        execute_scripted_runtime(
            self._script_naive_spec_chunked,
            **_spec_engine_kwargs(),
        )

    @staticmethod
    def _script_naive_spec_chunked(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=16)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2

    def test_spec_chunked_handoff_first_verify(self):
        """First spec verify after chunked prefill must include all chunked tokens with right prefix length."""
        execute_scripted_runtime(
            self._script_spec_chunked_handoff_first_verify,
            **_spec_engine_kwargs(),
        )

    # [a-Spec1] Spec handoff from chunked prefill — the first verify pass
    # after the last chunk must see prefix_len == prompt_len and include
    # the spec draft tokens for the freshly-finished prefill.
    @staticmethod
    def _script_spec_chunked_handoff_first_verify(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=16)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.chunks_done >= 2
        assert r.spec_first_verify_prefix_len == VERY_LONG_PROMPT_LEN, (
            f"first spec verify must see prefix_len == prompt_len, got "
            f"{r.spec_first_verify_prefix_len}"
        )
        assert r.spec_verify_count >= 1, (
            f"expected >=1 spec verify after chunked handoff, got "
            f"{r.spec_verify_count}"
        )

    def test_spec_acceptance_chunked_matches_baseline(self):
        """Chunked vs non-chunked spec acceptance rate must be approximately equal."""
        execute_scripted_runtime(
            self._script_spec_acceptance_chunked_matches_baseline,
            **_spec_engine_kwargs(),
        )

    # [a-Spec2] Spec acceptance — chunked vs non-chunked on the same
    # prompt should yield approximately the same accept rate (chunked must
    # not pollute spec state).
    @staticmethod
    def _script_spec_acceptance_chunked_matches_baseline(t: ScriptedRuntime):
        # Run a long chunked prompt and a short non-chunked baseline.
        r_chunked = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=32)
        yield from run_until_finished(r_chunked, max_steps=800)
        r_baseline = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE // 2, max_new_tokens=32)
        yield from run_until_finished(r_baseline, max_steps=400)
        assert r_chunked.finished and r_baseline.finished
        # Accept rates must be within 25% absolute; chunked path must not
        # leak draft state.
        delta = abs(r_chunked.spec_accept_rate - r_baseline.spec_accept_rate)
        assert delta < 0.25, (
            f"chunked vs baseline spec accept rate diverge by {delta:.3f} "
            f"(chunked={r_chunked.spec_accept_rate:.3f}, "
            f"baseline={r_baseline.spec_accept_rate:.3f})"
        )

    def test_spec_abort_during_chunked_prepare(self):
        """Spec draft prepare + chunked abort must reset draft state cleanly."""
        execute_scripted_runtime(
            self._script_spec_abort_during_chunked_prepare,
            **_spec_engine_kwargs(),
        )

    # [a-Spec3] Spec draft preparation racing with chunked abort —
    # cancellation must clean up partial draft state; no orphaned spec
    # buffers.
    @staticmethod
    def _script_spec_abort_during_chunked_prepare(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=16)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        t.abort(r)
        yield
        assert r.kv_pages == 0
        assert r.lock_refs == 0
        assert (
            r.spec_draft_state_cleared
        ), "spec draft state must be cleared after abort during chunked prefill"

    def test_spec_eagle_disagg_chunked(self):
        """EAGLE + disagg + chunked last chunk must land topk_p, topk_index, hidden_states."""
        execute_scripted_runtime(
            self._script_spec_eagle_disagg_chunked,
            **_spec_engine_kwargs(disaggregation_mode="prefill"),
        )

    # [c-D2] EAGLE + disagg + chunked — last chunk must capture all three
    # spec tensors (topk_p, topk_index, hidden_states) for the decode side
    # to verify against.
    @staticmethod
    def _script_spec_eagle_disagg_chunked(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=8)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.chunks_done >= 2
        assert r.eagle_topk_p_captured
        assert r.eagle_topk_index_captured
        assert r.eagle_hidden_states_captured


if __name__ == "__main__":
    unittest.main()
