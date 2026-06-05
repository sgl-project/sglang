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
# Real, public EAGLE3 draft for Qwen3-8B; the previous "Qwen/Qwen3-8B-EAGLE"
# path does not exist on HF (404).
_SPEC_DRAFT = "Tengyunw/qwen3_8b_eagle3"


def _spec_engine_kwargs(**overrides):
    # The harness KV-canary walker assumes a decode forward batch has one
    # `positions` entry per request, but the EAGLE draft decode expands each
    # request to `speculative_eagle_topk` draft-tree branches, so the draft
    # model's decode CUDA-graph capture feeds bs*topk positions into a bs-sized
    # canary buffer and crashes at launch (kv_canary/plan_input.py
    # _extract_prefix_lens_and_extend_seq_lens). The canary is a harness-side
    # integrity layer, not part of the v1 engine under test (same rationale as
    # TestRadixDisabled in test_scripted_radix.py), so turn it off here so the
    # spec path can use a realistic topk>1 tree. sweep_interval must also drop to
    # 0 or server_args validation rejects the combination.
    #
    # speculative params follow the Tengyunw/qwen3_8b_eagle3 model card's
    # recommended EAGLE3 configuration (steps=6, topk=10, draft_tokens=32).
    return base_engine_kwargs(
        model_path=_SPEC_MODEL,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path=_SPEC_DRAFT,
        speculative_num_steps=6,
        speculative_eagle_topk=10,
        speculative_num_draft_tokens=32,
        kv_canary="none",
        kv_canary_real_data="none",
        kv_canary_sweep_interval=0,
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


if __name__ == "__main__":
    unittest.main()
