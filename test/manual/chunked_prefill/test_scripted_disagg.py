"""Disagg × chunked: naive ScriptedRuntime smoke.

Submit a long-prompt request to a prefill-engine running with
``--chunked-prefill-size 256``. The prefill engine must chunk the
request and hand off KV to the decode engine cleanly.

The current ScriptedRuntime only spins up a single Engine; the
disagg topology requires running two Engines and routing through the
PD router (wishlist §4 P3 (16)). This test encodes the *intent* —
when disagg support lands the script body stays unchanged.

Requires 2-4 GPUs in PD configuration.
"""

import unittest

from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime.testcase import ScriptedRuntimeTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_STEPS,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_finished,
)
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST


class TestDisaggBasic(ScriptedRuntimeTestCase):
    # When ScriptedRuntime grows disagg topology support, the
    # decode-side engine kwargs go through ``decode_engine_kwargs=``
    # (or similar) — see wishlist §4 P3 (16).
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=DEFAULT_MODEL_NAME_FOR_TEST,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        disaggregation_mode="prefill",
    )

    def test_naive_disagg_chunked(self):
        """Disagg x chunked: prefill engine chunks long prompt and hands off to decode."""
        self.runtime.run(self._script_naive_disagg_chunked)

    @staticmethod
    def _script_naive_disagg_chunked(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2
        # Disagg-prefill chunked happy-path must finalize the send
        # cleanly: exactly one last_chunk=True send, no leftover
        # send-side state.
        assert r.kv_send_last_chunk_events == 1
        assert r.disagg_send_state in (None, "idle")

    def test_disagg_prefill_per_chunk_kv_send(self):
        """Disagg-prefill multi-chunk: each middle chunk sends KV with last_chunk=False."""
        self.runtime.run(self._script_disagg_prefill_per_chunk_kv_send)

    # Disagg-prefill per-chunk KV send — each middle chunk must call
    # send_kv_chunk(last_chunk=False); only the final chunk uses last_chunk=True.
    @staticmethod
    def _script_disagg_prefill_per_chunk_kv_send(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.chunks_done >= 2
        # Each middle chunk triggers a non-final KV send; the last chunk
        # is the only one tagged last_chunk=True. Total send events ==
        # chunks_done; non-final sends == chunks_done - 1.
        assert r.kv_send_events == r.chunks_done, (
            f"expected kv_send_events == chunks_done, got "
            f"kv_send_events={r.kv_send_events}, chunks_done={r.chunks_done}"
        )
        assert (
            r.kv_send_last_chunk_events == 1
        ), f"expected exactly one last_chunk=True send, got {r.kv_send_last_chunk_events}"

    def test_disagg_retract_resets_send_state(self):
        """Disagg-prefill chunked retract resets start_send_idx and tmp_end_idx."""
        self.runtime.run(self._script_disagg_retract_resets_send_state)

    # Disagg chunked retract — start_send_idx must reset to 0
    # and tmp_end_idx to -1 so the resumed prefill restarts the KV stream.
    @staticmethod
    def _script_disagg_retract_resets_send_state(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        t.force_retract(r)
        yield
        assert (
            r.start_send_idx == 0
        ), f"start_send_idx must reset on retract, got {r.start_send_idx}"
        assert (
            r.tmp_end_idx == -1
        ), f"tmp_end_idx must reset on retract, got {r.tmp_end_idx}"
        yield from run_until_finished(r, max_steps=800)
        assert r.finished

    def test_disagg_send_state_reset_on_retract_invariant(self):
        """Invariant D3: ``reset_for_retract`` must clear ``start_send_idx`` to 0 and ``tmp_end_idx`` to -1 on a disagg-prefill chunked req.

        Direct access to ``_scheduler`` internals is intentional; this is an
        invariant-tier test (see direct-internals-access plan).
        """
        self.runtime.run(self._script_disagg_send_state_reset_on_retract_invariant)

    # D3 — drive a disagg-prefill chunked req partway so its send-side
    # state has been advanced by ``send_kv_chunk`` (``start_send_idx >
    # 0``). Reach into ``t._scheduler`` to grab the raw ``Req``, then
    # call ``req.reset_for_retract()`` directly — this is the
    # canonical retract reset path; ``t.force_retract`` is a wishlist
    # stub that wraps the same call inside scheduler bookkeeping but
    # adds nothing to the invariant under test. After the reset the
    # disagg-prefill send fields must be back at their init values.
    # See commit 414efd4a27 ("Reset disagg send-side state on
    # chunked-resume retract") for the original bug.
    @staticmethod
    def _script_disagg_send_state_reset_on_retract_invariant(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        # Drive past at least one chunk completion so the disagg
        # prefill side has executed a ``send_kv_chunk`` and advanced
        # ``start_send_idx`` away from its init value of 0.
        yield from run_until(
            r,
            lambda h: h.is_chunking and h.chunks_done >= 1,
            max_steps=DEFAULT_MAX_STEPS,
        )
        req = t._find_req_by_rid(r.rid)
        assert req is not None, "req must still be live mid-chunk"
        # Sanity: precondition. If neither field ever advanced the
        # invariant assertion below is vacuously satisfied, so require
        # one to be in a non-init state. In disagg-prefill mode the
        # bootstrap or per-chunk send must have moved at least one.
        assert req.start_send_idx > 0 or req.tmp_end_idx >= 0, (
            f"setup expected disagg send-side state to have advanced "
            f"mid-chunk, got start_send_idx={req.start_send_idx}, "
            f"tmp_end_idx={req.tmp_end_idx}"
        )

        # Trigger the retract reset path directly. ``reset_for_retract``
        # is the single documented routine that retract callers (both
        # ``retract_decode`` and the disagg-prefill pause sweep) invoke
        # to bring the req back to a re-admissible state.
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


class TestDisaggOverlap(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=DEFAULT_MODEL_NAME_FOR_TEST,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        disaggregation_mode="prefill",
        disable_overlap_schedule=False,
    )

    def test_disagg_overlap_mid_chunk_tmp_end_idx(self):
        """Disagg overlap mode: tmp_end_idx updates per chunk per the documented formula."""
        self.runtime.run(self._script_disagg_overlap_mid_chunk_tmp_end_idx)

    # Disagg overlap + middle chunk — tmp_end_idx must advance
    # by exactly chunk_size each iter; send_kv_chunk uses tmp_end_idx as the
    # right end of the slice for the in-flight middle chunk.
    @staticmethod
    def _script_disagg_overlap_mid_chunk_tmp_end_idx(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        first_tmp = r.tmp_end_idx
        yield from run_until(r, lambda h: h.chunks_done >= 2)
        second_tmp = r.tmp_end_idx
        # Each chunk advances tmp_end_idx by chunk_size (modulo page alignment).
        assert second_tmp > first_tmp, (
            f"tmp_end_idx must advance across chunks, got "
            f"first_tmp={first_tmp}, second_tmp={second_tmp}"
        )
        yield from run_until_finished(r, max_steps=800)
        assert r.finished


if __name__ == "__main__":
    unittest.main()
