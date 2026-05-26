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

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_finished,
)
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST, CustomTestCase


class TestScriptedDisagg(CustomTestCase):
    def test_naive_disagg_chunked(self):
        """Disagg x chunked: prefill engine chunks long prompt and hands off to decode."""
        # When ScriptedRuntime grows disagg topology support, the
        # decode-side engine kwargs go through ``decode_engine_kwargs=``
        # (or similar) — see wishlist §4 P3 (16).
        execute_scripted_runtime(
            self._script_naive_disagg_chunked,
            **base_engine_kwargs(
                model_path=DEFAULT_MODEL_NAME_FOR_TEST,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                disaggregation_mode="prefill",
            ),
        )

    @staticmethod
    def _script_naive_disagg_chunked(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2

    @unittest.skip("requires real disagg topology — wire up when fixture lands")
    def test_disagg_prefill_per_chunk_kv_send(self):
        """Disagg-prefill multi-chunk: each middle chunk sends KV with last_chunk=False."""
        execute_scripted_runtime(
            self._script_disagg_prefill_per_chunk_kv_send,
            **base_engine_kwargs(
                model_path=DEFAULT_MODEL_NAME_FOR_TEST,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                disaggregation_mode="prefill",
            ),
        )

    # [c-D1] Disagg-prefill per-chunk KV send — each middle chunk must call
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

    @unittest.skip("requires real disagg topology — wire up when fixture lands")
    def test_disagg_overlap_mid_chunk_tmp_end_idx(self):
        """Disagg overlap mode: tmp_end_idx updates per chunk per the documented formula."""
        execute_scripted_runtime(
            self._script_disagg_overlap_mid_chunk_tmp_end_idx,
            **base_engine_kwargs(
                model_path=DEFAULT_MODEL_NAME_FOR_TEST,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                disaggregation_mode="prefill",
                disable_overlap_schedule=False,
            ),
        )

    # [c-D3 / B3] Disagg overlap + middle chunk — tmp_end_idx must advance
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

    @unittest.skip("requires real disagg topology — wire up when fixture lands")
    def test_disagg_retract_resets_send_state(self):
        """Disagg-prefill chunked retract resets start_send_idx and tmp_end_idx."""
        execute_scripted_runtime(
            self._script_disagg_retract_resets_send_state,
            **base_engine_kwargs(
                model_path=DEFAULT_MODEL_NAME_FOR_TEST,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                disaggregation_mode="prefill",
            ),
        )

    # [b-414efd4a27] Disagg chunked retract — start_send_idx must reset to 0
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

    @unittest.skip("requires real disagg topology — wire up when fixture lands")
    def test_disagg_decode_alloc_reuse_chunked_only_one(self):
        """Disagg decode alloc reusing list for chunked must hold at most one entry."""
        execute_scripted_runtime(
            self._script_disagg_decode_alloc_reuse_chunked_only_one,
            **base_engine_kwargs(
                model_path=DEFAULT_MODEL_NAME_FOR_TEST,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                disaggregation_mode="decode",
            ),
        )

    # [c-D6a] Disagg decode-side alloc — reusing list collected from inflight
    # chunked reqs must contain at most one entry per the alloc assert.
    @staticmethod
    def _script_disagg_decode_alloc_reuse_chunked_only_one(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.max_decode_alloc_reusing_len <= 1, (
            f"decode alloc reusing list must stay <=1, observed peak "
            f"{r.max_decode_alloc_reusing_len}"
        )

    @unittest.skip("requires real disagg topology — wire up when fixture lands")
    def test_disagg_decode_alloc_kv_committed_or_inflight(self):
        """Disagg decode alloc: either inflight_middle_chunks > 0 or kv_committed_len > 0."""
        execute_scripted_runtime(
            self._script_disagg_decode_alloc_kv_committed_or_inflight,
            **base_engine_kwargs(
                model_path=DEFAULT_MODEL_NAME_FOR_TEST,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                disaggregation_mode="decode",
            ),
        )

    # [c-D6b] Disagg decode alloc invariant — the disjunction
    # inflight_middle_chunks > 0 OR kv_committed_len > 0 must hold across
    # every alloc step for a chunked req on the decode side.
    @staticmethod
    def _script_disagg_decode_alloc_kv_committed_or_inflight(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(400):
            if r.is_chunking:
                assert r.inflight_middle_chunks > 0 or r.kv_committed_len > 0, (
                    f"chunked req on decode side must have either inflight "
                    f"middle chunks or committed KV, got "
                    f"inflight_middle_chunks={r.inflight_middle_chunks}, "
                    f"kv_committed_len={r.kv_committed_len}"
                )
            if r.finished:
                break
            yield
        assert r.finished

    @unittest.skip("requires real disagg topology — wire up when fixture lands")
    def test_disagg_send_kv_chunk_page_aligned_tail(self):
        """Disagg send_kv_chunk trims end_idx down to a page boundary on partial-page tails."""
        execute_scripted_runtime(
            self._script_disagg_send_kv_chunk_page_aligned_tail,
            **base_engine_kwargs(
                model_path=DEFAULT_MODEL_NAME_FOR_TEST,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                disaggregation_mode="prefill",
                page_size=16,
            ),
        )

    # [c-D5] Disagg send_kv_chunk page alignment — when end_idx is not a
    # page multiple the code does end_idx -= end_idx % page_size; verifies
    # no partial-page send is emitted before the final chunk.
    @staticmethod
    def _script_disagg_send_kv_chunk_page_aligned_tail(t: ScriptedRuntime):
        # Prompt length chosen so the penultimate chunk leaves a partial-page
        # tail that send_kv_chunk must trim.
        r = t.start_req(
            prompt_len=DEFAULT_CHUNK_SIZE * 3 + 17,
            max_new_tokens=2,
        )
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        # Every non-final send must land on a page boundary.
        assert r.kv_send_partial_page_events == 0, (
            f"non-final KV sends must be page-aligned, observed "
            f"{r.kv_send_partial_page_events} partial-page sends"
        )

    @unittest.skip("requires real disagg + EAGLE topology — wire up when fixture lands")
    def test_disagg_prefill_chunked_eagle_capture(self):
        """Disagg + EAGLE: topk_p, topk_index, hidden_states captured on the last chunk."""
        execute_scripted_runtime(
            self._script_disagg_prefill_chunked_eagle_capture,
            **base_engine_kwargs(
                model_path=DEFAULT_MODEL_NAME_FOR_TEST,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                disaggregation_mode="prefill",
                speculative_algorithm="EAGLE",
            ),
        )

    # [c-D2] Disagg + EAGLE + chunked — topk_p, topk_index, and hidden_states
    # must be captured during the last chunk so the decode-side draft has
    # the right context for verify.
    @staticmethod
    def _script_disagg_prefill_chunked_eagle_capture(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.eagle_topk_p_captured, "topk_p must be captured on last chunk"
        assert r.eagle_topk_index_captured, "topk_index must be captured on last chunk"
        assert (
            r.eagle_hidden_states_captured
        ), "hidden_states must be captured on last chunk"

    @unittest.skip("requires real disagg topology — wire up when fixture lands")
    def test_disagg_abort_during_mid_chunk(self):
        """Disagg prefill abort mid-chunk cancels decode receive and resets send state."""
        execute_scripted_runtime(
            self._script_disagg_abort_during_mid_chunk,
            **base_engine_kwargs(
                model_path=DEFAULT_MODEL_NAME_FOR_TEST,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                disaggregation_mode="prefill",
            ),
        )

    # [a-Disagg3] Disagg abort during middle chunk — decode side must cancel
    # the in-flight receive; prefill send state (start_send_idx, tmp_end_idx)
    # must reset cleanly.
    @staticmethod
    def _script_disagg_abort_during_mid_chunk(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        t.abort(r)
        yield
        assert r.start_send_idx == 0
        assert r.tmp_end_idx == -1
        assert (
            r.decode_receive_cancelled
        ), "decode side must mark its in-flight receive cancelled"
        assert r.kv_pages == 0
        assert r.lock_refs == 0


if __name__ == "__main__":
    unittest.main()
