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
        disaggregation_transfer_backend="fake",
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


class TestDisaggNonOverlap(ScriptedTestCase):
    # disable_overlap_schedule=True forces process_prefill_chunk down its
    # non-overlap else-branch (prefill.py:753-754), where send_kv_chunk is called
    # synchronously for the chunked_req instead of being deferred to the
    # batch-result step. Every existing disagg class runs overlap ON, so this is
    # the only class that exercises the immediate-send path.
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=DEFAULT_MODEL_NAME_FOR_TEST,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        disaggregation_mode="prefill",
        disaggregation_transfer_backend="fake",
        disable_overlap_schedule=True,
    )

    def test_nonoverlap_chunk_sends_synchronously(self):
        self.server.execute_script(self._script_nonoverlap_chunk_sends_synchronously)

    @staticmethod
    def _script_nonoverlap_chunk_sends_synchronously(t: ScriptedContext):
        """Non-overlap disagg advances start_send_idx synchronously per chunk; tmp_end_idx stays -1."""
        # GPU validation pending: scripted single-GPU harness only.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        # Reach at least the second chunk so process_prefill_chunk has executed
        # its non-overlap send_kv_chunk(self.chunked_req) for a prior chunk.
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 2)
        # The non-overlap else-branch sends immediately: start_send_idx has been
        # advanced to the previous chunk's page-aligned fill boundary (page_size=1
        # here, so it equals that boundary). This distinguishes it from the overlap
        # branch, which leaves start_send_idx at 0 mid-flight and defers the send.
        assert r.req.start_send_idx > 0, (
            f"non-overlap disagg must advance start_send_idx synchronously while "
            f"chunking; start_send_idx={r.req.start_send_idx}"
        )
        # tmp_end_idx is the overlap-only sentinel slice end; the non-overlap path
        # never assigns it, so it stays at its -1 default. This is the load-bearing
        # discriminator between the two process_prefill_chunk branches.
        assert r.req.tmp_end_idx == -1, (
            f"non-overlap disagg must never set tmp_end_idx (overlap-only field); "
            f"tmp_end_idx={r.req.tmp_end_idx}"
        )
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        # VERY_LONG_PROMPT_LEN is an exact multiple of DEFAULT_CHUNK_SIZE, so the
        # final chunk send (last_chunk=True, no page deferral) advances
        # start_send_idx all the way to the full prompt length.
        assert r.req.start_send_idx == VERY_LONG_PROMPT_LEN, (
            f"final send must reach the full prompt length; "
            f"start_send_idx={r.req.start_send_idx}"
        )

    def test_nonoverlap_retract_keeps_sent_send_idx(self):
        self.server.execute_script(self._script_nonoverlap_retract_keeps_sent_send_idx)

    @staticmethod
    def _script_nonoverlap_retract_keeps_sent_send_idx(t: ScriptedContext):
        """Retract of a partially-sent chunked disagg req preserves chunked_req and start_send_idx."""
        # GPU validation pending: scripted single-GPU harness only.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        # Non-overlap mode advances start_send_idx synchronously, so we can land
        # AFTER at least one KV chunk has actually been sent (start_send_idx > 0).
        # The existing overlap-mode retract tests land while start_send_idx is
        # still 0, so they only witness the "never sent" case; this witnesses the
        # "sent then retracted" case that reset_for_retract leaves untouched.
        yield from run_until(r, lambda h: h.is_chunking and h.req.start_send_idx > 0)
        sent = r.req.start_send_idx
        assert sent > 0

        t.pause_generation(mode="retract")
        yield

        # pause_generation(mode="retract") only retracts running_batch reqs and
        # clears chunked_req inside the `not running_batch.is_empty()` block
        # (scheduler.py:3728-3736). A disagg-prefill req that is still chunking
        # lives in chunked_req with an EMPTY running_batch, so that block is
        # skipped: the retract leaves chunked_req pointing at this same req rather
        # than clearing it. inflight_middle_chunks is already 0 in the non-overlap
        # path (it is only set on the overlap branch), so it stays 0 here.
        req = t.find_req_by_rid(r.rid)
        assert req is not None
        assert t.scheduler.chunked_req is req, (
            f"non-overlap disagg retract with empty running_batch must leave the "
            f"chunking req in chunked_req; got "
            f"{t.scheduler.chunked_req.rid if t.scheduler.chunked_req else None}"
        )
        assert req.inflight_middle_chunks == 0, (
            f"non-overlap path never sets inflight_middle_chunks; "
            f"got {req.inflight_middle_chunks}"
        )
        # The already-sent offset survives the retract: send_kv_chunk advanced
        # start_send_idx synchronously and nothing in the retract path resets it.
        assert req.start_send_idx == sent, (
            f"retract must NOT reset start_send_idx; "
            f"expected {sent}, got {req.start_send_idx}"
        )

        t.continue_generation()
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        # chunked_req survived the retract, so prefill resumes from the preserved
        # send offset and chunks the rest of the long prompt to completion.
        assert r.chunks_done >= 2


class TestDisaggInflightQueue(ScriptedTestCase):
    # Default overlap-ON disagg prefill. Exercises the
    # process_batch_result_disagg_prefill else-branch (prefill.py:580-602): a
    # non-final chunk decrements inflight_middle_chunks and is NOT appended to
    # disagg_prefill_inflight_queue; only the final chunk (inflight <= 0) appends.
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=DEFAULT_MODEL_NAME_FOR_TEST,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        disaggregation_mode="prefill",
        disaggregation_transfer_backend="fake",
    )

    def test_inflight_queue_membership_only_on_final_chunk(self):
        self.server.execute_script(
            self._script_inflight_queue_membership_only_on_final_chunk
        )

    @staticmethod
    def _script_inflight_queue_membership_only_on_final_chunk(t: ScriptedContext):
        """rid is never in disagg_prefill_inflight_queue while chunking, and is drained once finished."""
        # GPU validation pending: scripted single-GPU harness only.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)

        # The req is not chunked on the very first scheduler iteration after
        # submission, so advance to the point where it actually becomes the
        # chunked_req before sampling per-chunk state. Otherwise the loop below
        # would break immediately on the pre-chunking is_chunking==False.
        yield from run_until(r, lambda h: h.is_chunking)

        saw_inflight_positive = False
        # Step yield-by-yield through the chunked-prefill phase. On every non-final
        # chunk the else-branch keeps the rid OUT of the inflight queue and holds
        # inflight_middle_chunks >= 1 (get_next_batch_to_run increments it for the
        # chunked_req; the result branch only decrements it on the last chunk).
        for _ in range(800):
            if not r.is_chunking:
                break
            queued_rids = [req.rid for req in t.scheduler.disagg_prefill_inflight_queue]
            assert r.rid not in queued_rids, (
                f"non-final chunk must not enter disagg_prefill_inflight_queue; "
                f"queue={queued_rids}"
            )
            req = t.find_req_by_rid(r.rid)
            if req is not None and req.inflight_middle_chunks >= 1:
                saw_inflight_positive = True
            yield

        assert saw_inflight_positive, (
            "must observe at least one in-flight middle chunk "
            "(inflight_middle_chunks >= 1) during chunked prefill"
        )

        # The final chunk appends the rid to disagg_prefill_inflight_queue
        # (process_batch_result_disagg_prefill, prefill.py:533) and then, in the
        # SAME event-loop iteration, process_disagg_prefill_inflight_queue
        # (prefill.py:459) polls the sender and drains it. With the fake transfer
        # backend the sender concludes Success instantly, so the rid is appended and
        # removed between two consecutive script yield points and is never visible in
        # the queue at a yield boundary. We therefore cannot positively witness the
        # transient membership here; with a real network sender the rid would linger
        # in the queue until the transfer poll succeeds.
        #
        # What stays observable is the non-membership-while-chunking invariant
        # asserted above plus the consequence of the final-chunk append: the req
        # leaves the chunked phase, gets its prefill-completion output token, and
        # finishes.
        assert not r.is_chunking
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        # The req is no longer anywhere in the inflight queue once finished and
        # drained.
        final_queue = [req.rid for req in t.scheduler.disagg_prefill_inflight_queue]
        assert r.rid not in final_queue, (
            f"finished disagg-prefill req must have been drained from the inflight "
            f"queue; queue={final_queue}"
        )


class TestDisaggPartialPage(ScriptedTestCase):
    # page_size=16 with a prompt length that is NOT a multiple of 16 keeps the
    # final send non-page-aligned while every non-final chunk send stays page
    # aligned. The scheduler itself page-aligns each chunk boundary
    # (schedule_policy.py:932,949 round trunc_len/now_input_len down to page_size),
    # and the server-args check forbids a chunked_prefill_size that is not a
    # multiple of page_size (server_args.py:7279-7281), so chunked_prefill_size
    # must be a multiple of PAGE_SIZE. Non-overlap so the send (and start_send_idx
    # update) happens synchronously per chunk.
    PAGE_SIZE = 16
    PARTIAL_PROMPT_LEN = 60  # 60 % 16 == 12 -> final send is NOT page-aligned

    ENGINE_KWARGS = base_engine_kwargs(
        model_path=DEFAULT_MODEL_NAME_FOR_TEST,
        chunked_prefill_size=PAGE_SIZE,  # must be divisible by page_size
        page_size=PAGE_SIZE,
        disaggregation_mode="prefill",
        disaggregation_transfer_backend="fake",
        disable_overlap_schedule=True,
    )

    def test_partial_page_deferred_on_non_final_chunk(self):
        self.server.execute_script(
            self._script_partial_page_deferred_on_non_final_chunk
        )

    @staticmethod
    def _script_partial_page_deferred_on_non_final_chunk(t: ScriptedContext):
        """Non-final disagg sends stay page-aligned (start_send_idx % page_size == 0); the final send reaches the non-page-aligned prompt length."""
        # GPU validation pending: scripted single-GPU harness only.
        page_size = TestDisaggPartialPage.PAGE_SIZE
        prompt_len = TestDisaggPartialPage.PARTIAL_PROMPT_LEN
        r = t.start_req(prompt_len=prompt_len, max_new_tokens=2)

        # The req is not chunked on the very first scheduler iteration after
        # submission, so advance until it becomes the chunked_req before sampling
        # per-chunk send state.
        yield from run_until(r, lambda h: h.is_chunking)

        saw_nonzero_aligned_send = False
        # While chunking, the scheduler page-aligns each chunk boundary and the
        # non-final send rounds end_idx down to a page boundary, so start_send_idx
        # is always a multiple of page_size and never overshoots the prompt.
        for _ in range(800):
            if not r.is_chunking:
                break
            ssi = r.req.start_send_idx
            assert ssi % page_size == 0, (
                f"non-final disagg send must leave start_send_idx page-aligned; "
                f"start_send_idx={ssi}, page_size={page_size}"
            )
            assert ssi <= prompt_len
            if 0 < ssi < prompt_len:
                saw_nonzero_aligned_send = True
            yield

        assert saw_nonzero_aligned_send, (
            "must observe a non-final page-aligned partial send "
            "(0 < start_send_idx < prompt_len) during chunked prefill"
        )

        # Step to completion, capturing the last live start_send_idx. The final
        # send (last_chunk=True) skips the deferral and reaches the full prompt
        # length, which is NOT a multiple of page_size -- proving the non-final
        # page-alignment above came from the deferral, not an already-aligned
        # prompt.
        final_ssi = None
        for _ in range(800):
            req = t.find_req_by_rid(r.rid)
            if req is not None:
                final_ssi = req.start_send_idx
            if r.finished:
                break
            yield
        assert r.finished
        assert prompt_len % page_size != 0
        assert final_ssi == prompt_len, (
            f"final send must reach full (non-page-aligned) prompt length; "
            f"start_send_idx={final_ssi}, prompt_len={prompt_len}"
        )


class TestDisaggOverlap(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=DEFAULT_MODEL_NAME_FOR_TEST,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        disaggregation_mode="prefill",
        disaggregation_transfer_backend="fake",
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
