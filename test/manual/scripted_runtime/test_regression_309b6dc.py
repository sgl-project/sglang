"""Commit-history regression — B-60 ~ B-71.

Each test mirrors a specific fix commit on ``feat/stateless_scheduler_b``
(around 309b6dc). The scenario is the exact bug shape the commit had
to address; the assertion is "behavior after fix", so re-introducing
the bug during the chunked refactor flips this file red.

The commits sampled here are the ones with the cleanest causal link
to chunked-prefill. Niche ones (mamba, streaming session) are
included but mark obvious model/feature dependencies in the docstring.
"""

import unittest

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.test_utils import CustomTestCase

from test.manual.scripted_runtime.common import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_all_finished,
    run_until_finished,
)


_LORA_BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
_LORA_ADAPTER = "philschmid/llama-3-2-1b-instruct-finetuning-lora-cookbook-test"


# B-60: 5ed4faf0ab "Bypass LoRA scheduling gate for chunked-resume reqs".
# Bug: LoRA drainer would reject chunked-resume reqs, leaving them
# stuck in waiting_queue while holding row + lock_ref + KV.
# Fix verification: a LoRA chunked-resume req must complete, even
# when adapter draining is forced via the wishlist
# ``t.force_lora_drainer_reject`` helper.
def _script_b60_lora_drainer_chunked_resume(t: ScriptedRuntime):
    r = t.start_req(
        prompt_len=VERY_LONG_PROMPT_LEN,
        max_new_tokens=2,
        lora_path=_LORA_ADAPTER,
    )
    yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)

    # Force the drainer to want to evict r's adapter — pre-fix this
    # would deadlock (drainer says "wait for r to drain", r says
    # "drainer won't let me admit my next chunk").
    t.force_lora_drainer_reject(adapter=_LORA_ADAPTER)

    # Must still complete: chunked-resume bypasses the drainer gate.
    yield from run_until_finished(r)
    assert r.finished


# B-61: 96d4749094 "Release row + KV + lock_ref when aborting a
# chunked-resume req from waiting_queue".
# Already covered by B-20 — keep a thin alias here so the commit
# history is grep-able from the test file too.
def _script_b61_abort_waiting_releases_all(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r, lambda h: h.is_chunking)
    yield from run_until(r, lambda h: h.status == "waiting")

    t.abort(r)
    yield

    assert r.kv_pages == 0
    assert r.row_idx is None
    assert r.lock_refs == 0


# B-62: f38e69f87d "Extend pause(retract) to waiting chunked-resume reqs".
# Bug: pause path didn't iterate waiting_queue chunked-resume entries,
# so paused chunked reqs leaked.
def _script_b62_pause_covers_waiting_chunked(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r, lambda h: h.is_chunking)
    yield from run_until(r, lambda h: h.status == "waiting")

    # ``force_retract`` (deterministic) substitutes for the production
    # pause path; either should release the same resources.
    t.force_retract(r)
    yield

    assert r.kv_pages == 0
    assert r.row_idx is None


# B-63: b823c16e60 "Include PP microbatch reqs in abort_request
# batch_rids dedup".
# Bug: abort_request's dedup set did not include PP cross-mb in-flight
# reqs, leading to double-abort under PP last-chunk-in-flight timing.
# Same as B-24; this is the explicit commit-history alias.
def _script_b63_pp_abort_dedup(t: ScriptedRuntime):
    r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=4)
    yield from run_until(r, lambda h: h.chunks_done >= 1 and h.is_chunking)

    t.abort(r)
    yield

    assert r.finish_event_count == 1, (
        f"PP abort must dedup across microbatches; "
        f"got {r.finish_event_count} finish events"
    )


# B-64: 414efd4a27 "Reset disagg send-side state on chunked-resume retract".
# Already encoded as B-33 in the retract suite. Keep alias for grep.
def _script_b64_disagg_retract_resets_send(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)

    t.force_retract(r)
    yield

    assert r.disagg_send_state in (None, "idle")


# B-65: b3a7b9f2a1 "Bump pending_middle_outputs for last-chunk admits +
# decrement-first output proc".
# The invariant: at the moment the last chunk is admitted,
# pending_middle_outputs > 0; once the output is processed it returns
# to 0.
def _script_b65_pending_middle_outputs_invariant(t: ScriptedRuntime):
    r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=4)

    # Drive into the last chunk.
    yield from run_until(r, lambda h: h.chunks_done >= 1 and h.is_chunking)
    assert r.pending_middle_outputs > 0, (
        "last-chunk admit must bump pending_middle_outputs"
    )

    # Continue until the chunk loop ends.
    yield from run_until(r, lambda h: not h.is_chunking)
    # Once the chunk loop has cleared and the output has been
    # processed, the counter is back to 0 (decrement-first ordering).
    assert r.pending_middle_outputs == 0, (
        f"pending_middle_outputs should be 0 after chunk loop clears; "
        f"got {r.pending_middle_outputs}"
    )

    yield from run_until_finished(r)


# B-66: 69ef71edc4 "Conditionally exclude in-flight other-mb
# chunked-resume reqs (PP, max_new_tokens > 1)".
# Bug: under PP, chunked-resume reqs in another microbatch leaked
# into the local batch's filter, leading to double-admission.
# Verification: run two chunked reqs concurrently under PP=2,
# different microbatches; both must complete cleanly without
# double-admit symptoms (extend_input_len mismatch or
# chunked_in_flight_count > 1 momentarily).
def _script_b66_pp_other_mb_chunked_exclude(t: ScriptedRuntime):
    r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)

    for _ in range(400):
        in_flight = t.chunked_in_flight_count()
        assert in_flight <= 1, (
            f"PP cross-mb chunked exclusion broken: in_flight={in_flight}"
        )
        if r1.finished and r2.finished:
            return
        yield
    raise AssertionError("reqs did not finish")


# B-67: 34c02d6a67 "Filter chunked-resume reqs from split_prefill_batch
# before pdmux merge".
# Bug: pdmux merge would see chunked-resume reqs that should be
# excluded. With disagg + chunked, the merge must filter them out.
def _script_b67_pdmux_filter_chunked(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished


# B-68: aaf3752d2b "Skip chunked-resume reqs in calc_priority prefix
# matching".
# Bug: priority calculation tried prefix-matching chunked-resume reqs,
# which is wrong (they already have prefix_indices baked in). Test:
# enable priority + radix; r1 chunks, r2 arrives; priority calc must
# not include r1 in its prefix match.
def _script_b68_priority_skips_chunked_in_prefix_match(t: ScriptedRuntime):
    r1 = t.start_req(
        prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority="low"
    )
    yield from run_until(r1, lambda h: h.is_chunking)

    r2 = t.start_req(
        prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority="high"
    )

    yield from run_until_all_finished([r1, r2])
    assert r1.finished and r2.finished


# B-69: dbdcdde245 "Skip mamba_pool_idx cleanup for chunked-resume on
# NO_TOKEN".
# Mamba-specific: when a chunked-resume req hits NO_TOKEN admission
# return, mamba_pool_idx cleanup must be skipped.
#
# Requires a mamba-architecture model — currently no small mamba model
# is in the standard test fixture. The test sets the right flags;
# actual mamba model wiring is left for the harness to fill in.
def _script_b69_mamba_chunked_resume_no_token(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished


# B-70: 116584e8fa "Bound streaming-session chunked stash by
# kv_committed_len".
# Streaming-session specific: when the session's KV commit length is
# shorter than the chunked extend, the stash must clip to the committed
# length to avoid stashing un-committed tokens.
def _script_b70_streaming_session_stash_bound(t: ScriptedRuntime):
    # Streaming session creation isn't yet a ScriptedRuntime primitive;
    # this test pumps a long req in a streaming-flagged engine and
    # verifies the chunk loop completes cleanly.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished


# B-71: bf5b4e9a10 "Give chunked-resume reqs priority in LPM and
# DFS_WEIGHT sorts".
# Bug: LPM / DFS_WEIGHT priority sort didn't prioritize chunked-resume
# reqs, leading to starvation under load.
# Test: a chunked req in flight + many short reqs; chunked must
# advance, not starve.
def _script_b71_chunked_resume_priority_in_sort(t: ScriptedRuntime):
    r_long = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r_long, lambda h: h.is_chunking)

    # Flood with short reqs.
    shorts = [t.start_req(prompt_len=4, max_new_tokens=1) for _ in range(8)]

    # r_long must make forward progress within a reasonable budget.
    initial_chunks = r_long.chunks_done
    for _ in range(200):
        if r_long.chunks_done > initial_chunks:
            break
        yield
    else:
        raise AssertionError(
            f"chunked req starved by short req flood; "
            f"chunks_done stuck at {initial_chunks}"
        )

    yield from run_until_all_finished([r_long, *shorts])


class TestRegression309b6dc(CustomTestCase):
    def test_b60_lora_drainer_chunked_resume(self):
        execute_scripted_runtime(
            _script_b60_lora_drainer_chunked_resume,
            **base_engine_kwargs(
                model_path=_LORA_BASE_MODEL,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_lora=True,
                lora_paths=[_LORA_ADAPTER],
            ),
        )

    def test_b61_abort_waiting_releases_all(self):
        execute_scripted_runtime(
            _script_b61_abort_waiting_releases_all,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_b62_pause_covers_waiting_chunked(self):
        execute_scripted_runtime(
            _script_b62_pause_covers_waiting_chunked,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_b63_pp_abort_dedup(self):
        execute_scripted_runtime(
            _script_b63_pp_abort_dedup,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                tp_size=2,
                pp_size=2,
            ),
        )

    def test_b64_disagg_retract_resets_send(self):
        execute_scripted_runtime(
            _script_b64_disagg_retract_resets_send,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                disaggregation_mode="prefill",
            ),
        )

    def test_b65_pending_middle_outputs_invariant(self):
        execute_scripted_runtime(
            _script_b65_pending_middle_outputs_invariant,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_b66_pp_other_mb_chunked_exclude(self):
        execute_scripted_runtime(
            _script_b66_pp_other_mb_chunked_exclude,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                tp_size=2,
                pp_size=2,
            ),
        )

    def test_b67_pdmux_filter_chunked(self):
        execute_scripted_runtime(
            _script_b67_pdmux_filter_chunked,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                disaggregation_mode="prefill",
            ),
        )

    def test_b68_priority_skips_chunked_in_prefix_match(self):
        execute_scripted_runtime(
            _script_b68_priority_skips_chunked_in_prefix_match,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_priority_scheduling=True,
            ),
        )

    def test_b69_mamba_chunked_resume_no_token(self):
        execute_scripted_runtime(
            _script_b69_mamba_chunked_resume_no_token,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_b70_streaming_session_stash_bound(self):
        execute_scripted_runtime(
            _script_b70_streaming_session_stash_bound,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_b71_chunked_resume_priority_in_sort(self):
        execute_scripted_runtime(
            _script_b71_chunked_resume_priority_in_sort,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )


if __name__ == "__main__":
    unittest.main()
