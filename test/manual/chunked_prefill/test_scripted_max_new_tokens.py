import unittest
from typing import List

from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.scheduler_hook import ScriptedBatchRecord
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    base_engine_kwargs,
    run_until,
    run_until_finished,
)

# These are manual GPU tests: they spin up a real engine and cannot run in this
# environment. GPU validation is pending; the assertions below are derived from
# the scheduler source (see module docstring of facts).
#
# Verified facts (sglang/srt), for a single isolated request:
#   1. A request finishes when len(output_ids) >= max_new_tokens via FINISH_LENGTH
#      (schedule_batch.py Req.update_finish_state, ~L1278). With ignore_eos=True
#      the request always generates EXACTLY max_new_tokens tokens.
#   2. The FIRST output token is sampled on the LAST prefill (extend) chunk: only
#      when req.inflight_middle_chunks <= 0 does the prefill result append a token
#      (batch_result_processor.py process_batch_result_prefill, ~L223-227). Middle
#      chunks append nothing.
#   3. Every subsequent token comes from one DECODE forward batch. Therefore the
#      number of DECODE forward batches for a request == max(0, max_new_tokens - 1).
#      In particular max_new_tokens == 1 performs ZERO decode forwards.
#   4. For a single isolated request (no concurrent running decode batch), each
#      chunked-prefill chunk runs as pure ForwardMode.EXTEND, not MIXED: MIXED is
#      only set by ScheduleBatch.mix_with_running (schedule_batch.py ~L2192), which
#      requires a running decode batch to mix into. So chunk records here have
#      mode == 'extend'.


def _records_for_rid(
    batch_log: List[ScriptedBatchRecord], rid: str
) -> List[ScriptedBatchRecord]:
    return [rec for rec in batch_log if rid in rec.rids]


def _decode_records(
    batch_log: List[ScriptedBatchRecord], rid: str
) -> List[ScriptedBatchRecord]:
    return [rec for rec in _records_for_rid(batch_log, rid) if rec.mode == "decode"]


def _extend_records(
    batch_log: List[ScriptedBatchRecord], rid: str
) -> List[ScriptedBatchRecord]:
    # A single isolated request never mixes with a running decode batch, so its
    # prefill chunks are always pure EXTEND (never MIXED); see fact 4 above.
    return [rec for rec in _records_for_rid(batch_log, rid) if rec.mode == "extend"]


class TestMaxNewTokensDecodeForwardLaw(ScriptedTestCase):
    # prompt_len = 2 * chunk_size guarantees both chunking and decode occur, so a
    # single script exercises the extend (chunked prefill) and decode forward paths.
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_decode_forward_count_equals_mnt_minus_one(self):
        """Decode forward batches for a req == max_new_tokens - 1 across mnt in {1,2,3,4}."""
        self.server.execute_script(self._script_decode_forward_count_law)

    @staticmethod
    def _script_decode_forward_count_law(t: ScriptedContext):
        # The linear law decode_count == max_new_tokens - 1 is the key discriminator:
        # it catches an mnt==1 that wrongly triggers a decode, or an mnt==2 that
        # drops its single decode.
        for max_new_tokens in (1, 2, 3, 4):
            r = t.start_req(
                prompt_len=2 * DEFAULT_CHUNK_SIZE,
                max_new_tokens=max_new_tokens,
                ignore_eos=True,
            )
            yield from run_until_finished(r)
            assert r.finished

            output_ids = r.req.output_ids
            assert len(output_ids) == max_new_tokens, (
                f"max_new_tokens={max_new_tokens} must produce exactly "
                f"{max_new_tokens} tokens; got {len(output_ids)}"
            )

            batch_log = t._scheduler_hook._batch_log
            decode_records = _decode_records(batch_log, r.rid)
            extend_records = _extend_records(batch_log, r.rid)

            assert len(decode_records) == max_new_tokens - 1, (
                f"max_new_tokens={max_new_tokens} expected "
                f"{max_new_tokens - 1} decode forward batches, got "
                f"{len(decode_records)}"
            )
            # The prompt spans 2 chunks, so at least 2 extend records must exist;
            # the first output token is produced on the final extend chunk.
            assert len(extend_records) >= 2, (
                f"max_new_tokens={max_new_tokens} expected >= 2 extend (chunk) "
                f"records, got {len(extend_records)}"
            )

            yield


class TestMaxNewTokensOneSkipsDecode(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_mnt_one_does_zero_decode_forwards(self):
        """H1: mnt==1 produces its single token on the final prefill chunk, zero decodes."""
        self.server.execute_script(self._script_mnt_one_skips_decode)

    @staticmethod
    def _script_mnt_one_skips_decode(t: ScriptedContext):
        # Prompt spans >= 3 chunks so chunking is unambiguous; mnt=1 must finish
        # purely on prefill with no decode forward batch for this rid.
        r = t.start_req(
            prompt_len=3 * DEFAULT_CHUNK_SIZE,
            max_new_tokens=1,
            ignore_eos=True,
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2, (
            f"prompt spanning >=3 chunks should chunk at least twice, got "
            f"chunks_done={r.chunks_done}"
        )
        assert len(r.req.output_ids) == 1, (
            f"max_new_tokens=1 must produce exactly 1 token, got "
            f"{len(r.req.output_ids)}"
        )

        batch_log = t._scheduler_hook._batch_log
        decode_records = _decode_records(batch_log, r.rid)
        assert len(decode_records) == 0, (
            f"max_new_tokens=1 must perform ZERO decode forwards; got "
            f"{len(decode_records)}"
        )

        # The last record holding this rid must be an extend-family chunk: the
        # single token is produced on the final prefill chunk, not on a decode.
        rid_records = _records_for_rid(batch_log, r.rid)
        assert rid_records, "expected at least one batch record for the req"
        last_mode = rid_records[-1].mode
        assert last_mode in ("extend", "mixed"), (
            f"last record for mnt=1 req must be in the extend family; got "
            f"mode={last_mode!r}"
        )


class TestMaxNewTokensFirstDecodeAdjacent(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_first_decode_immediately_follows_last_chunk(self):
        """H4: first decode for the req directly follows its last extend chunk, no gap."""
        self.server.execute_script(self._script_first_decode_adjacent)

    @staticmethod
    def _script_first_decode_adjacent(t: ScriptedContext):
        max_new_tokens = 16
        r = t.start_req(
            prompt_len=3 * DEFAULT_CHUNK_SIZE,
            max_new_tokens=max_new_tokens,
            ignore_eos=True,
        )
        yield from run_until(r, lambda h: h.finished, max_steps=400)
        assert r.finished
        assert len(r.req.output_ids) == max_new_tokens, (
            f"max_new_tokens={max_new_tokens} must produce exactly "
            f"{max_new_tokens} tokens; got {len(r.req.output_ids)}"
        )

        batch_log = t._scheduler_hook._batch_log
        rid_records = _records_for_rid(batch_log, r.rid)

        decode_records = _decode_records(batch_log, r.rid)
        assert len(decode_records) == max_new_tokens - 1, (
            f"expected {max_new_tokens - 1} decode forwards, got "
            f"{len(decode_records)}"
        )

        # Within the subsequence of records that contain this rid, the first
        # 'decode' must be immediately preceded by the last 'extend' chunk: there
        # is no unrelated iteration for THIS rid between the final prefill chunk
        # and the first decode forward.
        rid_modes = [rec.mode for rec in rid_records]
        first_decode_pos = rid_modes.index("decode")
        assert first_decode_pos >= 1, (
            f"first decode must be preceded by an extend chunk; rid_modes="
            f"{rid_modes}"
        )
        assert rid_modes[first_decode_pos - 1] == "extend", (
            f"record immediately before the first decode (in this rid's "
            f"subsequence) must be the last extend chunk; got "
            f"{rid_modes[first_decode_pos - 1]!r}, rid_modes={rid_modes}"
        )
        # Everything before the first decode is the (chunked) extend prefill.
        assert all(mode == "extend" for mode in rid_modes[:first_decode_pos]), (
            f"all records before the first decode must be extend chunks; "
            f"rid_modes={rid_modes}"
        )


if __name__ == "__main__":
    unittest.main()
