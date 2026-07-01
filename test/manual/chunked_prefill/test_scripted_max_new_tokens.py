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
    return [rec for rec in _records_for_rid(batch_log, rid) if rec.mode == "extend"]


class TestMaxNewTokensDecodeForwardLaw(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_decode_forward_count_equals_mnt(self):
        self.server.execute_script(self._script_decode_forward_count_law)

    @staticmethod
    def _script_decode_forward_count_law(t: ScriptedContext):
        for max_new_tokens in (1, 2, 3, 4):
            r = t.start_req(
                prompt_len=2 * DEFAULT_CHUNK_SIZE,
                max_new_tokens=max_new_tokens,
                ignore_eos=True,
                prompt_token=10 + max_new_tokens,
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

            assert len(decode_records) == max_new_tokens, (
                f"max_new_tokens={max_new_tokens} expected "
                f"{max_new_tokens} decode forward batches, got "
                f"{len(decode_records)}"
            )
            assert len(extend_records) >= 2, (
                f"max_new_tokens={max_new_tokens} expected >= 2 extend (chunk) "
                f"records, got {len(extend_records)}"
            )

            yield


class TestMaxNewTokensOneSkipsDecode(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_mnt_one_emits_token_on_prefill_then_one_dead_decode(self):
        self.server.execute_script(self._script_mnt_one_skips_decode)

    @staticmethod
    def _script_mnt_one_skips_decode(t: ScriptedContext):
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
        assert len(decode_records) == 1, (
            f"max_new_tokens=1 under overlap launches exactly ONE trailing decode "
            f"forward whose token is discarded; got {len(decode_records)}"
        )

        rid_records = _records_for_rid(batch_log, r.rid)
        assert rid_records, "expected at least one batch record for the req"
        rid_modes = [rec.mode for rec in rid_records]
        first_decode_pos = rid_modes.index("decode")
        assert first_decode_pos >= 1, (
            f"the lone decode must be preceded by an extend chunk; rid_modes="
            f"{rid_modes}"
        )
        assert rid_modes[first_decode_pos - 1] in ("extend", "mixed"), (
            f"the record immediately before the lone decode must be the final "
            f"prefill chunk that emits the only token; got "
            f"{rid_modes[first_decode_pos - 1]!r}, rid_modes={rid_modes}"
        )


class TestMaxNewTokensFirstDecodeAdjacent(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_first_decode_immediately_follows_last_chunk(self):
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
        assert len(decode_records) == max_new_tokens, (
            f"expected {max_new_tokens} decode forwards, got " f"{len(decode_records)}"
        )

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
        assert all(mode == "extend" for mode in rid_modes[:first_decode_pos]), (
            f"all records before the first decode must be extend chunks; "
            f"rid_modes={rid_modes}"
        )


if __name__ == "__main__":
    unittest.main()
