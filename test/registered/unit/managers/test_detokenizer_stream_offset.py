"""Regression test for streaming detokenization offset bookkeeping.

Bug: across consecutive incomplete-UTF-8 ("�") recovery steps, sent_offset was
set to decoded_text_len + len(find_printable_text(new_text)). find_printable_text
is non-monotonic: its penultimate-CJK branch emits through a trailing CJK char
while the later last-space branch retreats to an earlier space, so len(printable)
can shrink between steps. The shrinking sent_offset made the next clean step's
`pending` too small, so it re-emitted already-streamed text and the client saw a
duplicated character (e.g. a CJK char duplicated in multilingual streaming).

Fix: sent_offset must never retreat within a recovery run (max with the prior
value). This drives the real DetokenizerManager._decode_batch_token_id_output;
only the tokenizer is mocked (the bug is in the offset arithmetic).
"""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _TableTokenizer:
    """decode(ids) -> string keyed by the exact id tuple. Used when no clean
    commit advances the read window (surr_ids stays empty), so only whole-prefix
    tuples are ever queried."""

    def __init__(self, table):
        self.table = table

    def decode(self, ids, skip_special_tokens=True, spaces_between_special_tokens=True):
        return self.table[tuple(ids)]


class _ConcatTokenizer:
    """decode(ids) -> concatenation of per-id pieces. Consistent for any slice,
    so it survives the middle-slice surr_ids that appear after clean commits."""

    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids, skip_special_tokens=True, spaces_between_special_tokens=True):
        return "".join(self.pieces[i] for i in ids)


def _recv(decode_ids):
    # Duck-typed BatchTokenIDOutput: only the fields the streaming branch reads.
    return SimpleNamespace(
        rids=["r1"],
        decoded_texts=[""],
        decode_ids=[decode_ids],
        read_offsets=[0],
        finished_reasons=[None],  # streaming
        no_stop_trim=[False],
        skip_special_tokens=[True],
        spaces_between_special_tokens=[True],
    )


class TestDetokenizerStreamOffset(CustomTestCase):
    def _make_manager(self, tokenizer):
        mgr = DetokenizerManager.__new__(DetokenizerManager)
        mgr.decode_status = {}
        mgr.disable_tokenizer_batch_decode = True
        mgr.tokenizer = tokenizer
        return mgr

    def _stream(self, mgr):
        client = ""
        for step_ids in ([0], [1], [2]):  # step 1 seeds [0]; 2/3 extend the window
            client += mgr._decode_batch_token_id_output(_recv(step_ids))[0]
        return client

    def test_recovery_retreat_does_not_duplicate(self):
        # new_text deltas "A 世�", "A 世a�", "A 世ab": a CJK char then a byte-split
        # codepoint (-> "�") resolving to a non-CJK char after the last space, so
        # find_printable_text retreats from "A 世" (len 3) to "A " (len 2). No clean
        # commit happens until the last step, so surr_ids stays empty and only the
        # whole-prefix tuples below are queried.
        tok = _TableTokenizer(
            {(): "", (0,): "A 世�", (0, 1): "A 世a�", (0, 1, 2): "A 世ab"}
        )
        client = self._stream(self._make_manager(tok))
        self.assertEqual(
            client, "A 世ab", f"streaming stream must not duplicate, got {client!r}"
        )

    def test_normal_ascii_streaming_unaffected(self):
        # A plain non-CJK stream that commits every step must be byte-exact
        # (the fix is a no-op when sent_offset never retreats).
        tok = _ConcatTokenizer({0: "hello", 1: " wor", 2: "ld"})
        client = self._stream(self._make_manager(tok))
        self.assertEqual(client, "hello world")


if __name__ == "__main__":
    unittest.main()
