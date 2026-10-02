"""Stop trimming must use each beam's own finish reason, not the leader's.

A group's returned beams mix stop-finished and length-finished ones, so a shared
reason either drops a real token from a length-finished beam or leaks a stop
token into a matched one.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.beam_search import BeamGroup, joint_select, select_final_topk
from sglang.srt.beam_search.output import (
    decode_beam_search_output,
    pack_beam_search_output,
)
from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

STOP_ID = 99


def _trim(output, finished_reason, no_stop_trim):
    stub = SimpleNamespace(is_tool_call_parser_gpt_oss=False)
    return DetokenizerManager.trim_matched_stop(
        stub, output, finished_reason, no_stop_trim
    )


class _IdTokenizer:
    """Renders the token list it was handed, so assertions read the trim result."""

    def decode(self, tokens, **kwargs):
        return ",".join(str(t) for t in tokens)

    def batch_decode(self, token_lists, **kwargs):
        return [self.decode(t) for t in token_lists]


def _select(cum, logprobs, tokens, k):
    return joint_select(
        torch.tensor(cum, dtype=torch.float32),
        torch.tensor(logprobs, dtype=torch.float32),
        torch.tensor(tokens, dtype=torch.int64),
        torch.tensor([STOP_ID], dtype=torch.int64),
        k,
    )


def _mixed_group(*, stop_wins: bool) -> BeamGroup:
    """A finished group with one stop-matched and one length-finished beam;
    stop_wins picks which of the two scores higher, i.e. the leader's reason."""
    group = BeamGroup(beam_width=2, stop_token_ids=[STOP_ID], max_new_tokens=3)
    group.advance(_select([0.0], [[-0.1, -0.2, -0.3, -0.4]], [[1, 2, 3, 4]], 2))

    # Bounded on both sides: low enough that the 3-token length beam outscores it
    # once normalized (-0.12), high enough to stay inside the examined window --
    # a stop candidate ranked past k survivors never finishes at all.
    stop_logprob = -0.05 if stop_wins else -0.15
    group.advance(
        _select(
            [-0.1, -0.2],
            [[stop_logprob, -0.2, -0.5, -0.9], [-0.11, -0.4, -0.8, -1.2]],
            [[STOP_ID, 5, 6, 7], [8, 9, 10, 11]],
            2,
        )
    )
    assert len(group.completed) == 1, "the stop candidate should have finished"

    # max_new_tokens: the surviving frontier finishes by length.
    group.advance_final(
        select_final_topk(
            group.frontier_cum_logprobs,
            torch.tensor([[-0.1, -0.9], [-0.05, -0.9]], dtype=torch.float32),
            torch.tensor([[12, 13], [14, 15]], dtype=torch.int64),
            2,
        )
    )
    group.final_results = group.finalize()
    return group


def _decode(group, *, disable_batch_decode):
    packed = pack_beam_search_output(SimpleNamespace(beam_group=group))
    recv_obj = SimpleNamespace(
        beam_search_output=[packed],
        # The leader's reason, which the trim must not consult.
        finished_reasons=[{"type": "stop", "matched": STOP_ID}],
        no_stop_trim=[False],
        skip_special_tokens=[True],
        spaces_between_special_tokens=[True],
    )
    decode_beam_search_output(
        recv_obj,
        tokenizer=_IdTokenizer(),
        disable_batch_decode=disable_batch_decode,
        trim_matched_stop=_trim,
    )
    return {tuple(s.tokens): s.text for s in packed.sequences}


class TestDecodeBeamSearchOutput(CustomTestCase):
    def _assert_mixed_group_trims(self, *, stop_wins, disable_batch_decode):
        group = _mixed_group(stop_wins=stop_wins)
        texts = _decode(group, disable_batch_decode=disable_batch_decode)
        matched = [r for r in group.final_results if r.matched_token is not None]
        length = [r for r in group.final_results if r.matched_token is None]
        self.assertEqual(len(matched), 1, group.final_results)
        self.assertEqual(len(length), 1, group.final_results)

        # The stop token is trimmed off the matched beam...
        stop_tokens = tuple(matched[0].tokens)
        self.assertEqual(stop_tokens[-1], STOP_ID)
        self.assertEqual(texts[stop_tokens], ",".join(map(str, stop_tokens[:-1])))
        # ...and the length-finished beam keeps every token.
        len_tokens = tuple(length[0].tokens)
        self.assertEqual(texts[len_tokens], ",".join(map(str, len_tokens)))

    def test_leader_matched_does_not_trim_the_length_beam(self):
        for disable_batch_decode in (True, False):
            with self.subTest(disable_batch_decode=disable_batch_decode):
                self._assert_mixed_group_trims(
                    stop_wins=True, disable_batch_decode=disable_batch_decode
                )

    def test_leader_length_still_trims_the_matched_beam(self):
        for disable_batch_decode in (True, False):
            with self.subTest(disable_batch_decode=disable_batch_decode):
                self._assert_mixed_group_trims(
                    stop_wins=False, disable_batch_decode=disable_batch_decode
                )

    def test_no_stop_trim_keeps_the_stop_token(self):
        group = _mixed_group(stop_wins=True)
        packed = pack_beam_search_output(SimpleNamespace(beam_group=group))
        recv_obj = SimpleNamespace(
            beam_search_output=[packed],
            finished_reasons=[{"type": "stop", "matched": STOP_ID}],
            no_stop_trim=[True],
            skip_special_tokens=[True],
            spaces_between_special_tokens=[True],
        )
        decode_beam_search_output(
            recv_obj,
            tokenizer=_IdTokenizer(),
            disable_batch_decode=True,
            trim_matched_stop=_trim,
        )
        for seq in packed.sequences:
            self.assertEqual(seq.text, ",".join(map(str, seq.tokens)))

    def test_non_beam_item_in_a_mixed_batch_is_skipped(self):
        group = _mixed_group(stop_wins=True)
        packed = pack_beam_search_output(SimpleNamespace(beam_group=group))
        recv_obj = SimpleNamespace(
            beam_search_output=[None, packed],
            finished_reasons=[None, {"type": "stop", "matched": STOP_ID}],
            no_stop_trim=[False, False],
            skip_special_tokens=[True, True],
            spaces_between_special_tokens=[True, True],
        )
        decode_beam_search_output(
            recv_obj,
            tokenizer=_IdTokenizer(),
            disable_batch_decode=False,
            trim_matched_stop=_trim,
        )
        self.assertTrue(all(s.text is not None for s in packed.sequences))


if __name__ == "__main__":
    unittest.main()
