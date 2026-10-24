"""A retracted input_embeds request must restart its output cursors (#39645).

reset_for_retract discards output_ids on the input_embeds branch and restarts
generation, but the streamer and detokenizer cursors used to survive with the
pre-restart offsets, so the first N tokens of the restarted generation were
silently dropped and the client assembled a splice of the two generations.
"""

import unittest
from array import array

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils.weight_versions import WeightVersionEvent
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _retracted_req(streamed: int = 40, prompt: int = 64) -> Req:
    req = Req(
        "r1",
        "x" * prompt,
        array("q", [1] * prompt),
        SamplingParams(),
        input_embeds=[[0.1] * 8] * prompt,
    )
    req.output_ids = array("q", [9000 + i for i in range(streamed)])
    req.weight_version_events = [
        WeightVersionEvent(old_version="v1", num_output_tokens=streamed)
    ]
    req.send_token_offset = streamed
    req.send_decode_id_offset = streamed
    req.send_output_token_logprobs_offset = streamed
    req.send_output_sampling_mask_offset = streamed
    req.surr_offset = max(streamed - 16, 0)
    req.read_offset = streamed
    return req


class TestRetractOutputCursorRestart(unittest.TestCase):
    def test_stream_and_detokenizer_cursors_restart_with_the_output(self):
        req = _retracted_req()
        req.reset_for_retract()

        # output is discarded for the shape-consistency restart (#14109)
        self.assertEqual(len(req.output_ids), 0)
        # the four stream cursors restart at zero, or the restarted
        # generation's first `streamed` tokens are dropped by the slicer
        self.assertEqual(req.send_token_offset, 0)
        self.assertEqual(req.send_decode_id_offset, 0)
        self.assertEqual(req.send_output_token_logprobs_offset, 0)
        self.assertEqual(req.send_output_sampling_mask_offset, 0)
        # the incremental detokenizer cursors go back to the uninitialized
        # sentinels so init_incremental_detokenize rebuilds for generation B
        self.assertIsNone(req.surr_offset)
        self.assertIsNone(req.read_offset)

    def test_truncation_still_reads_the_pre_restart_offset(self):
        # The weight_version_events truncation must keep the already-streamed
        # prefix's events, which only works if the cursor resets land after it.
        req = _retracted_req(streamed=40)
        req.reset_for_retract()
        self.assertEqual(len(req.weight_version_events), 1)
        self.assertEqual(req.weight_version_events[0].num_output_tokens, 40)

    def test_restarted_generation_streams_every_token(self):
        # end-to-end shape of the bug: generation B slices from offset 0, so
        # none of its tokens are skipped
        req = _retracted_req()
        req.reset_for_retract()
        generation_b = [7000 + i for i in range(60)]
        sliced = generation_b[req.send_token_offset :]
        self.assertEqual(sliced, generation_b)


if __name__ == "__main__":
    unittest.main()
