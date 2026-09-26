"""Regression: input_embeds retract must reset output-stream cursors.

KV retract discards output_ids for input_embeds requests (shape-mismatch
fix, #14110) but used to leave send_token_offset and sibling stream
cursors pointing into the old generation. The streamer then sliced the
restarted tokens, so the client assembled A[:N] + B[N:] as one success.

See https://github.com/sgl-project/sglang/issues/39645
"""

import unittest
from array import array

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.output_streamer import (
    _GenerationStreamAccumulator,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.weight_versions import WeightVersionEvent

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

N_STREAMED = 40
N_RESTARTED = 60
GEN_A = list(range(9000, 9000 + N_STREAMED))
GEN_B = list(range(10000, 10000 + N_RESTARTED))


def _make_req(*, input_embeds, n_output: int, stream_offset: int) -> Req:
    req = Req(
        rid="retract-input-embeds",
        origin_input_text="",
        origin_input_ids=array("q", [0] * 64),
        sampling_params=SamplingParams(max_new_tokens=128),
        input_embeds=input_embeds,
        stream=True,
    )
    req.output_ids = array("q", range(9000, 9000 + n_output))
    req.send_token_offset = stream_offset
    req.send_decode_id_offset = stream_offset
    req.send_output_token_logprobs_offset = stream_offset
    req.send_output_sampling_mask_offset = stream_offset
    req.surr_offset = 10
    req.read_offset = 64
    req.weight_version_events = [
        WeightVersionEvent(old_version="v0", num_output_tokens=n_output + 10)
    ]
    return req


def _make_accumulator() -> _GenerationStreamAccumulator:
    return _GenerationStreamAccumulator(
        return_logprob=False,
        return_hidden_states=False,
        return_routed_experts=False,
        return_indexer_topk=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        disaggregation_mode=DisaggregationMode.NULL,
        default_stream_interval=1,
        default_force_stream_interval=1,
        get_cached_tokens_details=lambda req: None,
        current_weight_version=None,
    )


class TestResetForRetractInputEmbeds(CustomTestCase):
    def test_input_embeds_retract_resets_stream_cursors(self):
        req = _make_req(
            input_embeds=[[0.0] * 8 for _ in range(64)],
            n_output=N_STREAMED,
            stream_offset=N_STREAMED,
        )
        req.reset_for_retract()

        self.assertEqual(list(req.output_ids), [])
        self.assertEqual(req.send_token_offset, 0)
        self.assertEqual(req.send_decode_id_offset, 0)
        self.assertEqual(req.send_output_token_logprobs_offset, 0)
        self.assertEqual(req.send_output_sampling_mask_offset, 0)
        self.assertIsNone(req.surr_offset)
        self.assertIsNone(req.read_offset)
        self.assertTrue(req.is_retracted)
        self.assertEqual(req.retraction_count, 1)
        # Truncation must use the pre-restart streamed length, not 0.
        self.assertEqual(len(req.weight_version_events), 1)
        self.assertEqual(req.weight_version_events[0].num_output_tokens, N_STREAMED)

    def test_input_embeds_retract_does_not_drop_restarted_tokens(self):
        req = _make_req(
            input_embeds=[[0.0] * 8 for _ in range(64)],
            n_output=N_STREAMED,
            stream_offset=N_STREAMED,
        )
        req.reset_for_retract()
        req.output_ids = array("q", GEN_B)

        accumulator = _make_accumulator()
        accumulator.accept(req=req)

        self.assertEqual(list(accumulator.output_ids[0]), GEN_B)
        self.assertEqual(req.send_token_offset, N_RESTARTED)

    def test_token_id_retract_keeps_output_and_stream_offset(self):
        req = _make_req(
            input_embeds=None, n_output=N_STREAMED, stream_offset=N_STREAMED
        )
        req.reset_for_retract()

        self.assertEqual(list(req.output_ids), GEN_A)
        self.assertEqual(req.send_token_offset, N_STREAMED)
        self.assertEqual(req.send_decode_id_offset, N_STREAMED)
        self.assertEqual(req.surr_offset, 10)
        self.assertEqual(req.read_offset, 64)
        self.assertEqual(
            req.weight_version_events[0].num_output_tokens, N_STREAMED + 10
        )


if __name__ == "__main__":
    unittest.main()
