"""input_embeds retraction must never splice two generations for a client."""

import unittest
from array import array
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils.weight_versions import WeightVersionEvent
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _input_embeds_req(*, stream: bool, sent: int, generated: int = 40) -> Req:
    prompt = 64
    req = Req(
        "r1",
        "x" * prompt,
        array("q", [1] * prompt),
        SamplingParams(),
        stream=stream,
        input_embeds=[[0.1] * 8] * prompt,
    )
    req.output_ids = array("q", [9000 + i for i in range(generated)])
    req.weight_version_events = [
        WeightVersionEvent(old_version="v1", num_output_tokens=generated)
    ]
    req.send_token_offset = sent
    req.send_decode_id_offset = sent
    req.send_output_token_logprobs_offset = sent
    req.send_output_sampling_mask_offset = sent
    req.surr_offset = max(sent - 16, 0)
    req.read_offset = sent
    return req


def _retract(req: Req):
    keep = Req("keep", "keep", array("q", [1]), SamplingParams())
    batch = ScheduleBatch(reqs=[keep, req])
    with (
        patch.object(batch, "_get_decode_retraction_order", return_value=[0, 1]),
        patch.object(batch, "check_decode_mem", return_value=True),
        patch.object(batch, "filter_batch"),
        patch(
            "sglang.srt.managers.schedule_batch.get_disagg",
            return_value=SimpleNamespace(disaggregation_mode="null"),
        ),
        patch("sglang.srt.managers.schedule_batch.release_kv_cache"),
        patch("sglang.srt.managers.schedule_batch.evict_from_tree_cache"),
    ):
        return batch.retract_decode()


class TestRetractOutputCursorRestart(CustomTestCase):
    def test_sent_output_aborts_instead_of_splicing_generations(self):
        """The tokenizer must not combine A's sent prefix with generation B."""
        for stream in (False, True):
            with self.subTest(stream=stream):
                req = _input_embeds_req(stream=stream, sent=40)

                retracted, _, aborted = _retract(req)

                self.assertEqual(retracted, [])
                self.assertEqual(aborted, [req])
                finish_reason = req.to_finish.to_json()
                self.assertEqual(finish_reason["type"], "abort")
                self.assertEqual(
                    finish_reason["status_code"], HTTPStatus.SERVICE_UNAVAILABLE
                )
                self.assertEqual(req.weight_version_events[0].num_output_tokens, 40)

    def test_request_without_output_can_restart(self):
        """A request with no emitted output can restart without an abort."""
        for stream in (False, True):
            with self.subTest(stream=stream):
                req = _input_embeds_req(stream=stream, sent=0)

                retracted, _, aborted = _retract(req)

                self.assertEqual(retracted, [req])
                self.assertEqual(aborted, [])
                self.assertEqual(len(req.output_ids), 0)
                self.assertEqual(req.send_token_offset, 0)
                self.assertEqual(req.send_decode_id_offset, 0)
                self.assertEqual(req.send_output_token_logprobs_offset, 0)
                self.assertEqual(req.send_output_sampling_mask_offset, 0)
                self.assertIsNone(req.surr_offset)
                self.assertIsNone(req.read_offset)


if __name__ == "__main__":
    unittest.main()
