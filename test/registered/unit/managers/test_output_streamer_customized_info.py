import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import unwrap_from_pickle
from sglang.srt.managers.scheduler_components.output_streamer import (
    _GenerationStreamAccumulator,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.weight_versions import (
    WeightVersionSpan,
    record_weight_version_events,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeReq:
    def __init__(self, rid, output_ids, customized_info=None, is_finished=False):
        self.rid = rid
        self.http_worker_ipc = None
        self.finished_reason = None
        self.finished_output = False
        self.finished_len = None
        self.stream = False
        self.sampling_params = SimpleNamespace(
            stream_interval=None,
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
            no_stop_trim=False,
        )
        self.output_ids = output_ids
        self.output_ids_through_stop = output_ids
        self.send_token_offset = 0
        self.send_output_token_logprobs_offset = 0
        self.send_decode_id_offset = 0
        self.decoded_text = ""
        self.origin_input_ids = []
        self.reasoning_tokens = 0
        self.cached_tokens = 0
        self.retraction_count = 0
        self.time_stats = None
        self.mm_image_tokens = 0
        self.mm_audio_tokens = 0
        self.mm_video_tokens = 0
        self.multimodal_inputs = None
        self.customized_info = customized_info
        self.weight_version_events = []
        self.is_finished = is_finished

    def finished(self):
        return self.is_finished

    def init_incremental_detokenize(self):
        return self.output_ids_through_stop, 0

    def check_match_stop_str_prefix(self):
        return False


def _accumulator(current_weight_version="default"):
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
        current_weight_version=current_weight_version,
    )


class TestOutputStreamerCustomizedInfo(unittest.TestCase):
    def test_customized_info_is_padded_for_mixed_batches(self):
        accumulator = _accumulator()

        accumulator.accept(req=_FakeReq("r0", [10, 11]))
        accumulator.accept(
            req=_FakeReq(
                "r1",
                [20, 21, 22],
                customized_info={"probe": [200, 201, 202]},
            )
        )
        accumulator.accept(req=_FakeReq("r2", [30], customized_info={"other": [300]}))

        payload = accumulator.to_payload(dp_rank=0, is_idle_batch=False)
        customized_info = unwrap_from_pickle(payload.customized_info)

        self.assertEqual(payload.output_ids, [[10, 11], [20, 21, 22], [30]])
        self.assertEqual(
            customized_info["probe"],
            [[None, None], [200, 201, 202], [None]],
        )
        self.assertEqual(
            customized_info["other"],
            [[None, None], [None, None, None], [300]],
        )


class TestOutputStreamerWeightVersions(unittest.TestCase):
    def test_payload_carries_spans_for_finished_requests(self):
        """Finished requests report their spans; still-generating ones report nothing."""
        streaming_req = _FakeReq("r0", [10, 11])
        finished_req = _FakeReq("r1", [20, 21, 22], is_finished=True)
        record_weight_version_events([finished_req], old_version="v1")
        finished_req.output_ids.extend([23, 24])

        accumulator = _accumulator(current_weight_version="v2")
        accumulator.accept(req=streaming_req)
        accumulator.accept(req=finished_req)
        payload = accumulator.to_payload(dp_rank=0, is_idle_batch=False)

        self.assertEqual(
            payload.weight_versions,
            [
                None,
                [
                    WeightVersionSpan(version="v1", start=0, end=3),
                    WeightVersionSpan(version="v2", start=3, end=5),
                ],
            ],
        )

    def test_payload_omits_spans_while_all_requests_stream(self):
        """A batch of unfinished requests puts nothing on the wire."""
        accumulator = _accumulator(current_weight_version="v2")
        accumulator.accept(req=_FakeReq("r0", [10]))
        payload = accumulator.to_payload(dp_rank=0, is_idle_batch=False)

        self.assertIsNone(payload.weight_versions)


if __name__ == "__main__":
    unittest.main()
