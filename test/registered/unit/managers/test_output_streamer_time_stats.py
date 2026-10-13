import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import unwrap_from_pickle
from sglang.srt.managers.scheduler_components.output_streamer import (
    _GenerationStreamAccumulator,
)
from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import enter_scope, published_topology

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeReq:
    def __init__(self, rid, output_ids, time_stats):
        self.rid = rid
        self.http_worker_ipc = None
        self.finished_reason = None
        self.finished_output = False
        self.finished_len = None
        self.beam_group = None
        self.stream = True
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
        self.time_stats = time_stats
        self.return_hidden_states = False
        self.return_routed_experts = False
        self.return_indexer_topk = False
        self.return_sampling_mask = False
        self.mm_image_tokens = 0
        self.mm_audio_tokens = 0
        self.mm_video_tokens = 0
        self.multimodal_inputs = None
        self.customized_info = None
        self.weight_version_events = []

    def finished(self):
        return False

    def init_incremental_detokenize(self):
        return self.output_ids_through_stop, 0

    def check_match_stop_str_prefix(self):
        return False


def _accumulator():
    return _GenerationStreamAccumulator(
        return_logprob=False,
        return_hidden_states=False,
        return_routed_experts=False,
        return_indexer_topk=False,
        return_sampling_mask=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        disaggregation_mode=DisaggregationMode.NULL,
        default_stream_interval=1,
        default_force_stream_interval=1,
        get_cached_tokens_details=lambda req: None,
        is_request_level_customized_info_key=frozenset().__contains__,
        current_weight_version="default",
    )


def _time_stats(*, enable_metrics=False, has_timing_data=False):
    time_stats = SchedulerReqTimeStats()
    time_stats.enable_metrics = enable_metrics
    time_stats.has_timing_data = has_timing_data
    return time_stats


class TestOutputStreamerTimeStats(unittest.TestCase):
    def setUp(self):
        serving_patch = patch(
            "sglang.srt.managers.scheduler_components.output_streamer.get_serving",
            return_value=SimpleNamespace(stream_interval=1, weight_version="default"),
        )
        observability_patch = patch(
            "sglang.srt.managers.scheduler_components.output_streamer.get_observability",
            return_value=SimpleNamespace(enable_request_time_stats_logging=False),
        )
        serving_patch.start()
        observability_patch.start()
        enter_scope(self, published_topology(ranks={"dp_rank": 0}))
        self.addCleanup(serving_patch.stop)
        self.addCleanup(observability_patch.stop)

    def _payload(self, *time_stats):
        accumulator = _accumulator()
        for i, stats in enumerate(time_stats):
            accumulator.accept(req=_FakeReq(f"r{i}", [10 + i], stats))
        return accumulator.to_payload(dp_rank=0, is_idle_batch=False)

    def test_time_stats_without_timing_data_are_not_shipped(self):
        payload = self._payload(_time_stats(), _time_stats())

        self.assertEqual(payload.rids, ["r0", "r1"])
        self.assertIsNone(payload.time_stats)

    def test_time_stats_with_metrics_are_shipped_for_every_request(self):
        with_metrics = _time_stats(enable_metrics=True)
        with_metrics.forward_entry_time = 1.5

        payload = self._payload(_time_stats(), with_metrics)
        shipped = unwrap_from_pickle(payload.time_stats)

        self.assertEqual(len(shipped), 2)
        self.assertEqual(shipped[1].forward_entry_time, 1.5)

    def test_received_timing_data_is_shipped(self):
        payload = self._payload(_time_stats(has_timing_data=True))

        self.assertIsNotNone(payload.time_stats)

    def test_has_wire_state_matches_getstate(self):
        for enable_metrics, has_timing_data in itertools.product(
            (False, True), repeat=2
        ):
            stats = _time_stats(
                enable_metrics=enable_metrics, has_timing_data=has_timing_data
            )
            self.assertEqual(stats.has_wire_state(), stats.__getstate__() != {})


if __name__ == "__main__":
    unittest.main()
