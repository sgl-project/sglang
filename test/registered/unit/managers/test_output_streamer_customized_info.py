import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import unwrap_from_pickle
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
    _GenerationStreamAccumulator,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _FakeReq:
    def __init__(
        self,
        rid,
        output_ids,
        customized_info=None,
        *,
        finished=False,
    ):
        self.rid = rid
        self.http_worker_ipc = None
        self._finished = finished
        self.finished_reason = (
            SimpleNamespace(to_json=lambda: {"type": "stop"}) if finished else None
        )
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
        self.return_hidden_states = False
        self.return_routed_experts = False
        self.return_indexer_topk = False
        self.return_sampling_mask = False
        self.mm_image_tokens = 0
        self.mm_audio_tokens = 0
        self.mm_video_tokens = 0
        self.multimodal_inputs = None
        self.customized_info = customized_info

    def finished(self):
        return self._finished

    def init_incremental_detokenize(self):
        return self.output_ids_through_stop, 0

    def check_match_stop_str_prefix(self):
        return False


class TestOutputStreamerCustomizedInfo(unittest.TestCase):
    def setUp(self):
        serving_patch = patch(
            "sglang.srt.managers.scheduler_components.output_streamer.get_serving",
            return_value=SimpleNamespace(stream_interval=1),
        )
        observability_patch = patch(
            "sglang.srt.managers.scheduler_components.output_streamer.get_observability",
            return_value=SimpleNamespace(enable_request_time_stats_logging=False),
        )
        serving_patch.start()
        observability_patch.start()
        self.addCleanup(serving_patch.stop)
        self.addCleanup(observability_patch.stop)

    @staticmethod
    def _accumulator():
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
        )

    def test_customized_info_is_padded_for_mixed_batches(self):
        accumulator = self._accumulator()

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

    def test_additional_customized_info_uses_the_existing_payload(self):
        class Streamer(SchedulerOutputStreamer):
            has_additional_customized_info = True

            def get_cached_tokens_details(self, req):
                return None

            def build_additional_customized_info(self, reqs):
                return {"request_info": [[req.rid] for req in reqs]}

        outputs = []
        streamer = Streamer(
            send_to_detokenizer=SimpleNamespace(send_output=outputs.append),
            tree_cache=None,
            ps=SimpleNamespace(dp_rank=0, attn_tp_rank=0),
            server_args=SimpleNamespace(
                stream_interval=1,
                enable_request_time_stats_logging=False,
            ),
            is_generation=True,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            disaggregation_mode=DisaggregationMode.NULL,
            enable_hicache_storage=lambda: False,
        )

        streamer._stream_output_generation([_FakeReq("r0", [], finished=True)], False)

        self.assertEqual(len(outputs), 1)
        self.assertEqual(
            unwrap_from_pickle(outputs[0].customized_info),
            {"request_info": [["r0"]]},
        )

    def test_additional_customized_info_only_indexes_emitted_requests(self):
        class Streamer(SchedulerOutputStreamer):
            has_additional_customized_info = True

            def get_cached_tokens_details(self, req):
                return None

            def build_additional_customized_info(self, reqs):
                return {"request_info": [[req.rid] for req in reqs]}

        outputs = []
        streamer = Streamer(
            send_to_detokenizer=SimpleNamespace(send_output=outputs.append),
            tree_cache=None,
            ps=SimpleNamespace(dp_rank=0, attn_tp_rank=0),
            server_args=SimpleNamespace(
                stream_interval=1,
                enable_request_time_stats_logging=False,
            ),
            is_generation=True,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            disaggregation_mode=DisaggregationMode.NULL,
            enable_hicache_storage=lambda: False,
        )
        quiet = _FakeReq("quiet", [10, 11])
        quiet.stream = True
        quiet.sampling_params.stream_interval = 2
        terminal = _FakeReq("terminal", [20], finished=True)

        streamer._stream_output_generation([quiet, terminal], False)

        self.assertEqual(outputs[0].rids, ["terminal"])
        self.assertEqual(
            unwrap_from_pickle(outputs[0].customized_info),
            {"request_info": [["terminal"]]},
        )

    def test_additional_customized_info_handles_suppressed_request_last(self):
        class Streamer(SchedulerOutputStreamer):
            has_additional_customized_info = True

            def get_cached_tokens_details(self, req):
                return None

            def build_additional_customized_info(self, reqs):
                return {"request_info": [[req.rid] for req in reqs]}

        outputs = []
        streamer = Streamer(
            send_to_detokenizer=SimpleNamespace(send_output=outputs.append),
            tree_cache=None,
            ps=SimpleNamespace(dp_rank=0, attn_tp_rank=0),
            server_args=SimpleNamespace(
                stream_interval=1,
                enable_request_time_stats_logging=False,
            ),
            is_generation=True,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            disaggregation_mode=DisaggregationMode.NULL,
            enable_hicache_storage=lambda: False,
        )
        terminal = _FakeReq("terminal", [20], finished=True)
        quiet = _FakeReq("quiet", [10, 11])
        quiet.stream = True
        quiet.sampling_params.stream_interval = 2

        streamer._stream_output_generation([terminal, quiet], False)

        self.assertEqual(outputs[0].rids, ["terminal"])
        self.assertEqual(
            unwrap_from_pickle(outputs[0].customized_info),
            {"request_info": [["terminal"]]},
        )

    def test_additional_customized_info_preserves_duplicate_rid_requests(self):
        accepted_reqs = []

        class Streamer(SchedulerOutputStreamer):
            has_additional_customized_info = True

            def get_cached_tokens_details(self, req):
                return None

            def build_additional_customized_info(self, reqs):
                accepted_reqs.extend(reqs)
                return {"request_info": [[req.rid] for req in reqs]}

        outputs = []
        streamer = Streamer(
            send_to_detokenizer=SimpleNamespace(send_output=outputs.append),
            tree_cache=None,
            ps=SimpleNamespace(dp_rank=0, attn_tp_rank=0),
            server_args=SimpleNamespace(
                stream_interval=1,
                enable_request_time_stats_logging=False,
            ),
            is_generation=True,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            disaggregation_mode=DisaggregationMode.NULL,
            enable_hicache_storage=lambda: False,
        )
        first = _FakeReq("duplicate", [10], finished=True)
        second = _FakeReq("duplicate", [20], finished=True)

        streamer._stream_output_generation([first, second], False)

        self.assertEqual(accepted_reqs, [first, second])

    def test_additional_customized_info_hook_is_opt_in(self):
        class Streamer(SchedulerOutputStreamer):
            build_additional_customized_info = Mock()

            def get_cached_tokens_details(self, req):
                return None

        outputs = []
        streamer = Streamer(
            send_to_detokenizer=SimpleNamespace(send_output=outputs.append),
            tree_cache=None,
            ps=SimpleNamespace(dp_rank=0, attn_tp_rank=0),
            server_args=SimpleNamespace(),
            is_generation=True,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            disaggregation_mode=DisaggregationMode.NULL,
            enable_hicache_storage=lambda: False,
        )

        streamer._stream_output_generation([_FakeReq("r0", [], finished=True)], False)

        Streamer.build_additional_customized_info.assert_not_called()
        self.assertIsNone(outputs[0].customized_info)

    def test_additional_customized_info_hook_can_skip_inactive_batches(self):
        class Streamer(SchedulerOutputStreamer):
            has_additional_customized_info = True
            build_additional_customized_info = Mock()
            should_build_additional_customized_info = Mock(return_value=False)

            def get_cached_tokens_details(self, req):
                return None

        outputs = []
        streamer = Streamer(
            send_to_detokenizer=SimpleNamespace(send_output=outputs.append),
            tree_cache=None,
            ps=SimpleNamespace(dp_rank=0, attn_tp_rank=0),
            server_args=SimpleNamespace(),
            is_generation=True,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            disaggregation_mode=DisaggregationMode.NULL,
            enable_hicache_storage=lambda: False,
        )

        streamer._stream_output_generation([_FakeReq("r0", [], finished=True)], False)

        Streamer.should_build_additional_customized_info.assert_called_once_with()
        Streamer.build_additional_customized_info.assert_not_called()
        self.assertIsNone(outputs[0].customized_info)

    def test_additional_customized_info_rejects_rust_egress(self):
        class Streamer(SchedulerOutputStreamer):
            has_additional_customized_info = True

        with self.assertRaisesRegex(ValueError, "Rust egress"):
            Streamer(
                send_to_detokenizer=SimpleNamespace(),
                tree_cache=None,
                ps=SimpleNamespace(),
                server_args=SimpleNamespace(),
                is_generation=True,
                spec_algorithm=SpeculativeAlgorithm.NONE,
                disaggregation_mode=DisaggregationMode.NULL,
                enable_hicache_storage=lambda: False,
                rust_server=object(),
            )


if __name__ == "__main__":
    unittest.main()
