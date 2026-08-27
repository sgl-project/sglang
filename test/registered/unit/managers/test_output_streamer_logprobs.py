import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler_components.output_streamer import (
    _GenerationStreamAccumulator,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeReq:
    def __init__(self, *, is_retracted: bool, max_new_tokens: int):
        self.rid = "req"
        self.http_worker_ipc = None
        self.finished_reason = None
        self.finished_output = False
        self.finished_len = None
        self.stream = False
        self.sampling_params = SimpleNamespace(
            max_new_tokens=max_new_tokens,
            stream_interval=None,
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
            no_stop_trim=False,
        )
        self.output_ids = []
        self.output_ids_through_stop = []
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
        self.customized_info = None
        self.is_retracted = is_retracted
        self.beam_group = None

        self.return_logprob = True
        self.input_logprob_sent = True
        self.logprob = SimpleNamespace(
            output_token_logprobs_val=[-0.5],
            output_token_logprobs_idx=[42],
            output_top_logprobs_val=[[(-0.5, 42)]],
            output_top_logprobs_idx=[[42]],
            output_token_ids_logprobs_val=[[-0.5]],
            output_token_ids_logprobs_idx=[[42]],
        )

    def finished(self):
        return False

    def init_incremental_detokenize(self):
        return self.output_ids_through_stop, 0


def _make_accumulator() -> _GenerationStreamAccumulator:
    return _GenerationStreamAccumulator(
        return_logprob=True,
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


class TestOutputStreamerLogprobs(unittest.TestCase):
    def test_retracted_empty_output_does_not_advance_logprob_offset(self):
        req = _FakeReq(is_retracted=True, max_new_tokens=16)
        accumulator = _make_accumulator()

        accumulator.accept(req=req)

        self.assertEqual(req.send_token_offset, 0)
        self.assertEqual(req.send_output_token_logprobs_offset, 0)
        self.assertEqual(accumulator.output_token_logprobs_val, [[]])

    def test_prefill_only_request_preserves_first_logprob(self):
        req = _FakeReq(is_retracted=False, max_new_tokens=0)
        accumulator = _make_accumulator()

        accumulator.accept(req=req)

        self.assertEqual(req.send_token_offset, 0)
        self.assertEqual(req.send_output_token_logprobs_offset, 1)
        self.assertEqual(accumulator.output_token_logprobs_val, [[-0.5]])


if __name__ == "__main__":
    unittest.main()
