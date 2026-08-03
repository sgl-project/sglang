import unittest
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import unwrap_from_pickle
from sglang.srt.managers.scheduler_components.output_streamer import (
    _GenerationStreamAccumulator,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeReq:
    def __init__(self, rid, output_ids, customized_info=None):
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
        self.drop_trailing_stop_token = False
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

    def finished(self):
        return False

    def init_incremental_detokenize(self):
        return self.output_ids_through_stop, 0

    def check_match_stop_str_prefix(self):
        return False


class TestOutputStreamerCustomizedInfo(unittest.TestCase):
    @staticmethod
    def _accumulator(
        *,
        return_logprob: bool = False,
        return_hidden_states: bool = False,
        return_sampling_mask: bool = False,
    ):
        return _GenerationStreamAccumulator(
            return_logprob=return_logprob,
            return_hidden_states=return_hidden_states,
            return_routed_experts=False,
            return_indexer_topk=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            disaggregation_mode=DisaggregationMode.NULL,
            default_stream_interval=1,
            default_force_stream_interval=1,
            get_cached_tokens_details=lambda req: None,
            return_sampling_mask=return_sampling_mask,
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

    def test_dropped_stop_token_is_removed_from_every_per_token_channel(self):
        for retained_ids in ([], [10]):
            with self.subTest(retained_ids=retained_ids):
                req = _FakeReq("r0", retained_ids + [99])
                req.output_ids_through_stop = retained_ids
                req.finished = lambda: True
                req.drop_trailing_stop_token = True
                req.return_logprob = True
                req.input_logprob_sent = False
                req.logprob = SimpleNamespace(
                    input_token_logprobs_val=None,
                    output_token_logprobs_val=[0.1, 0.2],
                    output_token_logprobs_idx=[10, 99],
                    output_top_logprobs_val=[[0.1], [0.2]],
                    output_top_logprobs_idx=[[10], [99]],
                    output_token_ids_logprobs_val=[[0.1], [0.2]],
                    output_token_ids_logprobs_idx=[[10], [99]],
                )
                req.return_sampling_mask = True
                req.send_output_sampling_mask_offset = 0
                req.output_token_sampling_mask = [True, False]
                req.output_token_sampling_logprobs = [0.1, 0.2]
                req.return_hidden_states = True
                req.hidden_states = torch.tensor([[1.0], [2.0]])
                accumulator = self._accumulator(
                    return_logprob=True,
                    return_hidden_states=True,
                    return_sampling_mask=True,
                )

                accumulator.accept(req=req)

                expected_len = len(retained_ids)
                self.assertEqual(accumulator.output_ids, [retained_ids])
                self.assertEqual(
                    len(accumulator.output_token_logprobs_val[0]), expected_len
                )
                self.assertEqual(accumulator.output_token_logprobs_idx[0], retained_ids)
                self.assertEqual(
                    len(accumulator.output_token_sampling_mask[0]), expected_len
                )
                self.assertEqual(
                    accumulator.output_token_sampling_mask[0],
                    [True][:expected_len],
                )
                self.assertEqual(
                    accumulator.output_hidden_states[0].shape[0], expected_len
                )
                torch.testing.assert_close(
                    accumulator.output_hidden_states[0],
                    torch.tensor([[1.0]])[:expected_len],
                )

    def test_default_per_token_channel_boundaries_are_unchanged(self):
        req = _FakeReq("r0", [10, 99])
        req.output_ids_through_stop = [10]
        req.finished = lambda: True
        req.finished_len = 1
        req.return_logprob = True
        req.input_logprob_sent = False
        req.logprob = SimpleNamespace(
            input_token_logprobs_val=None,
            output_token_logprobs_val=[0.1, 0.2],
            output_token_logprobs_idx=[10, 99],
            output_top_logprobs_val=[[0.1], [0.2]],
            output_top_logprobs_idx=[[10], [99]],
            output_token_ids_logprobs_val=[[0.1], [0.2]],
            output_token_ids_logprobs_idx=[[10], [99]],
        )
        req.return_sampling_mask = True
        req.send_output_sampling_mask_offset = 0
        req.output_token_sampling_mask = [True, False]
        req.output_token_sampling_logprobs = [0.1, 0.2]
        req.return_hidden_states = True
        req.hidden_states = torch.tensor([[1.0], [2.0]])
        accumulator = self._accumulator(
            return_logprob=True,
            return_hidden_states=True,
            return_sampling_mask=True,
        )

        accumulator.accept(req=req)

        self.assertEqual(accumulator.output_token_logprobs_idx, [[10]])
        self.assertEqual(accumulator.output_token_sampling_mask, [[True, False]])
        torch.testing.assert_close(
            accumulator.output_hidden_states[0], torch.tensor([[1.0]])
        )

    def test_default_empty_output_keeps_legacy_first_logprob(self):
        req = _FakeReq("r0", [99])
        req.output_ids_through_stop = []
        req.finished = lambda: True
        req.return_logprob = True
        req.input_logprob_sent = False
        req.logprob = SimpleNamespace(
            input_token_logprobs_val=None,
            output_token_logprobs_val=[0.2],
            output_token_logprobs_idx=[99],
            output_top_logprobs_val=[[0.2]],
            output_top_logprobs_idx=[[99]],
            output_token_ids_logprobs_val=[[0.2]],
            output_token_ids_logprobs_idx=[[99]],
        )
        req.return_sampling_mask = False
        req.return_hidden_states = False
        accumulator = self._accumulator(return_logprob=True)

        accumulator.accept(req=req)

        self.assertEqual(accumulator.output_token_logprobs_idx, [[99]])


if __name__ == "__main__":
    unittest.main()
