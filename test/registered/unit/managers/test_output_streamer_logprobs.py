import unittest
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.decode import DecodeRequest, DecodeTransferQueue
from sglang.srt.disaggregation.utils import DisaggregationMode, MetadataBuffers
from sglang.srt.managers.scheduler_components.output_streamer import (
    _GenerationStreamAccumulator,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


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

    def check_match_stop_str_prefix(self):
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
    def test_pd_handoff_preserves_decode_token_ids_logprobs(self):
        for rebootstrap, replay in ((False, False), (True, False), (True, True)):
            with self.subTest(rebootstrap=rebootstrap, replay=replay):
                req = _FakeReq(is_retracted=False, max_new_tokens=3)
                req.stream = True
                req.bootstrap_host = "127.0.0.1"
                req.bootstrap_room = 9
                req.pd_rebootstrap_forced_output_id = 42 if replay else None
                req.return_sampling_mask = False
                req.time_stats = SimpleNamespace(set_wait_queue_entry_time=lambda: None)
                req.logprob.top_logprobs_num = 1
                req.logprob.token_ids_logprob = [42]
                req.logprob.output_token_ids_logprobs_val = [[]] if replay else []
                req.logprob.output_token_ids_logprobs_idx = [[]] if replay else []
                if not replay:
                    req.logprob.output_token_logprobs_val = []
                    req.logprob.output_token_logprobs_idx = []
                    req.logprob.output_top_logprobs_val = []
                    req.logprob.output_top_logprobs_idx = []

                buffers = MetadataBuffers(
                    size=1,
                    hidden_size=2,
                    hidden_states_dtype=torch.float32,
                    max_sampling_mask_tokens=4,
                )
                buffers.bootstrap_room[0][0] = 9
                buffers.output_ids[0][0] = 99 if replay else 42
                buffers.output_token_logprobs_val[0][0] = -0.5
                buffers.output_token_logprobs_idx[0][0] = 42
                buffers.output_top_logprobs_val[0][0] = -0.5
                buffers.output_top_logprobs_idx[0][0] = 42
                queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
                queue.scheduler = SimpleNamespace(
                    kv_checksum_computer=None,
                    batch_result_processor=SimpleNamespace(
                        _maybe_update_reasoning_tokens=lambda req, token_id: None
                    ),
                )
                queue.spec_algorithm = SpeculativeAlgorithm.NONE
                queue.metadata_buffers = buffers
                queue._commit_transfer_to_req(
                    DecodeRequest(
                        req=req,
                        kv_receiver=SimpleNamespace(clear=lambda: None),
                        metadata_buffer_index=0,
                        is_rebootstrap=rebootstrap,
                    )
                )

                self.assertEqual(req.output_ids, [42])
                self.assertEqual(req.logprob.output_token_logprobs_val, [-0.5])
                self.assertEqual(req.logprob.output_token_ids_logprobs_val, [[]])
                self.assertEqual(req.logprob.output_token_ids_logprobs_idx, [[]])
                req.output_ids_through_stop = req.output_ids
                first = _make_accumulator()
                first.accept(req=req)
                self.assertEqual(first.output_token_ids_logprobs_val, [[[]]])

                req.output_ids.append(43)
                req.logprob.output_token_logprobs_val.append(-0.25)
                req.logprob.output_token_logprobs_idx.append(43)
                req.logprob.output_top_logprobs_val.append([-0.25])
                req.logprob.output_top_logprobs_idx.append([43])
                req.logprob.output_token_ids_logprobs_val.append([-1.25])
                req.logprob.output_token_ids_logprobs_idx.append([42])
                second = _make_accumulator()
                second.accept(req=req)
                self.assertEqual(second.output_token_ids_logprobs_val, [[[-1.25]]])
                self.assertEqual(second.output_token_ids_logprobs_idx, [[[42]]])
                self.assertEqual(req.send_output_token_logprobs_offset, 2)

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
