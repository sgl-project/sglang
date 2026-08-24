"""Unit tests for request construction in the encode-disaggregation path."""

import gc
import unittest
from array import array
from types import SimpleNamespace

import torch
import zmq

from sglang.srt.disaggregation.encoder.receiver import (
    EmbeddingData,
    MMReceiverBase,
    MultiModalEmbeddingData,
    SegmentedEmbedding,
    WaitingZmqRequest,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestEncodeReceiverRequestConstruction(CustomTestCase):
    def test_extra_key_and_cache_salt_are_forwarded(self):
        scheduler = SimpleNamespace(
            model_config=SimpleNamespace(hf_eos_token_id={2}, vocab_size=128),
            disaggregation_mode=DisaggregationMode.NULL,
            metrics_reporter=SimpleNamespace(enable_metrics=False),
            metrics_collector=None,
            dllm_config=None,
            tokenizer=object(),
        )
        receiver = SimpleNamespace(scheduler=scheduler)
        recv_req = SimpleNamespace(
            rid="request-1",
            input_text="hello",
            input_ids=array("q", [1, 2]),
            sampling_params=SamplingParams(max_new_tokens=1),
            return_logprob=False,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
            lora_id=None,
            input_embeds=None,
            custom_logit_processor=None,
            require_reasoning=False,
            return_hidden_states=False,
            return_routed_experts=False,
            routed_experts_start_len=0,
            bootstrap_host=None,
            bootstrap_port=None,
            bootstrap_room=None,
            routed_dp_rank=None,
            disagg_prefill_dp_rank=None,
            priority=None,
            extra_key="classification",
            cache_salt="tenant-a",
            http_worker_ipc=None,
        )

        req = MMReceiverBase.create_req(receiver, recv_req)

        self.assertEqual(req.extra_key, "classification")
        self.assertEqual(req.cache_salt, "tenant-a")


class TestSegmentedReceiverEmbedding(CustomTestCase):
    @staticmethod
    def _part(part_idx, embedding):
        return EmbeddingData(
            req_id="request-1",
            num_parts=2,
            part_idx=part_idx,
            grid_dim=torch.tensor([[1, 1, embedding.shape[0]]]),
            modality=Modality.IMAGE,
            embedding=embedding,
        )

    def test_segmented_embedding_slices(self):
        first = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        second = torch.arange(16, 28, dtype=torch.float32).reshape(3, 4)
        data = MultiModalEmbeddingData.from_embedding_data(self._part(0, first))
        data.add(self._part(1, second))

        segmented = data.get_embedding()[Modality.IMAGE]
        self.assertIsInstance(segmented, SegmentedEmbedding)

        item = segmented[4:7]
        self.assertTrue(torch.equal(item, second))
        self.assertEqual(
            item.untyped_storage().data_ptr(), second.untyped_storage().data_ptr()
        )
        self.assertTrue(
            torch.equal(segmented[3:5], torch.cat([first[3:4], second[:1]]))
        )

    def test_zmq_frame_tensor_survives_later_receives(self):
        context = zmq.Context()
        pull = context.socket(zmq.PULL)
        push = context.socket(zmq.PUSH)
        endpoint = "inproc://test-zmq-frame-tensor-lifetime"
        pull.bind(endpoint)
        push.connect(endpoint)
        try:
            expected = torch.arange(1024, dtype=torch.int32)
            push.send(expected.numpy().tobytes())
            frame = pull.recv(copy=False)
            frame_view = torch.frombuffer(frame.buffer, dtype=torch.int32)
            recv_obj = SimpleNamespace(dtype=torch.int32, shape=expected.shape)

            WaitingZmqRequest._extract_embedding_from_buffer(
                None, recv_obj, [None, frame]
            )
            self.assertEqual(recv_obj.embedding.data_ptr(), frame_view.data_ptr())
            del frame_view
            del frame
            gc.collect()

            for value in range(8):
                push.send(bytes([value]) * 8192)
                pull.recv(copy=False)

            torch.testing.assert_close(recv_obj.embedding, expected)
        finally:
            pull.close()
            push.close()
            context.term()

        gc.collect()
        torch.testing.assert_close(recv_obj.embedding, expected)


if __name__ == "__main__":
    unittest.main()
