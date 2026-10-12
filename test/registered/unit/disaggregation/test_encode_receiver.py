"""Unit tests for the encode-disaggregation receiver."""

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.encoder.receiver import (
    WaitingZmqRequest,
    _select_mm_processor_prompt,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSelectMMProcessorPrompt(CustomTestCase):
    def test_tokens_in_prompt_is_list(self):
        """A tokens-in prompt reaches the processor as list[int], not array("q");
        HTTP sends input_text=None and the gRPC servicer sends ""."""
        processor = SimpleNamespace(prefer_tokenized_input=False)
        for input_text in (None, ""):
            with self.subTest(input_text=input_text):
                recv_req = SimpleNamespace(
                    input_text=input_text, input_ids=array("q", [1, 2, 3])
                )

                prompt = _select_mm_processor_prompt(recv_req, processor)

                self.assertIsInstance(prompt, list)
                self.assertEqual(prompt, [1, 2, 3])


class TestEncodeReceiverRequestConstruction(CustomTestCase):
    def test_caller_mm_hashes_set_scheduler_built_pad_values(self):
        """With zmq_to_scheduler or mooncake the scheduler builds mm_inputs, so it
        must apply caller mm_hashes; otherwise pad values ignore the caller hash."""
        item = MultimodalDataItem(
            modality=Modality.IMAGE, precomputed_embeddings=torch.ones(4, 2)
        )
        request = WaitingZmqRequest.__new__(WaitingZmqRequest)
        request.recv_req = SimpleNamespace(
            input_text="<image>", input_ids=None, mm_hashes=["00c0ffee12345678"]
        )
        request.mm_processor = SimpleNamespace(
            prefer_tokenized_input=False,
            get_validated_mm_data=lambda _prompt, _embeddings, **_kwargs: (
                MultimodalProcessorOutput(input_ids=[1, 2, 3, 4], mm_items=[item])
            ),
        )
        request.recv_embedding_data = SimpleNamespace(get_mm_extra_meta=lambda: {})
        request.embedding_pool = None
        request._pool_slot_id = None

        request._finish_assemble({Modality.IMAGE: item.precomputed_embeddings})
        item.set_pad_value()

        expected = MultimodalDataItem(modality=Modality.IMAGE, hash=0x00C0FFEE12345678)
        expected.set_pad_value()
        self.assertEqual(item.padding_values(), expected.padding_values())


if __name__ == "__main__":
    unittest.main()
