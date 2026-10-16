"""Regression tests for SHM transport with token-ID-only multimodal requests."""

import pickle
import unittest
from array import array
from multiprocessing import shared_memory
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import mm_utils  # noqa: E402
from sglang.srt.managers.io_struct import (  # noqa: E402
    BatchTokenizedEmbeddingReqInput,
    TokenizedEmbeddingReqInput,
)
from sglang.srt.managers.schedule_batch import (  # noqa: E402
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.runtime_context import get_context  # noqa: E402
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _request(feature, precomputed_embeddings):
    return TokenizedEmbeddingReqInput(
        rid="shm-feature-request",
        input_text=None,
        input_ids=array("q", [1]),
        mm_inputs=MultimodalProcessorOutput(
            mm_items=[
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    hash=37,
                    feature=feature,
                    precomputed_embeddings=precomputed_embeddings,
                )
            ]
        ),
        token_type_ids=None,
        sampling_params=SamplingParams(),
    )


class TestShmFeatureTransport(unittest.TestCase):
    def test_single_node_round_trip_with_or_without_tokenizer(self):
        for skip_tokenizer_init in (False, True):
            for batched in (False, True):
                with (
                    self.subTest(
                        skip_tokenizer_init=skip_tokenizer_init, batched=batched
                    ),
                    get_context().override_server_args(
                        nnodes=1, skip_tokenizer_init=skip_tokenizer_init
                    ),
                    patch.object(mm_utils, "_is_default_tensor_transport", None),
                ):
                    tensors = {
                        "feature": torch.arange(12, dtype=torch.float32)
                        .reshape(3, 4)
                        .T,
                        "precomputed_embeddings": torch.arange(
                            8, dtype=torch.float32
                        ).reshape(2, 4),
                    }
                    sender = _request(**tensors)
                    receiver = None
                    shm_names = []
                    try:
                        mm_utils.wrap_shm_features(sender)
                        item = sender.mm_inputs.mm_items[0]
                        for field in tensors:
                            pointer = getattr(item, field)
                            self.assertIsInstance(pointer, mm_utils.ShmPointerMMData)
                            self.assertEqual(pointer.precomputed_hash, 37)
                            shm_names.append(pointer.shm_name)

                        receiver = pickle.loads(pickle.dumps(sender))
                        request = receiver
                        if batched:
                            receiver = BatchTokenizedEmbeddingReqInput(batch=[request])
                        mm_utils.unwrap_shm_features(receiver)
                        for field, expected in tensors.items():
                            actual = getattr(request.mm_inputs.mm_items[0], field)
                            self.assertIsInstance(actual, torch.Tensor)
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

                        for name in shm_names:
                            try:
                                handle = shared_memory.SharedMemory(name=name)
                            except FileNotFoundError:
                                continue
                            handle.close()
                            self.fail(f"SHM segment {name} was not unlinked")
                    finally:
                        mm_utils.discard_shm_features(sender)
                        if receiver is not None:
                            mm_utils.discard_shm_features(receiver)

    def test_multi_node_keeps_inline_features_with_or_without_tokenizer(self):
        for skip_tokenizer_init in (False, True):
            with (
                self.subTest(skip_tokenizer_init=skip_tokenizer_init),
                get_context().override_server_args(
                    nnodes=2, skip_tokenizer_init=skip_tokenizer_init
                ),
                patch.object(mm_utils, "_is_default_tensor_transport", None),
            ):
                feature = torch.ones(2, 3)
                embeddings = torch.ones(3, 4)
                request = _request(feature, embeddings)
                self.assertIs(mm_utils.wrap_shm_features(request), request)
                self.assertIs(mm_utils.unwrap_shm_features(request), request)
                item = request.mm_inputs.mm_items[0]
                self.assertIs(item.feature, feature)
                self.assertIs(item.precomputed_embeddings, embeddings)


if __name__ == "__main__":
    unittest.main()
