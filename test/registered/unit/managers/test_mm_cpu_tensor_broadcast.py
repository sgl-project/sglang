import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (  # noqa: E402
    BatchTokenizedGenerateReqInput,
    TokenizedGenerateReqInput,
    wrap_as_pickle,
)
from sglang.srt.managers.mm_transport import (  # noqa: E402
    _CpuTensorPlaceholder,
    broadcast_mm_cpu_tensors,
)
from sglang.srt.managers.schedule_batch import (  # noqa: E402
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_req(feature):
    mm_inputs = MultimodalProcessorOutput(
        mm_items=[MultimodalDataItem(modality=Modality.IMAGE, feature=feature)]
    )
    return TokenizedGenerateReqInput(
        input_text="",
        input_ids=None,
        input_embeds=None,
        mm_inputs=wrap_as_pickle(mm_inputs),
        token_type_ids=None,
        sampling_params=SamplingParams(),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
    )


class TestMMCpuTensorBroadcast(unittest.TestCase):
    def test_source_restores_feature_and_broadcasts_contiguous_payload(self):
        feature = torch.arange(2_200_000, dtype=torch.float32).reshape(2, -1).T
        self.assertFalse(feature.is_contiguous())
        req = _make_req(feature)
        sent = []

        def fake_broadcast(tensor, **kwargs):
            sent.append(tensor.clone())
            return SimpleNamespace(wait=lambda: None)

        with (
            patch(
                "sglang.srt.managers.mm_transport.broadcast_pyobj",
                side_effect=lambda data, *args, **kwargs: data,
            ),
            patch(
                "sglang.srt.managers.mm_transport.dist.broadcast",
                side_effect=fake_broadcast,
            ),
        ):
            result = broadcast_mm_cpu_tensors([req], rank=0, src=0)

        self.assertIs(result[0].mm_inputs.mm_items[0].feature, feature)
        self.assertEqual(len(sent), 1)
        self.assertTrue(sent[0].is_contiguous())
        torch.testing.assert_close(sent[0], feature)

    def test_peer_allocates_and_receives_feature(self):
        expected = torch.arange(2_200_000, dtype=torch.float32)
        req = _make_req(torch.empty(1))
        req.mm_inputs = MultimodalProcessorOutput(
            mm_items=[
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    feature=_CpuTensorPlaceholder(expected.shape, expected.dtype),
                )
            ]
        )

        def fake_broadcast(tensor, **kwargs):
            tensor.copy_(expected)
            return SimpleNamespace(wait=lambda: None)

        with (
            patch(
                "sglang.srt.managers.mm_transport.broadcast_pyobj",
                return_value=[req],
            ),
            patch(
                "sglang.srt.managers.mm_transport.dist.broadcast",
                side_effect=fake_broadcast,
            ),
        ):
            result = broadcast_mm_cpu_tensors(None, rank=1, src=0)

        received = result[0].mm_inputs.mm_items[0].feature
        self.assertIsInstance(received, torch.Tensor)
        torch.testing.assert_close(received, expected)

    def test_precomputed_embeddings_are_broadcast_directly(self):
        embeddings = torch.arange(5_000_000, dtype=torch.float16)
        req = _make_req(torch.empty(1))
        req.mm_inputs.mm_items[0].feature = None
        req.mm_inputs.mm_items[0].precomputed_embeddings = embeddings
        sent = []

        def fake_broadcast(tensor, **kwargs):
            sent.append(tensor.clone())
            return SimpleNamespace(wait=lambda: None)

        with (
            patch(
                "sglang.srt.managers.mm_transport.broadcast_pyobj",
                side_effect=lambda data, *args, **kwargs: data,
            ),
            patch(
                "sglang.srt.managers.mm_transport.dist.broadcast",
                side_effect=fake_broadcast,
            ),
        ):
            result = broadcast_mm_cpu_tensors([req], rank=0, src=0)

        item = result[0].mm_inputs.mm_items[0]
        self.assertIs(item.precomputed_embeddings, embeddings)
        self.assertEqual(len(sent), 1)
        torch.testing.assert_close(sent[0], embeddings)

    def test_small_feature_stays_in_metadata_broadcast(self):
        feature = torch.arange(16, dtype=torch.float32)
        req = _make_req(feature)
        with (
            patch(
                "sglang.srt.managers.mm_transport.broadcast_pyobj",
                side_effect=lambda data, *args, **kwargs: data,
            ),
            patch("sglang.srt.managers.mm_transport.dist.broadcast") as broadcast,
        ):
            result = broadcast_mm_cpu_tensors([req], rank=0, src=0)

        self.assertIs(result[0].mm_inputs.mm_items[0].feature, feature)
        broadcast.assert_not_called()

    def test_single_large_feature_below_aggregate_threshold_stays_in_metadata(self):
        # A single image-sized tensor should not pay the fixed Gloo collective
        # cost. The direct path is reserved for an aggregate scheduler batch.
        feature = torch.zeros(600_000, dtype=torch.float32)
        req = _make_req(feature)
        with (
            patch(
                "sglang.srt.managers.mm_transport.broadcast_pyobj",
                side_effect=lambda data, *args, **kwargs: data,
            ),
            patch("sglang.srt.managers.mm_transport.dist.broadcast") as broadcast,
        ):
            result = broadcast_mm_cpu_tensors([req], rank=0, src=0)

        self.assertIs(result[0].mm_inputs.mm_items[0].feature, feature)
        broadcast.assert_not_called()

    def test_transport_proxy_stays_on_metadata_path(self):
        # CUDA IPC/VMM proxies are intentionally opaque to this CPU-only
        # optimization.  A proxy-like object must not trigger a tensor
        # collective or be replaced by a CPU allocation.
        proxy = object()
        req = _make_req(proxy)
        with (
            patch(
                "sglang.srt.managers.mm_transport.broadcast_pyobj",
                side_effect=lambda data, *args, **kwargs: data,
            ),
            patch("sglang.srt.managers.mm_transport.dist.broadcast") as broadcast,
        ):
            result = broadcast_mm_cpu_tensors([req], rank=0, src=0)

        self.assertIs(result[0].mm_inputs.mm_items[0].feature, proxy)
        broadcast.assert_not_called()

    def test_batch_request_preserves_tensor_list_order(self):
        features = [
            torch.full((1_200_000,), value, dtype=torch.float32) for value in (1, 2)
        ]
        req = _make_req(features)
        batch = BatchTokenizedGenerateReqInput(batch=[req])
        sent = []

        def fake_broadcast(tensor, **kwargs):
            sent.append(tensor.clone())
            return SimpleNamespace(wait=lambda: None)

        with (
            patch(
                "sglang.srt.managers.mm_transport.broadcast_pyobj",
                side_effect=lambda data, *args, **kwargs: data,
            ),
            patch(
                "sglang.srt.managers.mm_transport.dist.broadcast",
                side_effect=fake_broadcast,
            ),
        ):
            result = broadcast_mm_cpu_tensors([batch], rank=0, src=0)

        self.assertIs(result[0].batch[0].mm_inputs.mm_items[0].feature, features)
        self.assertEqual([tensor[0].item() for tensor in sent], [1, 2])


if __name__ == "__main__":
    unittest.main()
