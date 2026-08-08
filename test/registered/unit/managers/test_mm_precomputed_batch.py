"""Regression test for mixed precomputed and raw multimodal batches."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import mm_utils  # noqa: E402
from sglang.srt.managers.schedule_batch import (  # noqa: E402
    Modality,
    MultimodalDataItem,
    MultimodalInputFormat,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import (  # noqa: E402
    ForwardBatch,
    ForwardMode,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeLanguageModel(nn.Module):
    def __init__(self, input_embedding):
        super().__init__()
        self.input_embedding = input_embedding

    def get_input_embeddings(self):
        return self.input_embedding

    def forward(self, input_ids=None, forward_batch=None, input_embeds=None, **kwargs):
        return input_embeds


def _run_mm_embedding(mm_inputs, input_ids, data_embedding_funcs, extend_seq_lens):
    mm_utils.init_mm_embedding_cache()
    forward_batch = ForwardBatch.__new__(ForwardBatch)
    forward_batch.forward_mode = ForwardMode.EXTEND
    forward_batch.mm_inputs = mm_inputs
    forward_batch.extend_prefix_lens_cpu = [0] * len(mm_inputs)
    forward_batch.extend_seq_lens_cpu = extend_seq_lens
    forward_batch.input_embeds = None

    input_embedding = nn.Embedding(128, 4)
    with torch.no_grad():
        input_embedding.weight.zero_()

    with patch.object(
        mm_utils,
        "get_server_args",
        return_value=SimpleNamespace(),
    ), patch.object(
        mm_utils,
        "get_disagg",
        return_value=SimpleNamespace(
            enable_adaptive_dispatch_to_encoder=False,
            language_only=False,
        ),
    ):
        return mm_utils.general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=_FakeLanguageModel(input_embedding),
            data_embedding_funcs=data_embedding_funcs,
        )


class TestMixedPrecomputedMultimodalBatch(unittest.TestCase):
    def test_mixes_precomputed_and_raw_requests_without_adaptive_dispatch(self):
        placeholder_token = 99
        precomputed_item = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(0, 0)],
            pad_value=placeholder_token,
            hash=1,
            precomputed_embeddings=torch.full((1, 4), 3.0),
        )
        raw_item = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(0, 0)],
            pad_value=placeholder_token,
            hash=2,
            feature=torch.ones((1, 2, 2, 2)),
        )
        mm_inputs = [
            MultimodalInputs(mm_items=[precomputed_item]),
            None,
            MultimodalInputs(mm_items=[raw_item]),
        ]

        def get_image_feature(items):
            return torch.full((len(items), 4), 7.0)

        output = _run_mm_embedding(
            mm_inputs,
            torch.tensor(
                [placeholder_token, 0, 1, 2, placeholder_token, 0],
                dtype=torch.long,
            ),
            {Modality.IMAGE: get_image_feature},
            extend_seq_lens=[2, 2, 2],
        )

        self.assertEqual(tuple(output.shape), (6, 4))
        torch.testing.assert_close(output[0], torch.full((4,), 3.0))
        torch.testing.assert_close(output[4], torch.full((4,), 7.0))

    def test_mixes_public_precomputed_format_and_raw_requests(self):
        placeholder_token = 99
        precomputed_item = MultimodalDataItem(
            modality=Modality.IMAGE,
            format=MultimodalInputFormat.PRECOMPUTED_EMBEDDING,
            offsets=[(0, 0)],
            pad_value=placeholder_token,
            hash=1,
            feature=torch.full((1, 4), 3.0),
        )
        raw_item = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(0, 0)],
            pad_value=placeholder_token,
            hash=2,
            feature=torch.ones((1, 2, 2, 2)),
        )
        mm_inputs = [
            MultimodalInputs(mm_items=[precomputed_item]),
            MultimodalInputs(mm_items=[raw_item]),
        ]

        def get_image_feature(items):
            if items and items[0].format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING:
                result = torch.cat([item.feature for item in items])
                return result.reshape(-1, result.shape[-1])
            return torch.full((len(items), 4), 7.0)

        output = _run_mm_embedding(
            mm_inputs,
            torch.tensor(
                [placeholder_token, 0, placeholder_token, 0],
                dtype=torch.long,
            ),
            {Modality.IMAGE: get_image_feature},
            extend_seq_lens=[2, 2],
        )

        self.assertEqual(tuple(output.shape), (4, 4))
        torch.testing.assert_close(output[0], torch.full((4,), 3.0))
        torch.testing.assert_close(output[2], torch.full((4,), 7.0))

    def test_splits_precomputed_and_raw_modalities_in_one_request(self):
        image_precomputed = MultimodalDataItem(
            modality=Modality.IMAGE,
            format=MultimodalInputFormat.PRECOMPUTED_EMBEDDING,
            offsets=[(0, 0)],
            pad_value=99,
            hash=1,
            feature=torch.full((1, 4), 3.0),
        )
        audio_raw = MultimodalDataItem(
            modality=Modality.AUDIO,
            offsets=[(1, 1)],
            pad_value=98,
            hash=2,
            feature=torch.ones((1, 2, 2)),
        )
        image_raw = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(0, 0)],
            pad_value=99,
            hash=3,
            feature=torch.ones((1, 2, 2, 2)),
        )
        mm_inputs = [
            MultimodalInputs(mm_items=[image_precomputed, audio_raw]),
            MultimodalInputs(mm_items=[image_raw]),
        ]

        def get_image_feature(items):
            features = torch.cat([item.feature for item in items], dim=0)
            if features.dim() == 2:
                return features
            return torch.full((features.shape[0], 4), 7.0)

        def get_audio_feature(items):
            return torch.full((len(items), 4), 8.0)

        output = _run_mm_embedding(
            mm_inputs,
            torch.tensor([99, 98, 0, 99, 0, 1], dtype=torch.long),
            {
                Modality.IMAGE: get_image_feature,
                Modality.AUDIO: get_audio_feature,
            },
            extend_seq_lens=[3, 3],
        )

        self.assertEqual(tuple(output.shape), (6, 4))
        torch.testing.assert_close(output[0], torch.full((4,), 3.0))
        torch.testing.assert_close(output[1], torch.full((4,), 8.0))
        torch.testing.assert_close(output[3], torch.full((4,), 7.0))


if __name__ == "__main__":
    unittest.main()
