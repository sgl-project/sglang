import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers import mm_utils, schedule_batch
from sglang.srt.managers.mm_utils import (
    _get_chunked_prefill_embedding,
    init_mm_embedding_cache,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.multimodal.evs import EVSEmbeddingResult, VideoEVSDataItem


def _make_proxy_with_reconstruct_result(tensor: torch.Tensor):
    proxy = mm_utils.CudaIpcTensorTransportProxy.__new__(
        mm_utils.CudaIpcTensorTransportProxy
    )
    proxy.reconstruct_on_target_device = Mock(return_value=tensor)
    return proxy


class TestMultimodalInputsFromDict(unittest.TestCase):
    def test_materialize_proxy(self):
        feature_tensor = torch.tensor([[7.0], [8.0]], dtype=torch.float32)
        proxy_feature = _make_proxy_with_reconstruct_result(feature_tensor)
        mm_item = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(0, 1), (1, 2)],
            feature=proxy_feature,
            model_specific_data={"image_grid_thw": [[1, 1, 1], [1, 1, 1]]},
        )

        with (
            patch.object(schedule_batch.torch.cuda, "is_available", return_value=True),
            patch.object(schedule_batch.torch.cuda, "current_device", return_value=0),
            patch.object(
                schedule_batch.envs.SGLANG_MM_BUFFER_SIZE_MB, "get", return_value=0
            ),
        ):
            mm_inputs = MultimodalInputs.from_dict({"mm_items": [mm_item]})

        # Splitting happens at the processor layer, not in from_dict.
        # from_dict just reconstructs and passes through.
        self.assertEqual(len(mm_inputs.mm_items), 1)
        self.assertTrue(torch.equal(mm_inputs.mm_items[0].feature, feature_tensor))
        proxy_feature.reconstruct_on_target_device.assert_called_once_with(0)

    def test_materialize_precomputed_embedding_proxy_without_feature(self):
        embedding_tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        proxy_embedding = _make_proxy_with_reconstruct_result(embedding_tensor)
        mm_item = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(0, 1)],
            precomputed_embeddings=proxy_embedding,
        )

        with (
            patch.object(schedule_batch.torch.cuda, "is_available", return_value=True),
            patch.object(schedule_batch.torch.cuda, "current_device", return_value=0),
            patch.object(
                schedule_batch.envs.SGLANG_MM_BUFFER_SIZE_MB, "get", return_value=0
            ),
        ):
            mm_inputs = MultimodalInputs.from_dict({"mm_items": [mm_item]})

        self.assertTrue(
            torch.equal(
                mm_inputs.mm_items[0].precomputed_embeddings,
                embedding_tensor,
            )
        )
        proxy_embedding.reconstruct_on_target_device.assert_called_once_with(0)

    def test_materialize_model_specific_proxy_without_feature(self):
        grid_tensor = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        proxy_grid = _make_proxy_with_reconstruct_result(grid_tensor)
        mm_item = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(0, 1)],
            model_specific_data={"image_grid_thw": proxy_grid},
        )

        with (
            patch.object(schedule_batch.torch.cuda, "is_available", return_value=True),
            patch.object(schedule_batch.torch.cuda, "current_device", return_value=0),
            patch.object(
                schedule_batch.envs.SGLANG_MM_BUFFER_SIZE_MB, "get", return_value=0
            ),
        ):
            mm_inputs = MultimodalInputs.from_dict({"mm_items": [mm_item]})

        self.assertTrue(
            torch.equal(
                mm_inputs.mm_items[0].model_specific_data["image_grid_thw"],
                grid_tensor,
            )
        )
        proxy_grid.reconstruct_on_target_device.assert_called_once_with(0)


class TestEVSChunkedPrefill(unittest.TestCase):
    """Regression test for https://github.com/sgl-project/sglang/issues/26507.

    _get_chunked_embedding_by_item assumed data_embedding_func always returns a
    plain tensor and called .reshape() on it. EVS video items return
    EVSEmbeddingResult instead, causing AttributeError. The fix routes
    VideoEVSDataItem through _get_chunked_embedding_full, which handles
    EVSEmbeddingResult correctly.
    """

    HIDDEN = 5120
    NUM_TOKENS = 256
    PAD_VALUE = 100001
    ITEM_HASH = 16885862069851955115

    def setUp(self):
        init_mm_embedding_cache(max_size=10)
        self.item = VideoEVSDataItem(
            modality=Modality.VIDEO,
            hash=self.ITEM_HASH,
            pad_value=self.PAD_VALUE,
            offsets=[(21, 276)],
            feature=torch.zeros(1, self.HIDDEN),
            thw_grids=[(1, 16, 16)],
            pre_chunked_input_ids=torch.full(
                (self.NUM_TOKENS,), self.PAD_VALUE, dtype=torch.long
            ),
        )

    def _make_evs_video(self):
        def evs_video(items):
            return EVSEmbeddingResult(
                embedding=torch.zeros(self.NUM_TOKENS, self.HIDDEN),
                num_tokens_per_frame=[self.NUM_TOKENS],
            )

        return evs_video

    def test_evs_video_item_does_not_crash_chunked_prefill(self):
        # Before the fix this raised:
        #   AttributeError: 'EVSEmbeddingResult' object has no attribute 'reshape'
        input_ids = torch.full((self.NUM_TOKENS,), self.PAD_VALUE, dtype=torch.long)
        try:
            _get_chunked_prefill_embedding(
                data_embedding_func=self._make_evs_video(),
                embedding_items=[self.item],
                items_size=[0, 1],
                prefix_length=[0],
                extend_length=[self.NUM_TOKENS],
                items_offset_list=[[(21, 276)]],
                input_ids=input_ids,
            )
        except AttributeError as e:
            self.fail(f"EVSEmbeddingResult crashed chunked prefill: {e}")

    def test_evs_video_item_returns_embedding_with_correct_shape(self):
        input_ids = torch.full((self.NUM_TOKENS,), self.PAD_VALUE, dtype=torch.long)
        embedding, _ = _get_chunked_prefill_embedding(
            data_embedding_func=self._make_evs_video(),
            embedding_items=[self.item],
            items_size=[0, 1],
            prefix_length=[0],
            extend_length=[self.NUM_TOKENS],
            items_offset_list=[[(21, 276)]],
            input_ids=input_ids,
        )
        self.assertIsNotNone(embedding)
        self.assertEqual(embedding.shape[-1], self.HIDDEN)


if __name__ == "__main__":
    unittest.main(verbosity=2)
