import pickle
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.disaggregation.encode_receiver import EmbeddingData
from sglang.srt.disaggregation.encode_server import MMEncoder, _get_mm_grid_dim
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.utils.common import safe_pickle_loads
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestKimiVLEPDGrid(unittest.TestCase):
    @staticmethod
    def _make_encoder(model_type="kimi_vl"):
        encoder = MMEncoder.__new__(MMEncoder)
        encoder.model_type = model_type
        encoder.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                vision_config=SimpleNamespace(merge_kernel_size=(2, 2))
            )
        )
        return encoder

    def test_kimi_vl_prefers_and_normalizes_hw_grid(self):
        mm_inputs = {
            "image_grid_hws": np.array([[40, 60]], dtype=np.int64),
            "image_grid_thw": torch.tensor([[1, 20, 30]]),
            "grid_thws": torch.tensor([[1, 10, 15]]),
        }

        grid = _get_mm_grid_dim(mm_inputs, Modality.IMAGE, "kimi_vl")

        self.assertIsInstance(grid, torch.Tensor)
        torch.testing.assert_close(grid, torch.tensor([[40, 60]]))

    def test_kimi_k25_keeps_thw_grid_preference(self):
        mm_inputs = {
            "image_grid_hws": np.array([[40, 60]], dtype=np.int64),
            "grid_thws": np.array([[1, 10, 15]], dtype=np.int64),
        }

        grid = _get_mm_grid_dim(mm_inputs, Modality.IMAGE, "kimi_k25")

        torch.testing.assert_close(grid, torch.tensor([[1, 10, 15]]))

    def test_kimi_vl_2d_grid_counting_and_slicing(self):
        encoder = self._make_encoder()
        grids = torch.tensor([[40, 60], [20, 40]])
        embedding = torch.arange(800 * 2).reshape(800, 2)

        self.assertEqual(
            encoder.get_num_patches(grids[0], Modality.IMAGE),
            2400,
        )
        self.assertEqual(
            encoder.get_num_tokens(grids[0], Modality.IMAGE),
            600,
        )

        slices = encoder.slice_embedding(embedding, grids, Modality.IMAGE)

        self.assertEqual([item.shape for item in slices], [(600, 2), (200, 2)])
        torch.testing.assert_close(slices[0], embedding[:600])
        torch.testing.assert_close(slices[1], embedding[600:])

    def test_kimi_3d_grid_remains_supported(self):
        encoder = self._make_encoder()
        grid = torch.tensor([1, 40, 60])

        self.assertEqual(encoder.get_num_patches(grid, Modality.IMAGE), 2400)
        self.assertEqual(encoder.get_num_tokens(grid, Modality.IMAGE), 600)

    def test_kimi_k25_3d_patch_counting_is_unchanged(self):
        encoder = self._make_encoder("kimi_k25")
        grid = torch.tensor([2, 12, 16])

        self.assertEqual(encoder.get_num_patches(grid, Modality.IMAGE), 384)
        self.assertEqual(encoder.get_num_tokens(grid, Modality.IMAGE), 48)

    def test_grid_metadata_is_safe_to_deserialize(self):
        grid = _get_mm_grid_dim(
            {"image_grid_hws": np.array([[40, 60]], dtype=np.int64)},
            Modality.IMAGE,
            "kimi_vl",
        )
        embedding_data = EmbeddingData(
            req_id="test-request",
            num_parts=1,
            part_idx=0,
            grid_dim=grid,
            modality=Modality.IMAGE,
            embedding=torch.zeros((600, 4)),
        )

        restored = safe_pickle_loads(
            pickle.dumps(embedding_data.copy_without_embedding())
        )

        torch.testing.assert_close(restored.grid_dim, torch.tensor([[40, 60]]))


if __name__ == "__main__":
    unittest.main()
