import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers import mm_utils, schedule_batch
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)


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

        with patch.object(
            schedule_batch.torch.cuda, "is_available", return_value=True
        ), patch.object(
            schedule_batch.torch.cuda, "current_device", return_value=0
        ), patch.object(
            schedule_batch.envs.SGLANG_ENABLE_MM_SPLITTING, "get", return_value=False
        ), patch.object(
            schedule_batch.envs.SGLANG_MM_BUFFER_SIZE_MB, "get", return_value=0
        ):
            mm_inputs = MultimodalInputs.from_dict({"mm_items": [mm_item]})

        self.assertEqual(len(mm_inputs.mm_items), 1)
        self.assertTrue(torch.equal(mm_inputs.mm_items[0].feature, feature_tensor))
        proxy_feature.reconstruct_on_target_device.assert_called_once_with(0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
