import unittest

import torch

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

IM_TOKEN_ID = 7


class TestMultimodalInputsMergeMropeDelta(CustomTestCase):
    def test_merge_invalidates_mrope_delta_cache(self):
        def _mm(delta):
            item = MultimodalDataItem(
                modality=Modality.IMAGE, hash=7, pad_value=7, offsets=[(0, 0)]
            )
            return MultimodalInputs(
                mm_items=[item],
                im_token_id=IM_TOKEN_ID,
                mrope_position_delta=torch.tensor([[delta]]),
            )

        base = _mm(3)
        base.mrope_position_delta_repeated_cache = torch.zeros(3, 1, dtype=torch.long)
        base.merge(_mm(5))
        self.assertIsNone(base.mrope_position_delta_repeated_cache)
        self.assertEqual(base.mrope_position_delta.flatten().tolist(), [3, 5])

        no_delta = _mm(1)
        no_delta.mrope_position_delta = None
        no_delta.merge(_mm(9))
        self.assertEqual(no_delta.mrope_position_delta.flatten().tolist(), [9])


if __name__ == "__main__":
    unittest.main()
