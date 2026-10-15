import unittest

import torch

from sglang.srt.hardware_backend.npu.moe.tp_fusion import shared_gmm1_weight_views
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")


class TestSharedGMM1WeightViews(unittest.TestCase):
    def test_single_expert_transpose_preserves_cann_batch_strides(self):
        h, n = 6144, 1024
        for packed in (False, True):
            with self.subTest(packed=packed):
                k = h // 2 if packed else h
                original_weight = torch.randint(0, 256, (n, k), dtype=torch.uint8)
                original_scale = torch.randint(
                    0, 256, (n, h // 64, 2), dtype=torch.uint8
                )
                weight = original_weight.transpose(0, 1)
                scale = original_scale.transpose(0, 1)
                grouped_weight, grouped_scale = shared_gmm1_weight_views(weight, scale)

                self.assertEqual(grouped_weight.shape, (1, k, n))
                self.assertEqual(grouped_scale.shape, (1, h // 64, n, 2))
                self.assertEqual(grouped_weight.stride(), (n * k, 1, k))
                self.assertEqual(grouped_scale.stride(), (n * h // 32, 2, h // 32, 1))
                self.assertNotEqual(weight.unsqueeze(0).stride()[0], n * k)
                self.assertNotEqual(scale.unsqueeze(0).stride()[0], n * h // 32)
                self.assertEqual(grouped_weight.data_ptr(), weight.data_ptr())
                self.assertEqual(grouped_scale.data_ptr(), scale.data_ptr())
                torch.testing.assert_close(grouped_weight[0], weight, atol=0, rtol=0)
                torch.testing.assert_close(grouped_scale[0], scale, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
