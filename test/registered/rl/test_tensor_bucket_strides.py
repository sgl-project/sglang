"""Weight buckets preserve logical values of non-contiguous tensor views."""

import unittest

import torch

from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestTensorBucketStrides(unittest.TestCase):
    def test_round_trip_tensor_views(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            base = torch.arange(24, dtype=dtype).reshape(4, 6)
            cases = {
                "contiguous": base,
                "strided_vector": base.flatten()[::2],
                "strided_matrix": base[:, ::2],
                "transposed": base.T,
                "scalar": base[0, 0],
                "empty": base[:0],
            }
            for name, tensor in cases.items():
                with self.subTest(dtype=dtype, layout=name):
                    bucket = FlattenedTensorBucket(named_tensors=[(name, tensor)])
                    [(actual_name, actual)] = bucket.reconstruct_tensors()
                    self.assertEqual(actual_name, name)
                    self.assertEqual(actual.dtype, tensor.dtype)
                    self.assertEqual(actual.shape, tensor.shape)
                    torch.testing.assert_close(actual, tensor, rtol=0, atol=0)
                    torch.testing.assert_close(
                        bucket.get_flattened_tensor(),
                        tensor.flatten().contiguous().view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )


if __name__ == "__main__":
    unittest.main()
