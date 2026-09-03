import unittest

import torch

from sglang.srt.mem_cache.shared_kv_ipc import CudaIpcTensorDescriptor


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestCudaIpcTensorDescriptor(unittest.TestCase):
    def test_rejects_non_cuda_and_non_contiguous_tensors(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            CudaIpcTensorDescriptor.export(torch.zeros(2))
        with self.assertRaisesRegex(ValueError, "contiguous"):
            CudaIpcTensorDescriptor.export(torch.zeros(2, 2, device="cuda").t())


if __name__ == "__main__":
    unittest.main()
