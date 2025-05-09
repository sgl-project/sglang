import unittest

import torch
from python.sglang.srt.memory_saver_tensors import DisposableTensor


class TestPPAccuracy(unittest.TestCase):
    def test_disposable_tensor(self):
        self._common_test(DisposableTensor(torch.tensor([3.0, 4.0, 5.0])))

    def test_lazy_tensor(self):
        self._common_test(TODO)

    def _common_test(self, x: torch.Tensor):
        self.assertEqual(torch.max(x).item(), 5.0)
        self.assertTrue(torch.allclose(x + torch.tensor([2.0, 2.0, 2.0]), torch.tensor([5.0, 6.0, 7.0])))
        self.assertTrue(torch.empty_like(x).shape, (3,))
        self.assertTrue(torch.allclose(torch.full_like(x, 42), torch.tensor([42.0, 42.0, 42.0])))


if __name__ == "__main__":
    unittest.main()
