import unittest

import torch

from sglang.srt.utils import DynamicGradMode


class TestDynamicGradMode(unittest.TestCase):
    def test_grad_mode(self):

        DynamicGradMode.set_inference_mode(True)

        @DynamicGradMode()
        def create_tensor_x():
            return torch.empty(0)

        X = create_tensor_x()
        self.assertTrue(not X.requires_grad and X.is_inference())

        DynamicGradMode.set_inference_mode(False)

        @DynamicGradMode()
        def create_tensor_y():
            return torch.empty(0)

        Y = create_tensor_y()
        self.assertTrue(not Y.requires_grad and not Y.is_inference())


if __name__ == "__main__":
    unittest.main(verbosity=2)
