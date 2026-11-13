from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


class TestWeightlessRMSNorm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_weightless_norm_registers_its_ones_as_a_buffer(self):
        # release_memory_occupation restores the model's buffers after resume; a
        # plain tensor attribute in the weights region would come back as zeros.
        norm = RMSNorm(16, has_weight=False)
        self.assertIn("weight", dict(norm.named_buffers()))
        self.assertNotIn("weight", dict(norm.named_parameters()))
        self.assertNotIn("weight", norm.state_dict())
        torch.testing.assert_close(norm.weight, torch.ones(16))

    def test_weightless_norm_follows_module_dtype_and_device_moves(self):
        norm = RMSNorm(16, has_weight=False).to(torch.bfloat16)
        self.assertEqual(norm.weight.dtype, torch.bfloat16)

    def test_weighted_norm_keeps_a_parameter(self):
        norm = RMSNorm(16)
        self.assertIn("weight", dict(norm.named_parameters()))
        self.assertNotIn("weight", dict(norm.named_buffers()))


if __name__ == "__main__":
    unittest.main()
