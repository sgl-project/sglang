import unittest

import torch

from sglang.kernels.ops.attention.kda_fused_decode import covered
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestKDAFusedDecodeContract(CustomTestCase):
    def _inputs(self):
        batch_size = 2
        num_slots = 4
        num_heads = 12
        head_dim = 128
        qkv_width = 3 * num_heads * head_dim
        dense_slot_stride = num_heads * head_dim * head_dim
        slot_stride = dense_slot_stride + 256
        storage_size = (num_slots - 1) * slot_stride + dense_slot_stride

        ssm_storage = torch.empty(storage_size, dtype=torch.float32)
        ssm_states = torch.as_strided(
            ssm_storage,
            (num_slots, num_heads, head_dim, head_dim),
            (slot_stride, head_dim * head_dim, head_dim, 1),
        )

        return (
            torch.empty(batch_size, qkv_width, dtype=torch.bfloat16),
            torch.empty(batch_size, num_heads * head_dim, dtype=torch.bfloat16),
            torch.empty(batch_size, num_heads, dtype=torch.bfloat16),
            torch.empty(num_slots, 3, qkv_width, dtype=torch.bfloat16),
            ssm_states,
            torch.arange(batch_size, dtype=torch.int32),
            torch.empty(batch_size, num_heads * head_dim, dtype=torch.bfloat16),
        )

    def test_accepts_inner_contiguous_state_with_padded_slot_pitch(self):
        self.assertTrue(covered(*self._inputs()))

    def test_rejects_noncontiguous_inner_state(self):
        inputs = list(self._inputs())
        inputs[4] = inputs[4].transpose(-1, -2)
        self.assertFalse(covered(*inputs))


if __name__ == "__main__":
    unittest.main()
