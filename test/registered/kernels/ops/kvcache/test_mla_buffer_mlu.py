"""MLU MLA gather correctness, row boundaries and strided output guards."""

import unittest

import torch

from sglang.kernels.ops.kvcache.mla_buffer_mlu import get_mla_kv_buffer_mlu
from sglang.test.ci.ci_register import register_mlu_ci
from sglang.test.test_utils import CustomTestCase

register_mlu_ci(est_time=60, suite="pr-test-1-mlu")


class TestMLABufferMLU(CustomTestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)

    @staticmethod
    def _allocate_guarded_tensor(rows, dim, dtype, padded, device="mlu"):
        """Return the data tensor and backing storage containing NaN guards."""
        padding = 8 if padded else 0
        # Storage layout (all guards remain NaN):
        #              padding      dim       padding
        #            +---------+-------------+---------+
        # first row  |          guard row              |
        #            +---------+-------------+---------+
        #            | guard   | data row 0  | guard   |
        # rows       | columns |     ...     | columns |
        #            |         | data row n  |         |
        #            +---------+-------------+---------+
        # last row   |          guard row              |
        #            +---------+-------------+---------+
        # Here n = rows - 1. With padded=False, the guard columns disappear;
        # the first and last guard rows remain. Widths are in elements.
        storage = torch.full(
            (rows + 2, dim + 2 * padding),
            float("nan"),
            dtype=dtype,
            device=device,
        )
        data = storage[1:-1, padding : padding + dim].unsqueeze(1)
        return data, storage

    def _check_gather(self, num_tokens, rope_dim, dst_dtype, padded, index_dtype):
        num_slots = 257
        source_cpu = torch.randn(num_slots, 1, 512 + rope_dim).to(torch.bfloat16)
        source, _ = self._allocate_guarded_tensor(
            num_slots, 512 + rope_dim, torch.bfloat16, padded
        )
        source.copy_(source_cpu)
        indices = torch.randint(num_slots, (num_tokens,), dtype=index_dtype)
        # Use 0 and num_slots - 1 to cover both boundaries, and two identical
        # middle-slot indices to cover duplicate reads and out-of-order access.
        middle_slot = num_slots // 2
        fixed_indices = torch.tensor([0, num_slots - 1, middle_slot, middle_slot])
        num_fixed_indices = min(num_tokens, len(fixed_indices))
        indices[:num_fixed_indices] = fixed_indices[:num_fixed_indices]
        expected = source_cpu[indices.long()].to(dst_dtype)
        nope, nope_storage = self._allocate_guarded_tensor(
            num_tokens, 512, dst_dtype, padded
        )
        if rope_dim:
            rope, rope_storage = self._allocate_guarded_tensor(
                num_tokens, rope_dim, dst_dtype, padded
            )
        else:
            rope, rope_storage = None, None

        get_mla_kv_buffer_mlu(source, indices.to("mlu"), nope, rope)
        for output, storage, reference in (
            (nope, nope_storage, expected[..., :512]),
            (rope, rope_storage, expected[..., 512:]),
        ):
            if output is None:
                continue
            expected_data, expected_storage = self._allocate_guarded_tensor(
                num_tokens, reference.shape[-1], dst_dtype, padded, device="cpu"
            )
            expected_data.copy_(reference)
            # Compare the data and all surrounding guard rows/columns.
            torch.testing.assert_close(
                storage.cpu(), expected_storage, rtol=0, atol=0, equal_nan=True
            )
        # Gathering must leave the source cache data unchanged.
        torch.testing.assert_close(source.cpu(), source_cpu, rtol=0, atol=0)

    def test_gather(self):
        """Gather preserves order and guards across empty, large and strided inputs."""
        cases = (
            (0, 64, torch.bfloat16, False, torch.int32),  # Empty input.
            (65537, 64, torch.bfloat16, False, torch.int32),  # Old grid limit.
            (129, 64, torch.float32, False, torch.int64),  # FP32 output.
            (129, 64, torch.bfloat16, True, torch.int64),  # Padded rows.
            (129, 0, torch.float32, True, torch.int64),  # Optional rope.
        )
        for case in cases:
            with self.subTest(case=case):
                self._check_gather(*case)


if __name__ == "__main__":
    unittest.main()
