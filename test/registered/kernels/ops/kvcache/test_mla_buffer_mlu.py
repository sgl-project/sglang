"""MLU MLA gather correctness, row boundaries and strided output guards."""

import unittest

import torch
import torch_mlu  # noqa: F401

from sglang.kernels.ops.kvcache.mla_buffer_mlu import get_mla_kv_buffer_mlu
from sglang.test.ci.ci_register import register_mlu_ci
from sglang.test.test_utils import CustomTestCase

register_mlu_ci(est_time=60, suite="pr-test-1-mlu")


class TestMLABufferMLU(CustomTestCase):
    @staticmethod
    def _allocate(rows, dim, dtype, padded):
        offset = 7 if padded else 0
        stride = dim + (16 if padded else 0)
        storage = torch.full(
            (offset + rows * stride + 16,),
            float("nan"),
            dtype=dtype,
            device="mlu",
        )
        view = storage.as_strided((rows, 1, dim), (stride, dim, 1), offset)
        return view, storage

    def _check_gather(self, num_tokens, rope_dim, dst_dtype, padded, index_dtype):
        torch.manual_seed(2026)
        num_slots = 257
        source_cpu = torch.randn(num_slots, 1, 512 + rope_dim).to(torch.bfloat16)
        source, _ = self._allocate(num_slots, 512 + rope_dim, torch.bfloat16, padded)
        source.copy_(source_cpu)
        indices = torch.randint(num_slots, (num_tokens,), dtype=index_dtype)
        if num_tokens:
            indices[0] = num_slots - 1
        if num_tokens >= 4:
            # Explicitly cover both endpoints and repeated, unsorted indices.
            indices[:4] = torch.tensor([num_slots - 1, 0, 17, 17], dtype=index_dtype)
        expected = source_cpu[indices.long()].to(dst_dtype)
        nope, nope_storage = self._allocate(num_tokens, 512, dst_dtype, padded)
        if rope_dim:
            rope, rope_storage = self._allocate(num_tokens, rope_dim, dst_dtype, padded)
        else:
            rope, rope_storage = None, None

        # Check the first invocation as well, including any autotune execution.
        get_mla_kv_buffer_mlu(source, indices.to("mlu"), nope, rope)
        for output, storage, reference in (
            (nope, nope_storage, expected[..., :512]),
            (rope, rope_storage, expected[..., 512:]),
        ):
            if output is None:
                continue
            expected_storage = torch.full(
                (storage.numel(),), float("nan"), dtype=dst_dtype
            )
            expected_storage.as_strided(
                output.shape, output.stride(), output.storage_offset()
            ).copy_(reference)
            # Also verifies the untouched leading, trailing and inter-row guards.
            torch.testing.assert_close(
                storage.cpu(), expected_storage, rtol=0, atol=0, equal_nan=True
            )
        torch.testing.assert_close(source.cpu(), source_cpu, rtol=0, atol=0)

    def test_row_boundaries(self):
        # Includes partial tiles and the old one-program-per-token grid limit.
        for num_tokens in (0, 1, 15, 16, 17, 63, 64, 65, 65535, 65536, 65537):
            with self.subTest(num_tokens=num_tokens):
                self._check_gather(num_tokens, 64, torch.bfloat16, False, torch.int32)

    def test_strides_dtypes_and_optional_rope(self):
        for rope_dim in (0, 64):
            for dst_dtype in (torch.bfloat16, torch.float32):
                for padded in (False, True):
                    with self.subTest(
                        rope_dim=rope_dim, dst_dtype=dst_dtype, padded=padded
                    ):
                        self._check_gather(
                            129, rope_dim, dst_dtype, padded, torch.int64
                        )


if __name__ == "__main__":
    unittest.main()
