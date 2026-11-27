import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.hardware_backend.npu.dsv4 import dsv41_int8_ops
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestInt8CsaOps(unittest.TestCase):
    def test_writer_masks_reserved_page_and_preserves_physical_view(self):
        backing = torch.zeros((3, 4 * 608 + 32), dtype=torch.int8)
        cache = backing[:, :-32].view(3, 4, 1, 608)
        slots = torch.tensor([-1, 0, 3, 4, 11, 12], dtype=torch.int32)
        latent = torch.zeros((6, 1, 512), dtype=torch.bfloat16)
        writer = Mock()
        with patch.object(
            dsv41_int8_ops, "int8_csa_ops", return_value=(writer, None, None)
        ):
            dsv41_int8_ops.write_int8_csa_rows(cache, slots, latent)

        dest, rows, valid_slots = writer.call_args.args
        self.assertIs(dest, cache)
        self.assertEqual(dest.data_ptr(), cache.data_ptr())
        self.assertEqual(dest.stride(), cache.stride())
        self.assertEqual(rows.shape, (6, 512))
        self.assertEqual(valid_slots.tolist(), [-1, -1, -1, 4, 11, -1])

    def test_empty_writer_does_not_load_the_native_ops(self):
        with patch.object(dsv41_int8_ops, "int8_csa_ops") as ops:
            dsv41_int8_ops.write_int8_csa_rows(
                None, torch.empty(0, dtype=torch.int32), torch.empty((0, 512))
            )
        ops.assert_not_called()

    def test_attention_and_metadata_delegate_to_kernel_package(self):
        metadata, attention = Mock(), Mock()
        q = torch.empty((1, 1, 512), dtype=torch.bfloat16)
        with patch.object(
            dsv41_int8_ops, "int8_csa_ops", return_value=(None, metadata, attention)
        ):
            dsv41_int8_ops.int8_csa_metadata(1, 1, 512, cmp_ratio=2)
            dsv41_int8_ops.int8_csa_attention(q, cmp_ratio=2)
        metadata.assert_called_once_with(1, 1, 512, cmp_ratio=2)
        attention.assert_called_once_with(q, cmp_ratio=2)
