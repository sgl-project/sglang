import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.kernels.ops.kvcache import pd_dcp_gather as gather
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDcpPackMetadata(unittest.TestCase):
    def test_reuses_layout_when_row_count_changes_and_replaces_changed_layout(self):
        cache = gather.MlaPackMetadataCache()
        kernel = MagicMock()
        with patch.object(gather, "_copy_mla_rows_into_pack_kernel", kernel):
            for rows in (1, 777, 8192):
                gather.copy_mla_rows_into_pack(
                    [4096, 8192],
                    torch.arange(rows),
                    torch.empty(rows * 12, dtype=torch.uint8),
                    [4, 8],
                    [16, 32],
                    metadata_cache=cache,
                )
        calls = kernel.__getitem__.return_value.call_args_list
        self.assertEqual(len(calls), 3)
        for call, rows in zip(calls, (1, 777, 8192)):
            self.assertIs(call.args[0], cache.tensor)
            self.assertEqual(call.args[3], rows)
        self.assertEqual(cache.tensor.tolist(), [4096, 4, 0, 16, 8192, 8, 4, 32])
        original = cache.tensor
        for pointers, widths, strides in (
            ([4097, 8192], [4, 8], [16, 32]),
            ([4097, 8192], [8, 8], [16, 32]),
            ([4097, 8192], [8, 8], [32, 32]),
        ):
            replacement = cache.get("cpu", pointers, widths, strides)
            self.assertIsNot(replacement, original)
            original = replacement

    def test_empty_rows_skip_kernel_and_invalid_widths_fail(self):
        with patch.object(gather, "_copy_mla_rows_into_pack_kernel") as kernel:
            gather.copy_mla_rows_into_pack(
                [4096],
                torch.empty(0, dtype=torch.int64),
                torch.empty(0, dtype=torch.uint8),
                [4],
            )
            kernel.__getitem__.assert_not_called()
            with self.assertRaisesRegex(ValueError, "positive"):
                gather.copy_mla_rows_into_pack(
                    [4096], torch.arange(1), torch.empty(4, dtype=torch.uint8), [0]
                )
            with self.assertRaisesRegex(ValueError, "length mismatch"):
                gather.copy_mla_rows_into_pack(
                    [4096], torch.arange(1), torch.empty(4, dtype=torch.uint8), [4, 8]
                )


if __name__ == "__main__":
    unittest.main()
