import itertools
import unittest

import torch

from sglang.kernels.ops.memory.allocator import (
    alloc_decode_kernel,
    alloc_extend_kernel,
    batch_scan_width,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=15, stage="jit-kernel-unit", runner_config="amd")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA or HIP")
class TestPagedAllocator(CustomTestCase):
    def test_masked_batch_scan_matches_reference(self):
        # Include batches below, at, and above the shared scan-width boundary.
        for batch_size, page_size, dtype, offset in itertools.product(
            (1, 3, 16, 63, 64, 65),
            (1, 16),
            (torch.int32, torch.int64),
            (0, 1),
        ):
            with self.subTest(
                batch_size=batch_size, page_size=page_size, dtype=dtype, offset=offset
            ):
                prefixes = [i % (2 * page_size + 1) for i in range(batch_size)]
                extensions = [1 + i % (2 * page_size + 3) for i in range(batch_size)]
                self._check_allocation(prefixes, extensions, page_size, dtype, offset)
                self._check_allocation(
                    prefixes, [1] * batch_size, page_size, dtype, offset, decode=True
                )

    def _check_allocation(
        self, prefixes, extensions, page_size, dtype, offset, decode=False
    ):
        seq_lens = [prefix + ext for prefix, ext in zip(prefixes, extensions)]
        last_locs = [
            (2000 + i * 4) * page_size + (prefix - 1) % page_size if prefix else -1
            for i, prefix in enumerate(prefixes)
        ]
        needed_pages = sum(
            (seq + page_size - 1) // page_size - (prefix + page_size - 1) // page_size
            for prefix, seq in zip(prefixes, seq_lens)
        )
        free_page_storage = (
            torch.arange(needed_pages + offset + 8, device="cuda", dtype=torch.int64)
            * 3
            + 100
        )
        free_pages = free_page_storage[offset:]
        expected = []
        page_cursor = 0
        # Scalar oracle: first consume an existing partial page, then allocate
        # distinct, deliberately non-contiguous pages from the free-page queue.
        for prefix, extension, last_loc in zip(prefixes, extensions, last_locs):
            old_page_tokens = min(extension, (-prefix) % page_size)
            expected.extend(last_loc + i + 1 for i in range(old_page_tokens))
            remaining = extension - old_page_tokens
            for token in range(remaining):
                page = 100 + 3 * (offset + page_cursor + token // page_size)
                expected.append(page * page_size + token % page_size)
            page_cursor += (remaining + page_size - 1) // page_size

        pre = torch.tensor(prefixes, dtype=dtype, device="cuda")
        seq = torch.tensor(seq_lens, dtype=dtype, device="cuda")
        last = torch.tensor(last_locs, dtype=torch.int64, device="cuda")
        # Sentinels also detect writes outside the requested output range.
        output = torch.full(
            (len(expected) + 2,), -777, dtype=torch.int64, device="cuda"
        )
        batch_size = len(prefixes)
        if decode:
            alloc_decode_kernel[(batch_size,)](
                seq,
                last,
                free_pages,
                output[1:-1],
                batch_scan_width(batch_size),
                page_size,
            )
        else:
            alloc_extend_kernel[(batch_size,)](
                pre,
                seq,
                last,
                free_pages,
                output[1:-1],
                batch_scan_width(batch_size),
                page_size,
            )
        self.assertEqual(output.cpu().tolist(), [-777, *expected, -777])
        self.assertEqual(page_cursor, needed_pages)


if __name__ == "__main__":
    unittest.main()
