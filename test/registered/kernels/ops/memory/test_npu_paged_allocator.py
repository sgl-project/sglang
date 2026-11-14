"""Regress NPU page mapping and avoid shape-dependent allocator variants."""

import itertools
import unittest

import torch
import torch_npu  # noqa: F401

from sglang.kernels.ops.memory.npu_allocator import alloc_extend_kernel
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=90, suite="base-b-test-1-npu-a3")


class TestNpuPagedAllocator(CustomTestCase):
    def test_mapping_and_specialization(self):
        for page_size in (96, 128):
            for dtype in (torch.int32, torch.int64):
                hashes = set()
                scenarios = [
                    (1, 0),
                    (1, 4097),
                    (1, 8193),
                    (4, 1),
                    (5, 129),
                    (257, 4),
                    (4, 128),
                ]
                for (batch, extension), (shift, last_shift) in itertools.product(
                    scenarios, ((0, 0), (0, 1), (1, 0), (1, 1))
                ):
                    base = (
                        2**24 if extension == 128 else 0 if extension == 4097 else 256
                    )
                    prefix = [
                        base + (0, 1, page_size // 2, page_size - 1)[i % 4]
                        for i in range(batch)
                    ]
                    growth = [extension if i % 3 != 1 else 0 for i in range(batch)]
                    if dtype == torch.int64 and extension == 128:
                        prefix = [p + 2**40 for p in prefix]
                    last = [
                        page_size * (8000 + 8 * i) + (p - 1) % page_size if p else -1
                        for i, p in enumerate(prefix)
                    ]
                    needed = sum(
                        (p + g + page_size - 1) // page_size
                        - (p + page_size - 1) // page_size
                        for p, g in zip(prefix, growth)
                    )
                    pages = [20000 + 3 * i for i in range(needed)]
                    expected, cursor = [], 0
                    for p, g, loc in zip(prefix, growth, last):
                        old = min(g, (-p) % page_size)
                        expected.extend(loc + j + 1 for j in range(old))
                        remaining = g - old
                        expected.extend(
                            pages[cursor + j // page_size] * page_size + j % page_size
                            for j in range(remaining)
                        )
                        cursor += (remaining + page_size - 1) // page_size
                    self.assertEqual(cursor, needed)
                    pre = torch.tensor(prefix, dtype=dtype, device="npu")
                    seq = torch.tensor(
                        [p + g for p, g in zip(prefix, growth)],
                        dtype=dtype,
                        device="npu",
                    )
                    last_base = torch.tensor(
                        [0] * last_shift + last, dtype=torch.int64, device="npu"
                    )
                    free_base = torch.tensor(
                        [0] * shift + pages + [0], dtype=torch.int64, device="npu"
                    )
                    out_base = torch.full(
                        (len(expected) + 64,), -99, dtype=torch.int64, device="npu"
                    )
                    compiled = alloc_extend_kernel[(batch,)](
                        pre,
                        seq,
                        last_base[last_shift:],
                        free_base[shift : shift + needed],
                        out_base[32:-32],
                        page_size,
                    )
                    torch.npu.synchronize()
                    self.assertIsNotNone(compiled)
                    hashes.add(compiled.hash)
                    self.assertEqual(
                        out_base.cpu().tolist(),
                        [-99] * 32 + expected + [-99] * 32,
                        (page_size, dtype, batch, extension, shift, last_shift),
                    )
                self.assertEqual(len(hashes), 1)


if __name__ == "__main__":
    unittest.main()
