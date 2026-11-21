"""Host allocation failure must not kill an otherwise recoverable retraction."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.memory_pool_npu import (
    NPUMHATokenToKVPool,
    NPUMLATokenToKVPool,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def copy_chunks(pool, buffers, indices):
    if isinstance(pool, NPUMLATokenToKVPool):
        # Preserve MLA's rank-local/global-index selection in the production path.
        pool.dcp_size = 1
        return pool._chunk_copy_npu_to_cpu(
            buffers, indices, [[False] * len(layer) for layer in buffers]
        )
    return pool._chunk_copy_npu_to_cpu(buffers, indices)


class TestNpuCpuRetractionHostOom(unittest.TestCase):
    def _copy(self, pool_type, failure):
        pool = object.__new__(pool_type)
        pool.cpu_offloading_chunk_size = 3
        data = torch.arange(64, dtype=torch.int32).reshape(16, 1, 4)
        empty = torch.empty((16, 1, 0), dtype=torch.bfloat16)
        indices = torch.tensor([1, 5, 9, 12, 14])
        calls = []
        real_to = torch.Tensor.to

        def transfer(tensor, *args, **kwargs):
            if args == ("cpu",):
                calls.append(kwargs["non_blocking"])
                # One successful chunk precedes the failing host allocation.
                if len(calls) == 3:
                    raise failure
            return real_to(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "to", transfer):
            result = copy_chunks(
                pool, [[data, empty, None], [data + 100, empty, None]], indices
            )
        for layer, base in zip(result, (data, data + 100), strict=True):
            torch.testing.assert_close(torch.cat([x[0] for x in layer]), base[indices])
            self.assertTrue(all(x[1].shape[-1] == 0 and len(x) == 2 for x in layer))
        return calls

    def test_host_oom_retries_and_keeps_remaining_chunks_pageable(self):
        for pool_type in (NPUMHATokenToKVPool, NPUMLATokenToKVPool):
            with self.subTest(pool_type=pool_type.__name__):
                calls = self._copy(pool_type, torch.OutOfMemoryError("host allocation"))
                self.assertEqual(calls[:3], [True, True, True])
                self.assertTrue(all(x is False for x in calls[3:]))

    def test_non_memory_errors_propagate(self):
        with self.assertRaisesRegex(RuntimeError, "transfer failed"):
            self._copy(NPUMLATokenToKVPool, RuntimeError("transfer failed"))

    def test_device_indexing_oom_does_not_retry(self):
        pool = object.__new__(NPUMLATokenToKVPool)
        pool.cpu_offloading_chunk_size = 3
        with patch.object(
            torch.Tensor,
            "__getitem__",
            side_effect=torch.OutOfMemoryError("device allocation"),
        ):
            with self.assertRaisesRegex(torch.OutOfMemoryError, "device allocation"):
                copy_chunks(pool, [[torch.empty((8, 4))]], torch.tensor([1]))

    def test_pageable_oom_propagates(self):
        pool = object.__new__(NPUMLATokenToKVPool)
        pool.cpu_offloading_chunk_size = 3
        with patch.object(
            torch.Tensor, "to", side_effect=torch.OutOfMemoryError("host exhausted")
        ) as copy:
            with self.assertRaisesRegex(torch.OutOfMemoryError, "host exhausted"):
                copy_chunks(pool, [[torch.empty((8, 4))]], torch.tensor([1]))
        self.assertEqual(copy.call_count, 2)
        self.assertEqual(copy.call_args_list[1].kwargs, {"non_blocking": False})


if __name__ == "__main__":
    unittest.main()
