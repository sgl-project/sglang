import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsv4 import dequant_k_cache
from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"
PAGE_SIZE = 4


class TestV41DequantNegativeIds(unittest.TestCase):
    """Sparse attention passes slot lists padded with -1; those must not read before the pool."""

    def _run(self, layout: KVLayout):
        page_bytes = layout.page_bytes(PAGE_SIZE)
        torch.manual_seed(0)
        # Page 0 of the storage sits before the pool and holds 0xFF (NaN scales).
        storage = torch.randint(
            0, 120, (3, page_bytes), dtype=torch.uint8, device=DEVICE
        )
        storage[0] = 0xFF
        pool = storage[1:]
        sentinel = torch.full((2, 1, 512), 3.0, dtype=torch.bfloat16, device=DEVICE)

        def dequant(ids):
            out = sentinel[: len(ids)].clone()
            return dequant_k_cache.dequantize_k_cache_paged_v41(
                pool,
                torch.tensor(ids, dtype=torch.int32, device=DEVICE),
                PAGE_SIZE,
                out=out,
                layout=layout,
            )

        if DEVICE == "cuda":
            if torch.cuda.get_device_capability()[0] < 10:
                self.skipTest("V4.1 KV layouts need SM100")
            padded, alone = dequant([-1, 5]), dequant([5])
        else:
            # The CPU interpreter stands in for an SM100 device.
            with (
                patch.object(
                    dequant_k_cache,
                    "get_jit_cuda_arch",
                    return_value=SimpleNamespace(major=10),
                ),
                patch.object(dequant_k_cache, "is_hip_runtime", return_value=False),
            ):
                padded, alone = dequant([-1, 5]), dequant([5])

        self.assertTrue(torch.equal(padded[0], sentinel[0]))
        self.assertTrue(torch.equal(padded[1], alone[0]))

    def test_fp8(self):
        self._run(KVLayout.V41)

    def test_fp4(self):
        self._run(KVLayout.V41_FP4)


if __name__ == "__main__":
    unittest.main()
