import unittest

import torch

from sglang.kernels.ops.memory.memcpy_triton import memcpy_triton
from sglang.srt.layers.dp_attention import memcpy_cpu
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_HIDDEN = 256
_GUARD_ROWS = 64
_SENTINEL = 7.0


def _guarded_rows(rows: int) -> torch.Tensor:
    """Rows followed by a sentinel-filled guard region that must stay untouched."""
    base = torch.full((rows + _GUARD_ROWS, _HIDDEN), _SENTINEL, device="cuda")
    base[:rows] = torch.randn(rows, _HIDDEN, device="cuda")
    return base


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestMemcpyTriton(CustomTestCase):
    def test_matches_cpu_and_stays_in_bounds(self):
        # (src_rows, dst_rows, offset, count, offset_src); count may exceed either tensor.
        for src_rows, dst_rows, offset, count, offset_src in [
            (5, 20, 3, 5, False),
            (2, 64, 10, 40, False),
            (2, 64, 30, 34, False),
            (20, 5, 3, 5, True),
            (64, 2, 10, 40, True),
            (64, 2, 30, 34, True),
        ]:
            with self.subTest(src_rows=src_rows, dst_rows=dst_rows, count=count):
                src = _guarded_rows(src_rows)
                dst = _guarded_rows(dst_rows)
                ref = dst.clone()
                memcpy_triton(
                    dst=dst[:dst_rows],
                    src=src[:src_rows],
                    dim=0,
                    offset=torch.tensor(offset, device="cuda"),
                    sz=torch.tensor(count, device="cuda"),
                    offset_src=offset_src,
                )
                memcpy_cpu(
                    dst=ref[:dst_rows],
                    src=src[:src_rows],
                    dim=0,
                    offset=offset,
                    sz=count,
                    offset_src=offset_src,
                )
                torch.testing.assert_close(dst, ref, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
