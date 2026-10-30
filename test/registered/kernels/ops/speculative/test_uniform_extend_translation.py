"""Fused write gathering must reuse kernels across pool sizes and sink bad IDs."""

import unittest

import torch

from sglang.kernels.ops.speculative.cache_locs import (
    assign_extend_cache_locs_uniform,
    assign_extend_cache_locs_uniform_func,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestUniformExtendTranslation(CustomTestCase):
    def test_pool_sizes_are_runtime_bounds(self):
        """Different virtual table lengths must share the warmed kernel.
        Negative, tombstoned and past-table slots map to the sink independently
        in FULL and SWA, including when their virtual tables differ in size.
        """
        ps = 128
        tokens = [
            -129,
            -1,
            0,
            3 * ps + 7,
            62 * ps + 1,
            126 * ps + 2,
            254 * ps + 3,
            511 * ps,
        ]
        req_to_token = torch.tensor([tokens], device="cuda", dtype=torch.int64)
        rows = torch.zeros(1, device="cuda", dtype=torch.int64)
        starts = torch.zeros_like(rows)
        full_out = torch.empty(8, device="cuda", dtype=torch.int64)
        swa_out = torch.empty_like(full_out)
        cache = assign_extend_cache_locs_uniform.device_caches[
            torch.cuda.current_device()
        ][0]
        saved = dict(cache)
        cache.clear()
        try:
            for full_size, swa_size in ((127, 63), (255, 127), (511, 255), (127, 511)):
                full = torch.arange(full_size, device="cuda", dtype=torch.int64)
                swa = torch.arange(swa_size, device="cuda", dtype=torch.int64) + 1
                full[0] = swa[0] = 0
                full[3] = swa[3] = -1
                got_virtual = assign_extend_cache_locs_uniform_func(
                    rows,
                    req_to_token,
                    starts,
                    1,
                    8,
                    "cuda",
                    full_v2p=full,
                    swa_v2p=swa,
                    full_out=full_out,
                    swa_out=swa_out,
                    page_size=ps,
                )
                torch.testing.assert_close(got_virtual, req_to_token.flatten())
                for mapping, output in ((full, full_out), (swa, swa_out)):
                    table = mapping.cpu().tolist()
                    expected = []
                    for token in tokens:
                        page, offset = divmod(token, ps)
                        physical = table[page] if 0 <= page < len(table) else -1
                        expected.append(max(physical * ps + offset, 0))
                    torch.testing.assert_close(output, output.new_tensor(expected))
                self.assertEqual(len(cache), 1, (full_size, swa_size))
        finally:
            cache.update(saved)


if __name__ == "__main__":
    unittest.main()
