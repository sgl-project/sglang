"""Level Zero specifics of XpuVmmBackend that the generic test_vmm_backend.py cannot see."""

import unittest

import torch

from sglang.srt.utils.vmm_backend import get_vmm_backend
from sglang.srt.utils.xpu_vmm_utils import XpuVmmBackend, query_page_size
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=10, suite="stage-b-test-1-gpu-xpu")


@unittest.skipUnless(torch.xpu.is_available(), "requires Intel XPU")
class TestXpuVmmUtils(CustomTestCase):
    def test_granularity_covers_every_allocation_size(self) -> None:
        """Level Zero reports a page size per allocation size rather than per
        device, and rejects a physical object whose size is not a multiple of the
        page it reports for that size. So the granularity DWDP aligns everything to
        has to be the coarse page, not whatever the smallest query returns."""
        granularity = get_vmm_backend(0).granularity
        for multiplier in (1, 2, 3, 33, 512):
            size = multiplier * granularity
            self.assertEqual(
                size % query_page_size(0, size),
                0,
                f"{size} B needs a {query_page_size(0, size)} B page but the "
                f"granularity is {granularity} B",
            )

    def test_platform_names_the_level_zero_backend(self) -> None:
        backend = get_vmm_backend(0)
        self.assertIsInstance(backend, XpuVmmBackend)
        self.assertFalse(backend.supports_aliased_mappings())
        self.assertFalse(backend.owns_exported_fds())


if __name__ == "__main__":
    unittest.main()
