"""Free-list prefix views must not compile new bind kernels after odd consumes."""

import unittest

import torch

from sglang.kernels.ops.memory.virtual_slot import (
    alloc_bind_inplace,
    alloc_bind_inplace_kernel,
    bind_inplace,
    bind_inplace_kernel,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestVirtualSlotBinding(CustomTestCase):
    def test_prefix_alignment_does_not_specialize_bind_kernels(self):
        """Taking one int64 page from a free list changes its pointer's 16-byte
        alignment. Copy-elided prefill must reuse the warmed bind kernel."""
        virtual = torch.arange(1, 35, device="cuda", dtype=torch.int64)
        physical = torch.arange(64, 98, device="cuda", dtype=torch.int64)
        v2p = torch.empty(128, device="cuda", dtype=torch.int64)
        p2v = torch.empty_like(v2p)
        for kernel, allocate in (
            (alloc_bind_inplace_kernel, True),
            (bind_inplace_kernel, False),
        ):
            with self.subTest(kernel=kernel.__name__):
                cache = kernel.device_caches[torch.cuda.current_device()][0]
                saved = dict(cache)
                cache.clear()
                try:
                    for voff, poff in ((0, 0), (1, 0), (0, 1), (1, 1)):
                        v2p.fill_(-1)
                        p2v.fill_(-1)
                        v = virtual[voff : voff + 32]
                        p = physical[poff : poff + 32]
                        if allocate:
                            p = alloc_bind_inplace(v, v2p, p2v, start_phys=64)
                        else:
                            bind_inplace(v, p, v2p, p2v)
                        torch.testing.assert_close(v2p[v], p)
                        torch.testing.assert_close(p2v[p], v)
                        self.assertEqual(len(cache), 1, (voff, poff))
                finally:
                    cache.update(saved)


if __name__ == "__main__":
    unittest.main()
