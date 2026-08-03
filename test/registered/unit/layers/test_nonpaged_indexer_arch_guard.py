"""Pin the architecture guard on the non-paged DSV4 indexer path.

deep_gemm's indexer kernels assert ``arch_major in {9, 10}`` inside the
kernel, so reaching them on any other architecture kills the scheduler
rather than raising something a fallback could catch. The guard has to
name the architectures that HAVE the kernel and reject everything else;
this test pins the enumeration and the module-level predicate so a
refactor cannot silently flip the polarity or widen the list.
"""

import unittest

import torch

from sglang.srt.layers.attention.dsv4 import indexer as indexer_mod
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestNonPagedIndexerArchGuard(CustomTestCase):
    def test_capability_list_names_only_kernel_bearing_arches(self):
        self.assertEqual(indexer_mod._DEEP_GEMM_INDEXER_CAPABILITIES, (9, 10))

    def test_predicate_matches_current_device(self):
        if torch.cuda.is_available():
            expect = torch.cuda.get_device_capability()[0] in (9, 10)
        else:
            expect = False
        self.assertEqual(indexer_mod._has_deep_gemm_indexer, expect)


if __name__ == "__main__":
    unittest.main()
