"""Bit-exact unified MXFP8 HiCache coverage on Blackwell GPUs."""

import unittest

import test_unified_radix_cache_kl_hybrid_bitexact as hybrid_kl

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, stage="extra-b", runner_config="4-gpu-b200")


class TestUnifiedMemoryMXFP8HiCacheBitExact(hybrid_kl.TestUnifiedHybridHiCacheBitExact):
    """MXFP8 scales require the kernel backend and page-first host layout."""

    io_backend = "kernel"
    mem_layout = "page_first"
    extra_args = ["--enable-unified-memory", "--kv-cache-dtype", "mxfp8"]


if __name__ == "__main__":
    unittest.main()
