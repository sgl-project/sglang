"""`test_unified_chain_realize.TestEndsPackOnTheDevice` over many more seeded
histories: device packs and reliefs against the host route."""

import unittest

import test_unified_chain_realize as realize

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="nightly", runner_config="1-gpu-small")


class TestEndsPackOnTheDeviceSweep(realize.TestEndsPackOnTheDevice):
    CPU_PACKS, CUDA_PACKS, CPU_RELIEFS, CUDA_RELIEFS = 200, 60, 400, 100


if __name__ == "__main__":
    unittest.main()
