"""
Basic PPU test: verifies the engine initializes on T-HEAD PPU and
completes at least one decode step. It runs run_bench_one_batch
(``python3 -m sglang.bench_one_batch``), the OFFLINE engine path:
no HTTP server is launched, so launch_server / HTTP routing are not
covered (test_anthropic_server.py covers that surface in stage-a).

Registered in suite stage-a-test-1-gpu-ppu so it gates stage-b before
the heavier tests run.

Usage:
python3 -m unittest test_ppu_basic.TestPPUBasic.test_basic_generation
"""

import unittest

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.ppu_pr_utils import small_model
from sglang.test.test_utils import (
    CustomTestCase,
    is_in_ci,
    run_bench_one_batch,
)

register_ppu_ci(est_time=120, suite="stage-a-test-1-gpu-ppu")


class TestPPUBasic(CustomTestCase):

    def test_basic_generation(self):
        """Engine initializes on PPU and completes at least one decode step."""
        args = [
            "--disable-radix-cache",
            "--mem-fraction-static",
            "0.6",
            "--batch-size",
            "1",
            # The offline bench_one_batch entrypoint does not call
            # load_plugins(), so the acext PPU-native FA3 override (which
            # accepts only_qv) is never installed. The default fa3 path then
            # resolves flash_attn_with_kvcache from the base image's sgl_kernel
            # binary, which predates the only_qv parameter and raises TypeError
            # during prefill CUDA-graph capture. Pin the self-contained triton
            # backend so this offline smoke exercises engine init + one decode
            # step without the plugin-provided fa3 kernel. Server-path fa3
            # coverage on the same model lives in test_ppu_fa3_eval (stage-b).
            "--attention-backend",
            "triton",
        ]
        if is_in_ci():
            args += ["--input", "64", "--output", "4"]

        _, decode_throughput, _ = run_bench_one_batch(small_model(), args)
        self.assertGreater(decode_throughput, 0, "PPU decode throughput must be > 0")


if __name__ == "__main__":
    unittest.main()
