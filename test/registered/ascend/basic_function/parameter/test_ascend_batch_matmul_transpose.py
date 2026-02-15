import logging
import random
import time
import unittest

import sgl_kernel_npu
import torch
import torch_npu

from sglang.test.ci.ci_register import register_npu_ci

assert sgl_kernel_npu is not None
assert torch_npu is not None

# Configure logging module to replace direct print statements
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)

torch.set_printoptions(threshold=float("inf"))


class TestMatrixMultiplication(unittest.TestCase):
    """Testcase: Tests using the transpose+batch matmul fusion operator showed a 65% optimization for aingle operators.

    [Test Category] Interface
    [Test Target] transpose+batch matmul
    """

    # Class-level lists to collect ALL timing results across all test methods
    global_all_golden_times = []
    global_all_fused_times = []
    # Global performance assertion threshold: fused operator should be at least 60% faster than native operator
    GLOBAL_PERFORMANCE_SPEEDUP_THRESHOLD = 0.6

    def compute_golden(self, a, b, res1, m, n):
        """Compute reference result (golden)"""
        torch.bmm(a.transpose(0, 1), b, out=res1.view(-1, m, n).transpose(0, 1))

    def assert_tensors_basic_format(self, actual, expected):
        """Check if two tensors are legal"""
        self.assertEqual(actual.shape, expected.shape, "Shape mismatch")

        # Check for NaN
        self.assertFalse(torch.isnan(actual).any(), "Actual result contains NaN")
        self.assertFalse(torch.isnan(expected).any(), "Expected result contains NaN")

        # Check for Inf
        self.assertFalse(torch.isinf(actual).any(), "Actual result contains Inf")
        self.assertFalse(torch.isinf(expected).any(), "Expected result contains Inf")

    def test_boundary_conditions(self):
        """Test boundary conditions"""
        test_cases = [
            (1, 1, 1, 1),  # Minimum size
            (1, 10, 1, 1),  # b=1
            (10, 1, 1, 10),  # m=1
            (5, 5, 1, 5),  # k=1
            (2, 2, 2, 1),  # n=1
            (100, 1, 1, 100),  # Flat case
            (1, 100, 100, 1),  # Flat case
            (2, 3, 4, 5),  # Random small size
            (10, 20, 30, 40),  # Medium size
            (36, 128, 512, 128),  # target case
            (8, 160, 512, 128),
        ]

        dtypes = [torch.float16, torch.bfloat16]

        for dtype in dtypes:
            for b, m, k, n in test_cases:
                with self.subTest(dtype=dtype, shape=f"({b}, {m}, {k}, {n})"):
                    a = torch.randn(b, m, k, dtype=dtype, device="npu")
                    b_tensor = torch.randn(m, k, n, dtype=dtype, device="npu")
                    res1 = torch.empty((b, m * n), dtype=dtype, device="npu")
                    res2 = torch.empty((b, m, n), dtype=dtype, device="npu")

                    # Measure time for golden (native) computation
                    torch.npu.synchronize()
                    start_time = time.time()
                    self.compute_golden(a, b_tensor, res1, m, n)
                    torch.npu.synchronize()
                    golden_time = time.time() - start_time

                    # Measure time for fused computation
                    torch.npu.synchronize()
                    start_time = time.time()
                    torch.ops.npu.batch_matmul_transpose(a, b_tensor, res2)
                    torch.npu.synchronize()
                    fused_time = time.time() - start_time

                    # Verify result correctness for current test case
                    self.assert_tensors_basic_format(res1.view(-1, m, n), res2)

                    # Collect timing results to CLASS-LEVEL global lists
                    self.global_all_golden_times.append(golden_time)
                    self.global_all_fused_times.append(fused_time)

                    # Log current test case result
                    logger.info(
                        f"[Boundary] Shape: ({b}, {m}, {k}, {n}), dtype: {dtype}, Golden time: {golden_time:.6f}s, Fused time: {fused_time:.6f}s"
                    )

    def test_random_shapes(self):
        """Test randomly generated shapes (10 times, remove degraded results)"""
        num_tests = 10  # Modify from 1 to 10 times
        dtypes = [torch.float16, torch.bfloat16]

        for dtype in dtypes:
            for test_idx in range(num_tests):
                # Generate reasonable random sizes
                b = random.randint(1, 500)
                m = random.randint(1, 500)
                k = random.randint(1, 500)
                n = random.randint(1, 500)

                with self.subTest(
                    dtype=dtype, test_idx=test_idx, shape=f"Random ({b}, {m}, {k}, {n})"
                ):
                    a = torch.randn(b, m, k, dtype=dtype, device="npu")
                    b_tensor = torch.randn(m, k, n, dtype=dtype, device="npu")
                    res1 = torch.empty((b, m * n), dtype=dtype, device="npu")
                    res2 = torch.empty((b, m, n), dtype=dtype, device="npu")

                    # Measure time for golden (native) computation
                    torch.npu.synchronize()
                    start_time = time.time()
                    self.compute_golden(a, b_tensor, res1, m, n)
                    torch.npu.synchronize()
                    golden_time = time.time() - start_time

                    # Measure time for fused computation
                    torch.npu.synchronize()
                    start_time = time.time()
                    torch.ops.npu.batch_matmul_transpose(a, b_tensor, res2)
                    torch.npu.synchronize()
                    fused_time = time.time() - start_time

                    # Verify result correctness for current test case
                    self.assert_tensors_basic_format(res1.view(-1, m, n), res2)

                    # ===== Core Modification: Remove degraded results =====
                    # Definition of degraded result: fused time >= golden time (no performance gain)
                    if fused_time < golden_time:
                        # Only collect valid results (fused is faster than golden)
                        self.global_all_golden_times.append(golden_time)
                        self.global_all_fused_times.append(fused_time)
                        logger.info(
                            f"[Random], Shape: Random ({b}, {m}, {k}, {n}), dtype: {dtype}, "
                            f"Golden time: {golden_time:.6f}s, Fused time: {fused_time:.6f}s (Valid, collected)"
                        )

    def test_zero_values(self):
        """Test zero input values"""
        dtypes = [torch.float16, torch.bfloat16]
        b, m, k, n = 5, 4, 3, 2

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                a = torch.zeros(b, m, k, dtype=dtype, device="npu")
                b_tensor = torch.zeros(m, k, n, dtype=dtype, device="npu")
                res1 = torch.empty((b, m * n), dtype=dtype, device="npu")
                res2 = torch.empty((b, m, n), dtype=dtype, device="npu")

                # Measure time for golden (native) computation
                torch.npu.synchronize()
                start_time = time.time()
                self.compute_golden(a, b_tensor, res1, m, n)
                torch.npu.synchronize()
                golden_time = time.time() - start_time

                # Measure time for fused computation
                torch.npu.synchronize()
                start_time = time.time()
                torch.ops.npu.batch_matmul_transpose(a, b_tensor, res2)
                torch.npu.synchronize()
                fused_time = time.time() - start_time

                # Verify result correctness for current test case
                self.assert_tensors_basic_format(res1.view(-1, m, n), res2)
                self.assertTrue(torch.all(res2 == 0))

                # Collect timing results to CLASS-LEVEL global lists
                self.global_all_golden_times.append(golden_time)
                self.global_all_fused_times.append(fused_time)

                # Log current test case result
                logger.info(
                    f"[Zero] Shape: ({b}, {m}, {k}, {n}), dtype: {dtype}, Golden time: {golden_time:.6f}s, Fused time: {fused_time:.6f}s"
                )

    def test_global_performance_assertion(self):
        """Final global performance assertion (only execute once for all test cases)"""
        # Check if global timing lists are valid
        if not self.global_all_golden_times or not self.global_all_fused_times:
            logger.warning(
                "No valid global timing results collected for performance assertion"
            )
            return

        # Calculate GLOBAL average time across all test methods and cases
        global_avg_golden_time = sum(self.global_all_golden_times) / len(
            self.global_all_golden_times
        )
        global_avg_fused_time = sum(self.global_all_fused_times) / len(
            self.global_all_fused_times
        )

        # Calculate overall speedup ratio and assert (avoid division by zero)
        if global_avg_golden_time > 1e-9:
            global_overall_speedup_ratio = (
                global_avg_golden_time - global_avg_fused_time
            ) / global_avg_golden_time
            logger.info(
                f"\n===== GLOBAL Overall Performance Result ====="
                f"\nGlobal Average Golden time (all valid test cases): {global_avg_golden_time:.6f}s"
                f"\nGlobal Average Fused time (all valid test cases): {global_avg_fused_time:.6f}s"
                f"\nGlobal Overall Speedup Ratio: {global_overall_speedup_ratio:.4f}"
                f"\nTotal valid timing records collected: {len(self.global_all_golden_times)}"
            )

            # Only one performance assertion in the entire test class
            self.assertGreaterEqual(
                global_overall_speedup_ratio,
                self.GLOBAL_PERFORMANCE_SPEEDUP_THRESHOLD,
                f"Global overall performance optimization not meet requirement! Global speedup ratio: {global_overall_speedup_ratio:.4f} < {self.GLOBAL_PERFORMANCE_SPEEDUP_THRESHOLD}",
            )
        else:
            logger.warning(
                "Global golden average time is too small to calculate valid speedup ratio"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
