"""CPU checks that DeepGEMM warmup budgets include every allocated buffer."""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.deep_gemm_wrapper.compile_utils import (
    DeepGemmKernelType,
    _BaseWarmupExecutor,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _allocate_on_meta(factory):
    def allocate(*args, **kwargs):
        kwargs["device"] = "meta"
        return factory(*args, **kwargs)

    return allocate


class TestWarmupMemory(CustomTestCase):
    def test_budget_covers_executor_buffers(self):
        for max_m, n, k, num_groups in (
            (9, 131, 259, 3),  # Scale dimensions require ceil division.
            (16384, 4096, 7168, 32),  # V3 DP8/EP8 startup OOM shape.
        ):
            for kernel_type in DeepGemmKernelType:
                with self.subTest(kernel_type=kernel_type, max_m=max_m):
                    with ExitStack() as stack:
                        for name in ("empty", "ones", "zeros"):
                            factory = getattr(torch, name)
                            stack.enter_context(
                                patch.object(
                                    torch, name, side_effect=_allocate_on_meta(factory)
                                )
                            )
                        executor = _BaseWarmupExecutor.create(
                            kernel_type, max_m=max_m, n=n, k=k, num_groups=num_groups
                        )
                    allocated_bytes = sum(
                        tensor.numel() * tensor.element_size()
                        for tensor in vars(executor).values()
                        if isinstance(tensor, torch.Tensor)
                    )
                    budget_gib = _BaseWarmupExecutor.get_memory_requirement(
                        kernel_type, max_m=max_m, n=n, k=k, num_groups=num_groups
                    )
                    self.assertEqual(budget_gib * (1 << 30), allocated_bytes)


if __name__ == "__main__":
    unittest.main()
