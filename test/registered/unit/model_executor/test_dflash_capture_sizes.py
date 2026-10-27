"""Automatic DFlash target graph buckets preserve capacity and explicit policy."""

import bisect
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    Phase,
    PhaseConfig,
)
from sglang.srt.model_executor.runner import base_cuda_graph_runner as mod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDFlashCaptureSizes(CustomTestCase):
    sizes = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]

    def select(
        self,
        *,
        sizes=None,
        capacity=16,
        alignment=1,
        width=8,
        explicit=False,
        no_padding=False,
        draft=False,
        dflash=True,
        backend=Backend.FULL,
        overlap=False,
        provenance=True,
        cuda=True,
        compact=True,
    ):
        sizes = list(self.sizes if sizes is None else sizes)
        original = list(sizes)
        args = NS()
        if provenance:
            args._cuda_graph_config_locked = (
                {(Phase.DECODE, "bs")} if explicit else {(Phase.DECODE, "max_bs")}
            )
        runner = NS(
            server_args=args,
            req_to_token_pool=NS(size=capacity),
            is_draft_worker=draft,
            spec_algorithm=NS(is_dflash=lambda: dflash),
        )
        execution = NS(
            graph=NS(
                cuda_graph_config=CudaGraphConfig(
                    decode=PhaseConfig(backend=backend, max_bs=max(sizes), bs=sizes)
                ),
                disable_cuda_graph_padding=no_padding,
                torch_compile_max_bs=8,
            ),
            overlap=NS(enable_two_batch_overlap=overlap),
        )
        with (
            mod.envs.SGLANG_DFLASH_COMPACT_CUDA_GRAPHS.override(compact),
            patch.object(mod, "get_exec", return_value=execution),
            patch.object(
                mod, "get_flags", return_value=NS(capture=NS(enable_torch_compile=True))
            ),
            patch.object(
                mod, "get_cuda_graph_batch_size_alignment", return_value=alignment
            ),
            patch.object(mod.DllmConfig, "from_server_args", return_value=None),
            patch.object(mod, "is_cuda", return_value=cuda),
        ):
            result = mod.get_batch_sizes_to_capture(runner, width)
        self.assertEqual(
            sizes, original, "Published configuration must remain immutable"
        )
        return result

    def test_default_preserves_intermediate_buckets(self):
        self.assertFalse(mod.envs.SGLANG_DFLASH_COMPACT_CUDA_GRAPHS.default)
        for capacity in (16, 64, 256):
            sizes = list(range(1, capacity + 1))
            dense, _ = self.select(sizes=sizes, capacity=capacity, compact=False)
            self.assertEqual(dense, sizes)
            for batch in (10, 24, 48, 129, 136):
                if batch <= capacity:
                    self.assertEqual(
                        mod.BaseCudaGraphRunner._pad_to_bucket(batch, dense), batch
                    )

    def test_automatic_target_preserves_endpoints_and_compile_subset(self):
        captures, compiled = self.select()
        self.assertEqual(captures, [1, 2, 4, 8, 16])
        self.assertEqual(compiled, [1, 2, 4, 8])
        for raw in range(1, 17):
            padded = mod.BaseCudaGraphRunner._pad_to_bucket(raw, captures)
            self.assertGreaterEqual(padded, raw)
            self.assertLessEqual(padded, 16)

    def test_explicit_sizes_and_other_modes_are_preserved(self):
        cases = [
            {"compact": False},
            {"explicit": True},
            {"no_padding": True},
            {"draft": True},
            {"dflash": False},
            {"backend": Backend.BREAKABLE},
            {"overlap": True},
            {"width": 1},
            {"provenance": False},
            {"cuda": False},
        ]
        for case in cases:
            with self.subTest(case=case):
                captures, compiled = self.select(**case)
                self.assertEqual(captures, self.sizes)
                self.assertEqual(compiled, list(range(1, 9)))

    def test_alignment_and_request_pool_limit_are_applied_first(self):
        captures, compiled = self.select(capacity=14, alignment=4, width=2)
        # Current main rounds the request-pool limit up to the TP alignment.
        self.assertEqual(captures, [2, 4, 8, 16])
        self.assertEqual(compiled, [2, 4, 8])
        self.assertTrue(all(size * 2 % 4 == 0 for size in captures))

    def test_non_power_of_two_capacity_and_single_bucket(self):
        self.assertEqual(self.select(capacity=13)[0], [1, 2, 4, 8, 13])
        self.assertEqual(self.select(sizes=[8], capacity=8)[0], [8])

    def test_capacity_coverage_across_aligned_configurations(self):
        for capacity in (8, 15, 16, 31, 64, 127, 192):
            for alignment in (1, 2, 4, 8):
                for width in (2, 4, 8):
                    source = list(range(1, capacity + 1))
                    baseline = [
                        size for size in source if size * width % alignment == 0
                    ]
                    captures, compiled = self.select(
                        sizes=source,
                        capacity=capacity,
                        alignment=alignment,
                        width=width,
                    )
                    self.assertEqual(captures[0], baseline[0])
                    self.assertEqual(captures[-1], baseline[-1])
                    self.assertEqual(captures, sorted(set(captures)))
                    self.assertTrue(set(captures).issubset(baseline))
                    self.assertTrue(set(compiled).issubset(captures))
                    for raw in range(1, baseline[-1] + 1):
                        original = baseline[bisect.bisect_left(baseline, raw)]
                        padded = mod.BaseCudaGraphRunner._pad_to_bucket(raw, captures)
                        self.assertGreaterEqual(padded, raw)
                        self.assertLessEqual(padded, 2 * original)


if __name__ == "__main__":
    unittest.main()
