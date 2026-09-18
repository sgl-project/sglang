"""Small-M Hopper block-FP8 MMA transpose with nonuniform UE8M0 scales."""

import unittest
from unittest.mock import patch

import torch
import triton

from sglang.kernels.ops.quantization import fp8_kernel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
    "Hopper required",
)
class TestHopperBlockFP8SwapAB(CustomTestCase):
    def test_graph_replay_matches_original(self):
        self._check_graph_replay(split_k=False)

    def test_split_k_replay_matches_original(self):
        self._check_graph_replay(split_k=True)

    def _check_graph_replay(self, split_k):
        original = dict(
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=32,
            BLOCK_SIZE_K=32,
            GROUP_SIZE_M=32,
            num_warps=4,
            num_stages=3,
        )
        for n, k in [
            (1152, 5120),
            (1792, 5120),
            (25600, 6144),
            (4096, 1280),
            (5120, 2048),
            (5120, 576),
            (8192, 1280),
        ]:
            with self.subTest(n=n, k=k):
                torch.manual_seed(17)
                a = torch.randn(1, k, device="cuda").to(torch.float8_e4m3fn)
                b = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
                sa = torch.exp2(
                    torch.randint(-5, 3, (1, k // 32), device="cuda").float()
                )
                sb = torch.exp2(
                    torch.randint(-5, 3, (n // 32, k // 32), device="cuda").float()
                )
                config = {
                    **original,
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128 if n == 25600 or k == 576 else 64,
                    "num_stages": 4,
                    "SWAP_AB": True,
                }

                if split_k and n != 25600:
                    splits = {
                        (1152, 5120): 16,
                        (1792, 5120): 8,
                        (4096, 1280): 4,
                        (5120, 2048): 8,
                        (5120, 576): 4,
                        (8192, 1280): 2,
                    }
                    config.update(SPLIT_K=splits[n, k], BLOCK_SIZE_N=64)

                def run():
                    return fp8_kernel.w8a8_block_fp8_matmul_triton(
                        a, b, sa, sb, [32, 32], torch.bfloat16
                    )

                with patch.object(
                    fp8_kernel, "get_w8a8_block_fp8_configs", return_value={1: original}
                ):
                    expected = run()
                with patch.object(
                    fp8_kernel, "get_w8a8_block_fp8_configs", return_value={1: config}
                ):
                    run()  # Compile outside graph capture.
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual = run()
                graph.replay()
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=0.008 if split_k else 0,
                    atol=0.0001 if split_k else 0,
                )
                # Replay must read fresh activation payload and scales.
                a.copy_(torch.randn(1, k, device="cuda").to(a.dtype))
                sa.mul_(2)
                with patch.object(
                    fp8_kernel, "get_w8a8_block_fp8_configs", return_value={1: original}
                ):
                    expected = run()
                graph.replay()
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=0.008 if split_k else 0,
                    atol=0.0001 if split_k else 0,
                )


def _specialization_count(kernel):
    """Number of compiled variants Triton is holding for `kernel`."""
    device_caches = getattr(kernel, "device_caches", None) or {}
    return sum(len(cache[0]) for cache in device_caches.values())


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
    "Hopper required",
)
class TestSplitKReduceSpecialization(CustomTestCase):
    """The split-K reduce must not mint a variant per token count.

    `elements` (= M*N) and `splits` (= SPLIT_K) are runtime arguments precisely
    so that a decode step whose M changes -- which under speculative decoding is
    nearly every step -- reuses one compiled variant instead of paying a fresh
    ~2s serving-time compile. Making either one a tl.constexpr again would
    reintroduce the stall without failing any correctness assertion, so pin the
    count directly.
    """

    def test_one_variant_per_split_k_regardless_of_M(self):
        from sglang.kernels.ops.quantization.fp8_kernel import (
            _reduce_block_fp8_split_k,
        )

        n, split_k = 512, 8
        distinct_ms = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

        # Measure the delta, not the absolute count: other tests in this module
        # also reach this kernel, so the cache is not necessarily empty here.
        before = _specialization_count(_reduce_block_fp8_split_k)
        for m in distinct_ms:
            parts = torch.randn(split_k, m, n, device="cuda")
            out = torch.empty(m, n, device="cuda")
            _reduce_block_fp8_split_k[(triton.cdiv(m * n, 256),)](
                parts, out, m * n, split_k, 256
            )
            torch.testing.assert_close(out, parts.sum(0), atol=1e-3, rtol=1e-4)
        compiled = _specialization_count(_reduce_block_fp8_split_k) - before

        # At most one new variant: zero if this SPLIT_K was already compiled by
        # an earlier test, one if not. What must never happen is one per M.
        self.assertLessEqual(
            compiled,
            1,
            f"{len(distinct_ms)} distinct M values compiled {compiled} new "
            "variants; elements/splits must stay runtime arguments.",
        )

    def test_masks_splits_beyond_split_k(self):
        """Padding rows past `splits` must be masked, not read."""
        from sglang.kernels.ops.quantization.fp8_kernel import (
            _MAX_SPLIT_K,
            _reduce_block_fp8_split_k,
        )

        m, n, split_k = 8, 64, 4
        parts = torch.full((_MAX_SPLIT_K, m, n), float("nan"), device="cuda")
        parts[:split_k] = 1.0
        out = torch.empty(m, n, device="cuda")
        _reduce_block_fp8_split_k[(triton.cdiv(m * n, 256),)](
            parts, out, m * n, split_k, 256
        )
        torch.testing.assert_close(out, torch.full_like(out, float(split_k)))


if __name__ == "__main__":
    unittest.main()
