"""Marlin MoE GEMV against the Marlin MoE GEMM on the same repacked NVFP4 / MXFP8
expert weights, plus its input validation."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

EXPERTS, N, K, TOP_K = 8, 512, 1024, 2


def _is_sm90() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


def _metadata(tokens, generator):
    ids = torch.stack(
        [torch.randperm(EXPERTS, generator=generator)[:TOP_K] for _ in range(tokens)]
    ).cuda()
    flat = ids.flatten()
    counts = torch.bincount(flat, minlength=EXPERTS).to(torch.int32)
    offsets = torch.zeros(EXPERTS + 1, dtype=torch.int32, device="cuda")
    offsets[1:] = torch.cumsum(counts, 0)
    dst2src = torch.sort(flat, stable=True)[1].to(torch.int32)
    return counts, offsets, dst2src


def _nvfp4_marlin(k=K):
    from sglang.kernels.ops.gemm import inkling2_marlin_sm90 as m

    weight = torch.randint(
        0, 256, (EXPERTS, N, k // 2), dtype=torch.uint8, device="cuda"
    )
    scales = (torch.rand(EXPERTS, N, k // 16, device="cuda") * 4 + 0.25).to(
        torch.float8_e4m3fn
    )
    return m.repack(weight, scales)


def _mxfp8_marlin():
    from sglang.kernels.ops.gemm import inkling2_marlin_sm90 as m

    weight = torch.randn(EXPERTS, N, K, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.randint(
        120, 131, (EXPERTS, N, K // 32), dtype=torch.uint8, device="cuda"
    )
    return m.repack_mxfp8(weight, scales)


@unittest.skipIf(not _is_sm90(), "Marlin MoE GEMV requires an SM90 GPU")
class TestMarlinMoeGemv(unittest.TestCase):
    def _check_matches_gemm(self, marlin, tokens, k=K):
        from sglang.kernels.ops.gemm import inkling2_marlin_sm90 as m

        generator = torch.Generator().manual_seed(tokens)
        counts, offsets, dst2src = _metadata(tokens, generator)
        x = torch.randn(tokens, k, dtype=torch.bfloat16, device="cuda")
        routing = m.route(counts, offsets, tokens * TOP_K, tokens, dst2src)
        ref = m.gemm(x, marlin, N, routing, TOP_K).float()
        out = m.gemv(x, marlin, N, (counts, offsets, dst2src), TOP_K, tokens).float()
        err = (out - ref).abs().max().item() / ref.abs().max().item()
        # a few outputs differ by several bf16 ulps (up to ~2e-3 of the max)
        self.assertLess(err, 1e-2, (tokens, err))

    def test_nvfp4_matches_gemm(self):
        marlin = _nvfp4_marlin()
        for tokens in (1, 2, 8):
            self._check_matches_gemm(marlin, tokens)

    def test_nvfp4_k_tiles_not_divisible_by_default_split(self):
        """K = 1152 has 72 K tiles, which the 4 x 4 small-batch split does not divide."""
        self._check_matches_gemm(_nvfp4_marlin(k=1152), tokens=1, k=1152)

    def test_nvfp4_small_batch_tuned_widths(self):
        for k in (1152, 2304):
            marlin = _nvfp4_marlin(k=k)
            for tokens in (1, 2):
                self._check_matches_gemm(marlin, tokens, k=k)

    def test_mxfp8_matches_gemm(self):
        marlin = _mxfp8_marlin()
        for tokens in (1, 2):
            self._check_matches_gemm(marlin, tokens)

    def test_rejects_bad_inputs(self):
        from sglang.kernels.ops.gemm import inkling2_marlin_sm90 as m

        marlin = _nvfp4_marlin()
        metadata = _metadata(2, torch.Generator().manual_seed(0))
        x = torch.randn(2, K, dtype=torch.bfloat16, device="cuda")
        with self.assertRaises(ValueError):
            m.gemv(x, marlin, N - 32, metadata, TOP_K, 2)
        strided = torch.randn(2, 2 * K, dtype=torch.bfloat16, device="cuda")[:, :K]
        with self.assertRaises(Exception):
            m.gemv(strided, marlin, N, metadata, TOP_K, 2)
        with self.assertRaises(Exception):
            m.gemv(x.half(), marlin, N, metadata, TOP_K, 2)
        counts, offsets, dst2src = metadata
        with self.assertRaises(Exception):
            m.gemv(x, marlin, N, (counts, offsets, dst2src.long()), TOP_K, 2)


if __name__ == "__main__":
    unittest.main()
