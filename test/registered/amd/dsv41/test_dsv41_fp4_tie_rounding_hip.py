"""Pin the e2m1 tie rounding of the HIP low-ratio FP4 indexer quantizers: ties to even on every path."""

import unittest

import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=40, suite="stage-b-test-1-gpu-small-amd-mi35x")

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
TIES = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
# What each convention returns for TIES (and their negatives, by symmetry).
RNE = [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0]
SCALES = (1.0, 2.0**-3)


def _tie_row(scale: float) -> torch.Tensor:
    """[128]: four 32-blocks of [6, +ties, -ties, 0...] times a power-of-two scale."""
    block = [6.0] + TIES + [-t for t in TIES]
    block += [0.0] * (32 - len(block))
    return (torch.tensor(block) * scale).repeat(4)


def _unpack(packed: torch.Tensor) -> torch.Tensor:
    """Packed e2m1 nibbles [..., 64] (low nibble first) -> values [..., 128]."""
    p = packed.view(torch.uint8).to(torch.int64).cpu()
    codes = torch.stack([p & 0xF, p >> 4], dim=-1).flatten(-2)
    mag = E2M1[codes & 7]
    return torch.where((codes & 8) != 0, -mag, mag)


def _expected(scale: float, convention) -> torch.Tensor:
    block = [6.0] + convention + [-v for v in convention]
    block += [0.0] * (32 - len(block))
    return (torch.tensor(block) * scale).repeat(4)


@unittest.skipUnless(is_hip() and torch.cuda.is_available(), "ROCm only")
class TestDsv41Fp4TieRoundingHip(CustomTestCase):
    def _check(self, name, scale, got, convention):
        self.assertTrue(
            torch.equal(got.float().cpu(), _expected(scale, convention)),
            f"{name} at scale {scale}: {got[1:8].tolist()} vs {convention}",
        )

    def _e8m0(self, scale: float) -> int:
        return 127 + int(torch.log2(torch.tensor(scale)))

    def test_low_ratio_triton_paths_round_half_to_even_like_cuda(self):
        """Same Triton quantizer as CUDA (`rne=True`): ties to even on both."""
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            quantize_fp4_indexer_tensor,
        )
        from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
            pack_fp4_query_flydsl,
            read_fp4_index_k_split,
            store_fp4_index_k_cache_split,
        )
        from sglang.kernels.ops.attention.dsv4.rope_fake_quant_fp4 import (
            rope_tail_fake_quant_fp4,
        )

        for scale in SCALES:
            row = _tie_row(scale).cuda().to(torch.bfloat16)
            self.assertTrue(torch.equal(row.float().cpu(), _tie_row(scale)))

            fp4, sf = quantize_fp4_indexer_tensor(row.view(1, 128), rne=True)
            self.assertEqual((sf.cpu() & 0xFF).item(), self._e8m0(scale))
            self._check(
                "quantize_fp4_indexer_tensor(rne=True)",
                scale,
                _unpack(fp4)[0] * scale,
                RNE,
            )
            q_fp4, q_scale = pack_fp4_query_flydsl(
                row.view(1, 1, 128).expand(1, 16, 128).contiguous()
            )
            self.assertEqual(q_scale.unique().tolist(), [0, self._e8m0(scale)])
            self._check(
                "pack_fp4_query_flydsl", scale, _unpack(q_fp4)[0, 0] * scale, RNE
            )

            payload = torch.zeros((1, 1, 4, 64, 16), dtype=torch.uint8, device="cuda")
            k_scale = torch.zeros((1, 1, 4, 64), dtype=torch.uint8, device="cuda")
            loc = torch.tensor([5], dtype=torch.int64, device="cuda")
            store_fp4_index_k_cache_split(
                row.view(1, 128), payload, k_scale, loc, page_size=64, rne=True
            )
            k_fp4, k_sf = read_fp4_index_k_split(payload, k_scale, loc, page_size=64)
            self.assertEqual((k_sf.cpu() & 0xFF).item(), self._e8m0(scale))
            self._check(
                "store_fp4_index_k_cache_split", scale, _unpack(k_fp4)[0] * scale, RNE
            )

            freqs = torch.ones(1, 32, dtype=torch.complex64, device="cuda")
            fq = rope_tail_fake_quant_fp4(row.view(1, 128), freqs, 64)
            self._check("rope_tail_fake_quant_fp4", scale, fq[0], RNE)


if __name__ == "__main__":
    unittest.main()
