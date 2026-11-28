"""The ROCm attention-residual read (attn_res_hip) on banks of more than 4 rows,
which the kernel walks in register tiles with the softmax taken online across them."""

import unittest

import torch

from sglang.kernels.ops.attention.attn_res_hip import attn_res_hip
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=15, stage="base-b", runner_config="1-gpu-small-amd-mi35x")

_HIDDEN_SIZE = 7168
_BANK_ROWS = 9
_EPS = 1e-6


def _reference(prefix, bank, nvb, cw, ow):
    rows = torch.cat([bank[:, :nvb], prefix.unsqueeze(1)], dim=1).double()
    scores = (rows * cw.double()).sum(-1) * torch.rsqrt(rows.square().mean(-1) + _EPS)
    mixed = (torch.softmax(scores, dim=-1).unsqueeze(-1) * rows).sum(1)
    if ow is not None:
        rms = torch.rsqrt(mixed.square().mean(-1, keepdim=True) + _EPS)
        mixed = mixed * rms * ow.double()
    return mixed


def _same_bits(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


class TestAttnResHipTiles(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not (torch.cuda.is_available() and is_hip()):
            raise unittest.SkipTest("ROCm only")
        cls.gen = torch.Generator(device="cuda").manual_seed(0)
        # Scores a few units apart: the running maximum moves from tile to tile.
        cls.cw = cls._randn(_HIDDEN_SIZE) * 4 * _HIDDEN_SIZE**-0.5
        cls.ow = (1 + 0.1 * cls._randn(_HIDDEN_SIZE)).to(torch.bfloat16)

    @classmethod
    def _randn(cls, *shape):
        return torch.randn(*shape, generator=cls.gen, device="cuda")

    def _inputs(self, tokens):
        prefix = self._randn(tokens, _HIDDEN_SIZE).to(torch.bfloat16)
        bank = self._randn(tokens, _BANK_ROWS, _HIDDEN_SIZE).to(torch.bfloat16)
        return prefix, bank

    def _read(self, prefix, bank, nvb, ow, addend=None, write=False):
        out = torch.empty_like(prefix)
        prefix_out = torch.empty_like(prefix) if addend is not None else None
        attn_res_hip(
            prefix,
            bank,
            self.cw,
            ow,
            out,
            nvb,
            _EPS,
            _EPS,
            addend=addend,
            prefix_out=prefix_out,
            write_prefix=write,
        )
        return out, prefix_out

    def test_tiled_bank_matches_reference(self):
        # The online softmax over the tiles (the last one half masked at an odd row
        # count) against the direct formula in float64, in every form of the read.
        for nvb in (5, 6, 7, 8):
            for with_add in (False, True):
                for ow in (self.ow, None):
                    prefix, bank = self._inputs(64)
                    addend = (
                        self._randn(64, _HIDDEN_SIZE).to(torch.bfloat16)
                        if with_add
                        else None
                    )
                    out, prefix_out = self._read(
                        prefix, bank, nvb, ow, addend=addend, write=True
                    )
                    msg = f"nvb={nvb} add={with_add} norm={ow is not None}"
                    if with_add:
                        summed = (prefix.float() + addend.float()).to(torch.bfloat16)
                        self.assertTrue(_same_bits(prefix_out, summed), msg)
                        prefix = summed
                    self.assertTrue(_same_bits(bank[:, nvb], prefix), msg)
                    torch.testing.assert_close(
                        out.double(),
                        _reference(prefix, bank, nvb, self.cw, ow),
                        rtol=4e-3,  # half a bf16 ulp
                        atol=1e-4,
                        msg=msg,
                    )

    def test_bits_do_not_depend_on_token_count(self):
        # A row reads the same whatever batch it is in (every TP rank and every
        # batching of a request must see one stream), on both sides of the 512
        # tokens that fill the CUs.
        for nvb in (5, 6, 7, 8):
            prefix, bank = self._inputs(513)
            out, _ = self._read(prefix, bank, nvb, self.ow)
            for tokens in (512, 3):
                part, _ = self._read(prefix[:tokens], bank[:tokens], nvb, self.ow)
                self.assertTrue(
                    _same_bits(part, out[:tokens]), f"nvb={nvb} tokens={tokens}"
                )


if __name__ == "__main__":
    unittest.main()
