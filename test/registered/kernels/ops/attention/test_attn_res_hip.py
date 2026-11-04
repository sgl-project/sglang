import unittest

import torch

from sglang.kernels.ops.attention.attn_res_hip import attn_res_hip
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

# backend-specific: attn_res_hip is the ROCm (no TMA) counterpart of the SM100 attn_res kernel.
register_amd_ci(est_time=15, stage="jit-kernel-unit", runner_config="amd")

_HIDDEN_SIZE = 7168
_EPS = 1e-6


def _reference(prefix, addend, bank, nvb, cw, ow):
    if addend is not None:
        prefix = (prefix.float() + addend.float()).to(prefix.dtype)
    rows = torch.cat([bank[:, :nvb].float(), prefix.unsqueeze(1).float()], dim=1)
    score = (rows * cw).sum(-1) / torch.sqrt(rows.pow(2).mean(-1) + _EPS)
    acc = (torch.softmax(score, dim=-1).unsqueeze(-1) * rows).sum(1)
    if ow is not None:
        acc = acc * torch.rsqrt(acc.pow(2).mean(-1, keepdim=True) + _EPS) * ow.float()
    return acc, prefix


class TestAttnResHip(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not (torch.cuda.is_available() and is_hip()):
            raise unittest.SkipTest("ROCm only")

    def test_matches_reference(self):
        # The folded residual add, bank snapshot, softmax mix and output RMSNorm against the direct
        # formula, with and without each optional stage.
        gen = torch.Generator(device="cuda").manual_seed(0)

        def randn(*shape):
            return torch.randn(*shape, generator=gen, device="cuda")

        cw = randn(_HIDDEN_SIZE) * _HIDDEN_SIZE**-0.5
        ow = (1 + 0.1 * randn(_HIDDEN_SIZE)).to(torch.bfloat16)
        for tokens in (1, 8, 33):
            for nvb in (1, 5, 8):
                for with_add in (False, True):
                    for norm in (ow, None):
                        prefix = randn(tokens, _HIDDEN_SIZE).to(torch.bfloat16)
                        bank = randn(tokens, 9, _HIDDEN_SIZE).to(torch.bfloat16)
                        addend = (
                            randn(tokens, _HIDDEN_SIZE).to(torch.bfloat16)
                            if with_add
                            else None
                        )
                        out = torch.empty_like(prefix)
                        prefix_out = torch.empty_like(prefix)
                        attn_res_hip(
                            prefix,
                            bank,
                            cw,
                            norm,
                            out,
                            nvb,
                            _EPS,
                            _EPS,
                            addend=addend,
                            prefix_out=prefix_out if with_add else None,
                            write_prefix=True,
                        )
                        want, want_prefix = _reference(
                            prefix, addend, bank, nvb, cw, norm
                        )
                        msg = f"tokens={tokens} nvb={nvb} add={with_add} norm={norm is not None}"
                        torch.testing.assert_close(
                            out.float(), want, atol=2e-2, rtol=2e-2, msg=msg
                        )
                        torch.testing.assert_close(
                            bank[:, nvb], want_prefix, atol=0, rtol=0, msg=msg
                        )
                        if with_add:
                            torch.testing.assert_close(
                                prefix_out, want_prefix, atol=0, rtol=0, msg=msg
                            )


if __name__ == "__main__":
    unittest.main()
