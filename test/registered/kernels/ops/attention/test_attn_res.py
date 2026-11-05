import unittest

import torch

from sglang.kernels.ops.attention.attn_res import attn_res_fused_tma
from sglang.kernels.ops.attention.attn_res_hip import attn_res_hip
from sglang.srt.utils import get_device_sm, is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

# backend-specific: the HIP kernel replaces SM100 TMA on ROCm.
register_amd_ci(est_time=15, stage="jit-kernel-unit", runner_config="amd")

_HIDDEN_SIZE = 7168
_EPS = 1e-6


def _reference(prefix, bank, nvb, cw, ow, addend=None):
    if addend is not None:
        prefix = (prefix.float() + addend.float()).to(prefix.dtype)
    rows = torch.cat([bank[:, :nvb].float(), prefix.unsqueeze(1).float()], dim=1)
    rms = torch.rsqrt(rows.square().mean(-1) + _EPS)
    scores = (rows * cw.float()).sum(-1) * rms
    mixed = (torch.softmax(scores, dim=-1).unsqueeze(-1) * rows).sum(1)
    if ow is not None:
        mixed = (
            mixed
            * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + _EPS)
            * ow.float()
        )
    return mixed, prefix


class TestAttentionResidual(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if is_hip() or get_device_sm() < 100:
            raise unittest.SkipTest("Kimi K3 compute kernels require SM100a+")

    def test_attn_residual_and_prefix_write(self):
        generator = torch.Generator(device="cuda").manual_seed(0)

        def randn(*shape):
            return torch.randn(*shape, generator=generator, device="cuda")

        num_tokens, num_bank_rows, num_valid_bank_rows = 5, 8, 5
        prefix = randn(num_tokens, _HIDDEN_SIZE).to(torch.bfloat16)
        bank = randn(num_tokens, num_bank_rows, _HIDDEN_SIZE).to(torch.bfloat16)
        combine_weight = (randn(_HIDDEN_SIZE) * _HIDDEN_SIZE**-0.5).to(torch.bfloat16)
        output_weight = (1 + 0.1 * randn(_HIDDEN_SIZE)).to(torch.bfloat16)
        output = torch.empty_like(prefix)

        expected, _ = _reference(
            prefix, bank, num_valid_bank_rows, combine_weight, output_weight
        )

        attn_res_fused_tma(
            prefix,
            bank,
            combine_weight,
            output_weight,
            output,
            num_valid_bank_rows,
            1e-6,
            write_prefix=True,
        )

        torch.testing.assert_close(output.float(), expected, rtol=2e-2, atol=4e-2)
        self.assertTrue(torch.equal(bank[:, num_valid_bank_rows], prefix))


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
        # Token count only changes the grid; keep every bank/add/norm specialization.
        for tokens, nvb in ((1, 1), (8, 5), (33, 8)):
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
                    want, want_prefix = _reference(prefix, bank, nvb, cw, norm, addend)
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
