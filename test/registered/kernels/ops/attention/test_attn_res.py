import unittest

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.attn_res import attn_res_fused_tma
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_HIDDEN_SIZE = 7168


@triton.jit
def _late_rows_kernel(
    dst_ptr, src_ptr, num_rows, row_elems, dst_stride, spin, BLOCK: tl.constexpr
):
    # Lets the next (PDL) kernel launch at once, then writes its rows late.
    tl.extra.cuda.gdc_launch_dependents()
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for _ in range(spin):
        acc = acc * 0.5 + 1.0
    late = tl.where(acc < 0.0, 1.0, 0.0)  # always 0; keeps the spin ahead of the stores
    for row in range(tl.program_id(0), num_rows, tl.num_programs(0)):
        for start in range(0, row_elems, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < row_elems
            vals = tl.load(src_ptr + row.to(tl.int64) * row_elems + offs, mask=mask)
            vals = (vals.to(tl.float32) + late).to(vals.dtype)
            tl.store(dst_ptr + row.to(tl.int64) * dst_stride + offs, vals, mask=mask)


class TestAttentionResidual(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() < 100:
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

        rows = torch.cat(
            [
                bank[:, :num_valid_bank_rows].float(),
                prefix.unsqueeze(1).float(),
            ],
            dim=1,
        )
        rms = torch.rsqrt(rows.square().mean(-1) + 1e-6)
        scores = (rows * combine_weight.float()).sum(-1) * rms
        mixed = (torch.softmax(scores, dim=-1).unsqueeze(-1) * rows).sum(1)
        expected = (
            mixed
            * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + 1e-6)
            * output_weight.float()
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

    def test_bank_written_by_the_preceding_kernel(self):
        """The kernel right before the aggregation may write the bank rows (a PP stage's entry copies the inherited
        rows in): every bank read must wait for it, not only the prefix row's."""
        generator = torch.Generator(device="cuda").manual_seed(0)
        num_tokens = 512
        for nvb in range(1, 8):
            with self.subTest(nvb=nvb):
                prefix = torch.randn(
                    num_tokens, _HIDDEN_SIZE, generator=generator, device="cuda"
                ).to(torch.bfloat16)
                rows = torch.randn(
                    num_tokens, nvb, _HIDDEN_SIZE, generator=generator, device="cuda"
                ).to(torch.bfloat16)
                combine_weight = (
                    torch.randn(_HIDDEN_SIZE, generator=generator, device="cuda")
                    * _HIDDEN_SIZE**-0.5
                ).to(torch.bfloat16)
                output_weight = torch.ones(
                    _HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
                )
                bank = torch.full(
                    (num_tokens, nvb + 1, _HIDDEN_SIZE),
                    float("nan"),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                output = torch.empty_like(prefix)
                args = (prefix, bank, combine_weight, output_weight, output, nvb, 1e-6)
                # A config's first call loads its module on the host, long enough for the writer to finish first.
                attn_res_fused_tma(*args, write_prefix=True)
                _late_rows_kernel[(16,)](
                    bank,
                    rows,
                    num_tokens,
                    nvb * _HIDDEN_SIZE,
                    bank.stride(0),
                    200000,
                    BLOCK=1024,
                )
                attn_res_fused_tma(*args, write_prefix=True)
                # The bank held NaN until the writer ran, so any read that did not wait for it is non-finite.
                self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
