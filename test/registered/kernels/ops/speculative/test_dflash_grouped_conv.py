import sys

import pytest
import torch

from sglang.kernels.ops.speculative.dflash import dflash_grouped_conv
from sglang.srt.models.dflash import _grouped_conv
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@pytest.mark.parametrize("num_tokens", [0, 1, 2304])
@pytest.mark.parametrize("block_size,taps", [(8, 2), (5, 3)])
def test_matches_reference(num_tokens, block_size, taps):
    hidden_size, group_size = 7168, 16
    num_groups = hidden_size // group_size
    gen = torch.Generator(device="cuda").manual_seed(num_tokens * 10 + taps)
    hidden = torch.randn(
        num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16, generator=gen
    )
    delta = torch.randn(
        num_tokens, taps, num_groups, device="cuda", dtype=torch.bfloat16, generator=gen
    )
    base = torch.randn(
        taps, hidden_size, device="cuda", dtype=torch.bfloat16, generator=gen
    )

    out = dflash_grouped_conv(hidden, delta, base, block_size, group_size, taps)
    if num_tokens == 0:
        assert out.shape == hidden.shape
        return
    ref = _grouped_conv(hidden, delta, base, block_size, num_groups, group_size, taps)
    # The reference returns extra rows for a batch shorter than `taps - 1`.
    ref = ref[:num_tokens]
    if taps == 2:
        # The shipped DFlash2 drafts (conv_kernel_size 2) match bit for bit.
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
    else:
        torch.testing.assert_close(out, ref)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
