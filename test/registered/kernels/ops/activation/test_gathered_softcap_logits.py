"""gathered_softcap_logits with no softcap: rank-major logits blocks (a TP
all-gather, or the TP LM-head all-to-all output) to vocab-major fp32 logits, bit for
bit the stock reassembling copy and fp32 cast."""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=4, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def _bits(x):
    return x.view(torch.int32)


@pytest.mark.parametrize(
    "world,rows,shard,vocab",
    [(2, 3, 5, 9), (4, 1, 64, 256), (4, 7, 1000, 3999), (8, 33, 31040, 248320)],
)
def test_cast_only_matches_the_stock_reassembly_and_cast(world, rows, shard, vocab):
    from sglang.kernels.ops.activation.softcap import (
        cast_to_float32_logits,
        gathered_softcap_logits,
    )

    torch.manual_seed(0)
    blocks = (torch.randn(world * rows, shard, device="cuda") * 16).bfloat16()
    blocks[0, :4] = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0])
    stock = (
        blocks.view(world, rows, shard)
        .permute(1, 0, 2)
        .reshape(rows, world * shard)[:, :vocab]
    )
    want = torch.empty((rows, vocab), dtype=torch.float32, device="cuda")
    cast_to_float32_logits(stock, want)
    # A row slice of a larger buffer, as the decode graphs pass theirs.
    buffer = torch.full((rows + 2, vocab), 7.0, device="cuda")
    out = buffer[:rows]

    got = gathered_softcap_logits(
        gathered=blocks.view(world, rows, shard),
        final_logit_softcapping=None,
        output=out,
    )

    assert got is out
    assert torch.all(buffer[rows:] == 7.0)
    assert torch.equal(_bits(got), _bits(want))
    eager = stock.float()
    keep = ~eager.isnan()
    assert torch.equal(got.isnan(), ~keep)
    assert torch.equal(_bits(got[keep]), _bits(eager[keep]))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
