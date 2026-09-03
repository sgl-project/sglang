import math

import pytest
import torch

from sglang.kernels.ops.diffusion import (
    can_use_vsa_block_sparse_sm100,
    vsa_block_sparse_sm100,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not can_use_vsa_block_sparse_sm100(
        torch.cuda.current_device(), torch.bfloat16, 128
    ),
    reason="needs the sm_100a / sm_103a block-sparse kernel",
)

BLOCK = 64


def _reference(q, k, v, q2k_idx, q2k_num, block_sizes, sm_scale):
    b, h, s, d = q.shape
    n = block_sizes.numel()
    key_valid = torch.arange(BLOCK, device=q.device)[None, :] < block_sizes[:, None]
    out = torch.zeros(b, h, s, d, dtype=torch.float32, device=q.device)
    for bi in range(b):
        for hi in range(h):
            for qi in range(n):
                row = (bi * h + hi) * n + qi
                count = int(q2k_num[row])
                if count == 0:
                    continue
                tiles = q2k_idx[row, :count].long()
                keys = torch.cat(
                    [k[bi, hi, t * BLOCK : (t + 1) * BLOCK] for t in tiles]
                ).float()
                vals = torch.cat(
                    [v[bi, hi, t * BLOCK : (t + 1) * BLOCK] for t in tiles]
                ).float()
                mask = key_valid[tiles].reshape(-1)
                scores = (
                    q[bi, hi, qi * BLOCK : (qi + 1) * BLOCK].float() @ keys.T * sm_scale
                )
                scores = scores.masked_fill(~mask[None, :], float("-inf"))
                out[bi, hi, qi * BLOCK : (qi + 1) * BLOCK] = (
                    torch.softmax(scores, -1) @ vals
                )
    return out


@pytest.mark.parametrize("batch,heads,num_blocks,max_kv", [(1, 2, 6, 4), (2, 3, 10, 7)])
def test_matches_masked_reference(batch, heads, num_blocks, max_kv):
    generator = torch.Generator(device="cuda").manual_seed(3)
    d, s = 128, num_blocks * BLOCK
    q, k, v = (
        torch.randn(
            batch, heads, s, d, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        for _ in range(3)
    )
    block_sizes = torch.randint(
        1, BLOCK + 1, (num_blocks,), device="cuda", generator=generator
    ).int()
    block_sizes[-1] = 0
    rows = batch * heads * num_blocks
    q2k_idx = torch.stack(
        [
            torch.randperm(num_blocks, device="cuda", generator=generator)[:max_kv]
            .sort()
            .values
            for _ in range(rows)
        ]
    ).int()
    q2k_num = torch.randint(
        0, max_kv + 1, (rows,), device="cuda", generator=generator
    ).int()
    q2k_num[0] = 0
    q2k_num[-1] = max_kv
    sm_scale = 1.0 / math.sqrt(d)

    out = torch.empty_like(q)
    vsa_block_sparse_sm100(q, k, v, q2k_idx, q2k_num, block_sizes, out, sm_scale)
    torch.cuda.synchronize()
    reference = _reference(q, k, v, q2k_idx, q2k_num, block_sizes, sm_scale)
    torch.testing.assert_close(out.float(), reference, atol=2e-2, rtol=2e-2)
    assert torch.all(out[0, 0, :BLOCK] == 0)


def test_rejects_odd_tile_count():
    q = torch.randn(1, 1, 3 * BLOCK, 128, device="cuda", dtype=torch.bfloat16)
    idx = torch.zeros(3, 1, device="cuda", dtype=torch.int32)
    num = torch.ones(3, device="cuda", dtype=torch.int32)
    sizes = torch.full((3,), BLOCK, device="cuda", dtype=torch.int32)
    with pytest.raises(Exception, match="even"):
        vsa_block_sparse_sm100(q, q, q, idx, num, sizes, torch.empty_like(q), 0.1)
