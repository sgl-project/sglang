"""ep_scatter_from_psum must forward expert_start/num_experts to the shared
_fwd_kernel_ep_scatter_2 exactly like ep_scatter does.

Regression guard: the psum entry point once omitted the (expert_start,
num_experts) positional arguments the kernel requires. With them missing the
call raises at launch, and had it not raised the expert-validity test
(0 <= expert_id < num_experts) would read the wrong operands. Both failure
modes are black-box observable through the scatter contract below, so this test
turns red on the pre-fix code and green on the fix.

The contract (independent of the non-deterministic atomic row assignment, since
every claim is reached through output_index rather than by fixing row order):
for each routed (token, k) slot mapping to expert e, output_index gives a valid
destination row dst, output_tensor[dst]/output_tensor_scale[dst] equal the
token's own fp8 payload/scale, and m_indices[dst] labels that row as expert e.
"""

import pytest
import torch

from sglang.kernels.ops.moe.ep_moe_kernels import ep_scatter_from_psum
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

BLOCK_E = 128
HIDDEN = 256  # multiple of the 128 quant block


def _build_inputs(num_tokens: int, num_experts: int, topk: int, seed: int):
    torch.manual_seed(seed)
    device = "cuda"

    # Each token routes to `topk` distinct experts in [0, num_experts).
    recv_topk = torch.stack(
        [torch.randperm(num_experts, device=device)[:topk] for _ in range(num_tokens)]
    ).to(torch.int64)

    valid = torch.bincount(recv_topk.flatten(), minlength=num_experts)
    # DeepGEMM contiguous layout pads every expert segment to a 128-row multiple.
    padded = ((valid + BLOCK_E - 1) // BLOCK_E) * BLOCK_E
    psum = torch.cumsum(padded, dim=0).to(torch.int32)
    all_tokens = int(psum[-1].item())

    scale_hidden = HIDDEN // 128
    recv_x = (torch.randn(num_tokens, HIDDEN, device=device) * 0.3).to(
        torch.float8_e4m3fn
    )
    recv_x_scale = torch.rand(num_tokens, scale_hidden, device=device) + 0.5

    output_tensor = torch.empty(all_tokens, HIDDEN, device=device, dtype=recv_x.dtype)
    output_tensor_scale = torch.empty(
        all_tokens, scale_hidden, device=device, dtype=torch.float32
    )
    m_indices = torch.empty(all_tokens, device=device, dtype=torch.int32)
    output_index = torch.empty_like(recv_topk)
    expert_start_loc = torch.empty_like(psum)

    return dict(
        recv_x=recv_x,
        recv_x_scale=recv_x_scale,
        recv_topk=recv_topk,
        psum_num_recv_tokens_per_expert=psum,
        expert_start_loc=expert_start_loc,
        output_tensor=output_tensor,
        output_tensor_scale=output_tensor_scale,
        m_indices=m_indices,
        output_index=output_index,
    )


@requires_cuda
@pytest.mark.parametrize("num_tokens", [64, 200, 512])
@pytest.mark.parametrize("topk", [1, 6])
def test_scatter_contract(num_tokens: int, topk: int):
    num_experts = 8
    args = _build_inputs(num_tokens, num_experts, topk, seed=num_tokens + topk)
    ep_scatter_from_psum(**args)

    recv_topk = args["recv_topk"]
    dst = args["output_index"]
    # Every routed slot must land on a real row (no -1 sentinel here: all experts
    # are local and expert_start=0, so every id is in range).
    assert torch.all(dst >= 0), dst[dst < 0]

    src_x = args["recv_x"].float()
    src_s = args["recv_x_scale"]
    out_x = args["output_tensor"].float()
    out_s = args["output_tensor_scale"]
    m_indices = args["m_indices"]

    for t in range(num_tokens):
        for k in range(topk):
            e = int(recv_topk[t, k].item())
            row = int(dst[t, k].item())
            assert torch.equal(out_x[row], src_x[t]), (t, k, e, row)
            assert torch.equal(out_s[row], src_s[t]), (t, k, e, row)
            assert int(m_indices[row].item()) == e, (t, k, e, row)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-x"]))
