import pytest
import torch

from sglang.kernels.ops.moe.ep_moe_kernels import ep_scatter_from_psum
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

dev = "cuda"
BLOCK_D = 128
BLOCK_E = 128


def _build(num_tokens, num_experts, top_k, pad_frac, expert_start, seed):
    """Route tokens to local experts, then describe the routing the way the
    DeepEPv2 permute path does: an INCLUSIVE prefix sum over local experts, and
    `recv_topk` holding expert ids offset by `expert_start` (-1 = padding)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    local_ids = torch.full((num_tokens, top_k), -1, dtype=torch.int32)
    counts = [0] * num_experts
    for t in range(num_tokens):
        experts = torch.randperm(num_experts, generator=g)[:top_k].tolist()
        for slot, e in enumerate(experts):
            if pad_frac > 0 and torch.rand(1, generator=g).item() < pad_frac:
                continue
            local_ids[t, slot] = e
            counts[e] += 1

    psum = torch.tensor(counts, dtype=torch.int32).cumsum(0).to(torch.int32)
    recv_topk = torch.where(local_ids >= 0, local_ids + expert_start, local_ids)
    return local_ids.to(dev), recv_topk.to(dev), psum.to(dev)


def _run(recv_topk, psum, hidden, dtype, expert_start):
    num_tokens = recv_topk.shape[0]
    total = int(psum[-1].item())
    scale_hidden = hidden // BLOCK_D
    is_fp8 = dtype != torch.bfloat16

    recv_x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=dev).to(dtype)
    if is_fp8:
        recv_x_scale = torch.rand(
            num_tokens, scale_hidden, dtype=torch.float32, device=dev
        )
        output_tensor_scale = torch.zeros(
            total, scale_hidden, dtype=torch.float32, device=dev
        )
    else:
        recv_x_scale = None
        output_tensor_scale = None

    output_tensor = torch.zeros(total, hidden, dtype=dtype, device=dev)
    # m_indices is padded up to BLOCK_E; -1 marks slots no expert owns.
    m_indices = torch.full(
        ((total + BLOCK_E - 1) // BLOCK_E * BLOCK_E,), -1, dtype=torch.int32, device=dev
    )
    expert_start_loc = torch.empty_like(psum)
    output_index = torch.empty_like(recv_topk)

    # Leave expert_start at its default in the local-id case, so this exercises
    # the call shape the DeepEPv2 permute path actually uses.
    kwargs = {"expert_start": expert_start} if expert_start else {}
    ep_scatter_from_psum(
        recv_x,
        recv_x_scale,
        recv_topk,
        psum,
        expert_start_loc,
        output_tensor,
        output_tensor_scale,
        m_indices,
        output_index,
        **kwargs,
    )
    return dict(
        recv_x=recv_x,
        recv_x_scale=recv_x_scale,
        output_tensor=output_tensor,
        output_tensor_scale=output_tensor_scale,
        m_indices=m_indices,
        expert_start_loc=expert_start_loc,
        output_index=output_index,
        total=total,
    )


def _check(out, local_ids, psum):
    """The scatter's destination slots are chosen by atomic_add, so the order
    within one expert is not deterministic. Assert the invariants instead — they
    pin the result down completely apart from that order."""
    output_index = out["output_index"]
    total = out["total"]
    valid = local_ids >= 0

    # 1. -1 sentinel exactly on the padding lanes. The post-permute kernel gates
    #    on this, so a stale value here silently reads a garbage row.
    assert torch.equal(output_index < 0, ~valid)

    # 2. Every valid lane lands inside its own expert's slab.
    starts = torch.cat([torch.zeros(1, dtype=psum.dtype, device=dev), psum[:-1]])
    e = local_ids[valid].long()
    dst = output_index[valid].long()
    assert torch.all(dst >= starts[e]) and torch.all(dst < psum[e])

    # 3. ... and the slabs are filled exactly, with no slot used twice.
    assert dst.numel() == total
    assert torch.equal(torch.sort(dst).values, torch.arange(total, device=dev))

    # 4. m_indices agrees with the routing, and the BLOCK_E tail stays -1.
    assert torch.equal(out["m_indices"][dst], local_ids[valid].to(torch.int32))
    assert torch.all(out["m_indices"][total:] == -1)

    # 5. expert_start_loc has been advanced to each expert's end.
    assert torch.equal(out["expert_start_loc"], psum)

    # 6. The rows themselves arrived, scales included.
    src = torch.nonzero(valid)[:, 0]
    assert torch.equal(
        out["output_tensor"][dst].view(torch.uint8),
        out["recv_x"][src].view(torch.uint8),
    )
    if out["recv_x_scale"] is not None:
        assert torch.equal(out["output_tensor_scale"][dst], out["recv_x_scale"][src])


@pytest.mark.parametrize("num_tokens", [1, 8, 256, 4096])
@pytest.mark.parametrize("pad_frac", [0.0, 0.3])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_ep_scatter_from_psum(num_tokens, pad_frac, dtype):
    num_experts, top_k, hidden = 8, 8, 512
    local_ids, recv_topk, psum = _build(
        num_tokens, num_experts, top_k, pad_frac, 0, seed=num_tokens
    )
    out = _run(recv_topk, psum, hidden, dtype, 0)
    _check(out, local_ids, psum)


@pytest.mark.parametrize("expert_start", [0, 8, 120])
def test_ep_scatter_from_psum_expert_start(expert_start):
    """`expert_start` shifts which global ids count as local. Offsetting both the
    ids and `expert_start` must not change the result, and out-of-range ids must
    be dropped like padding."""
    num_tokens, num_experts, top_k, hidden = 512, 8, 8, 512
    local_ids, recv_topk, psum = _build(
        num_tokens, num_experts, top_k, 0.2, expert_start, seed=7
    )
    out = _run(recv_topk, psum, hidden, torch.float8_e4m3fn, expert_start)
    _check(out, local_ids, psum)


def test_ep_scatter_from_psum_drops_non_local_experts():
    num_tokens, num_experts, top_k, hidden = 256, 8, 8, 512
    expert_start = 8
    local_ids, recv_topk, psum = _build(
        num_tokens, num_experts, top_k, 0.0, expert_start, seed=11
    )
    # Send half of slot 0 to an expert this rank does not own; it must be treated
    # exactly like padding, not scattered somewhere out of bounds.
    foreign = torch.arange(num_tokens, device=dev) % 2 == 0
    recv_topk[foreign, 0] = expert_start + num_experts + 3
    dropped = local_ids[foreign, 0].clone()
    local_ids[foreign, 0] = -1
    counts = torch.bincount(dropped.long(), minlength=num_experts).to(psum.dtype)
    psum = (
        (torch.cat([psum[:1], psum[1:] - psum[:-1]]) - counts).cumsum(0).to(psum.dtype)
    )

    out = _run(recv_topk, psum, hidden, torch.float8_e4m3fn, expert_start)
    _check(out, local_ids, psum)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
