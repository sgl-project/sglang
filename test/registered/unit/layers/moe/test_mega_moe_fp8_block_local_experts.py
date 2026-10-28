import socket
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HIDDEN, INTER, NUM_LOCAL, TOPK, TOKENS, BLOCK, CLAMP = 4096, 1536, 8, 8, 64, 128, 7.0


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def group():
    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{_free_port()}", rank=0, world_size=1
    )
    yield dist.group.WORLD
    from sglang.srt.layers.moe import mega_moe

    mega_moe._MEGA_MOE_SYMM_BUFFER.clear()
    dist.destroy_process_group()


def _block_fp8(num_experts, n, k):
    w = torch.randn(num_experts, n, k, device="cuda") * 0.02
    blocks = w.view(num_experts, n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    scale = blocks.abs().amax(dim=(2, 4)).clamp(min=1e-8) / 448.0
    q = (blocks / scale[:, :, None, :, None]).view_as(w).to(torch.float8_e4m3fn)
    return q, scale


def _dequant(q, scale):
    s = scale.repeat_interleave(BLOCK, 1).repeat_interleave(BLOCK, 2)
    return q.float() * s


def _experts(moe_ep_rank):
    w13, w13_s = _block_fp8(NUM_LOCAL, 2 * INTER, HIDDEN)
    w2, w2_s = _block_fp8(NUM_LOCAL, HIDDEN, INTER)
    ref = (_dequant(w13, w13_s), _dequant(w2, w2_s))
    experts = SimpleNamespace(
        w13_weight=SimpleNamespace(data=w13),
        w2_weight=SimpleNamespace(data=w2),
        w13_weight_scale_inv=SimpleNamespace(data=w13_s),
        w2_weight_scale_inv=SimpleNamespace(data=w2_s),
        quant_method=SimpleNamespace(
            quant_config=SimpleNamespace(weight_block_size=[BLOCK, BLOCK])
        ),
        num_local_experts=NUM_LOCAL,
        moe_ep_rank=moe_ep_rank,
        intermediate_size_per_partition=INTER,
    )
    return experts, ref


def _reference(x, ids, weights, ref_w13, ref_w2, offset):
    out = torch.zeros(x.shape[0], HIDDEN, device="cuda")
    for k in range(ids.shape[1]):
        local = ids[:, k].long() - offset
        mask = (local >= 0) & (local < NUM_LOCAL)
        if not mask.any():
            continue
        e = local[mask]
        h = torch.einsum("th,tnh->tn", x[mask].float(), ref_w13[e])
        g, u = h.chunk(2, -1)
        a = F.silu(g).clamp(max=CLAMP) * u.clamp(-CLAMP, CLAMP)
        out[mask] += weights[mask, k : k + 1] * torch.einsum("tn,thn->th", a, ref_w2[e])
    return out


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="Requires SM100-family GPU",
)
@pytest.mark.parametrize("moe_ep_rank", [0, 1])
def test_block_fp8_local_experts_match_reference(group, moe_ep_rank):
    from sglang.srt.layers.moe.mega_moe import (
        build_mega_moe_fp8_block_experts_weights,
        forward_mega_moe_local_experts,
    )

    torch.manual_seed(moe_ep_rank)
    experts, (ref_w13, ref_w2) = _experts(moe_ep_rank)
    build_mega_moe_fp8_block_experts_weights(experts)
    assert experts.mega_moe_mma_type == "fp8xfp8"

    num_global = 4 * NUM_LOCAL
    x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16)
    ids = torch.stack(
        [torch.randperm(num_global, device="cuda")[:TOPK] for _ in range(TOKENS)]
    ).to(torch.int32)
    weights = torch.rand(TOKENS, TOPK, device="cuda")
    offset = moe_ep_rank * NUM_LOCAL
    local_hits = ((ids >= offset) & (ids < offset + NUM_LOCAL)).sum().item()
    assert 0 < local_hits < ids.numel()

    y = forward_mega_moe_local_experts(
        experts, x, ids, weights, group=group, activation_clamp=CLAMP
    ).float()
    ref = _reference(x, ids, weights, ref_w13, ref_w2, offset)

    untouched = ~((ids >= offset) & (ids < offset + NUM_LOCAL)).any(dim=1)
    assert torch.all(y[untouched] == 0)
    assert F.cosine_similarity(y.flatten(), ref.flatten(), dim=0).item() > 0.995
    assert ((y - ref).norm() / ref.norm()).item() < 0.1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
