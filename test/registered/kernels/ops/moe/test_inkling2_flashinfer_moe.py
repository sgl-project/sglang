"""Inkling2 FlashInfer CuTe DSL MoE and DeepGEMM MegaMoE: the w13 row reorder matches FlashInfer's
own preparation, and both ops, reading the same prepared tensors, match a dequantized fp32 SiTU
reference."""

import socket

import pytest
import torch
import torch.distributed as dist
from flashinfer import block_scale_interleave
from flashinfer.fused_moe import interleave_cute_dsl_moe_weights

from sglang.kernels.arch import current_arch
from sglang.kernels.ops.moe.inkling2_flashinfer_moe import (
    Inkling2FlashInferMoE,
    up_gate_block16_rows_,
)
from sglang.kernels.ops.moe.inkling2_mega_moe import Inkling2MegaMoE
from sglang.srt.layers.quantization.utils import swizzle_blockscale
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _supported() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = current_arch()
    return arch.is_sm(10, 0) or arch.is_sm(10, 3)


pytestmark = pytest.mark.skipif(not _supported(), reason="SM100 / SM103 kernels")

NUM_EXPERTS, LOCAL, TOP_K, HIDDEN, INTERMEDIATE = 16, 8, 2, 512, 256
GATE_CAP, UP_CAP = 4.0, 25.0
FP4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _up_gate(t):
    return torch.cat([t[:, 1::2], t[:, 0::2]], 1)


def test_up_gate_block16_rows_matches_flashinfer():
    g = torch.Generator(device="cuda").manual_seed(0)
    w = torch.randint(
        0, 256, (3, 96, 40), dtype=torch.uint8, device="cuda", generator=g
    )
    s = torch.randint(0, 256, (3, 96, 5), dtype=torch.uint8, device="cuda", generator=g)
    ref_w, ref_s = interleave_cute_dsl_moe_weights(
        _up_gate(w), _up_gate(s), swap_ab=True
    )
    up_gate_block16_rows_(w)
    up_gate_block16_rows_(s)
    assert torch.equal(w, ref_w) and torch.equal(s, ref_s)


@pytest.mark.parametrize(
    "dtype", [torch.float8_e4m3fn, torch.uint8], ids=["nvfp4", "mxfp8"]
)
def test_swizzle_blockscale_matches_flashinfer(dtype):
    g = torch.Generator(device="cuda").manual_seed(2)
    s = torch.randint(0, 256, (3, 96, 5), dtype=torch.uint8, device="cuda", generator=g)
    ours = swizzle_blockscale(s.view(dtype)).view(torch.uint8)
    assert torch.equal(ours.flatten(), block_scale_interleave(s).flatten())


def _experts(nvfp4, rows, k, g):
    """Checkpoint storage: weights `[E, rows, k]` and row-major block scales, plus the fp32 dequant."""
    shape = (LOCAL, rows)
    if nvfp4:
        w = torch.randint(
            0, 256, (*shape, k // 2), dtype=torch.uint8, device="cuda", generator=g
        )
        s = torch.rand(*shape, k // 16, device="cuda", generator=g) * 0.5 + 0.5
        s = s.to(torch.float8_e4m3fn).view(torch.uint8)
        table = torch.tensor(FP4 + [-v for v in FP4], device="cuda")
        values = torch.stack(
            [table[(w & 15).long()], table[(w >> 4).long()]], -1
        ).flatten(-2)
        scale = s.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
    else:
        w = torch.randn(*shape, k, device="cuda", generator=g).to(torch.float8_e4m3fn)
        s = torch.randint(
            118, 122, (*shape, k // 32), dtype=torch.uint8, device="cuda", generator=g
        )
        values, scale = w.float(), torch.exp2(s.float() - 127).repeat_interleave(32, -1)
    return w, s, values * scale


def _reference(
    x, w13, w2, topk_weights, topk_ids, swap_gate_up=False, offset=NUM_EXPERTS - LOCAL
):
    out = torch.zeros(x.shape[0], HIDDEN, device="cuda")
    for t in range(x.shape[0]):
        for k in range(TOP_K):
            e = int(topk_ids[t, k]) - offset
            if e < 0:
                continue
            h = w13[e] @ x[t].float()
            gate, up = (h[1::2], h[0::2]) if swap_gate_up else (h[0::2], h[1::2])
            act = GATE_CAP * torch.tanh(gate / GATE_CAP) * torch.sigmoid(gate)
            act = act * UP_CAP * torch.tanh(up / UP_CAP)
            out[t] += topk_weights[t, k] * (w2[e] @ act)
    return out


def _route(g):
    """One local expert first, then any other expert (local or on another EP rank)."""
    perm = torch.randperm(NUM_EXPERTS, device="cuda", generator=g)
    local = perm[perm >= NUM_EXPERTS - LOCAL][0]
    return torch.cat([local.view(1), perm[perm != local][: TOP_K - 1]])


@pytest.mark.parametrize("nvfp4", [True, False], ids=["nvfp4", "mxfp8"])
def test_matches_reference(nvfp4):
    g = torch.Generator(device="cuda").manual_seed(1)
    w13, s13, ref13 = _experts(nvfp4, 2 * INTERMEDIATE, HIDDEN, g)
    w2, s2, ref2 = _experts(nvfp4, HIDDEN, INTERMEDIATE, g)
    alpha = torch.full((LOCAL,), 1.0 / 64.0, device="cuda") if nvfp4 else None
    if nvfp4:
        ref13, ref2 = ref13 * alpha[0], ref2 * alpha[0]
    up_gate_block16_rows_(w13)
    up_gate_block16_rows_(s13)
    moe = Inkling2FlashInferMoE(
        w13,
        swizzle_blockscale(s13),
        w2,
        swizzle_blockscale(s2),
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        local_expert_offset=NUM_EXPERTS - LOCAL,
        max_num_tokens=64,
        situ_beta=GATE_CAP,
        situ_linear_beta=UP_CAP,
        w13_alpha=alpha,
        w2_alpha=alpha,
    )
    for tokens in (1, 7, 64):
        x = torch.randn(tokens, HIDDEN, device="cuda", generator=g).bfloat16()
        ids = torch.stack([_route(g) for _ in range(tokens)])
        weights = torch.rand(tokens, TOP_K, device="cuda", generator=g)
        out = moe(x, weights, ids).float()
        ref = _reference(x, ref13, ref2, weights, ids)
        swapped = _reference(x, ref13, ref2, weights, ids, swap_gate_up=True)
        rel = ((out - ref).norm() / ref.norm()).item()
        control = ((out - swapped).norm() / swapped.norm()).item()
        bound = 0.25 if nvfp4 else 0.08
        assert rel < bound and control > 2 * bound, (
            f"{tokens} tokens: {rel=:.4f} {control=:.4f}"
        )


@pytest.fixture(scope="module")
def single_rank_group():
    if not dist.is_initialized():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        dist.init_process_group(
            "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1
        )
    return dist.group.WORLD


@pytest.mark.parametrize("nvfp4", [True, False], ids=["nvfp4", "mxfp8"])
def test_mega_moe_matches_reference(nvfp4, single_rank_group):
    g = torch.Generator(device="cuda").manual_seed(3)
    w13, s13, ref13 = _experts(nvfp4, 2 * INTERMEDIATE, HIDDEN, g)
    w2, s2, ref2 = _experts(nvfp4, HIDDEN, INTERMEDIATE, g)
    alpha = torch.full((LOCAL,), 1.0 / 64.0, device="cuda") if nvfp4 else None
    if nvfp4:
        ref13, ref2 = ref13 * alpha[0], ref2 * alpha[0]
    up_gate_block16_rows_(w13)
    up_gate_block16_rows_(s13)
    moe = Inkling2MegaMoE(
        w13,
        swizzle_blockscale(s13),
        w2,
        swizzle_blockscale(s2),
        group=single_rank_group,
        num_experts=LOCAL,
        top_k=TOP_K,
        max_num_tokens_per_rank=64,
        w13_alpha=alpha,
        w2_alpha=alpha,
    )
    for tokens in (1, 7, 64):
        x = torch.randn(tokens, HIDDEN, device="cuda", generator=g).bfloat16()
        ids = torch.stack(
            [
                torch.randperm(LOCAL, device="cuda", generator=g)[:TOP_K]
                for _ in range(tokens)
            ]
        )
        weights = torch.rand(tokens, TOP_K, device="cuda", generator=g)
        out = moe(x, weights, ids).float()
        ref = _reference(x, ref13, ref2, weights, ids, offset=0)
        swapped = _reference(x, ref13, ref2, weights, ids, swap_gate_up=True, offset=0)
        rel = ((out - ref).norm() / ref.norm()).item()
        control = ((out - swapped).norm() / swapped.norm()).item()
        bound = 0.25 if nvfp4 else 0.08
        assert rel < bound and control > 2 * bound, (
            f"{tokens} tokens: {rel=:.4f} {control=:.4f}"
        )
