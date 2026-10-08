"""Cake MiniMax-H3 gated-residual out-projection (SM100a / SM103a) through sglang.kernels.

Checks, for the BF16, MXFP8 and NVFP4 FlashInfer entries and their weight
preparations: the registry resolves the explicit FlashInfer backend; the
facade result is bitwise identical to calling FlashInfer directly; and the
fused result matches a pure-torch reference (unpack the Ulysses receive
layout, FP32 GEMM, ``bf16(residual + bf16(gate[idx] * bf16(A @ W^T)))``, gate 0
for out-of-range indices) within the precision's tolerance (BF16 1e-2; FP8
0.1; FP4 block-scaled atol 1.0 / rtol 0.1). Skips when FlashInfer lacks the
module or the GPU is not sm_100a / sm_103a.
"""

import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import diffusion_minimax_h3_proj as cake
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.diffusion.cake import (
    cake_minimax_h3_out_proj,
    cake_minimax_h3_out_proj_mxfp8,
    cake_minimax_h3_out_proj_nvfp4,
    cake_prepare_minimax_h3_o_weight_mxfp8,
    cake_prepare_minimax_h3_o_weight_nvfp4,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=600, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HIDDEN, NUM_HEADS, HEAD_DIM, ATTN_DIM, GATE_ROWS = 5376, 56, 128, 7168, 9
ROWS, DEGREE = 129, 8


@pytest.mark.parametrize(
    "op",
    [
        "diffusion.minimax_h3_out_proj",
        "diffusion.minimax_h3_out_proj_mxfp8",
        "diffusion.minimax_h3_out_proj_nvfp4",
        "diffusion.prepare_minimax_h3_o_weight_mxfp8",
        "diffusion.prepare_minimax_h3_o_weight_nvfp4",
    ],
)
def test_registry_resolves_flashinfer_backend(op):
    spec = select_kernel(op, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith(
        "sglang.kernels.cake_kernels.diffusion_minimax_h3_proj:"
    )


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(
        cake.FI_OUT_PROJ_MODULE, cake.FI_OUT_PROJ_JIT_MODULE
    ):
        pytest.skip("installed FlashInfer lacks diffusion_ops.minimax_h3_out_proj")
    cc = torch.cuda.get_device_capability()
    if cc not in cake.ARCHS:
        pytest.skip(f"Cake MiniMax-H3 out-proj needs sm_100a/103a, device is {cc}")


def make_model(device, seed=4616):
    g = torch.Generator(device=device).manual_seed(seed)
    o_weight = torch.empty((HIDDEN, ATTN_DIM), dtype=torch.bfloat16, device=device)
    o_weight.normal_(0.0, 0.005, generator=g)
    gate = torch.empty((GATE_ROWS, HIDDEN), dtype=torch.bfloat16, device=device)
    gate.uniform_(-1.0, 1.0, generator=g)
    return {"o_weight": o_weight, "gate": gate}


def make_inputs(rows, degree, device, seed=4616):
    g = torch.Generator(device=device).manual_seed(seed + 7919 * rows + 31 * degree)
    attn_out = torch.empty(
        (degree, rows, NUM_HEADS // degree, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).normal_(0.0, 0.5, generator=g)
    residual = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device).normal_(
        0.0, 0.5, generator=g
    )
    r = torch.arange(rows, device=device)
    idx = torch.div(r * GATE_ROWS, rows, rounding_mode="floor").clamp_max(GATE_ROWS - 1)
    idx = torch.where(r % 101 == 50, torch.full_like(idx, -1), idx)
    idx = torch.where(r % 103 == 60, torch.full_like(idx, GATE_ROWS), idx)
    return attn_out, idx.to(torch.int32), residual


def unpack(attn_out):
    degree, rows = attn_out.shape[0], attn_out.shape[1]
    return (
        attn_out.reshape(degree, rows, ATTN_DIM // degree)
        .permute(1, 0, 2)
        .reshape(rows, ATTN_DIM)
    )


def reference(attn_out, o_weight, gate, gate_index, residual):
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        o = (unpack(attn_out).float() @ o_weight.float().t()).to(torch.bfloat16)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    idx = gate_index.long()
    valid = (idx >= 0) & (idx < GATE_ROWS)
    g = gate.index_select(0, idx.clamp(0, GATE_ROWS - 1))
    g = torch.where(valid[:, None], g, torch.zeros_like(g))
    return (residual.float() + (g.float() * o.float()).to(torch.bfloat16).float()).to(
        torch.bfloat16
    )


def test_bf16_matches_flashinfer_and_reference():
    _skip_unless_supported()
    device = torch.device("cuda")
    model = make_model(device)
    attn_out, idx, residual = make_inputs(ROWS, DEGREE, device)
    out = torch.empty((ROWS, HIDDEN), dtype=torch.bfloat16, device=device)
    assert cake.supports_minimax_h3_out_proj(
        attn_out, model["o_weight"], model["gate"], idx, residual, out=out
    )
    result = cake_minimax_h3_out_proj(
        attn_out, model["o_weight"], model["gate"], idx, residual, out=out
    )
    assert result is out
    from flashinfer.diffusion_ops.minimax_h3_out_proj import minimax_h3_out_proj

    direct = minimax_h3_out_proj(
        attn_out, model["o_weight"], model["gate"], idx, residual
    )
    torch.cuda.synchronize()
    assert torch.equal(out, direct)
    expected = reference(attn_out, model["o_weight"], model["gate"], idx, residual)
    torch.testing.assert_close(out.float(), expected.float(), atol=1e-2, rtol=1e-2)
    # Out-of-range gate rows pass the residual through unchanged.
    assert torch.equal(out[50], residual[50]) and torch.equal(out[60], residual[60])


def test_mxfp8_matches_flashinfer_and_reference():
    _skip_unless_supported()
    device = torch.device("cuda")
    model = make_model(device)
    attn_out, idx, residual = make_inputs(ROWS, DEGREE, device)
    assert cake.supports_prepare_minimax_h3_o_weight(model["o_weight"])
    w_q, tiles = cake_prepare_minimax_h3_o_weight_mxfp8(model["o_weight"])
    from flashinfer.diffusion_ops.minimax_h3_out_proj import (
        minimax_h3_out_proj_mxfp8,
        prepare_minimax_h3_o_weight_mxfp8,
    )

    w_q_fi, tiles_fi = prepare_minimax_h3_o_weight_mxfp8(model["o_weight"])
    assert torch.equal(w_q.view(torch.uint8), w_q_fi.view(torch.uint8))
    assert torch.equal(tiles, tiles_fi)
    assert tiles.numel() == cake.MXFP8_O_SCALE_TILE_BYTES

    out = torch.empty((ROWS, HIDDEN), dtype=torch.bfloat16, device=device)
    ws_q = torch.empty((ROWS, ATTN_DIM), dtype=torch.float8_e4m3fn, device=device)
    ws_sf = torch.zeros(
        (cake.mxfp8_out_proj_activation_scale_workspace_bytes(ROWS),),
        dtype=torch.uint8,
        device=device,
    )
    assert cake.supports_minimax_h3_out_proj_mxfp8(
        attn_out, w_q, tiles, model["gate"], idx, residual,
        out=out, workspace_q=ws_q, workspace_sf=ws_sf,
    )  # fmt: skip
    result = cake_minimax_h3_out_proj_mxfp8(
        attn_out, w_q, tiles, model["gate"], idx, residual,
        out=out, workspace_q=ws_q, workspace_sf=ws_sf,
    )  # fmt: skip
    assert result is out
    direct = minimax_h3_out_proj_mxfp8(
        attn_out, w_q, tiles, model["gate"], idx, residual
    )
    torch.cuda.synchronize()
    assert torch.equal(out, direct)
    expected = reference(attn_out, model["o_weight"], model["gate"], idx, residual)
    torch.testing.assert_close(out.float(), expected.float(), atol=0.1, rtol=0.1)
    assert torch.equal(out[50], residual[50])


def test_nvfp4_matches_flashinfer_and_reference():
    _skip_unless_supported()
    device = torch.device("cuda")
    model = make_model(device)
    attn_out, idx, residual = make_inputs(ROWS, DEGREE, device)
    w_gs = cake.minimax_h3_nvfp4_global_scale(model["o_weight"])
    w_q, tiles = cake_prepare_minimax_h3_o_weight_nvfp4(model["o_weight"], w_gs)
    from flashinfer.diffusion_ops.minimax_h3_out_proj import (
        minimax_h3_out_proj_nvfp4,
        prepare_minimax_h3_o_weight_nvfp4,
    )

    w_q_fi, tiles_fi = prepare_minimax_h3_o_weight_nvfp4(model["o_weight"], w_gs)
    assert torch.equal(w_q, w_q_fi) and torch.equal(tiles, tiles_fi)
    assert tiles.numel() == cake.NVFP4_O_SCALE_TILE_BYTES

    a_gs = cake.minimax_h3_nvfp4_global_scale(attn_out)
    alpha = cake.minimax_h3_nvfp4_alpha(a_gs, w_gs)
    out = torch.empty((ROWS, HIDDEN), dtype=torch.bfloat16, device=device)
    ws_q = torch.empty((ROWS, ATTN_DIM // 2), dtype=torch.uint8, device=device)
    ws_sf = torch.zeros(
        (cake.nvfp4_out_proj_activation_scale_workspace_bytes(ROWS),),
        dtype=torch.uint8,
        device=device,
    )
    assert cake.supports_minimax_h3_out_proj_nvfp4(
        attn_out, a_gs, w_q, tiles, alpha, model["gate"], idx, residual,
        out=out, workspace_q=ws_q, workspace_sf=ws_sf,
    )  # fmt: skip
    result = cake_minimax_h3_out_proj_nvfp4(
        attn_out, a_gs, w_q, tiles, alpha, model["gate"], idx, residual,
        out=out, workspace_q=ws_q, workspace_sf=ws_sf,
    )  # fmt: skip
    assert result is out
    direct = minimax_h3_out_proj_nvfp4(
        attn_out, a_gs, w_q, tiles, alpha, model["gate"], idx, residual
    )
    torch.cuda.synchronize()
    assert torch.equal(out, direct)
    expected = reference(attn_out, model["o_weight"], model["gate"], idx, residual)
    torch.testing.assert_close(out.float(), expected.float(), atol=1.0, rtol=0.1)
    assert torch.equal(out[50], residual[50])


def test_admission_rejects_out_of_contract():
    _skip_unless_supported()
    device = torch.device("cuda")
    model = make_model(device)
    attn_out, idx, residual = make_inputs(8, 8, device)
    assert cake.supports_minimax_h3_out_proj(
        attn_out, model["o_weight"], model["gate"], idx, residual
    )
    # SM120-style [M, 7168] activation layout is not the receive layout.
    assert not cake.supports_minimax_h3_out_proj(
        unpack(attn_out).contiguous(), model["o_weight"], model["gate"], idx, residual
    )
    assert not cake.supports_minimax_h3_out_proj(
        attn_out, model["o_weight"], model["gate"], idx.long(), residual
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
