"""Cake MiniMax-H3 FC1 + SwiGLU (SM100a / SM103a) through sglang.kernels.

Checks, for the BF16, MXFP8 and NVFP4 FlashInfer entries and their weight
preparations: the registry resolves the explicit FlashInfer backend; the
facade result is bitwise identical to calling FlashInfer directly; and the
fused result matches a pure-torch reference (RMSNorm -> indexed AdaLN -> FP32
GEMM -> ``bf16(silu(gate) * up)``) within the precision's tolerance (BF16
1e-2; FP8 0.1; FP4 block-scaled atol 1.0 / rtol 0.1). The FC1 weight is drawn
with a small standard deviation so the quantized routes' error stays inside
those tolerances against the unquantized reference. Skips when FlashInfer
lacks the module or the GPU is not sm_100a / sm_103a.
"""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import diffusion_minimax_h3_proj as cake
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.diffusion.cake import (
    cake_minimax_h3_fc1_swiglu,
    cake_minimax_h3_fc1_swiglu_mxfp8,
    cake_minimax_h3_fc1_swiglu_nvfp4,
    cake_prepare_minimax_h3_fc1_weight_mxfp8,
    cake_prepare_minimax_h3_fc1_weight_nvfp4,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=600, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HIDDEN, FFN, FC1_ROWS, ADALN_ROWS, EPS = 5376, 14336, 28672, 9, 1.0e-5
ROWS = 129


@pytest.mark.parametrize(
    "op",
    [
        "diffusion.minimax_h3_fc1_swiglu",
        "diffusion.minimax_h3_fc1_swiglu_mxfp8",
        "diffusion.minimax_h3_fc1_swiglu_nvfp4",
        "diffusion.prepare_minimax_h3_fc1_weight_mxfp8",
        "diffusion.prepare_minimax_h3_fc1_weight_nvfp4",
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
    if not flashinfer_module_available(cake.FI_FC1_MODULE, cake.FI_FC1_JIT_MODULE):
        pytest.skip("installed FlashInfer lacks diffusion_ops.minimax_h3_fc1_swiglu")
    cc = torch.cuda.get_device_capability()
    if cc not in cake.ARCHS:
        pytest.skip(f"Cake MiniMax-H3 FC1+SwiGLU needs sm_100a/103a, device is {cc}")


def make_model(device, seed=4611):
    g = torch.Generator(device=device).manual_seed(seed)

    def uniform(shape, lo, hi):
        return torch.empty(shape, dtype=torch.bfloat16, device=device).uniform_(
            lo, hi, generator=g
        )

    def normal(shape, std):
        return torch.empty(shape, dtype=torch.bfloat16, device=device).normal_(
            0.0, std, generator=g
        )

    return {
        "x_norm_weight": uniform((HIDDEN,), 0.9, 1.1),
        "adaln_scale": uniform((ADALN_ROWS, HIDDEN), -0.05, 0.05),
        "adaln_shift": uniform((ADALN_ROWS, HIDDEN), -0.05, 0.05),
        # Small weights keep |gate|, |up| ~ 0.15 so FP8/FP4 noise is far inside tolerance.
        "fc1_weight": normal((FC1_ROWS, HIDDEN), 0.002),
    }


def make_inputs(rows, device, seed=4611):
    g = torch.Generator(device=device).manual_seed(seed + 7919 * rows)
    x = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device).normal_(
        0.0, 0.5, generator=g
    )
    r = torch.arange(rows, device=device)
    idx = torch.div(r * ADALN_ROWS, rows, rounding_mode="floor").clamp_max(8)
    idx = idx.to(torch.int32)
    idx[rows // 3] = -1  # device-guarded invalid row -> zero output row
    return x, idx


def reference_modulated(x, model, idx):
    norm = F.rms_norm(x, (HIDDEN,), model["x_norm_weight"], eps=EPS).to(torch.bfloat16)
    index = idx.long()
    valid = (index >= 0) & (index < ADALN_ROWS)
    safe = index.clamp(0, ADALN_ROWS - 1)
    a = torch.addcmul(
        model["adaln_shift"].index_select(0, safe),
        norm,
        (model["adaln_scale"].index_select(0, safe) + 1.0).to(torch.bfloat16),
    ).to(torch.bfloat16)
    return torch.where(valid[:, None], a, torch.zeros_like(a))


def reference(x, model, idx):
    a = reference_modulated(x, model, idx)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        h = (a.float() @ model["fc1_weight"].float().t()).to(torch.bfloat16)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    gate, up = h.chunk(2, dim=-1)
    return (F.silu(gate) * up).to(torch.bfloat16)


def test_bf16_matches_flashinfer_and_reference():
    _skip_unless_supported()
    device = torch.device("cuda")
    model = make_model(device)
    x, idx = make_inputs(ROWS, device)
    args = (x, model["x_norm_weight"], model["adaln_scale"], model["adaln_shift"], idx)
    out = torch.empty((ROWS, FFN), dtype=torch.bfloat16, device=device)
    workspace = torch.empty((ROWS, HIDDEN), dtype=torch.bfloat16, device=device)
    assert cake.supports_minimax_h3_fc1_swiglu(
        *args, model["fc1_weight"], out=out, workspace=workspace
    )
    result = cake_minimax_h3_fc1_swiglu(
        *args, model["fc1_weight"], out=out, workspace=workspace
    )
    assert result is out
    from flashinfer.diffusion_ops.minimax_h3_fc1_swiglu import minimax_h3_fc1_swiglu

    direct = minimax_h3_fc1_swiglu(*args, model["fc1_weight"])
    torch.cuda.synchronize()
    assert torch.equal(out, direct)
    expected = reference(x, model, idx)
    torch.testing.assert_close(out.float(), expected.float(), atol=1e-2, rtol=1e-2)
    assert torch.count_nonzero(out[ROWS // 3]).item() == 0


def test_mxfp8_matches_flashinfer_and_reference():
    _skip_unless_supported()
    device = torch.device("cuda")
    model = make_model(device)
    x, idx = make_inputs(ROWS, device)
    assert cake.supports_prepare_minimax_h3_fc1_weight(model["fc1_weight"])
    w_q, tiles = cake_prepare_minimax_h3_fc1_weight_mxfp8(model["fc1_weight"])
    from flashinfer.diffusion_ops.minimax_h3_fc1_swiglu import (
        minimax_h3_fc1_swiglu_mxfp8,
        prepare_minimax_h3_fc1_weight_mxfp8,
    )

    w_q_fi, tiles_fi = prepare_minimax_h3_fc1_weight_mxfp8(model["fc1_weight"])
    assert torch.equal(w_q.view(torch.uint8), w_q_fi.view(torch.uint8))
    assert torch.equal(tiles, tiles_fi)
    assert tiles.numel() == cake.MXFP8_FC1_SCALE_TILE_BYTES

    args = (x, model["x_norm_weight"], model["adaln_scale"], model["adaln_shift"], idx)
    out = torch.empty((ROWS, FFN), dtype=torch.bfloat16, device=device)
    ws_q = torch.empty((ROWS, HIDDEN), dtype=torch.float8_e4m3fn, device=device)
    ws_sf = torch.zeros(
        (cake.mxfp8_fc1_activation_scale_workspace_bytes(ROWS),),
        dtype=torch.uint8,
        device=device,
    )
    assert cake.supports_minimax_h3_fc1_swiglu_mxfp8(
        *args, w_q, tiles, out=out, workspace_q=ws_q, workspace_sf=ws_sf
    )
    result = cake_minimax_h3_fc1_swiglu_mxfp8(
        *args, w_q, tiles, out=out, workspace_q=ws_q, workspace_sf=ws_sf
    )
    assert result is out
    direct = minimax_h3_fc1_swiglu_mxfp8(*args, w_q, tiles)
    torch.cuda.synchronize()
    assert torch.equal(out, direct)
    expected = reference(x, model, idx)
    torch.testing.assert_close(out.float(), expected.float(), atol=0.1, rtol=0.1)
    assert torch.count_nonzero(out[ROWS // 3]).item() == 0


def test_nvfp4_matches_flashinfer_and_reference():
    _skip_unless_supported()
    device = torch.device("cuda")
    model = make_model(device)
    x, idx = make_inputs(ROWS, device)
    w_gs = cake.minimax_h3_nvfp4_global_scale(model["fc1_weight"])
    w_q, tiles = cake_prepare_minimax_h3_fc1_weight_nvfp4(model["fc1_weight"], w_gs)
    from flashinfer.diffusion_ops.minimax_h3_fc1_swiglu import (
        minimax_h3_fc1_swiglu_nvfp4,
        prepare_minimax_h3_fc1_weight_nvfp4,
    )

    w_q_fi, tiles_fi = prepare_minimax_h3_fc1_weight_nvfp4(model["fc1_weight"], w_gs)
    assert torch.equal(w_q, w_q_fi) and torch.equal(tiles, tiles_fi)
    assert tiles.numel() == cake.NVFP4_FC1_SCALE_TILE_BYTES

    a_ref = reference_modulated(x, model, idx)
    a_gs = cake.minimax_h3_nvfp4_global_scale(a_ref)
    alpha = cake.minimax_h3_nvfp4_alpha(a_gs, w_gs)
    args = (x, model["x_norm_weight"], model["adaln_scale"], model["adaln_shift"], idx)
    out = torch.empty((ROWS, FFN), dtype=torch.bfloat16, device=device)
    ws_q = torch.empty((ROWS, HIDDEN // 2), dtype=torch.uint8, device=device)
    ws_sf = torch.zeros(
        (cake.nvfp4_fc1_activation_scale_workspace_bytes(ROWS),),
        dtype=torch.uint8,
        device=device,
    )
    assert cake.supports_minimax_h3_fc1_swiglu_nvfp4(
        *args, a_gs, w_q, tiles, alpha, out=out, workspace_q=ws_q, workspace_sf=ws_sf
    )
    result = cake_minimax_h3_fc1_swiglu_nvfp4(
        *args, a_gs, w_q, tiles, alpha, out=out, workspace_q=ws_q, workspace_sf=ws_sf
    )
    assert result is out
    direct = minimax_h3_fc1_swiglu_nvfp4(*args, a_gs, w_q, tiles, alpha)
    torch.cuda.synchronize()
    assert torch.equal(out, direct)
    expected = reference(x, model, idx)
    torch.testing.assert_close(out.float(), expected.float(), atol=1.0, rtol=0.1)
    assert torch.count_nonzero(out[ROWS // 3]).item() == 0


def test_admission_rejects_out_of_contract():
    _skip_unless_supported()
    device = torch.device("cuda")
    model = make_model(device)
    x, idx = make_inputs(8, device)
    args = (x, model["x_norm_weight"], model["adaln_scale"], model["adaln_shift"], idx)
    assert cake.supports_minimax_h3_fc1_swiglu(*args, model["fc1_weight"])
    assert not cake.supports_minimax_h3_fc1_swiglu(*args, model["fc1_weight"], eps=1e-6)
    assert not cake.supports_minimax_h3_fc1_swiglu(*args, model["fc1_weight"][:, :8])
    assert not cake.supports_minimax_h3_fc1_swiglu(
        *args, model["fc1_weight"].to(torch.float8_e4m3fn)
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
