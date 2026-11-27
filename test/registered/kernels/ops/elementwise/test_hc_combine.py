import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.elementwise.hc_combine import (
    hc_combine,
    hc_combine_logits,
    hc_combine_logits_norm,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

HC_COUNT = 4
HIDDEN_SIZE = 2560


def _reference_hc_combine(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    normed_residual: torch.Tensor,
    inject_weight: torch.Tensor,
    hc: int,
    hs: int,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Eager reference mirroring ``GatedResidual._combine_compute``;
    ``compute_dtype=torch.float64`` is the near-exact reference."""
    R = residual.to(compute_dtype).unflatten(-1, (hc, hs))
    gates = 2 * torch.sigmoid(
        F.linear(normed_residual.to(compute_dtype), inject_weight.to(compute_dtype))
        / hc
    )
    injection = block_output.to(compute_dtype).unsqueeze(-2) * gates.unsqueeze(-1)
    return (R + injection).flatten(-2)


def _make_inputs(
    num_tokens: int, dtype: torch.dtype, hc: int = HC_COUNT, hs: int = HIDDEN_SIZE
):
    torch.manual_seed(0)
    block_output = torch.randn(num_tokens, hs, dtype=dtype, device="cuda")
    residual = torch.randn(num_tokens, hc * hs, dtype=dtype, device="cuda")
    normed_residual = torch.randn(num_tokens, hc * hs, dtype=dtype, device="cuda")
    inject_weight = torch.randn(hc, hc * hs, dtype=dtype, device="cuda") * 0.02
    return block_output, residual, normed_residual, inject_weight


# Worst case over M in {1, 7, 128, 8192} against the fp64 reference on B300 (sm103):
# bf16 max rel err 7.8e-3 (1 ulp at a binade edge), fp16 below 1e-3 (1 ulp = 9.8e-4);
# the residual is fp32 reordering flipping the final rounding.
_TOLERANCES = {
    torch.bfloat16: dict(rtol=1e-2, atol=5e-3),
    torch.float16: dict(rtol=1e-3, atol=1e-3),
}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [1, 7, 128, 8192])
def test_hc_combine_correctness(dtype, num_tokens):
    block_output, residual, normed_residual, inject_weight = _make_inputs(
        num_tokens, dtype
    )

    out = hc_combine(
        block_output,
        residual,
        normed_residual,
        inject_weight,
        HC_COUNT,
        HIDDEN_SIZE,
    )
    expected = _reference_hc_combine(
        block_output,
        residual,
        normed_residual,
        inject_weight,
        HC_COUNT,
        HIDDEN_SIZE,
        compute_dtype=torch.float64,
    ).to(dtype)

    torch.testing.assert_close(out, expected, **_TOLERANCES[dtype])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_hc_combine_out_param(dtype):
    block_output, residual, normed_residual, inject_weight = _make_inputs(64, dtype)
    out = torch.empty_like(residual)

    result = hc_combine(
        block_output,
        residual,
        normed_residual,
        inject_weight,
        HC_COUNT,
        HIDDEN_SIZE,
        out=out,
    )
    expected = _reference_hc_combine(
        block_output,
        residual,
        normed_residual,
        inject_weight,
        HC_COUNT,
        HIDDEN_SIZE,
        compute_dtype=torch.float64,
    ).to(dtype)

    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(result, expected, **_TOLERANCES[dtype])


def test_hc_combine_3d_input():
    dtype = torch.bfloat16
    block_output, residual, normed_residual, inject_weight = _make_inputs(32, dtype)
    block_output = block_output.reshape(4, 8, HIDDEN_SIZE)
    residual = residual.reshape(4, 8, HC_COUNT * HIDDEN_SIZE)
    normed_residual = normed_residual.reshape(4, 8, HC_COUNT * HIDDEN_SIZE)

    out = hc_combine(
        block_output,
        residual,
        normed_residual,
        inject_weight,
        HC_COUNT,
        HIDDEN_SIZE,
    )
    expected = _reference_hc_combine(
        block_output,
        residual,
        normed_residual,
        inject_weight,
        HC_COUNT,
        HIDDEN_SIZE,
        compute_dtype=torch.float64,
    ).to(dtype)

    assert out.shape == residual.shape
    torch.testing.assert_close(out, expected, **_TOLERANCES[dtype])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [0, 1, 7, 128, 8243])
@pytest.mark.parametrize("strided_logits", [False, True])
def test_hc_combine_logits_and_norm_correctness(dtype, num_tokens, strided_logits):
    block_output, residual, normed_residual, inject_weight = _make_inputs(
        max(num_tokens, 1), dtype
    )
    norm_weight = torch.randn(HC_COUNT * HIDDEN_SIZE, dtype=dtype, device="cuda") * 0.1
    if num_tokens == 0:
        block_output = block_output[:0]
        residual = residual[:0]
        normed_residual = normed_residual[:0]
    gate_logits_contig = F.linear(normed_residual, inject_weight)
    if strided_logits:
        wide = torch.randn(num_tokens, 80, dtype=dtype, device="cuda")
        wide[:, 64 : 64 + HC_COUNT] = gate_logits_contig
        gate_logits = wide[:, 64 : 64 + HC_COUNT]
    else:
        gate_logits = gate_logits_contig

    out = hc_combine_logits(
        block_output, residual, gate_logits, HC_COUNT, HIDDEN_SIZE
    )
    combined, normalized = hc_combine_logits_norm(
        block_output,
        residual,
        gate_logits,
        norm_weight,
        1e-6,
        HC_COUNT,
        HIDDEN_SIZE,
    )
    assert out.shape == residual.shape
    assert combined.shape == residual.shape
    assert normalized.shape == residual.shape
    if num_tokens == 0:
        return

    gates = 2.0 * torch.sigmoid(gate_logits_contig.float() / HC_COUNT)
    ref_combined = (
        residual.float().unflatten(-1, (HC_COUNT, HIDDEN_SIZE))
        + block_output.float().unsqueeze(-2) * gates.unsqueeze(-1)
    ).flatten(-2).to(dtype)
    # Match BF16/FP16 combined rounding before per-branch GemmaRMSNorm.
    c_f32 = ref_combined.float().unflatten(-1, (HC_COUNT, HIDDEN_SIZE))
    inv_rms = torch.rsqrt(c_f32.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    ref_norm = (
        (c_f32 * inv_rms).flatten(-2) * (1.0 + norm_weight.float())
    ).to(dtype)

    torch.testing.assert_close(out, ref_combined, **_TOLERANCES[dtype])
    torch.testing.assert_close(combined, ref_combined, **_TOLERANCES[dtype])
    norm_tol = (
        {"atol": 2e-3, "rtol": 2e-3}
        if dtype == torch.float16
        else _TOLERANCES[dtype]
    )
    torch.testing.assert_close(normalized, ref_norm, **norm_tol)


def test_hc_combine_logits_non_split_shape():
    # hc_count=5, hidden_size=2048 fails HcCombineSplitKernel's static_assert
    # (256 % 160 != 0) and must compile via the standalone logits JIT module.
    torch.manual_seed(7)
    hc_count, hidden_size, rows = 5, 2048, 4
    block_output = torch.randn(rows, hidden_size, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(
        rows, hc_count * hidden_size, device="cuda", dtype=torch.bfloat16
    )
    gate_logits = torch.randn(rows, hc_count, device="cuda", dtype=torch.bfloat16)
    out = hc_combine_logits(block_output, residual, gate_logits, hc_count, hidden_size)
    a = 2.0 * torch.sigmoid(gate_logits.float() / hc_count)
    ref = (
        residual.float().view(rows, hc_count, hidden_size)
        + a.unsqueeze(-1) * block_output.float().unsqueeze(1)
    ).view(rows, hc_count * hidden_size).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def test_gated_residual_inplace_weight_reload_updates_fused_prefill():
    from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig

    cfg = HyperConnectionConfig(
        hc_count=4,
        hidden_size=256,
        params_dtype=torch.bfloat16,
        hc_lowrank=16,
        hc_per_branch_norm=True,
    )
    hc = GatedResidual(cfg).to(device="cuda", dtype=torch.bfloat16).eval()
    with torch.no_grad():
        hc.input_mix_weight_down.weight.data.fill_(-0.01)
        hc.block_inject_weight.weight.data.fill_(1.0)
        hc.input_mix_weight_up.weight.data.fill_(0.1)

    x = torch.ones(260, 4 * 256, device="cuda", dtype=torch.bfloat16) * 0.01
    mixed_before, residuals_before = hc.mix(x)
    assert residuals_before[2] is not None and residuals_before[2].ndim == 2

    # Simulate default weight loader in-place copy_ hot reload.
    with torch.no_grad():
        hc.input_mix_weight_down.weight.data.copy_(
            torch.full_like(hc.input_mix_weight_down.weight.data, 0.01)
        )
        hc.block_inject_weight.weight.data.copy_(
            torch.full_like(hc.block_inject_weight.weight.data, 7.0)
        )

    fused = hc._get_fused_down_gate_weight()
    assert torch.all(fused[:16] == torch.tensor(0.01, dtype=torch.bfloat16, device="cuda"))
    assert torch.all(fused[16:20] == 7.0)
    mixed_after, residuals_after = hc.mix(x)
    assert not torch.equal(residuals_before[2], residuals_after[2])
    assert not torch.equal(mixed_before, mixed_after)


def test_gated_residual_batch_invariant_across_256_threshold(monkeypatch):
    from sglang.srt.layers import hyperconnection as hc_mod
    from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig

    monkeypatch.setattr(hc_mod, "is_batch_invariant_mode_enabled", lambda: True)
    cfg = HyperConnectionConfig(
        hc_count=4,
        hidden_size=512,
        params_dtype=torch.bfloat16,
        hc_lowrank=16,
        hc_per_branch_norm=True,
    )
    hc = GatedResidual(cfg).to(device="cuda", dtype=torch.bfloat16).eval()
    torch.manual_seed(0)
    x_257 = torch.randn(257, 4 * 512, device="cuda", dtype=torch.bfloat16) * 0.1
    y_257 = torch.ones(257, 512, device="cuda", dtype=torch.bfloat16)

    _, res_256 = hc.mix(x_257[:256])
    _, res_257 = hc.mix(x_257)
    assert res_256[2] is None and res_257[2] is None
    out_256 = hc.combine(y_257[:256], res_256)
    out_257 = hc.combine(y_257, res_257)
    assert torch.equal(out_256[0], out_257[0])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
