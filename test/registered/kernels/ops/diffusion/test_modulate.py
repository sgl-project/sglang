"""``diffusion.modulate``: adaLN modulation, gating and timestep conditioning.

The bit-exact kernels here (``modulate_scale_shift``, ``residual_gate_add``,
``ltx2_ada_values9``, ``try_fused_scaled_residual_add_exact``) reproduce every
aten rounding boundary, so they are asserted with ``torch.equal``.  The
select-0/1 LayerNorm fusions compute their statistics differently from the
reference chain and are asserted with a tolerance.
"""

import sys

import pytest
import torch

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.kernels.ops.diffusion import (
    can_use_modulate_scale_shift_cuda,
    can_use_residual_gate_add_cuda,
    can_use_rmsnorm_scale_shift_per_token,
    fuse_layernorm_scale_shift_gate_select01_kernel,
    fuse_residual_layernorm_scale_shift_gate_select01_kernel,
    fuse_scale_shift_kernel,
    ltx2_ada_values9,
    modulate_scale_shift,
    modulate_scale_shift_cuda,
    norm_infer,
    residual_gate_add,
    residual_gate_add_cuda,
    rmsnorm_scale_shift_per_token,
    timestep_embedding,
    try_fused_scaled_residual_add_exact,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=75, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
# Nightly is not redundant: it sets SGLANG_JIT_KERNEL_RUN_FULL_TESTS=1, which
# expands the get_ci_test_range sweeps below.
register_cuda_ci(est_time=50, stage="nightly", runner_config="1-gpu-large")
register_amd_ci(est_time=38, suite="nightly-amd-kernel-1-gpu", nightly=True)

DEVICE = "cuda"


@pytest.fixture(autouse=True)
def cuda_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)


# ---------------------------------------------------------------------------
# modulate: x * (1 + scale) + shift
# ---------------------------------------------------------------------------

# FLUX.1 1024^2 adaLN shapes (D=3072) plus batched and odd-length coverage.
MODULATE_CASES = [
    (1, 4096, 3072),
    (1, 512, 3072),
    (1, 4608, 3072),
    (2, 1024, 3072),
    (1, 17, 64),
]


def _eager_modulate(x, scale, shift):
    return x * (1 + scale[:, None]) + shift[:, None]


@pytest.mark.parametrize("shape", MODULATE_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_modulate_scale_shift_matches_eager(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    scale = torch.randn((shape[0], shape[-1]), device=DEVICE, dtype=dtype)
    shift = torch.randn_like(scale)
    assert torch.equal(
        modulate_scale_shift_cuda(x, scale, shift), _eager_modulate(x, scale, shift)
    )


def test_modulate_scale_shift_accepts_adaln_chunk_views():
    # Production feeds strided ``emb.chunk(6)`` views, not fresh tensors.
    x = torch.randn((1, 4096, 3072), device=DEVICE, dtype=torch.bfloat16)
    emb = torch.randn((1, 6 * 3072), device=DEVICE, dtype=torch.bfloat16)
    shift, scale = emb.chunk(6, dim=1)[:2]
    assert can_use_modulate_scale_shift_cuda(x, scale, shift)
    assert torch.equal(
        modulate_scale_shift_cuda(x, scale, shift), _eager_modulate(x, scale, shift)
    )


def test_modulate_scale_shift_guards_reject_fp32():
    x = torch.randn((1, 64, 64), device=DEVICE, dtype=torch.float32)
    row = torch.randn((1, 64), device=DEVICE, dtype=torch.float32)
    assert not can_use_modulate_scale_shift_cuda(x, row, row)
    # The public wrapper still returns the eager result on a rejected input.
    assert torch.equal(modulate_scale_shift(x, row, row), _eager_modulate(x, row, row))


# Causal Wan and LingBot use per-frame 4D modulation with a per-token shift.
SCALE_SHIFT_4D_CASES = [
    ((1, 18, 96), 3),
    ((2, 20, 384), 4),
    ((1, 9, 1536), 3),
    ((1, 4, 5120), 2),
]


@pytest.mark.parametrize("shape,num_frames", SCALE_SHIFT_4D_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scale_constant", [0, 1])
def test_scale_shift_4d_matches_torch(shape, num_frames, dtype, scale_constant):
    batch, seq_len, hidden = shape
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    scale = torch.randn((batch, num_frames, 1, hidden), device=DEVICE, dtype=dtype)
    shift = torch.randn_like(x)

    frame_seqlen = seq_len // num_frames
    expected = (
        x.unflatten(1, (num_frames, frame_seqlen)) * (scale_constant + scale)
        + shift.unflatten(1, (num_frames, frame_seqlen))
    ).flatten(1, 2)
    actual = fuse_scale_shift_kernel(x, scale, shift, scale_constant)

    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)


# ---------------------------------------------------------------------------
# residual + gate * update
# ---------------------------------------------------------------------------

GATE_CASES = [
    ((1, 1024, 4096), (1, 1, 4096)),
    ((1, 512, 4096), (1, 512, 4096)),
    ((1, 17, 65), (1, 1, 65)),
    ((1, 17, 65), (1, 17, 65)),
    # FLUX.1 / FLUX.2-klein 1024^2 shapes (D=3072): dual-stream image/text
    # and single-stream/joint concat; gates are [1, 1, D] modulation rows.
    ((1, 4096, 3072), (1, 1, 3072)),
    ((1, 512, 3072), (1, 1, 3072)),
    ((1, 4608, 3072), (1, 1, 3072)),
    # FLUX.2-dev (D=6144) joint sequence.
    ((1, 4608, 6144), (1, 1, 6144)),
    # ERNIE-4.5-VL 1024^2 image tokens plus text tokens.
    ((1, 4216, 4096), (1, 1, 4096)),
]


def _assert_gate_add(out, ref):
    if ref.dtype == torch.float32:
        # fp32 has no rounding boundary to reproduce; the kernel keeps the
        # accumulation in fp32 and only order may differ.
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    else:
        assert torch.equal(out, ref)


@pytest.mark.parametrize("residual_shape,gate_shape", GATE_CASES)
def test_residual_gate_add_matches_torch(residual_shape, gate_shape):
    residual = torch.randn(residual_shape, device=DEVICE, dtype=torch.bfloat16)
    update = torch.randn_like(residual)
    gate = torch.randn(gate_shape, device=DEVICE, dtype=torch.bfloat16)

    ref = residual + update * gate
    _assert_gate_add(residual_gate_add_cuda(residual, update, gate), ref)
    assert torch.equal(residual_gate_add(residual, update, gate), ref)


# LingBot per-token gates are [B, S, 1]: one scalar per token, broadcast
# along the hidden dimension.
PER_TOKEN_GATE_CASES = [
    ((1, 2560, 512), (1, 2560, 1)),
    ((1, 17, 65), (1, 17, 1)),
    ((2, 33, 128), (2, 33, 1)),
]


@pytest.mark.parametrize("residual_shape,gate_shape", PER_TOKEN_GATE_CASES)
def test_residual_gate_add_per_token_matches_torch(residual_shape, gate_shape):
    residual = torch.randn(residual_shape, device=DEVICE, dtype=torch.bfloat16)
    update = torch.randn_like(residual)
    gate = torch.randn(gate_shape, device=DEVICE, dtype=torch.bfloat16)

    assert can_use_residual_gate_add_cuda(residual, update, gate)
    ref = residual + update * gate
    _assert_gate_add(residual_gate_add_cuda(residual, update, gate), ref)
    assert torch.equal(residual_gate_add(residual, update, gate), ref)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_residual_gate_add_per_token_dtypes(dtype):
    residual = torch.randn((1, 2560, 512), device=DEVICE, dtype=dtype)
    update = torch.randn_like(residual)
    gate = torch.randn((1, 2560, 1), device=DEVICE, dtype=dtype)
    _assert_gate_add(
        residual_gate_add_cuda(residual, update, gate), residual + update * gate
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("gate_shape", [(1, 1, 64), (1, 9, 64)])
def test_residual_gate_add_dtypes(dtype, gate_shape):
    residual = torch.randn((1, 9, 64), device=DEVICE, dtype=dtype)
    update = torch.randn_like(residual)
    gate = torch.randn(gate_shape, device=DEVICE, dtype=dtype)
    _assert_gate_add(
        residual_gate_add_cuda(residual, update, gate), residual + update * gate
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 17, 65), (1, 7800, 2240), (2, 33, 128)])
def test_residual_gate_add_transposed_residual(dtype, shape):
    batch, tokens, hidden_size = shape
    residual = torch.randn(
        (batch, hidden_size, tokens), device=DEVICE, dtype=dtype
    ).transpose(1, 2)
    update = torch.randn(shape, device=DEVICE, dtype=dtype)
    gate = torch.randn((1, 1, hidden_size), device=DEVICE, dtype=dtype)

    assert not residual.is_contiguous()
    assert can_use_residual_gate_add_cuda(residual, update, gate)
    ref = residual + update * gate
    out = residual_gate_add_cuda(residual, update, gate)
    _assert_gate_add(out, ref)
    assert out.stride() == ref.stride() == residual.stride()


def test_residual_gate_add_transposed_storage_offsets():
    tokens, hidden_size = 33, 128
    residual = (
        torch.randn(1 + tokens * hidden_size, device=DEVICE, dtype=torch.bfloat16)[1:]
        .view(1, hidden_size, tokens)
        .transpose(1, 2)
    )
    update = torch.randn(1 + tokens * hidden_size, device=DEVICE, dtype=torch.bfloat16)[
        1:
    ].view(1, tokens, hidden_size)
    gate = torch.randn(1 + hidden_size, device=DEVICE, dtype=torch.bfloat16)[1:].view(
        1, 1, hidden_size
    )

    assert residual.storage_offset() > 0
    assert update.storage_offset() > 0
    assert gate.storage_offset() > 0
    assert can_use_residual_gate_add_cuda(residual, update, gate)
    out = residual_gate_add_cuda(residual, update, gate)
    assert torch.equal(out, residual + update * gate)


def test_residual_gate_add_transposed_torch_compile_fullgraph():
    residual = torch.randn((1, 128, 32), device=DEVICE, dtype=torch.bfloat16).transpose(
        1, 2
    )
    update = torch.randn_like(residual, memory_format=torch.contiguous_format)
    gate = torch.randn((1, 1, 128), device=DEVICE, dtype=torch.bfloat16)
    compiled = torch.compile(residual_gate_add, fullgraph=True)
    out = compiled(residual, update, gate)
    assert torch.equal(out, residual + update * gate)
    assert out.stride() == residual.stride()


def test_residual_gate_add_transposed_cuda_graph():
    residual = torch.randn((1, 128, 32), device=DEVICE, dtype=torch.bfloat16).transpose(
        1, 2
    )
    update = torch.randn_like(residual, memory_format=torch.contiguous_format)
    gate = torch.randn((1, 1, 128), device=DEVICE, dtype=torch.bfloat16)

    # Build the JIT module before capture; graph capture must contain only the
    # allocation and kernel launch used during steady-state replay.
    residual_gate_add_cuda(residual, update, gate)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = residual_gate_add_cuda(residual, update, gate)
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(out, residual + update * gate)
    assert out.stride() == residual.stride()


def test_residual_gate_add_guards_and_eager_fallback():
    residual = torch.randn((1, 8, 64), device=DEVICE, dtype=torch.bfloat16)
    update = torch.randn_like(residual)
    gate = torch.randn((1, 1, 64), device=DEVICE, dtype=torch.bfloat16)
    assert can_use_residual_gate_add_cuda(residual, update, gate)

    rejected = [
        (residual.cpu(), update, gate),  # not on device
        (residual, update.float(), gate),  # mixed dtypes
        (residual, update[:, ::2], gate),  # strided update
        (residual, update, gate[:, :, ::2]),  # strided gate
        (residual[:, :0], update[:, :0], gate),  # empty token dim
    ]
    for args in rejected:
        assert not can_use_residual_gate_add_cuda(*args)

    # Only [1, ..., 1, D] row-broadcast gates are supported; a batched
    # [B>1, 1, D] gate is not row-broadcast here and must fall back.
    batched = torch.randn((2, 8, 64), device=DEVICE, dtype=torch.bfloat16)
    batched_update = torch.randn_like(batched)
    batched_gate = torch.randn((2, 1, 64), device=DEVICE, dtype=torch.bfloat16)
    assert not can_use_residual_gate_add_cuda(batched, batched_update, batched_gate)
    assert torch.equal(
        residual_gate_add(batched, batched_update, batched_gate),
        batched + batched_update * batched_gate,
    )


def test_residual_gate_add_torch_compile_fullgraph():
    residual = torch.randn((1, 32, 128), device=DEVICE, dtype=torch.bfloat16)
    update = torch.randn_like(residual)
    gate = torch.randn((1, 1, 128), device=DEVICE, dtype=torch.bfloat16)
    compiled = torch.compile(residual_gate_add, fullgraph=True)
    assert torch.equal(compiled(residual, update, gate), residual + update * gate)


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scaled_residual_add_is_bit_exact(dtype):
    # fp32 residual accumulator + half-precision update, as the DiT blocks
    # that keep their residual stream in fp32 emit it.
    residual = torch.randn(2, 17, 64, device=DEVICE, dtype=torch.float32)
    x = torch.randn(2, 17, 64, device=DEVICE, dtype=dtype)
    scale = torch.randn(64, device=DEVICE, dtype=torch.float32)

    actual = try_fused_scaled_residual_add_exact(residual, x, scale)
    assert actual is not None
    assert torch.equal(actual, residual + x * scale)


@torch.no_grad()
def test_scaled_residual_add_rejects_unsupported_inputs():
    residual = torch.empty(2, 3, 8, device=DEVICE, dtype=torch.float32)
    x = torch.empty_like(residual)
    scale = torch.empty(8, device=DEVICE, dtype=torch.float32)
    # A too-small hidden dim and a mismatched scale length both bail out;
    # ``try_`` returning None is this helper's documented contract.
    assert try_fused_scaled_residual_add_exact(residual, x, scale) is None
    assert try_fused_scaled_residual_add_exact(residual, x.half(), scale[:-1]) is None


# ---------------------------------------------------------------------------
# LTX-2 nine-way adaLN value split
# ---------------------------------------------------------------------------


def _ltx2_reference(scale_shift_table, timestep):
    batch, seq, _ = timestep.shape
    hidden = scale_shift_table.shape[1]
    return (
        scale_shift_table.to(device=timestep.device, dtype=timestep.dtype)
        .view(1, 1, 9, hidden)
        .add(timestep.reshape(batch, seq, 9, hidden))
        .unbind(dim=2)
    )


@torch.no_grad()
@pytest.mark.parametrize("batch,seq,hidden", [(1, 1, 4096), (2, 3, 2048)])
@pytest.mark.parametrize("table_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("compiled", [False, True])
def test_ltx2_ada_values9(batch, seq, hidden, table_dtype, compiled):
    scale_shift_table = torch.randn(9, hidden, device=DEVICE, dtype=table_dtype)
    timestep = torch.randn(batch, seq, 9 * hidden, device=DEVICE, dtype=torch.bfloat16)

    fn = (
        torch.compile(ltx2_ada_values9, fullgraph=True)
        if compiled
        else ltx2_ada_values9
    )
    actual = fn(scale_shift_table, timestep)
    expected = _ltx2_reference(scale_shift_table, timestep)

    assert len(actual) == 9
    for got, want in zip(actual, expected, strict=True):
        # Each slice must come out naturally contiguous -- that is the point
        # of the kernel; a strided slice would re-add the downstream copy.
        assert got.is_contiguous()
        assert torch.equal(got, want)


@torch.no_grad()
def test_ltx2_ada_values9_rejects_unsupported_shape():
    scale_shift_table = torch.randn(8, 4096, device=DEVICE, dtype=torch.bfloat16)
    timestep = torch.randn(1, 1, 9 * 4096, device=DEVICE, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="scale_shift_table"):
        ltx2_ada_values9(scale_shift_table, timestep)


# ---------------------------------------------------------------------------
# select-0/1 LayerNorm modulation (Qwen-Image)
# ---------------------------------------------------------------------------

SELECT01_DTYPES = get_ci_test_range(
    [torch.float16, torch.bfloat16, torch.float32], [torch.float16, torch.bfloat16]
)
SELECT01_SHAPES = get_ci_test_range(
    [(b, s, h) for b in (1, 2, 4) for s in (6, 33, 128, 257) for h in (512, 3072)],
    [(1, 6, 512), (2, 128, 3072)],
)
EPS = 1e-6


def _select01_reference(normalized, mods, index):
    scale0, shift0, gate0, scale1, shift1, gate1 = mods
    idx = index.bool().unsqueeze(-1)
    scale = torch.where(idx, scale1.unsqueeze(1), scale0.unsqueeze(1))
    shift = torch.where(idx, shift1.unsqueeze(1), shift0.unsqueeze(1))
    gate = torch.where(idx, gate1.unsqueeze(1), gate0.unsqueeze(1))
    return normalized * (1 + scale) + shift, gate


@pytest.mark.parametrize("dtype", SELECT01_DTYPES)
@pytest.mark.parametrize("shape", SELECT01_SHAPES)
@pytest.mark.parametrize("with_residual", [False, True])
def test_layernorm_scale_shift_gate_select01(dtype, shape, with_residual):
    batch_size, seq_len, hidden_size = shape
    x = torch.randn(batch_size, seq_len, hidden_size, device=DEVICE, dtype=dtype)
    weight = torch.randn(hidden_size, device=DEVICE, dtype=dtype)
    bias = torch.randn(hidden_size, device=DEVICE, dtype=dtype)
    index = torch.randint(0, 2, (batch_size, seq_len), device=DEVICE, dtype=torch.int32)
    mods = tuple(
        torch.randn(batch_size, hidden_size, device=DEVICE, dtype=dtype)
        for _ in range(6)
    )
    scale0, shift0, gate0, scale1, shift1, gate1 = mods

    if with_residual:
        residual = torch.randn_like(x)
        residual_gate = torch.randn_like(x)
        residual_ref = residual + residual_gate * x
        normalized = norm_infer(
            residual_ref.flatten(0, 1), weight, bias, eps=EPS, is_rms_norm=False
        ).view_as(residual_ref)
        out_ref, gate_ref = _select01_reference(normalized, mods, index)
        out, residual_out, gate = (
            fuse_residual_layernorm_scale_shift_gate_select01_kernel(
                x.contiguous(),
                residual=residual.contiguous(),
                residual_gate=residual_gate.contiguous(),
                weight=weight,
                bias=bias,
                scale0=scale0,
                shift0=shift0,
                gate0=gate0,
                scale1=scale1,
                shift1=shift1,
                gate1=gate1,
                index=index,
                eps=EPS,
            )
        )
    else:
        normalized = norm_infer(
            x.flatten(0, 1), weight, bias, eps=EPS, is_rms_norm=False
        ).view_as(x)
        out_ref, gate_ref = _select01_reference(normalized, mods, index)
        residual_ref = residual_out = None
        out, gate = fuse_layernorm_scale_shift_gate_select01_kernel(
            x.contiguous(),
            weight=weight,
            bias=bias,
            scale0=scale0,
            shift0=shift0,
            gate0=gate0,
            scale1=scale1,
            shift1=shift1,
            gate1=gate1,
            index=index,
            eps=EPS,
        )

    tol = 1e-5 if dtype == torch.float32 else 5e-2
    torch.testing.assert_close(out, out_ref, atol=tol, rtol=tol)
    torch.testing.assert_close(gate, gate_ref, atol=tol, rtol=tol)
    if with_residual:
        torch.testing.assert_close(residual_out, residual_ref, atol=tol, rtol=tol)


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------

TIMESTEP_BATCHES = get_ci_test_range(
    [1, 2, 8, 128, 256, 512, 1536, 2048, 4096, 11008, 16384], [1, 128, 2048, 16384]
)
TIMESTEP_DIMS = get_ci_test_range(
    [32, 128, 256, 512, 1536, 2048, 4096, 8192], [32, 512, 8192]
)
TIMESTEP_DTYPES = get_ci_test_range(
    [torch.float16, torch.bfloat16, torch.float32], [torch.float16, torch.bfloat16]
)


def timestep_embedding_reference(
    timesteps,
    dim,
    *,
    flip_sin_to_cos=False,
    downscale_freq_shift=1,
    scale=1,
    max_period=10000,
):
    """diffusers' ``get_timestep_embedding``, kept verbatim as the oracle."""
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
    timesteps = timesteps.to(torch.float32)
    half_dim = dim // 2
    exponent = -torch.log(
        torch.tensor(max_period, dtype=torch.float32, device=timesteps.device)
    ) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


@pytest.mark.parametrize("batch_size", TIMESTEP_BATCHES)
@pytest.mark.parametrize("dim", TIMESTEP_DIMS)
@pytest.mark.parametrize("dtype", TIMESTEP_DTYPES)
@pytest.mark.parametrize(
    "flip_sin_to_cos,downscale_freq_shift,scale",
    [
        (True, 0, 1),  # the sgl-diffusion default
        (False, 1, 1),  # the diffusers default
        (True, 1, 0.01),  # scaled variant used by the SD-style embedders
    ],
)
def test_timestep_embedding_matches_diffusers(
    batch_size, dim, dtype, flip_sin_to_cos, downscale_freq_shift, scale
):
    t = torch.randint(low=0, high=1000, size=(batch_size,), device=DEVICE).to(dtype)
    kwargs = dict(
        flip_sin_to_cos=flip_sin_to_cos,
        downscale_freq_shift=downscale_freq_shift,
        scale=scale,
        max_period=10000,
    )
    torch.testing.assert_close(
        timestep_embedding(t, dim, **kwargs),
        timestep_embedding_reference(t, dim, **kwargs),
        atol=1e-3,
        rtol=1e-3,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

# ---------------------------------------------------------------------------
# fused RMSNorm + per-token adaLN scale/shift (quality-gated, LingBot)
# ---------------------------------------------------------------------------


def _eager_lingbot_norm_modulate(x, weight, scale, shift, eps):
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    normed = (weight.to(torch.float32) * xf).to(x.dtype)
    return (normed * (1.0 + scale.to(torch.float32)) + shift.to(torch.float32)).to(
        x.dtype
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(1, 4813, 2048), (1, 2560, 512), (2, 33, 128)])
def test_rmsnorm_scale_shift_per_token_matches_eager(shape, dtype):
    B, S, H = shape
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    weight = torch.randn(H, device=DEVICE, dtype=torch.float32)
    # scale/shift are non-contiguous chunk views of the [B, S, 6D] modulation,
    # matching the LingBot adaLN layout the kernel is built for.
    mod = torch.randn((B, S, 6 * H), device=DEVICE, dtype=torch.float32)
    shift, scale = mod.chunk(6, dim=-1)[0], mod.chunk(6, dim=-1)[1]
    eps = 1e-6

    assert can_use_rmsnorm_scale_shift_per_token(x, weight, scale, shift)
    ref = _eager_lingbot_norm_modulate(x, weight, scale, shift, eps)
    out = rmsnorm_scale_shift_per_token(x, weight, scale, shift, eps)
    assert out.dtype == x.dtype and out.shape == x.shape
    # Not bit-exact (single fp32 pass); assert bf16/fp16 rounding tolerance.
    torch.testing.assert_close(out, ref, atol=0.13, rtol=0.02)


def test_rmsnorm_scale_shift_per_token_guards():
    B, S, H = 1, 64, 128
    x = torch.randn((B, S, H), device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(H, device=DEVICE, dtype=torch.float32)
    scale = torch.randn((B, S, H), device=DEVICE, dtype=torch.float32)
    shift = torch.randn((B, S, H), device=DEVICE, dtype=torch.float32)
    assert can_use_rmsnorm_scale_shift_per_token(x, weight, scale, shift)

    assert not can_use_rmsnorm_scale_shift_per_token(
        x.cpu(), weight, scale, shift
    )  # not on device
    assert not can_use_rmsnorm_scale_shift_per_token(
        x, weight, scale, shift[:, :, ::2]
    )  # strided rows (stride(2) != 1)
    assert not can_use_rmsnorm_scale_shift_per_token(
        x, weight, scale.float(), shift.double()
    )  # mismatched scale/shift dtype
    assert not can_use_rmsnorm_scale_shift_per_token(
        x, weight[:-1], scale, shift
    )  # weight size mismatch
