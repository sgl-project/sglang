"""Correctness for the fused Qwen3.5 Gemma-RMSNorm + NeoX RoPE + gate kernel.

Covers both position contracts: the 1D ``[T]`` form the kernel was written for,
and the ``[3, T]`` mRoPE form used by the multimodal Qwen3.5 checkpoints
(Qwen3.6-27B / Qwen3.8-27B), whose image and video tokens carry genuinely
distinct temporal / height / width rows.

Regression for the case where ``[3, T]`` positions reached 1D indexing and only
the temporal row was applied, silently dropping height/width rotary dims for
visual tokens.

The reference builds cos/sin from an explicit per-dim axis map and a gather, so
it is an independent formulation rather than a transcription of the kernel's
mask arithmetic.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
    fused_qk_gemma_rmsnorm_rope_gate,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

DEV = "cuda"
DTYPE = torch.bfloat16
EPS = 1e-6
BASE = 10_000_000
MAX_POS = 512

# Qwen3.6-27B / Qwen3.8-27B geometry: head_dim 256, partial_rotary_factor 0.25.
QWEN35_HEAD_DIM, QWEN35_ROTARY_DIM = 256, 64
QWEN35_SECTION = [11, 11, 10]


def _build_cache(rotary_dim):
    inv_freq = 1.0 / (
        BASE
        ** (torch.arange(0, rotary_dim, 2, dtype=torch.float, device=DEV) / rotary_dim)
    )
    t = torch.arange(MAX_POS, dtype=torch.float, device=DEV)
    freqs = torch.outer(t, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()


def _axis_map(mrope_section, interleaved, half):
    """Which position row (0=T, 1=H, 2=W) each rotary pair draws its angle from."""
    axes = torch.zeros(half, dtype=torch.long, device=DEV)
    d = torch.arange(half, device=DEV)
    if interleaved:
        axes[(d % 3 == 1) & (d < 3 * mrope_section[1])] = 1
        axes[(d % 3 == 2) & (d < 3 * mrope_section[2])] = 2
    else:
        t_end = mrope_section[0]
        h_end = t_end + mrope_section[1]
        axes[t_end:h_end] = 1
        axes[h_end:] = 2
    return axes


def _ref_cos_sin(cache, positions, mrope_section, interleaved, half):
    cs = cache[positions].float()
    if positions.ndim == 1:
        return cs[:, :half], cs[:, half:]
    # cs: [3, T, rotary_dim] -> per-token [T, half] picked axis by axis.
    num_tokens = positions.shape[1]
    axes = _axis_map(mrope_section, interleaved, half)
    idx = axes.view(1, half, 1).expand(num_tokens, half, 1)
    cos = torch.gather(cs[..., :half].permute(1, 2, 0), 2, idx).squeeze(-1)
    sin = torch.gather(cs[..., half:].permute(1, 2, 0), 2, idx).squeeze(-1)
    return cos, sin


def _ref(
    q_gate,
    k,
    q_weight,
    k_weight,
    cache,
    positions,
    num_q_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    has_gate,
    mrope_section=None,
    mrope_interleaved=False,
):
    num_tokens = q_gate.shape[0]
    half = rotary_dim // 2
    cos, sin = _ref_cos_sin(cache, positions, mrope_section, mrope_interleaved, half)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)  # [T, 1, half]

    def gemma_norm(x, weight, num_heads):
        x = x.reshape(num_tokens, num_heads, head_dim)
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        normed = xf * torch.rsqrt(var + EPS) * (1.0 + weight.float())
        # The kernel rounds to the output dtype before applying RoPE.
        return normed.to(x.dtype).float()

    def rope(xn):
        out = xn.clone()
        x1, x2 = xn[..., :half], xn[..., half:rotary_dim]
        out[..., :half] = x1 * cos - x2 * sin
        out[..., half:rotary_dim] = x2 * cos + x1 * sin
        return out

    if has_gate:
        qg = q_gate.reshape(num_tokens, num_q_heads, 2, head_dim)
        q_in, gate = qg[:, :, 0, :].contiguous(), qg[:, :, 1, :].contiguous()
    else:
        q_in, gate = q_gate, None

    q_ref = rope(gemma_norm(q_in, q_weight, num_q_heads)).to(q_gate.dtype)
    k_ref = rope(gemma_norm(k, k_weight, num_kv_heads)).to(k.dtype)
    return q_ref.reshape(num_tokens, -1), k_ref.reshape(num_tokens, -1), gate


def _inputs(num_tokens, num_q_heads, num_kv_heads, head_dim, has_gate, seed=0):
    torch.manual_seed(seed)
    q_width = num_q_heads * (2 if has_gate else 1) * head_dim
    return (
        torch.randn(num_tokens, q_width, device=DEV, dtype=DTYPE),
        torch.randn(num_tokens, num_kv_heads * head_dim, device=DEV, dtype=DTYPE),
        torch.randn(head_dim, device=DEV, dtype=DTYPE),
        torch.randn(head_dim, device=DEV, dtype=DTYPE),
    )


def _run_case(
    positions,
    head_dim=QWEN35_HEAD_DIM,
    rotary_dim=QWEN35_ROTARY_DIM,
    num_q_heads=4,
    num_kv_heads=2,
    has_gate=True,
    mrope_section=None,
    mrope_interleaved=False,
    seed=0,
):
    num_tokens = positions.shape[-1]
    q_gate, k, q_weight, k_weight = _inputs(
        num_tokens, num_q_heads, num_kv_heads, head_dim, has_gate, seed
    )
    cache = _build_cache(rotary_dim)
    kwargs = {}
    if mrope_section is not None:
        kwargs = {
            "mrope_section": mrope_section,
            "mrope_interleaved": mrope_interleaved,
        }
    q_out, k_out, gate_out = fused_qk_gemma_rmsnorm_rope_gate(
        q_gate,
        k,
        q_weight,
        k_weight,
        cache,
        positions,
        EPS,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        has_gate=has_gate,
        **kwargs,
    )
    q_ref, k_ref, gate_ref = _ref(
        q_gate,
        k,
        q_weight,
        k_weight,
        cache,
        positions,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        has_gate,
        mrope_section,
        mrope_interleaved,
    )
    return (q_out, k_out, gate_out), (q_ref, k_ref, gate_ref)


def _assert_close(got, ref, name):
    torch.testing.assert_close(
        got.float(), ref.float(), atol=2e-2, rtol=2e-2, msg=lambda m: f"{name}: {m}"
    )


def _mrope_positions(num_tokens, seed=0):
    """[3, T] positions whose three rows genuinely differ, as for image tokens."""
    generator = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randint(
        0, MAX_POS, (3, num_tokens), device=DEV, dtype=torch.int64, generator=generator
    )


def test_positions_1d_matches_reference():
    """The pre-existing [T] contract still holds."""
    positions = torch.arange(17, device=DEV, dtype=torch.int64)
    (q, k, gate), (q_ref, k_ref, gate_ref) = _run_case(positions)
    _assert_close(q, q_ref, "q")
    _assert_close(k, k_ref, "k")
    assert torch.equal(gate, gate_ref)


def test_mrope_equal_rows_matches_1d():
    """[3, T] with identical rows must reduce exactly to the 1D result."""
    positions_1d = torch.arange(17, device=DEV, dtype=torch.int64)
    positions_3d = positions_1d.unsqueeze(0).repeat(3, 1).contiguous()
    (q_1d, k_1d, _), _ = _run_case(positions_1d)
    (q_3d, k_3d, _), _ = _run_case(
        positions_3d, mrope_section=QWEN35_SECTION, mrope_interleaved=True
    )
    assert torch.equal(q_1d, q_3d)
    assert torch.equal(k_1d, k_3d)


def test_mrope_interleaved_qwen35():
    """The reported bug: distinct T/H/W rows with Qwen3.5's interleaved sections."""
    positions = _mrope_positions(33)
    (q, k, gate), (q_ref, k_ref, gate_ref) = _run_case(
        positions, mrope_section=QWEN35_SECTION, mrope_interleaved=True
    )
    _assert_close(q, q_ref, "q")
    _assert_close(k, k_ref, "k")
    assert torch.equal(gate, gate_ref)


def test_mrope_interleaved_differs_from_temporal_only():
    """Keeps the mRoPE cases non-degenerate.

    If the sampled positions ever stopped differing across rows, the fused-vs-
    reference cases above would still pass while guarding nothing, which is
    exactly how the temporal-only bug survived text-only validation.
    """
    positions = _mrope_positions(33, seed=3)
    (q_mrope, _, _), _ = _run_case(
        positions, mrope_section=QWEN35_SECTION, mrope_interleaved=True
    )
    (q_temporal_only, _, _), _ = _run_case(positions[0].contiguous())
    assert not torch.allclose(q_mrope.float(), q_temporal_only.float(), atol=1e-3)


def test_mrope_sectioned():
    """Non-interleaved mRoPE: contiguous per-axis sections."""
    positions = _mrope_positions(21, seed=1)
    (q, k, _), (q_ref, k_ref, _) = _run_case(
        positions, mrope_section=[12, 10, 10], mrope_interleaved=False
    )
    _assert_close(q, q_ref, "q")
    _assert_close(k, k_ref, "k")


@pytest.mark.parametrize("mrope_interleaved", [True, False])
def test_mrope_padded_half_rotary(mrope_interleaved):
    """half_rotary=24 pads to ROT_HALF_BLOCK=32, exercising the masked loads."""
    positions = _mrope_positions(13, seed=2)
    (q, k, _), (q_ref, k_ref, _) = _run_case(
        positions,
        rotary_dim=48,
        mrope_section=[8, 8, 8],
        mrope_interleaved=mrope_interleaved,
    )
    _assert_close(q, q_ref, "q")
    _assert_close(k, k_ref, "k")


def test_mrope_positions_sliced_from_graph_buffer():
    """Row stride must come from the tensor, not be assumed equal to T.

    CUDA graph replay passes ``buffers.mrope_positions[:, :num_tokens]`` and two
    batch overlap passes ``[:, start:end]``, so the production tensor is a
    window into a wider buffer with a storage offset.
    """
    num_tokens, offset = 12, 8
    buffer = _mrope_positions(64, seed=7)
    positions = buffer[:, offset : offset + num_tokens]
    assert positions.stride(0) != num_tokens and positions.stride(1) == 1

    (q, k, _), _ = _run_case(
        positions, mrope_section=QWEN35_SECTION, mrope_interleaved=True
    )
    (q_contig, k_contig, _), (q_ref, k_ref, _) = _run_case(
        positions.contiguous(), mrope_section=QWEN35_SECTION, mrope_interleaved=True
    )
    assert torch.equal(q, q_contig)
    assert torch.equal(k, k_contig)
    _assert_close(q, q_ref, "q")
    _assert_close(k, k_ref, "k")


def test_no_gate():
    positions = _mrope_positions(11, seed=5)
    (q, k, gate), (q_ref, k_ref, _) = _run_case(
        positions,
        has_gate=False,
        mrope_section=QWEN35_SECTION,
        mrope_interleaved=True,
    )
    assert gate is None
    _assert_close(q, q_ref, "q")
    _assert_close(k, k_ref, "k")


def _call_with(positions, **kwargs):
    num_tokens = positions.shape[-1]
    q_gate, k, q_weight, k_weight = _inputs(num_tokens, 4, 2, QWEN35_HEAD_DIM, True)
    return fused_qk_gemma_rmsnorm_rope_gate(
        q_gate,
        k,
        q_weight,
        k_weight,
        _build_cache(QWEN35_ROTARY_DIM),
        positions,
        EPS,
        4,
        2,
        QWEN35_HEAD_DIM,
        QWEN35_ROTARY_DIM,
        **kwargs,
    )


def test_contract_2d_positions_without_mrope_section_raises():
    """The original bug: [3, T] must never be accepted as 1D positions."""
    with pytest.raises(ValueError, match="require mrope_section"):
        _call_with(_mrope_positions(8))


def test_contract_rejections():
    positions = _mrope_positions(8)
    section = {"mrope_section": QWEN35_SECTION, "mrope_interleaved": True}

    with pytest.raises(ValueError, match="must be \\[3, T\\]"):
        _call_with(positions[:2].contiguous(), **section)

    with pytest.raises(ValueError, match="require \\[3, T\\] positions"):
        _call_with(torch.arange(8, device=DEV, dtype=torch.int64), **section)

    with pytest.raises(ValueError, match="must have 3 entries"):
        _call_with(positions, mrope_section=[16, 16], mrope_interleaved=True)

    with pytest.raises(ValueError, match="!= rotary_dim // 2"):
        _call_with(positions, mrope_section=[11, 11, 11], mrope_interleaved=True)

    with pytest.raises(ValueError, match="stride\\(1\\) == 1"):
        _call_with(_mrope_positions(16)[:, ::2], **section)

    with pytest.raises(ValueError, match="GLM-interleaved"):
        _call_with(positions, mrope_interleaved_glm=True, **section)

    with pytest.raises(ValueError, match="must be \\[T\\] or \\[3, T\\]"):
        _call_with(positions.unsqueeze(0), **section)


def test_dispatch_forwards_mrope_metadata():
    """qwen3_5 must hand the kernel the metadata, not just the tensor.

    The reported bug was in this dispatch, not in the kernel, so assert the
    metadata derived from a real MRotaryEmbedding and that feeding it to the
    kernel reproduces that embedding's own RoPE.
    """
    from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
    from sglang.srt.models.qwen3_5 import _fused_mrope_kwargs

    rotary_emb = MRotaryEmbedding(
        head_size=QWEN35_HEAD_DIM,
        rotary_dim=QWEN35_ROTARY_DIM,
        max_position_embeddings=MAX_POS,
        base=BASE,
        is_neox_style=True,
        dtype=DTYPE,
        mrope_section=QWEN35_SECTION,
        mrope_interleaved=True,
    ).to(DEV)

    positions = _mrope_positions(12, seed=6)
    kwargs = _fused_mrope_kwargs(rotary_emb=rotary_emb, positions=positions)
    assert kwargs == {
        "mrope_section": QWEN35_SECTION,
        "mrope_interleaved": True,
    }

    # Cases the fused kernel must not be handed at all.
    assert (
        _fused_mrope_kwargs(rotary_emb=rotary_emb, positions=positions[:, ::2]) is None
    )
    assert (
        _fused_mrope_kwargs(
            rotary_emb=rotary_emb, positions=torch.arange(4, device=DEV)
        )
        == {}
    )
    rotary_emb.mrope_interleaved_glm = True
    assert _fused_mrope_kwargs(rotary_emb=rotary_emb, positions=positions) is None
    rotary_emb.mrope_interleaved_glm = False

    # End to end: kernel + dispatch metadata == the embedding's own RoPE.
    num_q_heads, num_kv_heads, num_tokens = 4, 2, positions.shape[1]
    q_gate, k, q_weight, k_weight = _inputs(
        num_tokens, num_q_heads, num_kv_heads, QWEN35_HEAD_DIM, True, seed=6
    )
    cache = rotary_emb.cos_sin_cache.float().contiguous()
    q_out, k_out, _ = fused_qk_gemma_rmsnorm_rope_gate(
        q_gate,
        k,
        q_weight,
        k_weight,
        cache,
        positions,
        EPS,
        num_q_heads,
        num_kv_heads,
        QWEN35_HEAD_DIM,
        QWEN35_ROTARY_DIM,
        has_gate=True,
        **kwargs,
    )

    def gemma_norm(x, weight, num_heads):
        x = x.reshape(num_tokens, num_heads, QWEN35_HEAD_DIM)
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        return (xf * torch.rsqrt(var + EPS) * (1.0 + weight.float())).to(x.dtype)

    q_in = q_gate.reshape(num_tokens, num_q_heads, 2, QWEN35_HEAD_DIM)[
        :, :, 0, :
    ].contiguous()
    q_normed = gemma_norm(q_in, q_weight, num_q_heads).reshape(num_tokens, -1)
    k_normed = gemma_norm(k, k_weight, num_kv_heads).reshape(num_tokens, -1)
    q_expected, k_expected = rotary_emb.forward_native(
        positions, q_normed.clone(), k_normed.clone()
    )
    _assert_close(q_out, q_expected, "q vs MRotaryEmbedding")
    _assert_close(k_out, k_expected, "k vs MRotaryEmbedding")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
