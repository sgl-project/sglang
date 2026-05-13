"""Numerical equivalence test for the Path B fused swiglu kernel.

Loads a small MoE model, runs the fused gate_qmv + silu + ×x_up kernel and
compares against the unfused reference (``mx.gather_qmm`` + ``nn.silu(gate) * x_up``).
Both the unsorted (small-batch) and sorted (large-batch) paths exercised
through the patched SwitchGLU.__call__ are covered.

Gated by SGLANG_MLX_TEST_MODEL like ``test_fused_switch_glu.py`` so CI hosts
without an MLX model cache skip it.
"""

import os

import pytest

mx = pytest.importorskip("mlx.core")


pytestmark = pytest.mark.skipif(
    not os.environ.get("SGLANG_MLX_TEST_MODEL"),
    reason="Set SGLANG_MLX_TEST_MODEL to a HuggingFace model id to enable",
)


def _max_rel_diff(a, b):
    diff = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
    max_abs = diff.max().item()
    ref_max = mx.abs(a.astype(mx.float32)).max().item()
    return max_abs, max_abs / max(ref_max, 1e-9)


def test_fused_gate_qmv_silu_mul_matches_unfused():
    """Kernel output matches ``nn.silu(gate_qmv) * x_up`` within bf16 ULP."""
    import mlx.nn as nn
    from mlx_lm import load

    from sglang.srt.hardware_backend.mlx.moe.fused_swiglu import (
        can_fuse,
        fused_gate_qmv_silu_mul,
    )

    model, _ = load(os.environ["SGLANG_MLX_TEST_MODEL"])
    sw = model.model.layers[0].mlp.switch_mlp
    assert can_fuse(sw), "layer 0 not eligible for fused swiglu"

    up = sw.up_proj
    gate = sw.gate_proj
    in_dim = up.scales.shape[-1] * up.group_size
    out_dim = up.weight.shape[-2]
    num_experts = up.weight.shape[0]
    dtype = up.scales.dtype

    # Two batch sizes both take the unsorted path (indices.size < 64).
    for B, TOPK in [(1, 8), (4, 8)]:
        x = mx.random.normal(shape=(B, 1, 1, in_dim)).astype(dtype)
        indices = mx.random.randint(0, num_experts, shape=(B, TOPK)).astype(mx.uint32)

        x_up = up(x, indices, sorted_indices=False)
        x_gate = gate(x, indices, sorted_indices=False)
        y_ref = nn.silu(x_gate) * x_up

        y_fused = fused_gate_qmv_silu_mul(
            x, gate["weight"], gate["scales"], gate.get("biases"), indices, x_up
        )
        mx.eval(y_ref, y_fused)

        assert y_ref.shape == y_fused.shape

        max_abs, rel = _max_rel_diff(y_ref, y_fused)
        # 2 % relative covers ~2 bf16 ULPs at typical activation magnitudes;
        # the kernel's fp32 accumulation order matches MLX's qmv_fast_impl so
        # most elements should land within 1 ULP.
        assert rel < 2e-2, f"B={B} TOPK={TOPK}: max_abs={max_abs:.3e} rel={rel:.2%}"


def test_patched_switchglu_matches_unpatched():
    """Full SwitchGLU forward equivalence on both sorted and unsorted paths."""
    from mlx_lm import load

    from sglang.srt.hardware_backend.mlx.moe.fused_swiglu import (
        patch_switch_glu_with_fused_swiglu,
    )

    model, _ = load(os.environ["SGLANG_MLX_TEST_MODEL"])
    sw = model.model.layers[0].mlp.switch_mlp
    in_dim = sw.up_proj.scales.shape[-1] * sw.up_proj.group_size
    num_experts = sw.up_proj.weight.shape[0]
    dtype = sw.up_proj.scales.dtype

    cases = []
    # B=2 TOPK=8 -> indices.size=16 < 64 -> unsorted
    # B=8 TOPK=8 -> indices.size=64 -> sorted
    for B, TOPK, label in [(2, 8, "unsorted"), (8, 8, "sorted")]:
        x = mx.random.normal(shape=(B, in_dim)).astype(dtype)
        indices = mx.random.randint(0, num_experts, shape=(B, TOPK)).astype(mx.uint32)
        out_ref = sw(x, indices)
        mx.eval(out_ref)
        cases.append((label, x, indices, out_ref))

    n_patched = patch_switch_glu_with_fused_swiglu(model)
    assert n_patched > 0, "no SwitchGLU layers were patched"

    for label, x, indices, out_ref in cases:
        out_fused = sw(x, indices)
        mx.eval(out_fused)
        max_abs, rel = _max_rel_diff(out_ref, out_fused)
        # 5 % is generous; in practice we see <0.6 % on 48-layer Qwen3-MoE.
        # The looser bound here absorbs cross-layer ULP propagation through
        # down_proj's quantized matmul.
        assert rel < 5e-2, f"full forward {label}: max_abs={max_abs:.3e} rel={rel:.2%}"
