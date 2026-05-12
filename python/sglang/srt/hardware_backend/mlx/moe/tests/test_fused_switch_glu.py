"""Numerical equivalence test for FusedSwitchUpGate.

Loads a small MoE model, runs one MoE block both ways (original + patched),
and asserts outputs match within fp16 tolerance.
"""

import os

import pytest

mx = pytest.importorskip("mlx.core")


# Skip if MLX not available or if running in a CI that doesn't have model cache
pytestmark = pytest.mark.skipif(
    not os.environ.get("SGLANG_MLX_TEST_MODEL"),
    reason="Set SGLANG_MLX_TEST_MODEL to a HuggingFace model id to enable",
)


def test_fused_switch_up_gate_matches_unfused():
    """Fused gather_qmm output equals separate up_proj + gate_proj outputs.

    Uses the model identified by env var SGLANG_MLX_TEST_MODEL, e.g.
    'mlx-community/Qwen1.5-MoE-A2.7B-4bit'.
    """
    from mlx_lm import load

    from sglang.srt.hardware_backend.mlx.moe.fused_switch_glu import (
        FusedSwitchUpGate,
    )

    model_id = os.environ["SGLANG_MLX_TEST_MODEL"]
    model, _ = load(model_id)

    # Get the first MoE layer's switch_mlp
    block = model.model.layers[0].mlp
    switch_mlp = block.switch_mlp

    # Save the original projections before patching mutates the layer
    orig_up = switch_mlp.up_proj
    orig_gate = switch_mlp.gate_proj

    # Build the fused module manually (don't patch the model yet)
    fused = FusedSwitchUpGate(orig_up, orig_gate)

    # Make a fake input matching what SwitchGLU expects after expand_dims:
    # shape (batch, 1, 1, input_dim)
    input_dim = orig_up.weight.shape[2] * 8  # packed -> unpacked
    x = mx.random.normal(shape=(1, 1, 1, input_dim)).astype(mx.float16)

    # Pick top_k=4 expert indices (matches Qwen1.5-MoE-A2.7B)
    num_experts = orig_up.weight.shape[0]
    top_k = min(4, num_experts)
    indices = mx.array([[0, 1, 2, 3][:top_k]])  # shape (1, top_k)

    # Run the unfused path
    x_up_orig = orig_up(x, indices, sorted_indices=False)
    x_gate_orig = orig_gate(x, indices, sorted_indices=False)

    # Run the fused path
    x_up_fused, x_gate_fused = fused(x, indices, sorted_indices=False)

    # Force materialization
    mx.eval(x_up_orig, x_gate_orig, x_up_fused, x_gate_fused)

    # Compare
    up_diff = mx.abs(x_up_orig - x_up_fused).max().item()
    gate_diff = mx.abs(x_gate_orig - x_gate_fused).max().item()

    # fp16 quantization tolerance: should be 0 since we're computing the same
    # values in the same order. Allow a tiny epsilon for any floating-point
    # noise just in case.
    assert up_diff < 1e-3, f"up_proj output mismatch: max abs diff = {up_diff}"
    assert gate_diff < 1e-3, f"gate_proj output mismatch: max abs diff = {gate_diff}"

    print(f"PASS: up max diff = {up_diff:.2e}, gate max diff = {gate_diff:.2e}")
