from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.attention.fla import chunk_delta_h, kda
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="jit-kernel-unit-test-amd")


def _inputs(tokens: int, heads: int, seed: int, state_index: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (1, tokens, heads, 128)

    def randn(*size, dtype=torch.bfloat16, scale=0.02):
        return (
            torch.randn(
                *size,
                device="cuda",
                dtype=dtype,
                generator=generator,
            )
            * scale
        )

    return dict(
        q=randn(*shape),
        k=randn(*shape),
        v=randn(*shape),
        g=randn(*shape),
        beta=randn(1, tokens, heads),
        initial_state=randn(1, heads, 128, 128, scale=0.01),
        initial_state_indices=torch.tensor(
            [state_index],
            device="cuda",
            dtype=torch.int64,
        ),
        cu_seqlens=torch.tensor(
            [0, tokens],
            device="cuda",
            dtype=torch.int64,
        ),
        A_log=randn(1, 1, heads, 1, dtype=torch.float32, scale=0.01),
        dt_bias=randn(heads * 128, dtype=torch.float32, scale=0.01),
    )


def _run(inputs, *, enable_fusion):
    arguments = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    state = arguments["initial_state"]
    with patch.object(
        chunk_delta_h,
        "is_gfx95_supported",
        return_value=enable_fusion,
    ):
        output = kda.chunk_kda(
            **arguments,
            use_qk_l2norm_in_kernel=True,
        )
    return output, state


@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("tokens", [128, 192, 256])
@pytest.mark.parametrize("state_index", [0, -1])
@pytest.mark.parametrize("seed", [0, 37])
def test_fused_kda_state_output_matches_current(
    heads,
    tokens,
    state_index,
    seed,
):
    inputs = _inputs(tokens, heads, seed, state_index)
    expected_output, expected_state = _run(inputs, enable_fusion=False)
    actual_output, actual_state = _run(inputs, enable_fusion=True)

    torch.testing.assert_close(
        actual_output,
        expected_output,
        atol=2e-5,
        rtol=2e-3,
    )
    torch.testing.assert_close(actual_state, expected_state, atol=0, rtol=0)
    assert torch.isfinite(actual_output).all()


def test_long_context_keeps_current_path():
    inputs = _inputs(tokens=8192, heads=8, seed=0, state_index=0)
    with patch.object(
        kda,
        "chunk_gated_delta_rule_fwd_o_128",
    ) as fused:
        output = kda.chunk_kda(
            **inputs,
            use_qk_l2norm_in_kernel=True,
        )
    fused.assert_not_called()
    assert torch.isfinite(output).all()


def test_tracked_call_keeps_current_path():
    inputs = _inputs(tokens=128, heads=8, seed=0, state_index=0)
    track_state = torch.empty(
        (1, 8, 128, 128),
        device="cuda",
        dtype=torch.float32,
    )
    track_chunk_idx = torch.tensor([0], device="cuda", dtype=torch.int64)
    with patch.object(
        kda,
        "chunk_gated_delta_rule_fwd_o_128",
    ) as fused:
        output, intermediate = kda.chunk_kda(
            **inputs,
            use_qk_l2norm_in_kernel=True,
            output_intermediate_states=True,
            track_state=track_state,
            track_chunk_idx=track_chunk_idx,
        )
    fused.assert_not_called()
    assert output.shape == inputs["v"].shape
    assert intermediate.shape == (1, 2, 8, 128, 128)


@pytest.mark.parametrize("layout", ["fixed", "ragged"])
def test_unsupported_layout_keeps_current_path(layout):
    inputs = _inputs(tokens=128, heads=8, seed=0, state_index=0)
    if layout == "fixed":
        inputs["cu_seqlens"] = None
    else:
        inputs["cu_seqlens"] = torch.tensor(
            [0, 48, 128],
            device="cuda",
            dtype=torch.int64,
        )
        inputs["initial_state"] = inputs["initial_state"].repeat(2, 1, 1, 1)
        inputs["initial_state_indices"] = torch.tensor(
            [0, 1],
            device="cuda",
            dtype=torch.int64,
        )
    with patch.object(
        kda,
        "chunk_gated_delta_rule_fwd_o_128",
    ) as fused:
        output = kda.chunk_kda(
            **inputs,
            use_qk_l2norm_in_kernel=True,
        )
    fused.assert_not_called()
    assert torch.isfinite(output).all()
