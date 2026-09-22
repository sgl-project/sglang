from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.attention.fla import kda
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=180, suite="jit-kernel-unit-test-amd")


def _inputs(tokens, heads, seed, zero_state=False):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (1, tokens, heads, 128)

    def randn(*size, dtype=torch.bfloat16, scale=1.0):
        return (
            torch.randn(
                *size,
                device="cuda",
                dtype=dtype,
                generator=generator,
            )
            * scale
        )

    state = randn(
        1,
        heads,
        128,
        128,
        dtype=torch.float32,
        scale=0.01,
    )
    if zero_state:
        state.zero_()
    return {
        "q": randn(*shape),
        "k": randn(*shape),
        "v": randn(*shape, scale=0.1),
        "g": -torch.rand(
            shape,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.01,
        "beta": randn(1, tokens, heads),
        "initial_state": state,
        "initial_state_indices": torch.tensor(
            [0],
            device="cuda",
            dtype=torch.int64,
        ),
        "cu_seqlens": torch.tensor(
            [0, tokens],
            device="cuda",
            dtype=torch.int64,
        ),
        "use_qk_l2norm_in_kernel": True,
        "beta_is_raw": True,
    }


def _run(inputs, enable_fusion):
    arguments = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    state = arguments["initial_state"]
    with patch.object(
        kda,
        "_use_gfx950_glm_fused_intra",
        return_value=enable_fusion,
    ):
        output = kda.chunk_kda(**arguments)
    return output, state


@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize(
    "chunks",
    [17, 32, 33, 64, 128, 129, 256, 512, 1024, 2048],
)
def test_fused_intra_matches_current(heads, chunks):
    inputs = _inputs(chunks * 64, heads, seed=0)
    expected_output, expected_state = _run(inputs, enable_fusion=False)
    actual_output, actual_state = _run(inputs, enable_fusion=True)
    torch.testing.assert_close(
        actual_output,
        expected_output,
        atol=2e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        actual_state,
        expected_state,
        atol=2e-2,
        rtol=1e-2,
    )
    assert torch.isfinite(actual_output).all()
    assert torch.isfinite(actual_state).all()


@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("chunks", [33, 129])
def test_fused_intra_incomplete_chunk_and_zero_prefix(heads, chunks):
    inputs = _inputs(chunks * 64 - 7, heads, seed=37, zero_state=True)
    expected_output, expected_state = _run(inputs, enable_fusion=False)
    actual_output, actual_state = _run(inputs, enable_fusion=True)
    torch.testing.assert_close(
        actual_output,
        expected_output,
        atol=2e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        actual_state,
        expected_state,
        atol=2e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize("heads", [8, 16])
def test_fused_intra_cuda_graph_replay(heads):
    inputs = _inputs(17 * 64, heads, seed=11)
    state_seed = inputs["initial_state"].clone()
    output = None

    with patch.object(
        kda,
        "_use_gfx950_glm_fused_intra",
        return_value=True,
    ):
        for _ in range(2):
            inputs["initial_state"].copy_(state_seed)
            output = kda.chunk_kda(**inputs)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            inputs["initial_state"].copy_(state_seed)
            output = kda.chunk_kda(**inputs)
        graph.replay()
        torch.cuda.synchronize()
        first_output = output.clone()
        first_state = inputs["initial_state"].clone()
        graph.replay()
        torch.cuda.synchronize()

    torch.testing.assert_close(
        output,
        first_output,
        atol=2e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        inputs["initial_state"],
        first_state,
        atol=2e-2,
        rtol=1e-2,
    )
