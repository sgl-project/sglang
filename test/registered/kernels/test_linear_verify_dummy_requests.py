"""Dummy verify requests must not read inputs or own persistent/scratch state."""

import sys
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")


@pytest.mark.parametrize("steps", [1, 6])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tree", [False, True])
def test_conv_dummy_output_is_initialized(steps, dtype, tree):
    torch.manual_seed(17)
    x = torch.randn(2, steps, 65, device="cuda", dtype=dtype).transpose(1, 2)
    x[1].fill_(float("nan"))
    state = torch.randn(4, 3, 65, device="cuda", dtype=dtype).transpose(1, 2)
    before = state.clone()
    weight = torch.randn(65, 4, device="cuda", dtype=dtype)
    indices = torch.tensor([1, -1], device="cuda", dtype=torch.int32)
    intermediate_indices = torch.tensor([0], device="cuda", dtype=torch.int32)
    window = torch.full((2, steps, 3, 65), 73.0, device="cuda", dtype=dtype).transpose(
        2, 3
    )
    reference_window = window.clone()
    tree_kwargs = {}
    if tree:
        tree_kwargs = {
            "retrieve_next_token": torch.tensor(
                [list(range(1, steps)) + [-1]], device="cuda", dtype=torch.int32
            ),
            "retrieve_next_sibling": torch.full(
                (1, steps), -1, device="cuda", dtype=torch.int32
            ),
            "retrieve_parent_token": torch.empty(
                (1, steps), device="cuda", dtype=torch.int32
            ),
        }
    reference_state = before.clone()
    reference = causal_conv1d_update(
        x[:1],
        reference_state,
        weight,
        activation="silu",
        conv_state_indices=indices[:1],
        intermediate_conv_window=reference_window,
        intermediate_state_indices=intermediate_indices,
        **tree_kwargs,
    )
    empty_like = torch.empty_like

    def poison_allocation(*args, **kwargs):
        return empty_like(*args, **kwargs).fill_(float("nan"))

    with patch("torch.empty_like", side_effect=poison_allocation):
        output = causal_conv1d_update(
            x,
            state,
            weight,
            activation="silu",
            conv_state_indices=indices,
            intermediate_conv_window=window,
            intermediate_state_indices=intermediate_indices,
            **tree_kwargs,
        )
    torch.testing.assert_close(output[:1], reference, atol=0, rtol=0)
    torch.testing.assert_close(output[1], torch.zeros_like(output[1]), atol=0, rtol=0)
    torch.testing.assert_close(state, reference_state, atol=0, rtol=0)
    torch.testing.assert_close(window, reference_window, atol=0, rtol=0)
    torch.testing.assert_close(state[[0, 2, 3]], before[[0, 2, 3]], atol=0, rtol=0)


@pytest.mark.parametrize("is_kda", [False, True])
@pytest.mark.parametrize("tree", [False, True])
@pytest.mark.parametrize("real_requests", [0, 3])
def test_recurrent_dummy_skips_inputs_and_short_metadata(is_kda, tree, real_requests):
    torch.manual_seed(23)
    physical_requests, steps, heads, dim = 4, 6, 2, 32
    tokens = physical_requests * steps
    real_tokens = real_requests * steps
    kwargs = {"device": "cuda", "dtype": torch.bfloat16}
    q, k, v = [torch.randn(1, tokens, heads, dim, **kwargs) * 0.2 for _ in range(3)]
    a = torch.randn(tokens, heads * dim if is_kda else heads, **kwargs)
    b = torch.randn(tokens, heads, **kwargs)
    for tensor in (q, k, v):
        tensor[:, real_tokens:].fill_(float("nan"))
    a[real_tokens:].fill_(float("nan"))
    b[real_tokens:].fill_(float("nan"))
    state = torch.randn(7, heads, dim, dim, device="cuda") * 0.2
    before = state.clone()
    indices = torch.tensor(
        [1, 3, 5][:real_requests] + [-1] * (physical_requests - real_requests),
        device="cuda",
        dtype=torch.int32,
    )
    # Deliberately allocate only real request rows. Run under compute-sanitizer
    # with PYTORCH_NO_CUDA_MEMORY_CACHING=1 to detect even speculative OOB reads.
    intermediate_indices = torch.arange(real_requests, device="cuda", dtype=torch.int32)
    parents = (
        torch.tensor(
            [[-1, 0, 0, 1, 1, 2]] * real_requests, device="cuda", dtype=torch.int32
        ).reshape(real_requests, steps)
        if tree
        else None
    )
    scratch = torch.full((5, steps, heads, dim, dim), 73.0, device="cuda")
    reference_scratch = scratch.clone()
    a_log = torch.randn(heads, device="cuda") * 0.2
    dt_bias = torch.randn(a.shape[-1], device="cuda") * 0.2

    def run(count, cache):
        n = count * steps
        return fused_sigmoid_gating_delta_rule_update(
            A_log=a_log,
            a=a[:n],
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q[:, :n],
            k=k[:, :n],
            v=v[:, :n],
            b=b[:n],
            initial_state_source=state,
            initial_state_indices=indices[:count],
            cu_seqlens=torch.arange(0, n + 1, steps, device="cuda", dtype=torch.int32),
            is_kda=is_kda,
            use_qk_l2norm_in_kernel=True,
            disable_state_update=True,
            intermediate_states_buffer=cache,
            intermediate_state_indices=intermediate_indices,
            cache_steps=steps,
            retrieve_parent_token=parents,
        )

    reference = run(real_requests, reference_scratch) if real_requests else None
    output = run(physical_requests, scratch)
    torch.cuda.synchronize()
    if reference is not None:
        torch.testing.assert_close(output[:, :real_tokens], reference, atol=0, rtol=0)
    torch.testing.assert_close(
        output[:, real_tokens:],
        torch.zeros_like(output[:, real_tokens:]),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(state, before, atol=0, rtol=0)
    torch.testing.assert_close(scratch, reference_scratch, atol=0, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
