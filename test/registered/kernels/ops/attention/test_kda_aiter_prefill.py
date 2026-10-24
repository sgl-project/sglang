from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.attention.fla import chunk_kda_aiter
from sglang.kernels.ops.attention.fla.chunk_kda_aiter import (
    aiter_chunk_kda_fwd,
    can_use_aiter_chunk_kda,
)
from sglang.kernels.ops.attention.fla.kda import chunk_kda
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, suite="jit-kernel-unit-test-amd")

requires_aiter_chunk_kda = pytest.mark.skipif(
    chunk_kda_aiter._get_aiter_chunk_kda() is None,
    reason="requires merged ROCm/aiter#5866 and ROCm/aiter#6672",
)


def _inputs(lengths, heads, seed=0, state_indices=None):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    total = sum(lengths)

    def randn(*shape, dtype=torch.bfloat16, scale=0.02):
        return (
            torch.randn(
                *shape,
                device="cuda",
                dtype=dtype,
                generator=generator,
            )
            * scale
        )

    shape = (1, total, heads, 128)
    if state_indices is None:
        state_indices = list(range(len(lengths)))
    return {
        "q": randn(*shape),
        "k": randn(*shape),
        "v": randn(*shape),
        "g": randn(*shape),
        "beta": randn(1, total, heads),
        "A_log": torch.log(
            torch.empty(heads, device="cuda").uniform_(1, 16, generator=generator)
        ),
        "dt_bias": randn(heads * 128, dtype=torch.float32),
        "initial_state": randn(
            max(len(lengths), max(state_indices) + 1),
            heads,
            128,
            128,
            scale=0.01,
        ),
        "initial_state_indices": torch.tensor(
            state_indices,
            device="cuda",
            dtype=torch.int64,
        ),
        "cu_seqlens": torch.tensor(
            [0] + torch.tensor(lengths).cumsum(0).tolist(),
            device="cuda",
            dtype=torch.int64,
        ),
    }


def _clone(inputs):
    return {key: value.clone() for key, value in inputs.items()}


def _err_ratio(expected, actual):
    expected = expected.double()
    actual = actual.double()
    return (
        (expected - actual).square().mean().sqrt()
        / (expected.square().mean().sqrt() + 1e-12)
    ).item()


@requires_aiter_chunk_kda
@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("lengths", [[1], [63, 64], [65, 127, 192]])
@pytest.mark.parametrize("seed", [0, 37])
def test_aiter_chunk_kda_matches_current(heads, lengths, seed):
    inputs = _inputs(lengths, heads, seed)
    expected_args = _clone(inputs)
    expected_output = chunk_kda(
        **expected_args,
        lower_bound=-5.0,
        beta_is_raw=True,
        use_qk_l2norm_in_kernel=True,
    )
    actual_args = _clone(inputs)
    actual_output = aiter_chunk_kda_fwd(
        **actual_args,
        lower_bound=-5.0,
        scale=128**-0.5,
    )
    assert torch.isfinite(actual_output).all()
    assert _err_ratio(expected_output, actual_output) < 0.008
    valid = actual_args["initial_state_indices"] >= 0
    assert (
        _err_ratio(
            expected_args["initial_state"][actual_args["initial_state_indices"][valid]],
            actual_args["initial_state"][actual_args["initial_state_indices"][valid]],
        )
        < 0.008
    )


@requires_aiter_chunk_kda
def test_aiter_chunk_kda_invalid_state_index():
    inputs = _inputs([65, 127], heads=8, seed=11, state_indices=[0, -1])
    before = inputs["initial_state"].clone()
    expected_args = _clone(inputs)
    expected_output = chunk_kda(
        **expected_args,
        lower_bound=-5.0,
        beta_is_raw=True,
        use_qk_l2norm_in_kernel=True,
    )
    actual_args = _clone(inputs)
    actual_output = aiter_chunk_kda_fwd(
        **actual_args,
        lower_bound=-5.0,
        scale=128**-0.5,
    )
    assert _err_ratio(expected_output, actual_output) < 0.008
    assert torch.equal(actual_args["initial_state"][1], before[1])


@requires_aiter_chunk_kda
def test_aiter_chunk_kda_deterministic():
    inputs = _inputs([65, 127, 192], heads=8, seed=23)
    first_args = _clone(inputs)
    first_output = aiter_chunk_kda_fwd(
        **first_args,
        lower_bound=-5.0,
        scale=128**-0.5,
    )
    second_args = _clone(inputs)
    second_output = aiter_chunk_kda_fwd(
        **second_args,
        lower_bound=-5.0,
        scale=128**-0.5,
    )
    assert torch.equal(first_output, second_output)
    assert torch.equal(first_args["initial_state"], second_args["initial_state"])


@requires_aiter_chunk_kda
def test_aiter_chunk_kda_graph_replay():
    args = _inputs([65, 127], heads=8, seed=29)
    value_seed = args["v"].clone()
    state_seed = args["initial_state"].clone()
    for _ in range(3):
        args["v"].copy_(value_seed)
        args["initial_state"].copy_(state_seed)
        aiter_chunk_kda_fwd(
            **args,
            lower_bound=-5.0,
            scale=128**-0.5,
        )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        args["v"].copy_(value_seed)
        args["initial_state"].copy_(state_seed)
        output = aiter_chunk_kda_fwd(
            **args,
            lower_bound=-5.0,
            scale=128**-0.5,
        )
    graph.replay()
    torch.cuda.synchronize()
    first_output = output.clone()
    first_state = args["initial_state"].clone()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output, first_output)
    assert torch.equal(args["initial_state"], first_state)


def _dispatch_args(heads, sequences):
    q = torch.empty((1, 16384, heads, 128), device="meta", dtype=torch.bfloat16)
    return {
        "q": q,
        "k": torch.empty_like(q),
        "v": torch.empty_like(q),
        "g": torch.empty_like(q),
        "beta": torch.empty((1, 16384, heads), device="meta", dtype=torch.bfloat16),
        "A_log": torch.empty(heads, device="meta"),
        "dt_bias": torch.empty(heads * 128, device="meta"),
        "lower_bound": -5.0,
        "initial_state": torch.empty(
            sequences,
            heads,
            128,
            128,
            device="meta",
            dtype=torch.bfloat16,
        ),
        "initial_state_indices": torch.arange(
            sequences,
            device="meta",
            dtype=torch.int64,
        ),
        "cu_seqlens": torch.empty(
            sequences + 1,
            device="meta",
            dtype=torch.int64,
        ),
        "use_qk_l2norm_in_kernel": True,
        "output_intermediate_states": False,
        "beta_is_raw": True,
    }


@pytest.mark.parametrize("heads,sequences", [(16, 3), (8, 2), (8, 3), (8, 4)])
def test_dispatch_supported_cells(heads, sequences):
    with ExitStack() as stack:
        stack.enter_context(envs.SGLANG_OPT_KDA_FUSED_PREFILL.override(True))
        stack.enter_context(
            patch.object(chunk_kda_aiter, "is_gfx95_supported", return_value=True)
        )
        stack.enter_context(
            patch.object(chunk_kda_aiter, "_get_aiter_chunk_kda", return_value=object())
        )
        assert can_use_aiter_chunk_kda(**_dispatch_args(heads, sequences))


@pytest.mark.parametrize(
    "mutation",
    [
        "disabled",
        "wrong_heads",
        "wrong_tokens",
        "no_safe_gate",
        "intermediate",
        "beta_not_raw",
        "no_aiter",
    ],
)
def test_dispatch_fallback(mutation):
    args = _dispatch_args(8, 3)
    enabled = mutation != "disabled"
    available = mutation != "no_aiter"
    if mutation == "wrong_heads":
        args = _dispatch_args(4, 3)
    elif mutation == "wrong_tokens":
        args["q"] = torch.empty(
            (1, 8192, 8, 128),
            device="meta",
            dtype=torch.bfloat16,
        )
    elif mutation == "no_safe_gate":
        args["lower_bound"] = None
    elif mutation == "intermediate":
        args["output_intermediate_states"] = True
    elif mutation == "beta_not_raw":
        args["beta_is_raw"] = False
    with ExitStack() as stack:
        stack.enter_context(envs.SGLANG_OPT_KDA_FUSED_PREFILL.override(enabled))
        stack.enter_context(
            patch.object(chunk_kda_aiter, "is_gfx95_supported", return_value=True)
        )
        stack.enter_context(
            patch.object(
                chunk_kda_aiter,
                "_get_aiter_chunk_kda",
                return_value=(object() if available else None),
            )
        )
        assert not can_use_aiter_chunk_kda(**args)
