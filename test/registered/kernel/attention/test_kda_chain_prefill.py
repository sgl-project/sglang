from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.attention.fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from sglang.kernels.ops.attention.fla.chunk_kda_chain_gfx950 import (
    can_use_kda_chain_prefill,
    chunk_kda_chain_fwd,
)
from sglang.kernels.ops.attention.fla.index import prepare_chunk_indices
from sglang.kernels.ops.attention.fla.kda import chunk_gla_fwd_o_gk
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=240, suite="jit-kernel-unit-test-amd")

OUTPUT_ATOL = 6.2e-5
OUTPUT_RTOL = 8e-3


def _inputs(lengths, heads, seed=0, state_indices=None, envelope=False):
    total = sum(lengths)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (1, total, heads, 128)

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

    if envelope:
        pitch = heads * 128 * 128 + 257
        backing = randn(len(lengths), pitch, scale=0.01)
        state = torch.as_strided(
            backing,
            (len(lengths), heads, 128, 128),
            (pitch, 128 * 128, 128, 1),
        )
    else:
        backing = None
        state = randn(len(lengths), heads, 128, 128, scale=0.01)
    if state_indices is None:
        state_indices = list(range(len(lengths)))
    cu_seqlens = torch.tensor(
        [0] + torch.tensor(lengths).cumsum(0).tolist(),
        device="cuda",
        dtype=torch.int64,
    )
    return {
        "q": randn(*shape),
        "k": randn(*shape),
        "v": randn(*shape),
        "w": randn(*shape),
        "gk": randn(*shape, dtype=torch.float32, scale=0.001),
        "A": randn(1, total, heads, 64),
        "initial_state": state,
        "initial_state_indices": torch.tensor(
            state_indices,
            device="cuda",
            dtype=torch.int64,
        ),
        "cu_seqlens": cu_seqlens,
        "backing": backing,
    }


def _clone_inputs(inputs):
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
        if key != "backing"
    }


def _reference(inputs):
    args = _clone_inputs(inputs)
    chunk_indices = prepare_chunk_indices(args["cu_seqlens"], 64)
    h, v_new = chunk_gated_delta_rule_fwd_h(
        k=args["k"],
        w=args["w"],
        u=args["v"],
        gk=args["gk"],
        initial_state=args["initial_state"],
        initial_state_indices=args["initial_state_indices"],
        cu_seqlens=args["cu_seqlens"],
        chunk_indices=chunk_indices,
        use_exp2=True,
    )
    output = chunk_gla_fwd_o_gk(
        q=args["q"],
        v=v_new,
        g=args["gk"],
        A=args["A"],
        h=h,
        o=args["v"],
        scale=128**-0.5,
        cu_seqlens=args["cu_seqlens"],
        chunk_indices=chunk_indices,
    )
    return output, args["initial_state"], h


def _chain(inputs, *, track_chunk_idx=None, **config):
    args = _clone_inputs(inputs)
    track_state = None
    if track_chunk_idx is not None:
        track_state = torch.full(
            (len(track_chunk_idx), args["q"].shape[-2], 128, 128),
            float("nan"),
            device="cuda",
            dtype=torch.float32,
        )
        track_chunk_idx = torch.tensor(
            track_chunk_idx,
            device="cuda",
            dtype=torch.int64,
        )
    output = chunk_kda_chain_fwd(
        **args,
        scale=128**-0.5,
        track_state=track_state,
        track_chunk_idx=track_chunk_idx,
        **config,
    )
    return output, args["initial_state"], track_state


@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize(
    "lengths",
    [[1], [63, 64], [65, 127, 64], [4096, 4096, 4096, 4096]],
)
def test_chain_matches_current(heads, lengths):
    seed = 37
    inputs = _inputs(lengths, heads, seed)
    expected_o, expected_state, _ = _reference(inputs)
    actual_o, actual_state, _ = _chain(inputs)
    torch.testing.assert_close(actual_o, expected_o, atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL)
    torch.testing.assert_close(actual_state, expected_state, atol=0, rtol=0)
    assert torch.isfinite(actual_o).all()


def test_minus_one_index_and_envelope_stride():
    inputs = _inputs(
        [65, 127, 64],
        heads=8,
        seed=11,
        state_indices=[0, -1, 2],
        envelope=True,
    )
    expected_o, expected_state, _ = _reference(inputs)
    actual_o, actual_state, _ = _chain(inputs)
    torch.testing.assert_close(actual_o, expected_o, atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL)
    torch.testing.assert_close(actual_state, expected_state, atol=0, rtol=0)


def test_track_snapshot_matches_chunk_state():
    lengths = [65, 127, 192]
    track_indices = [0, 1, 2]
    inputs = _inputs(lengths, heads=8, seed=5)
    expected_o, expected_state, h = _reference(inputs)
    actual_o, actual_state, track_state = _chain(
        inputs,
        track_chunk_idx=track_indices,
    )
    offsets = [0]
    for length in lengths[:-1]:
        offsets.append(offsets[-1] + (length + 63) // 64)
    expected_track = torch.stack(
        [h[0, offset + index] for offset, index in zip(offsets, track_indices)]
    )
    torch.testing.assert_close(actual_o, expected_o, atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL)
    torch.testing.assert_close(actual_state, expected_state, atol=0, rtol=0)
    # The chain snapshots its FP32 accumulator before the state-pool round.
    assert torch.equal(track_state.to(torch.bfloat16), expected_track)


def test_deterministic():
    inputs = _inputs([5462, 5461, 5461], heads=8, seed=23)
    first_o, first_state, _ = _chain(inputs, bv=16, num_warps=4, num_stages=3)
    second_o, second_state, _ = _chain(inputs, bv=16, num_warps=4, num_stages=3)
    assert torch.equal(first_o, second_o)
    assert torch.equal(first_state, second_state)


def test_graph_replay():
    inputs = _inputs([65, 127], heads=8, seed=29)
    args = _clone_inputs(inputs)
    state_seed = args["initial_state"].clone()
    value_seed = args["v"].clone()
    for _ in range(3):
        args["initial_state"].copy_(state_seed)
        args["v"].copy_(value_seed)
        chunk_kda_chain_fwd(**args, scale=128**-0.5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        args["initial_state"].copy_(state_seed)
        args["v"].copy_(value_seed)
        output = chunk_kda_chain_fwd(**args, scale=128**-0.5)
    graph.replay()
    torch.cuda.synchronize()
    first_output = output.clone()
    first_state = args["initial_state"].clone()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output, first_output)
    assert torch.equal(args["initial_state"], first_state)


@pytest.mark.parametrize(
    "heads,num_sequences", [(8, 2), (8, 3), (8, 4), (16, 3), (16, 4)]
)
def test_dispatch_supported_cells(heads, num_sequences):
    lengths = [16384 // num_sequences] * num_sequences
    lengths[-1] += 16384 - sum(lengths)
    args = _clone_inputs(_inputs(lengths, heads))
    with ExitStack() as stack:
        stack.enter_context(envs.SGLANG_OPT_KDA_CHAIN_PREFILL.override(True))
        stack.enter_context(
            patch(
                "sglang.kernels.ops.attention.fla.chunk_kda_chain_gfx950.is_gfx95_supported",
                return_value=True,
            )
        )
        assert can_use_kda_chain_prefill(
            q=args["q"],
            k=args["k"],
            v=args["v"],
            w=args["w"],
            gk=args["gk"],
            A=args["A"],
            initial_state=args["initial_state"],
            cu_seqlens=args["cu_seqlens"],
            output_intermediate_states=False,
        )


@pytest.mark.parametrize(
    "mutation",
    ["disabled", "wrong_dtype", "wrong_heads", "fixed", "intermediate"],
)
def test_dispatch_fallback(mutation):
    inputs = _inputs([8192, 8192], heads=8)
    args = _clone_inputs(inputs)
    output_intermediate_states = False
    if mutation == "wrong_dtype":
        args["q"] = args["q"].float()
    elif mutation == "wrong_heads":
        args = _clone_inputs(_inputs([8192, 8192], heads=4))
    elif mutation == "fixed":
        args["cu_seqlens"] = None
    elif mutation == "intermediate":
        output_intermediate_states = True
    with ExitStack() as stack:
        stack.enter_context(
            envs.SGLANG_OPT_KDA_CHAIN_PREFILL.override(mutation != "disabled")
        )
        stack.enter_context(
            patch(
                "sglang.kernels.ops.attention.fla.chunk_kda_chain_gfx950.is_gfx95_supported",
                return_value=True,
            )
        )
        assert not can_use_kda_chain_prefill(
            q=args["q"],
            k=args["k"],
            v=args["v"],
            w=args["w"],
            gk=args["gk"],
            A=args["A"],
            initial_state=args["initial_state"],
            cu_seqlens=args["cu_seqlens"],
            output_intermediate_states=output_intermediate_states,
        )
