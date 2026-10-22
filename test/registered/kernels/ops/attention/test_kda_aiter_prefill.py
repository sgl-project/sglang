import sys
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.attention.linear.kernels import kda_aiter
from sglang.srt.layers.attention.linear.kernels.kda_aiter import (
    AiterKDAPrefillKernel,
    aiter_kda_prefill_available,
)
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=240, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx95_supported()),
    reason="requires one gfx950 GPU",
)


def _inputs(lengths, heads, seed=0, state_indices=None, state_dtype=torch.bfloat16):
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
        "ssm_states": randn(
            max(len(lengths), max(state_indices) + 1),
            heads,
            128,
            128,
            dtype=state_dtype,
            scale=0.01,
        ),
        "cache_indices": torch.tensor(
            state_indices,
            device="cuda",
            dtype=torch.int64,
        ),
        "query_start_loc": torch.tensor(
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


def _run(kernel, inputs, **kwargs):
    return kernel.extend(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        ssm_states=inputs["ssm_states"],
        cache_indices=inputs["cache_indices"],
        query_start_loc=inputs["query_start_loc"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        lower_bound=-5.0,
        beta_is_raw=True,
        **kwargs,
    )


def _assert_matches(
    lengths,
    heads,
    seed=0,
    state_indices=None,
    state_dtype=torch.bfloat16,
):
    inputs = _inputs(lengths, heads, seed, state_indices, state_dtype)
    expected_args = _clone(inputs)
    expected = _run(TritonKDAKernel(), expected_args)
    actual_args = _clone(inputs)
    actual = _run(AiterKDAPrefillKernel(), actual_args)
    assert torch.isfinite(actual).all()
    assert _err_ratio(expected, actual) < 0.008
    valid = actual_args["cache_indices"] >= 0
    rows = actual_args["cache_indices"][valid]
    assert (
        _err_ratio(
            expected_args["ssm_states"][rows],
            actual_args["ssm_states"][rows],
        )
        < 0.008
    )


@pytest.mark.skipif(
    not aiter_kda_prefill_available(),
    reason="requires merged ROCm/aiter#5866 and ROCm/aiter#6672",
)
@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("lengths", [[1], [63, 64], [65, 127, 192]])
@pytest.mark.parametrize("seed", [0, 37])
def test_aiter_chunk_kda_matches_current(heads, lengths, seed):
    _assert_matches(lengths, heads, seed)


@pytest.mark.skipif(
    not aiter_kda_prefill_available(),
    reason="requires merged ROCm/aiter#5866 and ROCm/aiter#6672",
)
def test_production_single_sequence_dispatch_matches_current():
    _assert_matches([16384], heads=16, seed=5)


@pytest.mark.skipif(
    not aiter_kda_prefill_available(),
    reason="requires merged ROCm/aiter#5866 and ROCm/aiter#6672",
)
def test_production_fp32_state_dispatch_matches_current():
    _assert_matches([16384], heads=16, seed=7, state_dtype=torch.float32)


@pytest.mark.skipif(
    not aiter_kda_prefill_available(),
    reason="requires merged ROCm/aiter#5866 and ROCm/aiter#6672",
)
def test_invalid_state_index_and_determinism():
    inputs = _inputs([65, 127], heads=8, seed=11, state_indices=[0, -1])
    before = inputs["ssm_states"].clone()
    first_args = _clone(inputs)
    first = _run(AiterKDAPrefillKernel(), first_args)
    second_args = _clone(inputs)
    second = _run(AiterKDAPrefillKernel(), second_args)
    assert torch.equal(first, second)
    assert torch.equal(first_args["ssm_states"], second_args["ssm_states"])
    assert torch.equal(first_args["ssm_states"][1], before[1])


@pytest.mark.skipif(
    not aiter_kda_prefill_available(),
    reason="requires merged ROCm/aiter#5866 and ROCm/aiter#6672",
)
def test_graph_replay():
    args = _inputs([65, 127], heads=8, seed=29)
    value_seed = args["v"].clone()
    state_seed = args["ssm_states"].clone()
    kernel = AiterKDAPrefillKernel()
    for _ in range(3):
        args["v"].copy_(value_seed)
        args["ssm_states"].copy_(state_seed)
        _run(kernel, args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        args["v"].copy_(value_seed)
        args["ssm_states"].copy_(state_seed)
        output = _run(kernel, args)
    graph.replay()
    torch.cuda.synchronize()
    first_output = output.clone()
    first_state = args["ssm_states"].clone()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output, first_output)
    assert torch.equal(args["ssm_states"], first_state)


@pytest.mark.skipif(
    not aiter_kda_prefill_available(),
    reason="requires merged ROCm/aiter#5866 and ROCm/aiter#6672",
)
@pytest.mark.parametrize("fallback", ["track_state", "track_chunk_idx"])
def test_fallback_contract(fallback):
    args = _inputs([128], heads=8, state_dtype=torch.float32)
    args["ssm_states"] = args["ssm_states"].bfloat16()
    kwargs = {}
    if fallback == "track_state":
        kwargs["track_state"] = torch.empty_like(
            args["ssm_states"], dtype=torch.float32
        )
    elif fallback == "track_chunk_idx":
        kwargs["track_chunk_idx"] = torch.zeros(1, device="cuda", dtype=torch.int64)
    kernel = AiterKDAPrefillKernel()
    sentinel = args["v"]
    with patch.object(kernel._triton, "extend", return_value=sentinel) as triton:
        output = _run(kernel, args, **kwargs)
    assert output is sentinel
    triton.assert_called_once()


def test_not_gfx950_rejected():
    with patch.object(kda_aiter, "is_gfx95_supported", return_value=False):
        with pytest.raises(ValueError, match="requires gfx950"):
            AiterKDAPrefillKernel()


def test_source_inspection_failure_disables_bf16_capability():
    with patch.object(kda_aiter.inspect, "getsource", side_effect=OSError):
        operator, supports_bf16 = kda_aiter._load_chunk_kda()
    assert operator is not None
    assert not supports_bf16


@pytest.mark.skipif(
    not aiter_kda_prefill_available(),
    reason="requires merged ROCm/aiter#5866 and ROCm/aiter#6672",
)
def test_metadata_forwarded():
    args = _inputs([65, 127], heads=8)
    chunk_indices = torch.empty(4, 2, device="cuda", dtype=torch.int64)
    chunk_offsets = torch.empty(3, device="cuda", dtype=torch.int64)
    captured = {}

    def fake_chunk_kda(**kwargs):
        captured.update(kwargs)
        return kwargs["out"], None

    with patch.object(kda_aiter, "_AITER_CHUNK_KDA", fake_chunk_kda):
        _run(
            AiterKDAPrefillKernel(),
            args,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
        )
    assert captured["chunk_indices"] is chunk_indices
    assert captured["chunk_offsets"] is chunk_offsets


@pytest.mark.skipif(
    not aiter_kda_prefill_available(),
    reason="requires merged ROCm/aiter#5866 and ROCm/aiter#6672",
)
def test_single_sequence_tracking_matches_current():
    inputs = _inputs([8192], heads=16, seed=41)
    expected_args = _clone(inputs)
    expected_track = torch.empty_like(expected_args["ssm_states"], dtype=torch.float32)
    expected, _ = _run(
        TritonKDAKernel(),
        expected_args,
        return_intermediate_states=True,
        track_state=expected_track,
        track_chunk_idx=torch.tensor([127], device="cuda", dtype=torch.int32),
    )

    actual_args = _clone(inputs)
    actual_track = torch.empty_like(actual_args["ssm_states"], dtype=torch.float32)
    actual, intermediate = _run(
        AiterKDAPrefillKernel(),
        actual_args,
        return_intermediate_states=True,
        track_state=actual_track,
        track_chunk_idx=torch.tensor([127], device="cuda", dtype=torch.int32),
        track_chunk_idx_cpu=[127],
    )

    assert intermediate is None
    assert _err_ratio(expected, actual) < 0.008
    assert _err_ratio(expected_track, actual_track) < 0.008
    assert _err_ratio(expected_args["ssm_states"], actual_args["ssm_states"]) < 0.008


@pytest.mark.skipif(
    not aiter_kda_prefill_available(),
    reason="requires merged ROCm/aiter#5866 and ROCm/aiter#6672",
)
def test_aligned_tracking_tuple_contract_matches_current():
    inputs = _inputs([16384], heads=16, seed=43, state_dtype=torch.float32)
    expected_args = _clone(inputs)
    expected, _ = _run(
        TritonKDAKernel(),
        expected_args,
        return_intermediate_states=True,
    )
    actual_args = _clone(inputs)
    actual, intermediate = _run(
        AiterKDAPrefillKernel(),
        actual_args,
        return_intermediate_states=True,
    )
    assert intermediate is None
    assert _err_ratio(expected, actual) < 0.008
    assert _err_ratio(expected_args["ssm_states"], actual_args["ssm_states"]) < 0.008


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
