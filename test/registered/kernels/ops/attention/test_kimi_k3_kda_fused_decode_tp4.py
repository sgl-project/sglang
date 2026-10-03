"""Kimi K3 fused KDA decode coverage for TP8 and TP4 head shards."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from sglang.kernels.ops.attention.fla.fused_norm_gate import rms_norm_gated
from sglang.kernels.ops.attention.kda_fused_decode import (
    covered,
    kda_fused_decode,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_D = 128
_LOWER_BOUND = -5.0
_EPS = 1e-6


@dataclass
class Inputs:
    mixed_qkv: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    onorm_g: torch.Tensor
    conv_states: torch.Tensor
    ssm_states: torch.Tensor
    cache_indices: torch.Tensor
    conv_weight: torch.Tensor
    conv_weight_t_q: torch.Tensor
    conv_weight_t_k: torch.Tensor
    conv_weight_t_v: torch.Tensor
    conv_bias: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    onorm_weight: torch.Tensor


def _strided_zeros(shape, stride, *, dtype):
    storage_size = 1 + sum((size - 1) * step for size, step in zip(shape, stride))
    storage = torch.zeros(storage_size, dtype=dtype, device="cuda")
    return torch.as_strided(storage, shape, stride)


def _make_inputs(
    heads: int,
    batch: int,
    *,
    strided_state: bool,
    cache_indices: torch.Tensor | None = None,
    seed: int = 32541,
) -> Inputs:
    generator = torch.Generator(device="cuda").manual_seed(seed + heads + batch)
    seg = heads * _D
    conv_dim = 3 * seg
    slots = batch + 5

    def randn(*shape, dtype=torch.bfloat16, scale=0.1):
        return (
            torch.randn(
                *shape,
                dtype=dtype,
                device="cuda",
                generator=generator,
            )
            * scale
        )

    if strided_state:
        conv_stride = (3 * conv_dim + 256, conv_dim, 1)
        conv_states = _strided_zeros(
            (slots, 3, conv_dim), conv_stride, dtype=torch.bfloat16
        )
        ssm_stride = (heads * _D * _D + 256, _D * _D, _D, 1)
        ssm_states = _strided_zeros(
            (slots, heads, _D, _D), ssm_stride, dtype=torch.float32
        )
        conv_states.copy_(randn(slots, 3, conv_dim))
        ssm_states.copy_(
            randn(slots, heads, _D, _D, dtype=torch.float32, scale=0.01)
        )
    else:
        conv_states = randn(slots, 3, conv_dim).contiguous()
        ssm_states = randn(
            slots, heads, _D, _D, dtype=torch.float32, scale=0.01
        ).contiguous()

    if cache_indices is None:
        cache_indices = torch.randperm(
            slots, device="cuda", generator=generator, dtype=torch.int64
        )[:batch].to(torch.int32)

    conv_weight = randn(
        conv_dim, 4, dtype=torch.float32, scale=0.05
    ).contiguous()
    conv_weight_t = conv_weight.t().contiguous()
    return Inputs(
        mixed_qkv=randn(batch, conv_dim).contiguous(),
        a=randn(batch, seg).contiguous(),
        b=randn(batch, heads).contiguous(),
        onorm_g=randn(batch, seg).contiguous(),
        conv_states=conv_states,
        ssm_states=ssm_states,
        cache_indices=cache_indices.contiguous(),
        conv_weight=conv_weight,
        conv_weight_t_q=conv_weight_t[:, :seg].contiguous(),
        conv_weight_t_k=conv_weight_t[:, seg : 2 * seg].contiguous(),
        conv_weight_t_v=conv_weight_t[:, 2 * seg :].contiguous(),
        conv_bias=randn(conv_dim, dtype=torch.float32, scale=0.01).contiguous(),
        a_log=randn(heads, dtype=torch.float32).contiguous(),
        dt_bias=randn(seg, dtype=torch.float32).contiguous(),
        onorm_weight=(
            1.0 + randn(_D, dtype=torch.float32, scale=0.1)
        ).contiguous(),
    )


def _clone_state_view(tensor: torch.Tensor) -> torch.Tensor:
    clone = torch.empty_strided(
        tensor.shape, tensor.stride(), dtype=tensor.dtype, device=tensor.device
    )
    clone.copy_(tensor)
    return clone


def _clone_inputs(inputs: Inputs) -> Inputs:
    values = vars(inputs).copy()
    values["conv_states"] = _clone_state_view(inputs.conv_states)
    values["ssm_states"] = _clone_state_view(inputs.ssm_states)
    return Inputs(**values)


def _reference(inputs: Inputs) -> torch.Tensor:
    heads = inputs.ssm_states.shape[-3]
    qkv = causal_conv1d_update(
        inputs.mixed_qkv,
        inputs.conv_states.transpose(-1, -2),
        inputs.conv_weight,
        inputs.conv_bias,
        activation="silu",
        conv_state_indices=inputs.cache_indices,
    )
    out = TritonKDAKernel().packed_decode(
        qkv,
        inputs.a,
        inputs.b,
        A_log=inputs.a_log,
        dt_bias=inputs.dt_bias,
        scale=_D**-0.5,
        ssm_states=inputs.ssm_states,
        cache_indices=inputs.cache_indices,
        num_v_heads=heads,
        head_v_dim=_D,
        lower_bound=_LOWER_BOUND,
    )
    return rms_norm_gated(
        x=out,
        g=inputs.onorm_g.view(1, -1, heads, _D),
        weight=inputs.onorm_weight,
        bias=None,
        activation="sigmoid",
        eps=_EPS,
    )


def _candidate(inputs: Inputs) -> torch.Tensor:
    return kda_fused_decode(
        inputs.mixed_qkv,
        inputs.a,
        inputs.b,
        inputs.conv_states,
        inputs.conv_weight_t_q,
        inputs.conv_weight_t_k,
        inputs.conv_weight_t_v,
        inputs.conv_bias,
        inputs.a_log,
        inputs.dt_bias,
        inputs.onorm_g,
        inputs.onorm_weight,
        inputs.ssm_states,
        inputs.cache_indices,
        scale=_D**-0.5,
        onorm_eps=_EPS,
        lower_bound=_LOWER_BOUND,
    )


def _assert_matches(reference: Inputs, candidate: Inputs) -> None:
    reference_out = _reference(reference)
    candidate_out = _candidate(candidate)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        candidate_out.float(), reference_out.float(), rtol=1e-2, atol=2e-3
    )
    torch.testing.assert_close(
        candidate.ssm_states,
        reference.ssm_states,
        rtol=1e-4,
        atol=5e-5,
    )
    torch.testing.assert_close(
        candidate.conv_states.float(),
        reference.conv_states.float(),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("heads", [12, 24])
@pytest.mark.parametrize("strided_state", [False, True])
@torch.inference_mode()
def test_fused_decode_matches_full_chain(heads: int, strided_state: bool) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required.")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("The fused KDA decode kernel requires Blackwell.")

    source = _make_inputs(heads, 8, strided_state=strided_state)
    reference = _clone_inputs(source)
    candidate = _clone_inputs(source)
    assert covered(
        candidate.mixed_qkv,
        candidate.a,
        candidate.b,
        candidate.conv_states,
        candidate.ssm_states,
        candidate.cache_indices,
        candidate.onorm_g,
    )
    _assert_matches(reference, candidate)


@torch.inference_mode()
def test_tp4_padded_slots_are_zero_and_do_not_mutate_state() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required.")

    indices = torch.tensor([4, -1, 1, -1], dtype=torch.int32, device="cuda")
    inputs = _make_inputs(
        24,
        4,
        strided_state=True,
        cache_indices=indices,
        seed=32607,
    )
    conv_before = inputs.conv_states.clone()
    state_before = inputs.ssm_states.clone()
    out = _candidate(inputs)
    torch.cuda.synchronize()

    torch.testing.assert_close(out[:, [1, 3]], torch.zeros_like(out[:, [1, 3]]))
    untouched = torch.tensor([0, 2, 3, 5, 6, 7, 8], device="cuda")
    torch.testing.assert_close(
        inputs.ssm_states[untouched], state_before[untouched], rtol=0, atol=0
    )
    torch.testing.assert_close(
        inputs.conv_states[untouched].float(),
        conv_before[untouched].float(),
        rtol=0,
        atol=0,
    )


@torch.inference_mode()
def test_tp4_changing_input_cuda_graph_replay() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required.")

    captured = _make_inputs(24, 8, strided_state=True, seed=400)
    _candidate(captured)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = _candidate(captured)

    changed = _make_inputs(24, 8, strided_state=True, seed=401)
    reference = _clone_inputs(changed)
    for name in (
        "mixed_qkv",
        "a",
        "b",
        "onorm_g",
        "conv_states",
        "ssm_states",
        "cache_indices",
        "conv_weight",
        "conv_weight_t_q",
        "conv_weight_t_k",
        "conv_weight_t_v",
        "conv_bias",
        "a_log",
        "dt_bias",
        "onorm_weight",
    ):
        getattr(captured, name).copy_(getattr(changed, name))

    graph.replay()
    reference_out = _reference(reference)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        graph_out.float(), reference_out.float(), rtol=1e-2, atol=2e-3
    )
    torch.testing.assert_close(
        captured.ssm_states,
        reference.ssm_states,
        rtol=1e-4,
        atol=5e-5,
    )
    torch.testing.assert_close(
        captured.conv_states.float(),
        reference.conv_states.float(),
        rtol=0,
        atol=0,
    )
