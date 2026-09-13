# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for the fused FlyDSL Kimi-K3 KDA decode path."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("flydsl")
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.utils import is_flydsl_available

from sglang.kernels.ops.kimi_k3.flydsl.kimi_k3_kda_decode import (
    _fb_build_options,
    flydsl_kimi_k3_kda_decode,
    flydsl_kimi_k3_kda_decode_with_f_b,
    is_flydsl_kimi_k3_kda_decode_supported,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _gfx950_flydsl_available() -> bool:
    if not torch.cuda.is_available() or not is_flydsl_available():
        return False
    try:
        return get_gfx() == "gfx950"
    except (AssertionError, KeyError, RuntimeError):
        return False


pytestmark = pytest.mark.skipif(
    not _gfx950_flydsl_available(),
    reason="gfx950 FlyDSL required",
)

_DEVICE = torch.device("cuda")
_HEADS = 12
_DIM = 128
_CHANNELS = 3 * _HEADS * _DIM
_CONV_WIDTH = 4
_LOWER_BOUND = -5.0
_NORM_EPS = 1e-5


@dataclass
class Inputs:
    x: torch.Tensor
    conv_weight: torch.Tensor
    conv_state: torch.Tensor
    raw_g: torch.Tensor
    raw_beta: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    state: torch.Tensor
    state_indices: torch.Tensor
    output_gate: torch.Tensor
    norm_weight: torch.Tensor


def _make_inputs(batch: int, seed: int = 20260728) -> Inputs:
    generator = torch.Generator(device=_DEVICE).manual_seed(seed + batch)
    slots = batch + 2

    x_storage = torch.randn(
        (batch, _CHANNELS + 17),
        dtype=torch.bfloat16,
        device=_DEVICE,
        generator=generator,
    )
    x = x_storage[:, :_CHANNELS]
    conv_weight = 0.1 * torch.randn(
        (_CHANNELS, _CONV_WIDTH),
        dtype=torch.float32,
        device=_DEVICE,
        generator=generator,
    )

    # Kimi's hybrid cache can pad the slot stride. Exercise that layout
    # explicitly while keeping each slot's inner dimensions contiguous.
    conv_storage = torch.randn(
        (slots, _CHANNELS * (_CONV_WIDTH - 1) + 19),
        dtype=torch.bfloat16,
        device=_DEVICE,
        generator=generator,
    )
    conv_state = conv_storage[:, : _CHANNELS * (_CONV_WIDTH - 1)].view(
        slots, _CHANNELS, _CONV_WIDTH - 1
    )
    state_storage = 0.01 * torch.randn(
        (slots, _HEADS * _DIM * _DIM + 23),
        dtype=torch.float32,
        device=_DEVICE,
        generator=generator,
    )
    state = state_storage[:, : _HEADS * _DIM * _DIM].view(slots, _HEADS, _DIM, _DIM)

    raw_beta_storage = torch.randn(
        (1, batch, _HEADS + 1),
        dtype=torch.bfloat16,
        device=_DEVICE,
        generator=generator,
    )
    output_gate_storage = torch.randn(
        (batch, _HEADS * _DIM + 7),
        dtype=torch.bfloat16,
        device=_DEVICE,
        generator=generator,
    )
    return Inputs(
        x=x,
        conv_weight=conv_weight,
        conv_state=conv_state,
        raw_g=torch.randn(
            (1, batch, _HEADS, _DIM),
            dtype=torch.bfloat16,
            device=_DEVICE,
            generator=generator,
        ),
        raw_beta=raw_beta_storage[:, :, :_HEADS],
        A_log=0.5
        * torch.randn(
            (_HEADS,),
            dtype=torch.float32,
            device=_DEVICE,
            generator=generator,
        ),
        dt_bias=0.1
        * torch.randn(
            (_HEADS * _DIM,),
            dtype=torch.float32,
            device=_DEVICE,
            generator=generator,
        ),
        state=state,
        state_indices=torch.arange(
            1,
            batch + 1,
            dtype=torch.int32,
            device=_DEVICE,
        ),
        output_gate=output_gate_storage[:, : _HEADS * _DIM].view(batch, _HEADS, _DIM),
        norm_weight=torch.randn(
            (_DIM,),
            dtype=torch.bfloat16,
            device=_DEVICE,
            generator=generator,
        ),
    )


def _copy_inputs(inputs: Inputs) -> Inputs:
    def clone_preserving_strides(tensor: torch.Tensor) -> torch.Tensor:
        clone = torch.empty_strided(
            tensor.shape,
            tensor.stride(),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        clone.copy_(tensor)
        return clone

    return Inputs(
        **{
            name: clone_preserving_strides(getattr(inputs, name))
            for name in Inputs.__dataclass_fields__
        }
    )


def _reference(inputs: Inputs) -> torch.Tensor:
    batch = inputs.x.shape[0]
    output = torch.zeros(
        (1, batch, _HEADS, _DIM),
        dtype=torch.bfloat16,
        device=_DEVICE,
    )
    dt_bias = inputs.dt_bias.view(_HEADS, _DIM)

    for batch_idx in range(batch):
        state_idx = int(inputs.state_indices[batch_idx])
        if state_idx <= 0:
            continue

        history = inputs.conv_state[state_idx]
        conv_values = torch.cat(
            (
                history.float(),
                inputs.x[batch_idx, :, None].float(),
            ),
            dim=-1,
        )
        packed_qkv = F.silu((conv_values * inputs.conv_weight).sum(dim=-1)).to(
            torch.bfloat16
        )
        history[:, 0].copy_(history[:, 1])
        history[:, 1].copy_(history[:, 2])
        history[:, 2].copy_(inputs.x[batch_idx])

        q, k, v = packed_qkv.view(
            3,
            _HEADS,
            _DIM,
        ).unbind(0)
        for head_idx in range(_HEADS):
            q_head = q[head_idx].float()
            k_head = k[head_idx].float()
            q_head = q_head * torch.rsqrt(q_head.square().sum() + 1e-6)
            q_head = q_head * (_DIM**-0.5)
            k_head = k_head * torch.rsqrt(k_head.square().sum() + 1e-6)

            a = inputs.A_log[head_idx].exp()
            decay = (
                _LOWER_BOUND
                * torch.sigmoid(
                    (
                        inputs.raw_g[
                            0,
                            batch_idx,
                            head_idx,
                        ].float()
                        + dt_bias[head_idx]
                    )
                    * a
                )
            ).exp()
            decayed_state = inputs.state[state_idx, head_idx] * decay[None, :]
            state_dot_k = decayed_state @ k_head
            state_dot_q = decayed_state @ q_head
            k_dot_q = torch.dot(k_head, q_head)
            beta = torch.sigmoid(
                inputs.raw_beta[
                    0,
                    batch_idx,
                    head_idx,
                ].float()
            )
            v_new = (v[head_idx].float() - state_dot_k) * beta
            inputs.state[state_idx, head_idx].copy_(
                decayed_state + v_new[:, None] * k_head[None, :]
            )

            # The model materializes recurrent output in BF16 before the
            # normalization/gating operation.
            recurrent = (state_dot_q + v_new * k_dot_q).to(torch.bfloat16)
            recurrent_f32 = recurrent.float()
            inv_rms = torch.rsqrt(recurrent_f32.square().mean() + _NORM_EPS)
            output[0, batch_idx, head_idx] = (
                recurrent_f32
                * inv_rms
                * inputs.norm_weight.float()
                * torch.sigmoid(
                    inputs.output_gate[
                        batch_idx,
                        head_idx,
                    ].float()
                )
            ).to(torch.bfloat16)
    return output


def _relative_rmse(
    reference: torch.Tensor,
    actual: torch.Tensor,
) -> float:
    delta = actual.float() - reference.float()
    return float(
        delta.square().mean().sqrt() / (reference.float().square().mean().sqrt() + 1e-8)
    )


def _run(inputs: Inputs) -> torch.Tensor:
    return flydsl_kimi_k3_kda_decode(
        x=inputs.x,
        conv_weight=inputs.conv_weight,
        conv_bias=None,
        conv_state=inputs.conv_state,
        raw_g=inputs.raw_g,
        raw_beta=inputs.raw_beta,
        A_log=inputs.A_log,
        dt_bias=inputs.dt_bias,
        lower_bound=_LOWER_BOUND,
        state=inputs.state,
        state_indices=inputs.state_indices,
        output_gate=inputs.output_gate,
        norm_weight=inputs.norm_weight,
        norm_eps=_NORM_EPS,
    )


def _make_fb_inputs(
    batch: int,
    seed: int = 20260728,
) -> tuple[torch.Tensor, torch.Tensor, Inputs]:
    generator = torch.Generator(device=_DEVICE).manual_seed(seed + 10_000 + batch)
    f_a_storage = torch.randn(
        (batch, _DIM + 5),
        dtype=torch.bfloat16,
        device=_DEVICE,
        generator=generator,
    )
    f_a = f_a_storage[:, :_DIM]
    f_b_weight = (
        0.05
        * torch.randn(
            (_HEADS, _DIM, _DIM),
            dtype=torch.bfloat16,
            device=_DEVICE,
            generator=generator,
        )
    ).to(torch.bfloat16)
    inputs = _make_inputs(batch, seed)
    projected = F.linear(
        f_a.float(),
        f_b_weight.view(_HEADS * _DIM, _DIM).float(),
    ).to(torch.bfloat16)
    inputs.raw_g.copy_(projected.view(1, batch, _HEADS, _DIM))
    return f_a, f_b_weight, inputs


def _run_with_f_b(
    f_a: torch.Tensor,
    f_b_weight: torch.Tensor,
    inputs: Inputs,
) -> torch.Tensor:
    return flydsl_kimi_k3_kda_decode_with_f_b(
        f_a=f_a,
        f_b_weight=f_b_weight,
        x=inputs.x,
        conv_weight=inputs.conv_weight,
        conv_bias=None,
        conv_state=inputs.conv_state,
        raw_beta=inputs.raw_beta,
        A_log=inputs.A_log,
        dt_bias=inputs.dt_bias,
        lower_bound=_LOWER_BOUND,
        state=inputs.state,
        state_indices=inputs.state_indices,
        output_gate=inputs.output_gate,
        norm_weight=inputs.norm_weight,
        norm_eps=_NORM_EPS,
    )


def test_public_api_and_support_predicate() -> None:
    import sglang.kernels.ops.kimi_k3.flydsl as flydsl_ops

    assert flydsl_ops.flydsl_kimi_k3_kda_decode is flydsl_kimi_k3_kda_decode
    assert (
        flydsl_ops.is_flydsl_kimi_k3_kda_decode_supported
        is is_flydsl_kimi_k3_kda_decode_supported
    )
    assert is_flydsl_kimi_k3_kda_decode_supported(0)
    assert not is_flydsl_kimi_k3_kda_decode_supported("cpu")


def test_f_b_public_api() -> None:
    import sglang.kernels.ops.kimi_k3.flydsl as flydsl_ops

    assert (
        flydsl_ops.flydsl_kimi_k3_kda_decode_with_f_b
        is flydsl_kimi_k3_kda_decode_with_f_b
    )


def test_f_b_batch_two_dispatch_is_guarded() -> None:
    assert _fb_build_options(1) == {}
    assert _fb_build_options(4) == {}
    options = _fb_build_options(2)
    assert options == {
        "waves_per_eu": 3,
        "cooperative_f_a": True,
        "parallel_front": True,
        "fused_norm_reduce": True,
        "projection_fdot2": True,
    }


@pytest.mark.parametrize("batch", [1, 8, 16])
def test_kimi_k3_kda_decode_matches_reference(batch: int) -> None:
    seed = _make_inputs(batch)
    reference_inputs = _copy_inputs(seed)
    actual_inputs = _copy_inputs(seed)

    reference = _reference(reference_inputs)
    actual = _run(actual_inputs)
    torch.cuda.synchronize()

    assert is_flydsl_kimi_k3_kda_decode_supported(_DEVICE)
    assert not torch.isnan(actual).any()
    assert _relative_rmse(reference, actual) < 1e-3
    assert (
        _relative_rmse(
            reference_inputs.state,
            actual_inputs.state,
        )
        < 1e-3
    )
    assert torch.equal(
        reference_inputs.conv_state,
        actual_inputs.conv_state,
    )


def test_non_positive_slots_do_not_modify_caches() -> None:
    inputs = _make_inputs(batch=2)
    inputs.state_indices.copy_(torch.tensor([0, -1], dtype=torch.int32, device=_DEVICE))
    conv_before = inputs.conv_state.clone()
    state_before = inputs.state.clone()

    actual = _run(inputs)
    torch.cuda.synchronize()

    assert torch.count_nonzero(actual) == 0
    assert torch.equal(inputs.conv_state, conv_before)
    assert torch.equal(inputs.state, state_before)


@pytest.mark.parametrize("batch", [1, 8, 16])
def test_kimi_k3_kda_decode_with_f_b_matches_reference(batch: int) -> None:
    f_a, f_b_weight, seed = _make_fb_inputs(batch)
    reference_inputs = _copy_inputs(seed)
    actual_inputs = _copy_inputs(seed)

    reference = _reference(reference_inputs)
    actual = _run_with_f_b(f_a, f_b_weight, actual_inputs)
    torch.cuda.synchronize()

    assert not torch.isnan(actual).any()
    assert _relative_rmse(reference, actual) < 1e-3
    assert _relative_rmse(reference_inputs.state, actual_inputs.state) < 1e-3
    assert torch.equal(reference_inputs.conv_state, actual_inputs.conv_state)


def test_kimi_k3_kda_decode_with_f_b_recurrent_sequence() -> None:
    f_a, f_b_weight, seed = _make_fb_inputs(batch=2)
    reference_inputs = _copy_inputs(seed)
    actual_inputs = _copy_inputs(seed)
    for _ in range(32):
        reference = _reference(reference_inputs)
        actual = _run_with_f_b(f_a, f_b_weight, actual_inputs)
    torch.cuda.synchronize()
    assert _relative_rmse(reference, actual) < 1e-3
    assert _relative_rmse(reference_inputs.state, actual_inputs.state) < 1e-3
    assert torch.equal(reference_inputs.conv_state, actual_inputs.conv_state)


def test_kimi_k3_kda_decode_with_f_b_graph_replay() -> None:
    f_a, f_b_weight, inputs = _make_fb_inputs(batch=2)
    out = torch.empty((1, 2, _HEADS, _DIM), dtype=torch.bfloat16, device=_DEVICE)
    _run_with_f_b(f_a, f_b_weight, inputs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flydsl_kimi_k3_kda_decode_with_f_b(
            f_a=f_a,
            f_b_weight=f_b_weight,
            x=inputs.x,
            conv_weight=inputs.conv_weight,
            conv_bias=None,
            conv_state=inputs.conv_state,
            raw_beta=inputs.raw_beta,
            A_log=inputs.A_log,
            dt_bias=inputs.dt_bias,
            lower_bound=_LOWER_BOUND,
            state=inputs.state,
            state_indices=inputs.state_indices,
            output_gate=inputs.output_gate,
            norm_weight=inputs.norm_weight,
            norm_eps=_NORM_EPS,
            out=out,
        )
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.isnan(out).any()


def test_f_b_non_positive_slots_do_not_modify_caches() -> None:
    f_a, f_b_weight, inputs = _make_fb_inputs(batch=2)
    inputs.state_indices.copy_(torch.tensor([0, -1], dtype=torch.int32, device=_DEVICE))
    conv_before = inputs.conv_state.clone()
    state_before = inputs.state.clone()

    actual = _run_with_f_b(f_a, f_b_weight, inputs)
    torch.cuda.synchronize()

    assert torch.count_nonzero(actual) == 0
    assert torch.equal(inputs.conv_state, conv_before)
    assert torch.equal(inputs.state, state_before)


def test_f_b_api_rejects_invalid_projection_inputs() -> None:
    f_a, f_b_weight, inputs = _make_fb_inputs(batch=1)

    with pytest.raises(ValueError, match="`f_a` must have rank 2"):
        _run_with_f_b(f_a.unsqueeze(0), f_b_weight, inputs)
    with pytest.raises(ValueError, match="`f_a` must have dtype"):
        _run_with_f_b(f_a.float(), f_b_weight, inputs)
    with pytest.raises(ValueError, match="`f_b_weight` must have shape"):
        _run_with_f_b(f_a, f_b_weight[:, :, :-1], inputs)
    with pytest.raises(ValueError, match="`f_b_weight` must have inner strides"):
        _run_with_f_b(f_a, f_b_weight.transpose(1, 2), inputs)


def test_decode_api_rejects_invalid_input_rank() -> None:
    inputs = _make_inputs(batch=1)
    inputs.x = inputs.x.unsqueeze(0)

    with pytest.raises(ValueError, match="`x` must have rank 2"):
        _run(inputs)


def test_fused_kda_backend_is_opt_in(monkeypatch):
    from sglang.kernels.ops.attention import kda_fused_decode_aiter_hip

    monkeypatch.delenv("SGLANG_K3_KDA_FUSED_BACKEND", raising=False)
    assert not kda_fused_decode_aiter_hip.enabled()
    monkeypatch.setenv("SGLANG_K3_KDA_FUSED_BACKEND", "aiter")
    assert kda_fused_decode_aiter_hip.enabled()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
