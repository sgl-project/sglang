# Adapted from sgl-kernel/tests/test_causal_conv1d.py (AOT) and
# https://github.com/vllm-project/vllm/blob/main/tests/kernels/mamba/test_causal_conv1d.py
import sys
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from sglang.jit_kernel.causal_conv1d import causal_conv1d_fwd, causal_conv1d_update
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=128, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=512, suite="nightly-kernel-1-gpu", nightly=True)

PAD_SLOT_ID = -1


# ---------------------------------------------------------------------------
# Thin Python wrappers over the JIT kernel that match the AOT test signatures.
# ---------------------------------------------------------------------------


def _causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    conv_states: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
) -> torch.Tensor:
    """Run the JIT forward kernel and return the in-place updated ``x``."""
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    if x.stride(-1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None
    causal_conv1d_fwd(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        activation in ["silu", "swish"],
        pad_slot_id,
    )
    return x


def _causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
) -> torch.Tensor:
    """Run the JIT update kernel and return the in-place updated ``x`` (with optional unsqueeze handling)."""
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError(
            f"activation must be None, silu, or swish, actual: {activation}"
        )
    activation_val = activation in ["silu", "swish"]
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation_val,
        cache_seqlens,
        conv_state_indices,
        pad_slot_id,
    )
    if unsqueeze:
        x = x.squeeze(-1)
    return x


# ---------------------------------------------------------------------------
# PyTorch reference implementations (copied verbatim from the AOT test).
# ---------------------------------------------------------------------------


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    return_final_states: bool = False,
    final_states_out: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
):
    """PyTorch reference for ``causal_conv1d_fn`` using ``F.conv1d`` with optional initial states."""
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(dtype_in)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def causal_conv1d_update_ref(
    x, conv_state, weight, bias=None, activation=None, cache_seqlens=None
):
    """PyTorch reference for ``causal_conv1d_update`` with optional circular-buffer ``cache_seqlens``."""
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)
    if cache_seqlens is None:
        x_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)
        conv_state.copy_(x_new[:, :, -state_len:])
    else:
        width_idx = torch.arange(
            -(width - 1), 0, dtype=torch.long, device=x.device
        ).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        width_idx = (
            torch.remainder(width_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        )
        x_new = torch.cat([conv_state.gather(2, width_idx), x], dim=-1).to(weight.dtype)
        copy_idx = torch.arange(seqlen, dtype=torch.long, device=x.device).unsqueeze(
            0
        ) + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        conv_state.scatter_(2, copy_idx, x)
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[
        :, :, -seqlen:
    ]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


def _tolerances(itype: torch.dtype):
    """Return ``(rtol, atol)`` matched to the AOT test for a given input dtype."""
    if itype == torch.bfloat16:
        return 1e-2, 5e-2
    if itype == torch.float32:
        return 3e-4, 1e-3
    return 3e-3, 5e-3  # float16


# ---------------------------------------------------------------------------
# Tests (mirrors the AOT suite + adds a few sanity cases).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("silu_activation", [True, False])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("has_initial_state", [True, False])
@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("seqlen", [1, 8, 16, 32, 64, 128, 256, 512, 1024, 1025, 4096])
@pytest.mark.parametrize("dim", [64])
@pytest.mark.parametrize("batch", [1, 2])
def test_causal_conv1d(
    batch, dim, seqlen, width, has_bias, silu_activation, has_initial_state, itype
):
    """Forward kernel matches the PyTorch reference across dtypes / widths / seqlens."""
    device = "cuda"
    rtol, atol = _tolerances(itype)
    torch.manual_seed(0)

    x = torch.randn(batch, dim, seqlen, device=device, dtype=itype).contiguous()
    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = torch.randn(dim, device=device, dtype=itype) if has_bias else None

    if has_initial_state:
        initial_states = torch.randn(batch, dim, width - 1, device=device, dtype=itype)
        has_initial_state_tensor = torch.ones(batch, dtype=torch.bool, device=device)
    else:
        initial_states = None
        has_initial_state_tensor = None

    x_ref = x.clone()
    initial_states_ref = (
        initial_states.clone() if initial_states is not None else None
    )
    activation = "silu" if silu_activation else None

    out = _causal_conv1d_fn(
        x,
        weight,
        bias,
        activation=activation,
        conv_states=initial_states,
        has_initial_state=has_initial_state_tensor,
    )
    out_ref, final_states_ref = causal_conv1d_ref(
        x_ref,
        weight,
        bias,
        initial_states=initial_states_ref,
        return_final_states=True,
        activation=activation,
    )

    if has_initial_state:
        assert initial_states is not None and final_states_ref is not None
        torch.testing.assert_close(
            initial_states, final_states_ref, rtol=rtol, atol=atol
        )
    torch.testing.assert_close(out, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("silu_activation", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("seqlen", [1])
@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
def test_causal_conv1d_update(dim, width, seqlen, has_bias, silu_activation, itype):
    """Single-step update kernel matches reference for a fresh ``conv_state``."""
    device = "cuda"
    rtol, atol = _tolerances(itype)
    torch.manual_seed(0)

    batch = 2
    x = torch.randn(batch, dim, seqlen, device=device, dtype=itype)
    x_ref = x.clone()
    conv_state = torch.randn(batch, dim, width - 1, device=device, dtype=itype)

    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = torch.randn(dim, device=device, dtype=itype) if has_bias else None
    conv_state_ref = conv_state.detach().clone()
    activation = "silu" if silu_activation else None

    out = _causal_conv1d_update(x, conv_state, weight, bias, activation=activation)
    out_ref = causal_conv1d_update_ref(
        x_ref, conv_state_ref, weight, bias, activation=activation
    )

    torch.testing.assert_close(conv_state, conv_state_ref, rtol=0, atol=0)
    torch.testing.assert_close(out, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("silu_activation", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("seqlen", [1, 4, 5])
@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
@pytest.mark.parametrize("with_padding", [True, False])
def test_causal_conv1d_update_with_batch_gather(
    with_padding, dim, width, seqlen, has_bias, silu_activation, itype
):
    """Update kernel correctly gathers ``conv_state`` rows by index and skips padded slots."""
    device = "cuda"
    rtol, atol = _tolerances(itype)
    torch.manual_seed(0)

    batch_size = 3
    padding = 5 if with_padding else 0
    padded_batch_size = batch_size + padding
    total_entries = 10 * batch_size

    x = torch.randn(padded_batch_size, dim, 1, device=device, dtype=itype)
    x_ref = x.clone()

    conv_state_indices = torch.randperm(total_entries)[:batch_size].to(
        dtype=torch.int32, device=device
    )
    unused_states_bool = torch.ones(total_entries, dtype=torch.bool, device=device)
    unused_states_bool[conv_state_indices] = False
    padded_state_indices = torch.concat(
        [
            conv_state_indices,
            torch.as_tensor([PAD_SLOT_ID] * padding, dtype=torch.int32, device=device),
        ],
        dim=0,
    )
    conv_state = torch.randn(total_entries, dim, width - 1, device=device, dtype=itype)
    conv_state_for_padding_test = conv_state.clone()

    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = torch.randn(dim, device=device, dtype=itype) if has_bias else None
    conv_state_ref = conv_state[conv_state_indices, :].detach().clone()
    activation = "silu" if silu_activation else None

    out = _causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation=activation,
        conv_state_indices=padded_state_indices,
        pad_slot_id=PAD_SLOT_ID,
    )
    out_ref = causal_conv1d_update_ref(
        x_ref[:batch_size], conv_state_ref, weight, bias, activation=activation
    )

    torch.testing.assert_close(
        conv_state[conv_state_indices, :], conv_state_ref, rtol=0, atol=0
    )
    torch.testing.assert_close(out[:batch_size], out_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(
        conv_state[unused_states_bool],
        conv_state_for_padding_test[unused_states_bool],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("silu_activation", [True, False])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize(
    "seqlen", [8, 16, 32, 64, 128, 256, 512, 784, 1024, 2048, 2049, 4096]
)
@pytest.mark.parametrize("dim", [64, 4096])
@pytest.mark.parametrize("with_padding", [True, False])
def test_causal_conv1d_varlen(
    with_padding, dim, seqlen, width, has_bias, silu_activation, itype
):
    """Forward kernel handles continuous-batch (``query_start_loc``) input with optional padded slots."""
    device = "cuda"
    rtol, atol = _tolerances(itype)
    torch.manual_seed(0)
    torch.cuda.empty_cache()

    seqlens = []
    batch_size = 4 if seqlen >= 10 else 1
    padding = 3 if with_padding else 0
    padded_batch_size = batch_size + padding
    nsplits = padded_batch_size - 1

    eos_pos = torch.randperm(seqlen - 1)[:nsplits].sort().values
    seqlens.append(
        torch.diff(
            torch.cat([torch.tensor([-1]), eos_pos, torch.tensor([seqlen - 1])])
        ).tolist()
    )
    assert sum(seqlens[-1]) == seqlen
    assert all(s > 0 for s in seqlens[-1])

    total_entries = batch_size * 10
    cumsum = torch.cumsum(torch.tensor(seqlens[0]), dim=0).to(torch.int32)
    cumsum = torch.concat([torch.tensor([0], dtype=torch.int32), cumsum], dim=0)
    x = torch.randn(1, 4096 + dim + 64, seqlen, device=device, dtype=itype)[
        :, 4096 : 4096 + dim, :
    ]
    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = torch.randn(dim, device=device, dtype=itype) if has_bias else None
    x_ref = x.clone()
    activation = "silu" if silu_activation else None
    final_states = torch.randn(
        total_entries, dim, width - 1, device=device, dtype=itype
    )
    final_states_ref = final_states.clone()
    has_initial_states = torch.randint(
        0, 2, (cumsum.shape[0] - 1,), dtype=torch.bool, device=device
    )
    state_indices = torch.randperm(total_entries, dtype=torch.int32, device=device)[
        :batch_size
    ]
    padded_state_indices = torch.concat(
        [
            state_indices,
            torch.as_tensor([PAD_SLOT_ID] * padding, dtype=torch.int32, device=device),
        ],
        dim=-1,
    )

    out = _causal_conv1d_fn(
        x.squeeze(0),
        weight,
        bias,
        cumsum.cuda(),
        padded_state_indices,
        has_initial_states,
        final_states,
        activation,
        PAD_SLOT_ID,
    )

    out_ref_b = []
    splits = [torch.split(var, seqlens[0], dim=-1) for var in (x_ref,)]
    for i in range(len(seqlens[0])):
        x_s = [v[i].unsqueeze(0) for v in splits][0]
        if padded_state_indices[i] == PAD_SLOT_ID:
            continue
        out_ref_b.append(
            causal_conv1d_ref(
                x_s,
                weight,
                bias,
                activation=activation,
                return_final_states=True,
                final_states_out=final_states_ref[padded_state_indices[i]].unsqueeze(0),
                initial_states=(
                    final_states_ref[padded_state_indices[i]].unsqueeze(0)
                    if has_initial_states[i]
                    else None
                ),
            )
        )
    out_ref_tensor = torch.cat(
        [torch.cat([t[0] for t in out_ref_b], dim=2)], dim=0
    )

    unpadded_out = out[:, : out_ref_tensor.shape[-1]]
    torch.testing.assert_close(unpadded_out, out_ref_tensor, rtol=rtol, atol=atol)
    torch.testing.assert_close(
        final_states[state_indices],
        final_states_ref[state_indices],
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("itype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [2, 3, 4])
def test_causal_conv1d_update_circular_buffer(width, itype):
    """Update kernel handles ``cache_seqlens`` (circular-buffer) mode for streaming decode."""
    device = "cuda"
    rtol, atol = _tolerances(itype)
    torch.manual_seed(0)

    batch, dim, state_len = 2, 64, 8
    x = torch.randn(batch, dim, 1, device=device, dtype=itype)
    x_ref = x.clone()
    conv_state = torch.randn(batch, dim, state_len, device=device, dtype=itype)
    conv_state_ref = conv_state.clone()
    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = torch.randn(dim, device=device, dtype=itype)
    cache_seqlens = torch.tensor([3, 5], dtype=torch.int32, device=device)

    out = _causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation="silu",
        cache_seqlens=cache_seqlens,
    )
    out_ref = causal_conv1d_update_ref(
        x_ref,
        conv_state_ref,
        weight,
        bias,
        activation="silu",
        cache_seqlens=cache_seqlens,
    )

    torch.testing.assert_close(conv_state, conv_state_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(out, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.float16, torch.bfloat16])
def test_causal_conv1d_no_activation(itype):
    """Forward kernel returns the raw conv output (no SiLU) when ``silu_activation=False``."""
    device = "cuda"
    rtol, atol = _tolerances(itype)
    torch.manual_seed(0)

    batch, dim, seqlen, width = 2, 64, 128, 4
    x = torch.randn(batch, dim, seqlen, device=device, dtype=itype).contiguous()
    weight = torch.randn(dim, width, device=device, dtype=itype)
    x_ref = x.clone()

    out = _causal_conv1d_fn(x, weight, None, activation=None)
    out_ref, _ = causal_conv1d_ref(x_ref, weight, None, activation=None)

    torch.testing.assert_close(out, out_ref, rtol=rtol, atol=atol)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
