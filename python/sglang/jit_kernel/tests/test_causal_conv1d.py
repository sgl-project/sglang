# Adapted from sgl-kernel/tests/test_causal_conv1d.py (AOT) and
# https://github.com/vllm-project/vllm/blob/main/tests/kernels/mamba/test_causal_conv1d.py
import sys
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from sglang.jit_kernel.causal_conv1d import causal_conv1d_fwd, causal_conv1d_update
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=32, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=128, suite="nightly-kernel-1-gpu", nightly=True)

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
# PyTorch reference implementations.
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
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
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


def _tol(dtype: torch.dtype):
    """Return ``(rtol, atol)`` matched to the AOT test for a given input dtype."""
    if dtype == torch.bfloat16:
        return 1e-2, 5e-2
    if dtype == torch.float32:
        return 3e-4, 1e-3
    return 3e-3, 5e-3  # float16


# ---------------------------------------------------------------------------
# Tests — each parametrize set picks the *minimum* configs that still cover
# every dispatch axis (dtype × width × seqlen × bias × init_state × silu × varlen).
# Total: ~14 cases.
# ---------------------------------------------------------------------------

# 7 fwd configs: cover all 3 widths, all 3 dtypes, bias/init/silu on+off, large dim,
# long seqlen, batch=2 + non-aligned, seqlen < width edge case.
# (batch, dim, seqlen, width, has_bias, has_initial, silu, dtype)
FWD_CONFIGS = [
    (1, 64, 128, 4, True, False, True, torch.float16),       # baseline (w=4, fp16)
    (2, 64, 1024, 3, True, True, True, torch.bfloat16),      # w=3, bf16, initial_state path
    (1, 64, 1025, 2, False, False, False, torch.float32),    # w=2, fp32, non-aligned, no bias, no silu
    (1, 64, 4096, 4, True, False, True, torch.bfloat16),     # long seqlen, bf16
    (1, 4096, 128, 4, True, False, True, torch.float16),     # large dim
    (2, 64, 1025, 4, True, False, True, torch.float16),      # batch=2 + non-aligned (kIsVecLoad=false)
    (1, 64, 2, 4, True, True, True, torch.float16),          # seqlen < width-1 (state pad edge case)
]


@pytest.mark.parametrize(
    "batch,dim,seqlen,width,has_bias,has_initial,silu,dtype", FWD_CONFIGS
)
def test_fwd(batch, dim, seqlen, width, has_bias, has_initial, silu, dtype):
    """Forward kernel matches PyTorch reference across all dispatch axes."""
    device = "cuda"
    rtol, atol = _tol(dtype)
    torch.manual_seed(0)

    x = torch.randn(batch, dim, seqlen, device=device, dtype=dtype).contiguous()
    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype) if has_bias else None

    if has_initial:
        initial_states = torch.randn(batch, dim, width - 1, device=device, dtype=dtype)
        has_init_tensor = torch.ones(batch, dtype=torch.bool, device=device)
    else:
        initial_states = None
        has_init_tensor = None

    x_ref = x.clone()
    initial_states_ref = (
        initial_states.clone() if initial_states is not None else None
    )
    activation = "silu" if silu else None

    out = _causal_conv1d_fn(
        x,
        weight,
        bias,
        activation=activation,
        conv_states=initial_states,
        has_initial_state=has_init_tensor,
    )
    out_ref, final_states_ref = causal_conv1d_ref(
        x_ref,
        weight,
        bias,
        initial_states=initial_states_ref,
        return_final_states=True,
        activation=activation,
    )

    if has_initial:
        torch.testing.assert_close(
            initial_states, final_states_ref, rtol=rtol, atol=atol
        )
    torch.testing.assert_close(out, out_ref, rtol=rtol, atol=atol)


# 3 update configs: cover all 3 widths, all 3 dtypes, bias/silu on+off, aligned/non-aligned dim.
# (batch, dim, width, has_bias, silu, dtype)
UPDATE_CONFIGS = [
    (2, 2048, 4, True, True, torch.float16),          # baseline
    (2, 2048, 2, False, False, torch.bfloat16),       # w=2, bf16, no bias, no silu
    (2, 2048 + 16, 3, True, True, torch.float32),     # w=3, fp32, non-aligned dim
]


@pytest.mark.parametrize("batch,dim,width,has_bias,silu,dtype", UPDATE_CONFIGS)
def test_update(batch, dim, width, has_bias, silu, dtype):
    """Update kernel matches reference across all dispatch axes (single-step decode)."""
    device = "cuda"
    rtol, atol = _tol(dtype)
    torch.manual_seed(0)

    x = torch.randn(batch, dim, 1, device=device, dtype=dtype)
    x_ref = x.clone()
    conv_state = torch.randn(batch, dim, width - 1, device=device, dtype=dtype)
    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype) if has_bias else None
    conv_state_ref = conv_state.clone()
    activation = "silu" if silu else None

    out = _causal_conv1d_update(x, conv_state, weight, bias, activation=activation)
    out_ref = causal_conv1d_update_ref(
        x_ref, conv_state_ref, weight, bias, activation=activation
    )

    torch.testing.assert_close(conv_state, conv_state_ref, rtol=0, atol=0)
    torch.testing.assert_close(out, out_ref, rtol=rtol, atol=atol)


# 2 batch-gather configs: with/without padding, two dtypes.
@pytest.mark.parametrize(
    "with_padding,dtype",
    [(False, torch.float16), (True, torch.bfloat16)],
)
def test_update_batch_gather(with_padding, dtype):
    """Update kernel correctly gathers ``conv_state`` rows by index and skips padded slots."""
    device = "cuda"
    rtol, atol = _tol(dtype)
    torch.manual_seed(0)

    dim, width, batch_size = 2048, 4, 3
    padding = 5 if with_padding else 0
    padded_batch_size = batch_size + padding
    total_entries = 10 * batch_size

    x = torch.randn(padded_batch_size, dim, 1, device=device, dtype=dtype)
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
    conv_state = torch.randn(total_entries, dim, width - 1, device=device, dtype=dtype)
    conv_state_for_padding_test = conv_state.clone()

    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype)
    conv_state_ref = conv_state[conv_state_indices, :].detach().clone()

    out = _causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation="silu",
        conv_state_indices=padded_state_indices,
        pad_slot_id=PAD_SLOT_ID,
    )
    out_ref = causal_conv1d_update_ref(
        x_ref[:batch_size], conv_state_ref, weight, bias, activation="silu"
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


# 2 circular-buffer configs: covers the kIsCircularBuffer kernel branch.
@pytest.mark.parametrize(
    "width,dtype", [(4, torch.float16), (2, torch.bfloat16)]
)
def test_update_circular_buffer(width, dtype):
    """Update kernel handles ``cache_seqlens`` (circular-buffer) mode for streaming decode."""
    device = "cuda"
    rtol, atol = _tol(dtype)
    torch.manual_seed(0)

    batch, dim, state_len = 2, 64, 8
    x = torch.randn(batch, dim, 1, device=device, dtype=dtype)
    x_ref = x.clone()
    conv_state = torch.randn(batch, dim, state_len, device=device, dtype=dtype)
    conv_state_ref = conv_state.clone()
    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype)
    cache_seqlens = torch.tensor([3, 5], dtype=torch.int32, device=device)

    out = _causal_conv1d_update(
        x, conv_state, weight, bias, activation="silu", cache_seqlens=cache_seqlens
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


# 2 varlen configs: with/without padding, exercises the kVarlen kernel branch.
@pytest.mark.parametrize(
    "with_padding,dtype",
    [(False, torch.float16), (True, torch.bfloat16)],
)
def test_varlen(with_padding, dtype):
    """Forward kernel handles continuous-batch (``query_start_loc``) input with optional padded slots."""
    device = "cuda"
    rtol, atol = _tol(dtype)
    torch.manual_seed(0)

    dim, seqlen, width = 64, 1024, 4
    batch_size = 4
    padding = 3 if with_padding else 0
    padded_batch_size = batch_size + padding
    nsplits = padded_batch_size - 1

    eos_pos = torch.randperm(seqlen - 1)[:nsplits].sort().values
    seq_lens = torch.diff(
        torch.cat([torch.tensor([-1]), eos_pos, torch.tensor([seqlen - 1])])
    ).tolist()
    assert sum(seq_lens) == seqlen
    assert all(s > 0 for s in seq_lens)

    total_entries = batch_size * 10
    cumsum = torch.cumsum(torch.tensor(seq_lens), dim=0).to(torch.int32)
    cumsum = torch.concat([torch.tensor([0], dtype=torch.int32), cumsum], dim=0)

    x = torch.randn(1, 4096 + dim + 64, seqlen, device=device, dtype=dtype)[
        :, 4096 : 4096 + dim, :
    ]
    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype)
    x_ref = x.clone()
    final_states = torch.randn(total_entries, dim, width - 1, device=device, dtype=dtype)
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
        "silu",
        PAD_SLOT_ID,
    )

    # Reference: split the varlen tensor back into per-sequence 3D tensors.
    x_2d = x_ref.squeeze(0)  # (dim, seqlen)
    splits = torch.split(x_2d, seq_lens, dim=-1)  # tuple of (dim, seg_len)
    out_ref_b = []
    for i in range(len(seq_lens)):
        if padded_state_indices[i] == PAD_SLOT_ID:
            continue
        x_s = splits[i].unsqueeze(0)  # (1, dim, seg_len)
        out_ref_b.append(
            causal_conv1d_ref(
                x_s,
                weight,
                bias,
                activation="silu",
                return_final_states=True,
                final_states_out=final_states_ref[padded_state_indices[i]].unsqueeze(0),
                initial_states=(
                    final_states_ref[padded_state_indices[i]].unsqueeze(0)
                    if has_initial_states[i]
                    else None
                ),
            )
        )
    # out is 2D (dim, seqlen); ref segments are 3D (1, dim, seg_len), so squeeze for shape match.
    out_ref_tensor = torch.cat([t[0] for t in out_ref_b], dim=2).squeeze(0)

    unpadded_out = out[:, : out_ref_tensor.shape[-1]]
    torch.testing.assert_close(unpadded_out, out_ref_tensor, rtol=rtol, atol=atol)
    torch.testing.assert_close(
        final_states[state_indices],
        final_states_ref[state_indices],
        rtol=rtol,
        atol=atol,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
