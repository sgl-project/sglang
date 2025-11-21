# Adapted from https://github.com/vllm-project/vllm/blob/633f943e30a4444d890d26b81850f7217736f840/tests/kernels/mamba/test_mamba_ssm_ssd.py


import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from sglang.srt.layers.attention.mamba.causal_conv1d_triton import PAD_SLOT_ID
from sglang.srt.layers.attention.mamba.ops import selective_state_update


def selective_state_update_ref(
    state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False
):
    """
    Argument:
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        x: (batch, dim) or (batch, nheads, dim)
        dt: (batch, dim) or (batch, nheads, dim)
        A: (dim, dstate) or (nheads, dim, dstate)
        B: (batch, dstate) or (batch, ngroups, dstate)
        C: (batch, dstate) or (batch, ngroups, dstate)
        D: (dim,) or (nheads, dim)
        z: (batch, dim) or (batch, nheads, dim)
        dt_bias: (dim,) or (nheads, dim)
    Return:
        out: (batch, dim) or (batch, nheads, dim)
    """
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    batch, nheads, dim, dstate = state.shape
    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[1]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
        dt = dt + dt_bias
    dt = F.softplus(dt) if dt_softplus else dt
    dA = torch.exp(
        rearrange(dt, "b h d -> b h d 1") * A
    )  # (batch, nheads, dim, dstate)
    B = repeat(B, "b g n -> b (g h) n", h=nheads // ngroups)  # (batch, nheads, dstate)
    C = repeat(C, "b g n -> b (g h) n", h=nheads // ngroups)  # (batch, nheads, dstate)
    dB = rearrange(dt, "b h d -> b h d 1") * rearrange(
        B, "b h n -> b h 1 n"
    )  # (batch, nheads, dim, dstate)
    state.copy_(
        state * dA + dB * rearrange(x, "b h d -> b h d 1")
    )  # (batch, dim, dstate
    out = torch.einsum("bhdn,bhn->bhd", state.to(C.dtype), C)
    if D is not None:
        out += (x * D).to(out.dtype)
    out = (out if z is None else out * F.silu(z)).to(x.dtype)
    if not has_heads:
        out = out.squeeze(1)
    return out


@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_z", [False, True])
@pytest.mark.parametrize("dstate", [16, 32, 64])
@pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
def test_selective_state_update(dim, dstate, has_z, itype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")

    device = "cuda"

    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (5e-3, 1e-2)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
        if torch.version.hip:
            atol *= 2
    # set seed
    torch.manual_seed(0)
    batch_size = 1
    state = torch.randn(batch_size, dim, dstate, dtype=itype, device=device)
    x = torch.randn(batch_size, dim, device=device, dtype=itype)
    out = torch.empty_like(x)
    dt = torch.randn(batch_size, dim, device=device, dtype=itype)
    dt_bias = torch.rand(dim, device=device) - 4.0
    A = -torch.rand(dim, dstate, device=device) - 1.0
    B = torch.randn(batch_size, dstate, device=device)
    C = torch.randn(batch_size, dstate, device=device)
    D = torch.randn(dim, device=device)
    z = torch.randn_like(x) if has_z else None
    state_ref = state.detach().clone()
    selective_state_update(
        state, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True, out=out
    )
    out_ref = selective_state_update_ref(
        state_ref, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True
    )

    assert torch.allclose(state, state_ref, rtol=rtol, atol=atol)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_z", [True])
@pytest.mark.parametrize("dstate", [16, 32, 64])
@pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
# tests correctness in case subset of the sequences are padded
@pytest.mark.parametrize("with_padding", [True, False])
def test_selective_state_update_with_batch_indices(
    with_padding, dim, dstate, has_z, itype
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")

    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (5e-3, 1e-2)
    if itype == torch.bfloat16:
        rtol, atol = 1e-1, 1e-1
        if torch.version.hip:
            atol *= 2
    # set seed
    torch.random.manual_seed(0)
    batch_size = 3
    padding = 5 if with_padding else 0
    padded_batch_size = batch_size + padding
    total_entries = 10 * batch_size
    state = torch.randn(total_entries, dim, dstate, dtype=itype, device=device)
    state_indices = torch.randperm(total_entries)[:batch_size].to(
        dtype=torch.int32, device=device
    )
    unused_states_bool = torch.ones(total_entries, dtype=torch.bool, device=device)
    unused_states_bool[state_indices] = False
    padded_state_indices = torch.concat(
        [
            state_indices,
            torch.as_tensor([PAD_SLOT_ID] * padding, dtype=torch.int32, device=device),
        ],
        dim=0,
    )
    x = torch.randn(padded_batch_size, dim, device=device, dtype=itype)
    out = torch.empty_like(x)
    dt = torch.randn(padded_batch_size, dim, device=device, dtype=itype)
    dt_bias = torch.rand(dim, device=device) - 4.0
    A = -torch.rand(dim, dstate, device=device) - 1.0
    B = torch.randn(padded_batch_size, dstate, device=device)
    C = torch.randn(padded_batch_size, dstate, device=device)
    D = torch.randn(dim, device=device)
    z = torch.randn_like(x) if has_z else None
    state_ref = state[state_indices, :].clone()
    state_before = state.clone()
    selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=padded_state_indices,
        pad_slot_id=PAD_SLOT_ID,
        out=out,
    )
    out_ref = selective_state_update_ref(
        state_ref,
        x[:batch_size],
        dt[:batch_size],
        A,
        B[:batch_size],
        C[:batch_size],
        D=D,
        z=z[:batch_size],
        dt_bias=dt_bias,
        dt_softplus=True,
    )

    print("Output diff max", (out[:batch_size] - out_ref).max())
    print("Output diff mean", (out[:batch_size] - out_ref).mean())
    print("Output state diff max", (state[state_indices, :] - state_ref).max())
    print("Output state diff mean", (state[state_indices, :] - state_ref).mean())
    # test padded entries stay the same
    if with_padding:
        assert torch.equal(state_before[unused_states_bool], state[unused_states_bool])
        assert torch.equal(x[batch_size + 1 :], x[batch_size + 1 :])
        assert torch.equal(dt[batch_size + 1 :], dt[batch_size + 1 :])
        assert torch.equal(B[batch_size + 1 :], B[batch_size + 1 :])
        assert torch.equal(C[batch_size + 1 :], C[batch_size + 1 :])

    # test "real" entries
    assert torch.allclose(state[state_indices, :], state_ref, rtol=rtol, atol=atol)
    assert torch.allclose(out[:batch_size], out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_z", [False, True])
@pytest.mark.parametrize("tie_hdim", [False, True])
@pytest.mark.parametrize("ngroups", [1, 2, 4])
@pytest.mark.parametrize("dstate", [16, 32, 64])
@pytest.mark.parametrize("dim", [2048, 4096])
def test_selective_state_update_with_heads_with_batch_indices(
    dim, dstate, ngroups, has_z, tie_hdim, itype
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")

    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (5e-3, 3e-2)
    if itype == torch.bfloat16:
        rtol, atol = 1e-1, 1e-1
    # set seed
    torch.random.manual_seed(0)
    batch_size = 3
    headdim = 64
    nheads = dim // headdim

    total_entries = 10 * batch_size
    state = torch.randn(
        total_entries, nheads, headdim, dstate, dtype=itype, device=device
    )
    state_indices = torch.randperm(total_entries)[:batch_size].to(
        dtype=torch.int32, device=device
    )

    x = torch.randn(batch_size, nheads, headdim, device=device, dtype=itype)
    out = torch.empty_like(x)
    if not tie_hdim:
        dt = torch.randn(batch_size, nheads, headdim, device=device, dtype=itype)
        dt_bias = torch.rand(nheads, headdim, device=device) - 4.0
        A = -torch.rand(nheads, headdim, dstate, device=device) - 1.0
        D = torch.randn(nheads, headdim, device=device)
    else:
        dt = repeat(
            torch.randn(batch_size, nheads, device=device, dtype=itype),
            "b h -> b h p",
            p=headdim,
        )
        dt_bias = repeat(torch.rand(nheads, device=device) - 4.0, "h -> h p", p=headdim)
        A = repeat(
            -torch.rand(nheads, device=device) - 1.0, "h -> h p n", p=headdim, n=dstate
        )
        D = repeat(torch.randn(nheads, device=device), "h -> h p", p=headdim)
    B = torch.randn(batch_size, ngroups, dstate, device=device)
    C = torch.randn(batch_size, ngroups, dstate, device=device)
    z = torch.randn_like(x) if has_z else None
    state_ref = state[state_indices, :].detach().clone()
    selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=state_indices,
        pad_slot_id=PAD_SLOT_ID,
        out=out,
    )
    out_ref = selective_state_update_ref(
        state_ref, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True
    )

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert torch.allclose(state[state_indices, :], state_ref, rtol=rtol, atol=atol)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__])
