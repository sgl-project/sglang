# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits import kandinsky6


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("shape", [(1, 1, 1, 1), (2, 3, 5, 7), (1, 31, 32, 48)])
def test_3d_rope_broadcast_matches_materialized_axes(device, shape, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    rope = kandinsky6.Kandinsky6RoPE3D((64, 32, 32)).to(device)
    batch, frames, height, width = shape
    pos = [torch.arange(n * 2, device=device)[::2] for n in shape[1:]]
    args = torch.cat(
        [
            rope.args_0[pos[0]]
            .view(1, frames, 1, 1, -1)
            .repeat(batch, 1, height, width, 1),
            (rope.args_1[pos[1]] / 2.0)
            .view(1, 1, height, 1, -1)
            .repeat(batch, frames, 1, width, 1),
            (rope.args_2[pos[2]] / 3.16)
            .view(1, 1, 1, width, -1)
            .repeat(batch, frames, height, 1, 1),
        ],
        dim=-1,
    )
    cosine, sine = args.cos(), args.sin()
    expected = torch.stack([cosine, -sine, sine, cosine], dim=-1)
    expected = expected.view(*expected.shape[:-1], 2, 2).unsqueeze(-4)

    def no_repeat(*args, **kwargs):
        pytest.fail("RoPE axes should broadcast without materializing repeats")

    monkeypatch.setattr(torch.Tensor, "repeat", no_repeat)
    actual = rope(shape, pos, (1.0, 2.0, 3.16))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.is_contiguous()


@pytest.mark.parametrize("strided", [False, True])
def test_rotary_cpu_and_autograd(strided):
    x = torch.randn(2, 17, 3, 64, requires_grad=True)
    if strided:
        x = x[:, ::2]
    rope = torch.randn(1, x.shape[1], 1, 32, 2, 2)
    expected = (rope * x.reshape(2, x.shape[1], 3, 32, 1, 2)).sum(-1).reshape_as(x)
    actual = kandinsky6._apply_rotary(x, rope)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), x, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), x)[0],
        rtol=0,
        atol=0,
    )


@torch.no_grad()
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="CUDA required",
)
def test_rotary_uses_fused_kernel(monkeypatch):
    original = kandinsky6.apply_matrix_rope
    calls = []

    def record(x, rope, dtype=None):
        calls.append(x.shape)
        return original(x, rope, dtype)

    monkeypatch.setattr(kandinsky6, "apply_matrix_rope", record)
    x = torch.randn(1, 257, 8, 128, dtype=torch.bfloat16, device="cuda")
    rope = torch.randn(1, 257, 1, 64, 2, 2, device="cuda")
    expected = (
        (rope * x.float().reshape(1, 257, 8, 64, 1, 2))
        .sum(-1)
        .reshape_as(x)
        .to(x.dtype)
    )
    assert torch.equal(kandinsky6._apply_rotary(x, rope), expected)
    assert len(calls) == 1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
def test_rotary_cast_preserves_rounding_and_autograd(device, dtype, strided):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    x = torch.randn(2, 17, 3, 64, device=device, requires_grad=True)
    if strided:
        x = x[:, ::2]
    rope = torch.randn(1, x.shape[1], 1, 32, 2, 2, device=device)
    rounded = x.to(dtype).float().reshape(2, x.shape[1], 3, 32, 1, 2)
    expected = (rope * rounded).sum(-1).reshape_as(x).to(dtype)
    actual = kandinsky6._apply_rotary(x, rope, dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), x, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), x)[0],
        rtol=0,
        atol=0,
    )
    with torch.no_grad():
        torch.testing.assert_close(
            kandinsky6._apply_rotary(x, rope, dtype), expected, rtol=0, atol=0
        )


@torch.no_grad()
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="CUDA required",
)
def test_gate_sum_preserves_fp32_intermediates():
    x = torch.randn(2, 257, 128, dtype=torch.bfloat16, device="cuda")
    update = torch.randn_like(x)
    gate = torch.randn(2, 1, 1152, dtype=x.dtype, device=x.device)[..., 256:384]
    expected = (x.float() + gate.float() * update.float()).to(x.dtype)
    assert not torch.equal(expected, x + gate * update)
    assert torch.equal(kandinsky6._apply_gate_sum(x, update, gate), expected)
