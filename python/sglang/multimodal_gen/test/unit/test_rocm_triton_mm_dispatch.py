import pytest
import torch

from sglang.multimodal_gen.runtime.layers import linear, triton_mm


def _pair(m=256, n=128, k=256, dtype=torch.bfloat16, contiguous=True):
    x = torch.empty((m, k), dtype=dtype)
    weight = torch.empty((n, k), dtype=dtype)
    if not contiguous:
        x = torch.empty((m, 2 * k), dtype=dtype)[:, ::2]
    return x, weight


@pytest.mark.parametrize(
    "m,n,k",
    [
        (256, 128, 256),  # the smallest tile that fills the kernel
        (16448, 21504, 5376),  # MiniMax-H3 qkv at 2048x2048
        (16448, 28672, 5376),  # gate/up
        (16448, 5376, 5376),  # out_proj
    ],
)
def test_gate_accepts_aligned_bf16(m, n, k):
    x, weight = _pair(m, n, k)
    assert triton_mm.can_use_triton_mm(x, weight, None)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"m": 128},  # below _MIN_M: launch overhead dominates
        {"k": 256 + 32},  # K not a multiple of the K tile
        {"n": 128 + 64},  # N not a multiple of the N tile
        {"dtype": torch.float32},  # only bf16/fp16 is tuned
        {"contiguous": False},
    ],
)
def test_gate_rejects_unsupported_operands(kwargs):
    x, weight = _pair(**kwargs)
    assert not triton_mm.can_use_triton_mm(x, weight, None)


def test_gate_rejects_mismatched_bias():
    x, weight = _pair()
    assert not triton_mm.can_use_triton_mm(x, weight, torch.empty(64))
    assert not triton_mm.can_use_triton_mm(
        x, weight, torch.empty(128, dtype=torch.float32)
    )


def test_apply_unquantized_linear_dispatches_when_enabled(monkeypatch):
    x, weight = _pair()
    calls = []

    def fake_triton_mm(x_, weight_, bias_=None):
        calls.append((tuple(x_.shape), tuple(weight_.shape)))
        return torch.empty((*x_.shape[:-1], weight_.shape[0]), dtype=x_.dtype)

    monkeypatch.setattr(linear, "_USE_TRITON_MM", True)
    monkeypatch.setattr(linear, "can_use_triton_mm", lambda *a: True, raising=False)
    monkeypatch.setattr(linear, "triton_mm", fake_triton_mm, raising=False)
    out = linear.apply_unquantized_linear(x, weight, None)

    assert calls == [((256, 256), (128, 256))]
    assert out.shape == (256, 128)


def test_apply_unquantized_linear_keeps_reference_path_by_default():
    x = torch.randn(256, 256, dtype=torch.bfloat16)
    weight = torch.randn(128, 256, dtype=torch.bfloat16)
    out = linear.apply_unquantized_linear(x, weight, None)
    assert torch.equal(out, torch.nn.functional.linear(x, weight, None))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton matmul needs a ROCm GPU"
)
def test_triton_mm_matches_reference_on_gfx1151():
    if not linear.current_platform.is_gfx1151():
        pytest.skip("tuned for gfx1151")
    x, weight = _pair(1024, 4096, 5376)
    x = x.normal_().to("cuda")
    weight = weight.normal_().to("cuda")
    ref = torch.nn.functional.linear(x, weight)
    out = triton_mm.triton_mm(x, weight)
    # Same fp32 accumulation, different reduction order: every element is within
    # one bf16 step of the reference, not bit-equal.
    tol = (ref.abs().max() * 2**-8).item()
    assert (out.float() - ref.float()).abs().max().item() <= tol
