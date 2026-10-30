"""``diffusion.activation``: activation-function fusions.

All of these are bit-exact by construction -- they are elementwise chains with
no reduction, so reproducing aten's per-op fp32-opmath / round-to-bf16
boundaries is enough and ``torch.equal`` is the assertion.

The cublasLt linear+tanh-GELU epilogue is *not* here: it is not bit-exact and
is therefore quality-gated, so it is tested through its mount protocol in
``test_sites.py``.
"""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion import (
    can_use_fused_situ_softcap_mul,
    fused_bias_glu,
    fused_bias_silu,
    fused_packed_silu_mul_bitexact,
    fused_silu_mul_bitexact,
    fused_situ_softcap_mul_bitexact,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("channels", [2240, 11200])
def test_sana_bias_silu_is_bit_exact(channels):
    torch.manual_seed(0)
    x = torch.randn(
        (1, channels, 7, 5),
        device="cuda",
        dtype=torch.bfloat16,
    ).to(memory_format=torch.channels_last)
    bias = torch.randn(channels, device="cuda", dtype=torch.bfloat16)

    actual = fused_bias_silu(x, bias)
    expected = F.silu(x + bias[None, :, None, None])

    assert actual.is_contiguous(memory_format=torch.channels_last)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("channels", [2240, 5600])
def test_sana_bias_glu_is_bit_exact(channels):
    torch.manual_seed(1)
    x = torch.randn(
        (1, 2 * channels, 7, 5),
        device="cuda",
        dtype=torch.bfloat16,
    ).to(memory_format=torch.channels_last)
    bias = torch.randn(2 * channels, device="cuda", dtype=torch.bfloat16)

    actual = fused_bias_glu(x, bias)
    biased = x + bias[None, :, None, None]
    hidden, gate = torch.chunk(biased, 2, dim=1)
    expected = hidden * F.silu(gate)

    assert actual.is_contiguous(memory_format=torch.channels_last)
    assert torch.equal(actual, expected)


# ---------------------------------------------------------------------------
# silu(a) * b for split-projection SwiGLU
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(1, 4096, 3072), (2, 17, 512)])
def test_silu_mul_is_bit_exact(shape):
    # Separate gate/up GEMMs, so the concatenated ``silu_and_mul`` kernels do
    # not apply without an extra full-width cat -- this kernel replaces the
    # eager ``F.silu(a) * b`` pair instead.
    torch.manual_seed(0)
    a = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)

    assert torch.equal(fused_silu_mul_bitexact(a, b), F.silu(a) * b)


@pytest.mark.parametrize("hidden", [384, 3072])
@pytest.mark.parametrize("strided", [False, True])
def test_packed_silu_mul_is_bit_exact(hidden, strided):
    # The packed form splits one [.., 2 * hidden] projection in-kernel; it must
    # accept the strided view a wider projection slice produces.
    torch.manual_seed(1)
    if strided:
        x = torch.randn(1, 19, 3 * hidden, device="cuda", dtype=torch.bfloat16)
        x = x[..., : 2 * hidden]
    else:
        x = torch.randn(1, 19, 2 * hidden, device="cuda", dtype=torch.bfloat16)

    expected = F.silu(x[..., :hidden]) * x[..., hidden:]
    assert torch.equal(fused_packed_silu_mul_bitexact(x), expected)


def _situ_reference(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (
        25.0 * torch.tanh(hidden / 25.0)
    )


@pytest.mark.skipif(torch.version.hip is not None, reason="Requires CUDA PTX")
@pytest.mark.parametrize("shape", [(1, 13, 128), (2, 65, 8192), (1, 2524, 8192)])
@pytest.mark.parametrize("packed", [False, True])
def test_situ_preserves_bf16_rounding_and_projection_strides(shape, packed):
    torch.manual_seed(0)
    if packed:
        projection = (
            torch.randn(
                (*shape[:-1], 2 * shape[-1]), device="cuda", dtype=torch.bfloat16
            )
            * 5
        )
        hidden, gate = projection.chunk(2, -1)
    else:
        hidden = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 5
        gate = torch.randn_like(hidden)
    assert can_use_fused_situ_softcap_mul(device=hidden.device, dtype=hidden.dtype)
    actual = fused_situ_softcap_mul_bitexact(hidden, gate)
    assert actual.is_contiguous()
    assert torch.equal(actual, _situ_reference(hidden, gate))


@pytest.mark.skipif(torch.version.hip is not None, reason="Requires CUDA PTX")
@pytest.mark.parametrize("operand", ["hidden", "gate"])
@pytest.mark.parametrize(
    "other", [-25.0, -4.0, -1.0, 0.0, 1.0, 4.0, 25.0, float("inf"), -float("inf")]
)
def test_situ_all_bf16_encodings(operand, other):
    # Check the full lookup domain, including subnormals and signed zero,
    # against positive, negative, zero and infinite companion factors.
    values = (
        torch.arange(65536, device="cuda", dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
        .reshape(1, 1, -1)
    )
    if operand == "hidden":
        hidden, gate = values, torch.full_like(values, other)
    else:
        hidden, gate = torch.full_like(values, other), values
    actual = fused_situ_softcap_mul_bitexact(hidden, gate)
    expected = _situ_reference(hidden, gate)
    nan = torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), nan)
    assert torch.equal(actual[~nan].view(torch.int16), expected[~nan].view(torch.int16))


@pytest.mark.skipif(torch.version.hip is not None, reason="Requires CUDA PTX")
def test_situ_cold_and_warm_tables_preserve_cuda_graph_replay(monkeypatch):
    from sglang.kernels.ops.diffusion.activation import (
        situ_softcap_mul_bitexact as impl,
    )

    tables = {}
    monkeypatch.setattr(impl, "_SITU_TABLES", tables)
    hidden = torch.randn(1, 32, 8192, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(hidden)
    # Warm the direct kernel without constructing tables.
    small = torch.randn(1, 1, 8192, device="cuda", dtype=torch.bfloat16)
    fused_situ_softcap_mul_bitexact(small, small)
    cold = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cold):
        cold_out = fused_situ_softcap_mul_bitexact(hidden, gate)
    assert not tables
    fused_situ_softcap_mul_bitexact(hidden, gate)
    pointers = tuple(t.data_ptr() for t in tables[hidden.device])
    warm = torch.cuda.CUDAGraph()
    with torch.cuda.graph(warm):
        warm_out = fused_situ_softcap_mul_bitexact(hidden, gate)
    for scale in (1.0, 5.0):
        hidden.copy_(torch.randn_like(hidden) * scale)
        gate.copy_(torch.randn_like(gate) * scale)
        cold.replay()
        warm.replay()
        expected = _situ_reference(hidden, gate)
        assert torch.equal(cold_out, expected)
        assert torch.equal(warm_out, expected)
        assert tuple(t.data_ptr() for t in tables[hidden.device]) == pointers


@pytest.mark.skipif(torch.version.hip is not None, reason="Requires CUDA PTX")
def test_situ_tables_are_ready_for_other_streams(monkeypatch):
    from sglang.kernels.ops.diffusion.activation import (
        situ_softcap_mul_bitexact as impl,
    )

    hidden = torch.randn(1, 32, 8192, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(hidden)
    fused_situ_softcap_mul_bitexact(hidden, gate)  # Compile the lookup kernel first.
    monkeypatch.setattr(impl, "_SITU_TABLES", {})
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    consumer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        torch.cuda._sleep(50_000_000)
        fused_situ_softcap_mul_bitexact(hidden, gate)
    # No wait on producer: published tables must already be usable.
    with torch.cuda.stream(consumer):
        actual = fused_situ_softcap_mul_bitexact(hidden, gate)
    consumer.synchronize()
    assert torch.equal(actual, _situ_reference(hidden, gate))


def test_situ_rejects_unsupported_operands():
    hidden = torch.randn(1, 8, 64, device="cuda", dtype=torch.bfloat16)
    for gate in (hidden.float(), hidden[:, :-1], hidden.cpu()):
        with pytest.raises(ValueError, match="matching CUDA BF16"):
            fused_situ_softcap_mul_bitexact(hidden, gate)
    strided = hidden[..., ::2]
    for x in (strided, hidden[:, :0]):
        with pytest.raises(ValueError, match="matching CUDA BF16"):
            fused_situ_softcap_mul_bitexact(x, x)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
