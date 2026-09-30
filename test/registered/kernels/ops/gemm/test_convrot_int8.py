import math
import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.gemm.convrot_int8 import (
    GROUP_SIZES,
    SUPPORTED_CAPABILITIES,
    convrot_int8_fused_linear,
    convrot_int8_linear_prequant,
    convrot_int8_supported_capabilities,
    convrot_rotate_quantize_activation,
)
from sglang.test.ci.ci_register import register_cuda_ci

# One registration per GEMM path: H100 (sm_90a), B200 (sm_100a), RTX 5090 (CC 12.0).
register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def _device_supported() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() in SUPPORTED_CAPABILITIES
    )


pytestmark = pytest.mark.skipif(
    not _device_supported(),
    reason="convrot_int8 kernels carry no code for this GPU (see SUPPORTED_CAPABILITIES)",
)

GROUP_SIZE = 256

# Qwen-Image DiT linears at 1024x1024: text-stream rows (3, 20), image-stream
# rows (2048, 4096); (K, N) = attention projections, FFN up, FFN down. Together
# they hit every tile branch: SM90 small-M narrow/wide-N and large-M, SM100
# default and wide-N, and the small/large-M rows of the CC 12.x mma.sync table.
PRODUCTION_M = [3, 20, 2048, 4096]
PRODUCTION_KN = [(3072, 3072), (3072, 12288), (12288, 3072)]


def _channel_shift(z, gen):
    K = z.shape[-1]
    scale = (1 + 0.3 * torch.randn(K, device="cuda", generator=gen)).abs()
    shift = 0.5 + 0.3 * torch.randn(K, device="cuda", generator=gen)
    return (z * scale + shift).to(z.dtype)


# Activation statistics a data-free rotation must be indifferent to. GELU
# outputs and shifted rows are what the DiT feeds the down-projections; the
# Sylvester transform concentrated a group's mean into one coefficient and
# ran 1.8-2.8x the Gaussian error on these, the regular Hadamard stays within
# 1.15x.
INPUT_FAMILIES = {
    "gauss": lambda z, gen: z,
    "gelu": lambda z, gen: torch.nn.functional.gelu(2 * z, approximate="tanh"),
    "dc": lambda z, gen: z + 1.0,
    "chan_shift": _channel_shift,
}


def _make_inputs(M, K, N, with_bias, seed=0, family="gauss"):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, generator=gen)
    x = INPUT_FAMILIES[family](x, gen)
    weight = (
        torch.randn(N, K, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.02
    )
    bias = (
        torch.randn(N, device="cuda", dtype=torch.bfloat16, generator=gen)
        if with_bias
        else None
    )
    weight_q, weight_scale = convrot_rotate_quantize_activation(
        weight, group_size=GROUP_SIZE
    )
    return x, weight, weight_q, weight_scale, bias


def _regular_hadamard(n, device):
    """kron([H2,] H4, H4, ...) / sqrt(n): 4-point stages from the lowest index digit
    up, one 2-point stage on the top bit when log2(n) is odd. Every row sums to +1 (up to
    the 2-point factor), unlike the Sylvester transform whose first row is all ones."""
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        device=device,
        dtype=torch.float64,
    )
    h2 = torch.tensor([[1, 1], [1, -1]], device=device, dtype=torch.float64)
    bits = int(math.log2(n))
    h = torch.ones(1, 1, device=device, dtype=torch.float64)
    for _ in range(bits // 2):
        h = torch.kron(h4, h)  # the lowest digit is the rightmost Kronecker factor
    if bits % 2:
        h = torch.kron(h2, h)
    return (h / math.sqrt(n)).float()


def _reference_rotate_quantize(x, group_size):
    M, K = x.shape
    h = _regular_hadamard(group_size, x.device)
    rotated = (x.float().view(M, K // group_size, group_size) @ h).view(M, K)
    amax = rotated.abs().amax(dim=1)
    scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    x_q = torch.round(rotated / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return x_q, scale


@pytest.mark.parametrize("group_size", [64, 128, 256, 512])
@pytest.mark.parametrize("M", [3, 2048])
def test_rotate_quantize_matches_hadamard_reference(M, group_size):
    """The in-kernel stages must be the orthonormal regular Hadamard (Kronecker
    power of the 4x4) the weight side was quantized with, not merely some
    consistent orthogonal map."""
    x, *_ = _make_inputs(M, 3072, 8, with_bias=False)
    x_q, x_scale = convrot_rotate_quantize_activation(x, group_size=group_size)
    ref_q, ref_scale = _reference_rotate_quantize(x, group_size=group_size)
    # The butterfly sums the same terms in a different fp32 order than the
    # dense reference matmul; measured drift on the row absmax is ~1e-4.
    torch.testing.assert_close(x_scale, ref_scale, rtol=1e-3, atol=0)
    assert (x_q.int() - ref_q.int()).abs().max().item() <= 1


def _fused_rel_l2(M, K, N, family):
    x, weight, weight_q, weight_scale, bias = _make_inputs(
        M, K, N, with_bias=True, family=family
    )
    out = convrot_int8_fused_linear(
        x, weight_q, weight_scale, bias=bias, group_size=GROUP_SIZE
    )
    ref = torch.nn.functional.linear(x, weight, bias)
    err = torch.linalg.vector_norm(out.float() - ref.float())
    return (err / torch.linalg.vector_norm(ref.float())).item()


@pytest.mark.parametrize("family", [f for f in INPUT_FAMILIES if f != "gauss"])
@pytest.mark.parametrize("K,N", PRODUCTION_KN)
def test_error_is_independent_of_input_statistics(family, K, N):
    M = 2048
    gauss = _fused_rel_l2(M, K, N, "gauss")
    other = _fused_rel_l2(M, K, N, family)
    assert other < 2e-2, other
    assert other / gauss < 1.25, (family, other, gauss)


@pytest.mark.parametrize("group_size", [64, 128, 256, 512])
@pytest.mark.parametrize("K", [1024, 3072, 12288, 16384, 33792, 57856])
def test_rotate_quantize_matches_hadamard_reference_wide_rows(K, group_size):
    # K + group_size leaves a partial last tile; 16384 < K selects the
    # 1024-thread launch and K > 32768 the two-pass one.
    gen = torch.Generator(device="cuda").manual_seed(0)
    for k in (K, K + group_size):
        x = torch.randn(3, k, device="cuda", dtype=torch.bfloat16, generator=gen)
        x_q, x_scale = convrot_rotate_quantize_activation(x, group_size=group_size)
        ref_q, ref_scale = _reference_rotate_quantize(x, group_size=group_size)
        torch.testing.assert_close(x_scale, ref_scale, rtol=1e-3, atol=0)
        assert (x_q.int() - ref_q.int()).abs().max().item() <= 1


def test_rejects_misaligned_view():
    base = torch.randn(20 * 3072 + 8, device="cuda", dtype=torch.bfloat16)
    # Contiguous, but the storage offset is one element: the vectorized loads
    # need 16-byte alignment and the op says so instead of faulting.
    x = base[1 : 1 + 20 * 3072].view(20, 3072)
    with pytest.raises(RuntimeError, match="aligned"):
        convrot_rotate_quantize_activation(x, group_size=GROUP_SIZE)


@pytest.mark.parametrize("M", PRODUCTION_M)
@pytest.mark.parametrize("K,N", PRODUCTION_KN)
@pytest.mark.parametrize("with_bias", [True, False])
def test_matches_bf16_reference(M, K, N, with_bias):
    x, weight, weight_q, weight_scale, bias = _make_inputs(M, K, N, with_bias=with_bias)
    out = convrot_int8_fused_linear(
        x, weight_q, weight_scale, bias=bias, group_size=GROUP_SIZE
    )
    ref = torch.addmm(bias, x, weight.t()) if with_bias else x @ weight.t()
    err = torch.linalg.vector_norm(out.float() - ref.float())
    rel_l2 = (err / torch.linalg.vector_norm(ref.float())).item()
    assert rel_l2 < 2e-2, rel_l2


@pytest.mark.parametrize("M", PRODUCTION_M)
@pytest.mark.parametrize("K,N", PRODUCTION_KN)
@pytest.mark.parametrize("with_bias", [True, False])
def test_prequant_bitwise_equals_fused(M, K, N, with_bias):
    x, _, weight_q, weight_scale, bias = _make_inputs(M, K, N, with_bias=with_bias)
    x_q, x_scale = convrot_rotate_quantize_activation(x, group_size=GROUP_SIZE)
    fused = convrot_int8_fused_linear(
        x, weight_q, weight_scale, bias=bias, group_size=GROUP_SIZE
    )
    prequant = convrot_int8_linear_prequant(
        x_q, x_scale, weight_q, weight_scale, bias=bias, group_size=GROUP_SIZE
    )
    assert torch.equal(fused, prequant)


@pytest.mark.parametrize("M", [20, 2048])
@pytest.mark.parametrize("K,N", [(3072, 3072), (3072, 12288)])
def test_out_variants_bitwise_equal_plain(M, K, N):
    x, _, weight_q, weight_scale, bias = _make_inputs(M, K, N, with_bias=True)
    x_q, x_scale = convrot_rotate_quantize_activation(x, group_size=GROUP_SIZE)
    plain = convrot_int8_fused_linear(
        x, weight_q, weight_scale, bias=bias, group_size=GROUP_SIZE
    )

    # A row slice of a larger joint buffer is the production target of `out`.
    joint = torch.empty(M + 7, N, device="cuda", dtype=torch.bfloat16)
    out = joint[7:]
    ret = convrot_int8_fused_linear(
        x, weight_q, weight_scale, bias=bias, group_size=GROUP_SIZE, out=out
    )
    assert ret.data_ptr() == out.data_ptr()
    assert torch.equal(out, plain)

    out2 = torch.empty_like(plain)
    ret2 = convrot_int8_linear_prequant(
        x_q, x_scale, weight_q, weight_scale, bias=bias, group_size=GROUP_SIZE, out=out2
    )
    assert ret2.data_ptr() == out2.data_ptr()
    assert torch.equal(out2, plain)


def test_serialized_scale_layout_bitwise_equal_flat():
    """A serialized Comfy ConvRot checkpoint stores weight_scale as [N, 1]; the
    ops must accept it unchanged and compute exactly what the flat [N] gives."""
    M, K, N = 20, 3072, 3072
    x, _, weight_q, weight_scale, bias = _make_inputs(M, K, N, with_bias=True)
    flat = convrot_int8_fused_linear(
        x, weight_q, weight_scale, bias=bias, group_size=GROUP_SIZE
    )
    column = convrot_int8_fused_linear(
        x, weight_q, weight_scale.view(N, 1), bias=bias, group_size=GROUP_SIZE
    )
    assert torch.equal(flat, column)


@pytest.mark.parametrize("M", [3, 20, 2048, 4096])
def test_gelu_input_bitwise_equals_eager_gelu(M):
    K, N = 12288, 3072
    x, _, weight_q, weight_scale, bias = _make_inputs(M, K, N, with_bias=True)
    fused = convrot_int8_fused_linear(
        x, weight_q, weight_scale, bias=bias, group_size=GROUP_SIZE, gelu_input=True
    )
    x_gelu = F.gelu(x, approximate="tanh")
    ref = convrot_int8_fused_linear(
        x_gelu, weight_q, weight_scale, bias=bias, group_size=GROUP_SIZE
    )
    assert torch.equal(fused, ref)


def test_empty_batch():
    x, _, weight_q, weight_scale, bias = _make_inputs(0, 3072, 3072, with_bias=True)
    out = convrot_int8_fused_linear(
        x, weight_q, weight_scale, bias=bias, group_size=GROUP_SIZE
    )
    assert out.shape == (0, 3072) and out.dtype == torch.bfloat16
    x_q, x_scale = convrot_rotate_quantize_activation(x, group_size=GROUP_SIZE)
    assert x_q.shape == (0, 3072) and x_scale.shape == (0,)


def test_rejects_invalid_arguments():
    M, K, N = 20, 3072, 3072
    x, _, weight_q, weight_scale, bias = _make_inputs(M, K, N, with_bias=True)
    x_q, x_scale = convrot_rotate_quantize_activation(x, group_size=GROUP_SIZE)

    with pytest.raises(RuntimeError, match="unsupported group_size"):
        convrot_int8_fused_linear(x, weight_q, weight_scale, bias=bias, group_size=96)
    with pytest.raises(RuntimeError, match="multiple of group_size"):
        convrot_rotate_quantize_activation(
            x[:, :3000].contiguous(), group_size=GROUP_SIZE
        )
    with pytest.raises(
        RuntimeError, match=r"\[float32\] not in the allowed options: \[bfloat16\]"
    ):
        convrot_int8_fused_linear(
            x.float(), weight_q, weight_scale, bias=bias, group_size=GROUP_SIZE
        )
    x_strided = x.t().contiguous().t()
    with pytest.raises(RuntimeError, match="Tensor is not contiguous"):
        convrot_int8_fused_linear(
            x_strided, weight_q, weight_scale, bias=bias, group_size=GROUP_SIZE
        )
    with pytest.raises(
        RuntimeError, match=r"\[int32\] not in the allowed options: \[int8\]"
    ):
        convrot_int8_fused_linear(
            x, weight_q.to(torch.int32), weight_scale, bias=bias, group_size=GROUP_SIZE
        )
    with pytest.raises(RuntimeError, match="multiple of 8"):
        convrot_int8_fused_linear(
            x,
            weight_q[: N - 4].contiguous(),
            weight_scale[: N - 4],
            bias=bias[: N - 4],
            group_size=GROUP_SIZE,
        )
    narrow_weight_q = weight_q[:, :2048].contiguous()
    with pytest.raises(
        RuntimeError, match=r"shape#1\('K'\): expected 3072 but got 2048"
    ):
        convrot_int8_fused_linear(
            x, narrow_weight_q, weight_scale, bias=bias, group_size=GROUP_SIZE
        )
    with pytest.raises(
        RuntimeError, match=r"shape#0\('N'\): expected 3072 but got 3071"
    ):
        convrot_int8_fused_linear(
            x, weight_q, weight_scale[:-1], bias=bias, group_size=GROUP_SIZE
        )
    with pytest.raises(
        RuntimeError, match=r"\[float32\] not in the allowed options: \[bfloat16\]"
    ):
        convrot_int8_fused_linear(
            x, weight_q, weight_scale, bias=bias.float(), group_size=GROUP_SIZE
        )
    wrong_out = torch.empty(M, N + 1, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(
        RuntimeError, match=r"shape#1\('N'\): expected 3072 but got 3073"
    ):
        convrot_int8_fused_linear(
            x, weight_q, weight_scale, bias=bias, group_size=GROUP_SIZE, out=wrong_out
        )
    fp16_out = torch.empty(M, N, device="cuda", dtype=torch.float16)
    with pytest.raises(
        RuntimeError, match=r"\[float16\] not in the allowed options: \[bfloat16\]"
    ):
        convrot_int8_fused_linear(
            x, weight_q, weight_scale, bias=bias, group_size=GROUP_SIZE, out=fp16_out
        )
    strided_out = torch.empty(M, 2 * N, device="cuda", dtype=torch.bfloat16)[:, :N]
    with pytest.raises(RuntimeError, match="Tensor is not contiguous"):
        convrot_int8_fused_linear(
            x, weight_q, weight_scale, bias=bias, group_size=GROUP_SIZE, out=strided_out
        )
    x_q_i32 = x_q.to(torch.int32)
    with pytest.raises(
        RuntimeError, match=r"\[int32\] not in the allowed options: \[int8\]"
    ):
        convrot_int8_linear_prequant(
            x_q_i32, x_scale, weight_q, weight_scale, bias=bias, group_size=GROUP_SIZE
        )
    with pytest.raises(RuntimeError, match=r"shape#0\('M'\): expected 20 but got 19"):
        convrot_int8_linear_prequant(
            x_q, x_scale[:-1], weight_q, weight_scale, bias=bias, group_size=GROUP_SIZE
        )
    with pytest.raises(
        RuntimeError, match=r"\[cpu\] not in the allowed options: \[cuda\]"
    ):
        convrot_int8_fused_linear(
            x.cpu(), weight_q, weight_scale, bias=bias, group_size=GROUP_SIZE
        )


def test_supported_capabilities_is_the_published_table():
    """The op module is the single source of truth for the supported parts and
    group widths; the quantization method and the harness read these tables.
    Extending them is a deliberate change, so pin them here."""
    assert convrot_int8_supported_capabilities() == frozenset(
        {(9, 0), (10, 0), (12, 0), (12, 1)}
    )
    assert GROUP_SIZES == (64, 128, 256, 512)
    assert torch.cuda.get_device_capability() in convrot_int8_supported_capabilities()


@pytest.mark.parametrize("with_bias", [True, False])
def test_ops_compile_fullgraph_and_match_eager(with_bias):
    """torch.compile routes through the registered custom ops instead of the
    eager module call; the compiled graph must not break and must reproduce
    eager bit for bit, including the out= write into a row slice."""
    x, _, weight_q, weight_scale, bias = _make_inputs(
        64, 1024, 512, with_bias=with_bias
    )
    # Zeros, not empty: the seven rows above the slice are compared too.
    joint = torch.zeros(64 + 7, 512, device="cuda", dtype=torch.bfloat16)

    def fn(x, weight_q, weight_scale, bias, out):
        x_q, x_scale = convrot_rotate_quantize_activation(x, group_size=GROUP_SIZE)
        gelu = convrot_int8_fused_linear(
            x, weight_q, weight_scale, bias=bias, group_size=GROUP_SIZE, gelu_input=True
        )
        convrot_int8_linear_prequant(
            x_q,
            x_scale,
            weight_q,
            weight_scale,
            bias=bias,
            group_size=GROUP_SIZE,
            out=out[7:],
        )
        return x_q, x_scale, gelu, out

    args = (x, weight_q, weight_scale, bias)
    eager = fn(*args, joint.clone())
    compiled = torch.compile(fn, fullgraph=True)(*args, joint.clone())
    for got, want in zip(compiled, eager, strict=True):
        assert torch.equal(got, want)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
