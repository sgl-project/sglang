import sys
from types import SimpleNamespace

import pytest
import torch
from sgl_kernel import rmsnorm as sgl_rmsnorm

import sglang.jit_kernel.rmsnorm_per_token_group_quant_fp8 as fused_rmsnorm_quant
from sglang.jit_kernel.per_token_group_quant_8bit_v2 import (
    per_token_group_quant_8bit_v2,
)
from sglang.jit_kernel.rmsnorm_per_token_group_quant_fp8 import (
    rmsnorm_per_token_group_quant_fp8,
    rmsnorm_per_token_group_quant_fp8_out,
)
from sglang.jit_kernel.tests.rmsnorm_per_token_group_quant_fp8_test_utils import (
    DEVICE,
    DTYPE,
    EPS,
    GROUP_SIZE,
    TRACE_HIDDEN_SIZE,
)
from sglang.jit_kernel.tests.rmsnorm_per_token_group_quant_fp8_test_utils import (
    alloc_outputs as _alloc_outputs,
)
from sglang.jit_kernel.tests.rmsnorm_per_token_group_quant_fp8_test_utils import (
    make_input as _make_input,
)
from sglang.kernels.ops.quantization.fp8_kernel import (
    fp8_dtype,
    fp8_max,
    fp8_min,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=35, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HIDDEN_SIZES = [128, 640, 1536, TRACE_HIDDEN_SIZE, 4096]


def _reference(
    x: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = x.shape[0]
    output_q, output_s, output_norm = _alloc_outputs(num_tokens, x.shape[1])
    if num_tokens == 0:
        return output_q, output_s, output_norm

    sgl_rmsnorm(x, weight, EPS, out=output_norm)
    per_token_group_quant_8bit_v2(
        output_norm,
        output_q,
        output_s,
        GROUP_SIZE,
        1e-10,
        float(fp8_min),
        float(fp8_max),
        scale_ue8m0=True,
    )
    return output_q, output_s, output_norm


def _assert_matches_reference(x: torch.Tensor, weight: torch.Tensor) -> None:
    q_ref, s_ref, norm_ref = _reference(x, weight)
    output_q, output_s, output_norm = rmsnorm_per_token_group_quant_fp8(x, weight, EPS)
    torch.cuda.synchronize()

    hidden_size = x.shape[1]
    num_groups = hidden_size // GROUP_SIZE
    packed_scale_cols = (num_groups + 3) // 4

    assert output_q.shape == x.shape
    assert output_q.dtype == fp8_dtype
    assert output_s.shape == (x.shape[0], packed_scale_cols)
    assert output_s.dtype == torch.int32
    assert output_s.stride(0) == 1
    if x.shape[0] > 0:
        assert output_s.stride(1) >= x.shape[0]
        assert output_s.stride(1) % 4 == 0
    assert output_norm.shape == x.shape
    assert output_norm.dtype == DTYPE

    # The fused row-reduction topology may differ from production RMSNorm by a
    # BF16 ULP; quantization parity is checked against the emitted norm below.
    torch.testing.assert_close(output_norm, norm_ref, atol=1e-3, rtol=1e-2)

    # The power-of-two scale is insensitive to the occasional BF16 ULP in the
    # independent RMS reduction. Quantized values should also agree directly
    # almost everywhere; retain the exact emitted-norm check below as the hard
    # contract for the fused dataflow.
    assert torch.equal(output_s, s_ref)
    if output_q.numel() > 0:
        exact_q_fraction = (
            output_q.view(torch.int8).eq(q_ref.view(torch.int8)).float().mean().item()
        )
        assert exact_q_fraction >= 0.99, (
            f"only {exact_q_fraction:.2%} of fused FP8 values matched the "
            "unfused end-to-end reference"
        )

    scale_remainder = num_groups % 4
    if output_s.shape[0] > 0 and scale_remainder:
        # Scale bytes are packed least-significant byte first. DeepGEMM may
        # consume a whole final int32, so all bytes beyond the valid groups
        # must be deterministic zeros.
        valid_mask = (1 << (8 * scale_remainder)) - 1
        padding_mask = 0xFFFFFFFF ^ valid_mask
        padding_bits = output_s[:, -1].to(torch.int64) & padding_mask
        assert torch.count_nonzero(padding_bits).item() == 0

    # Quantization must consume the BF16-rounded norm output, not the transient
    # FP32 normalized value held during RMSNorm.
    q_from_norm, s_from_norm, _ = _alloc_outputs(x.shape[0], hidden_size)
    if x.shape[0] > 0:
        per_token_group_quant_8bit_v2(
            output_norm,
            q_from_norm,
            s_from_norm,
            GROUP_SIZE,
            1e-10,
            float(fp8_min),
            float(fp8_max),
            scale_ue8m0=True,
        )
        torch.cuda.synchronize()
        assert torch.equal(output_q.view(torch.int8), q_from_norm.view(torch.int8))
        assert torch.equal(output_s, s_from_norm)


@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_tokens", [1, 6])
def test_matches_unfused_reference(num_tokens: int, hidden_size: int) -> None:
    torch.manual_seed(1000 + num_tokens + hidden_size)
    x = _make_input(num_tokens, hidden_size)
    weight = torch.randn(hidden_size, device=DEVICE, dtype=DTYPE)
    _assert_matches_reference(x, weight)


@pytest.mark.parametrize("num_tokens", [2, 4, 8, 16, 38])
def test_trace_hidden_size_token_sweep(num_tokens: int) -> None:
    torch.manual_seed(2000 + num_tokens)
    x = _make_input(num_tokens)
    weight = torch.randn(TRACE_HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    _assert_matches_reference(x, weight)


def test_trace_m6_row_stride_2624() -> None:
    torch.manual_seed(6)
    x = _make_input(6, trace_stride=True)
    weight = torch.randn(TRACE_HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    _assert_matches_reference(x, weight)


def test_zero_input_matches_unfused_reference() -> None:
    x = torch.zeros(6, TRACE_HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    weight = torch.randn(TRACE_HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    _assert_matches_reference(x, weight)


def test_tiny_input_matches_unfused_reference() -> None:
    torch.manual_seed(7)
    x = torch.randn(6, TRACE_HIDDEN_SIZE, device=DEVICE, dtype=DTYPE) * 1e-12
    weight = torch.randn(TRACE_HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    _assert_matches_reference(x, weight)


def test_empty_input_is_a_noop() -> None:
    x = torch.empty(0, TRACE_HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    weight = torch.randn(TRACE_HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)

    output_q, output_s, output_norm = rmsnorm_per_token_group_quant_fp8(x, weight, EPS)
    assert output_q.shape == (0, TRACE_HIDDEN_SIZE)
    assert output_s.shape == (0, TRACE_HIDDEN_SIZE // GROUP_SIZE // 4)
    assert output_norm.shape == (0, TRACE_HIDDEN_SIZE)

    output_q, output_s, output_norm = _alloc_outputs(0)
    assert (
        rmsnorm_per_token_group_quant_fp8_out(
            x, weight, output_q, output_s, output_norm, EPS
        )
        is None
    )


@pytest.mark.parametrize(
    "make_invalid",
    [
        pytest.param(
            lambda: torch.randn(2, 1025, device=DEVICE, dtype=DTYPE),
            id="hidden-size",
        ),
        pytest.param(
            lambda: torch.randn(
                2, TRACE_HIDDEN_SIZE, device=DEVICE, dtype=torch.float16
            ),
            id="input-dtype",
        ),
        pytest.param(
            lambda: torch.randn(2, TRACE_HIDDEN_SIZE * 2, device=DEVICE, dtype=DTYPE)[
                :, ::2
            ],
            id="inner-stride",
        ),
        pytest.param(
            lambda: torch.randn(2, TRACE_HIDDEN_SIZE + 8, device=DEVICE, dtype=DTYPE)[
                :, :TRACE_HIDDEN_SIZE
            ],
            id="row-alignment",
        ),
        pytest.param(
            lambda: torch.randn(
                2, TRACE_HIDDEN_SIZE + 17, device=DEVICE, dtype=DTYPE
            ).as_strided((2, TRACE_HIDDEN_SIZE), (TRACE_HIDDEN_SIZE + 16, 1), 1),
            id="storage-alignment",
        ),
    ],
)
def test_rejects_invalid_input_contract(make_invalid) -> None:
    x = make_invalid()
    weight = torch.randn(x.shape[1], device=DEVICE, dtype=DTYPE)
    with pytest.raises((AssertionError, RuntimeError, TypeError, ValueError)):
        rmsnorm_per_token_group_quant_fp8(x, weight, EPS)


def test_rejects_invalid_rank_before_allocation_or_jit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.randn(TRACE_HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    weight = torch.randn(TRACE_HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)

    def fail_jit_lookup(*args, **kwargs):
        pytest.fail("invalid rank reached JIT compilation")

    monkeypatch.setattr(fused_rmsnorm_quant, "_jit_module", fail_jit_lookup)
    with pytest.raises(ValueError, match="rank 2"):
        rmsnorm_per_token_group_quant_fp8(x, weight, EPS)


@pytest.mark.parametrize(
    "weight",
    [
        pytest.param(lambda: torch.randn(1024, device=DEVICE, dtype=DTYPE), id="shape"),
        pytest.param(
            lambda: torch.randn(TRACE_HIDDEN_SIZE, device=DEVICE, dtype=torch.float16),
            id="dtype",
        ),
        pytest.param(
            lambda: torch.randn(TRACE_HIDDEN_SIZE + 1, device=DEVICE, dtype=DTYPE)[1:],
            id="storage-alignment",
        ),
    ],
)
def test_rejects_invalid_weight_contract(weight) -> None:
    x = _make_input(2)
    with pytest.raises((AssertionError, RuntimeError, TypeError, ValueError)):
        rmsnorm_per_token_group_quant_fp8(x, weight(), EPS)


def test_out_variant_rejects_incompatible_outputs() -> None:
    x = _make_input(2)
    weight = torch.randn(TRACE_HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    output_q, output_s, output_norm = _alloc_outputs(2)

    with pytest.raises((AssertionError, RuntimeError, ValueError)):
        rmsnorm_per_token_group_quant_fp8_out(
            x,
            weight,
            output_q[:, :-1],
            output_s,
            output_norm,
            EPS,
        )


def test_out_variant_rejects_overlapping_scale_layout_before_jit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_tokens = 6
    x = _make_input(num_tokens)
    weight = torch.randn(TRACE_HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    output_q, output_s, output_norm = _alloc_outputs(num_tokens)
    overlapping_scale = torch.empty(
        output_s.numel(), device=DEVICE, dtype=torch.int32
    ).as_strided(output_s.shape, (1, 4))

    def fail_jit_lookup(*args, **kwargs):
        pytest.fail("overlapping scale layout reached JIT compilation")

    monkeypatch.setattr(fused_rmsnorm_quant, "_jit_module", fail_jit_lookup)
    with pytest.raises(ValueError, match="packed column-major"):
        rmsnorm_per_token_group_quant_fp8_out(
            x,
            weight,
            output_q,
            overlapping_scale,
            output_norm,
            EPS,
        )


def test_torch_compile() -> None:
    torch.manual_seed(17)
    x = _make_input(6, trace_stride=True)
    weight = torch.randn(TRACE_HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)

    @torch.compile(fullgraph=True)
    def compiled(input: torch.Tensor, norm_weight: torch.Tensor):
        return rmsnorm_per_token_group_quant_fp8(input, norm_weight, EPS)

    q_ref, s_ref, norm_ref = rmsnorm_per_token_group_quant_fp8(x, weight, EPS)
    output_q, output_s, output_norm = compiled(x, weight)
    torch.cuda.synchronize()

    assert torch.equal(output_norm, norm_ref)
    assert torch.equal(output_q.view(torch.int8), q_ref.view(torch.int8))
    assert torch.equal(output_s, s_ref)


def test_cuda_graph_replay() -> None:
    torch.manual_seed(23)
    x = _make_input(6, trace_stride=True)
    weight = torch.randn(TRACE_HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    output_q, output_s, output_norm = _alloc_outputs(6)

    # Raw torch.cuda.graph does not insert the runner's warmup forwards, so
    # emulate one ordinary invocation before entering the capture context.
    rmsnorm_per_token_group_quant_fp8_out(
        x, weight, output_q, output_s, output_norm, EPS
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rmsnorm_per_token_group_quant_fp8_out(
            x, weight, output_q, output_s, output_norm, EPS
        )

    graph.replay()
    torch.cuda.synchronize()
    _, _, norm_ref = _reference(x, weight)
    torch.testing.assert_close(output_norm, norm_ref, atol=1e-3, rtol=1e-2)

    q_from_norm, s_from_norm, _ = _alloc_outputs(x.shape[0])
    per_token_group_quant_8bit_v2(
        output_norm,
        q_from_norm,
        s_from_norm,
        GROUP_SIZE,
        1e-10,
        float(fp8_min),
        float(fp8_max),
        scale_ue8m0=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(output_q.view(torch.int8), q_from_norm.view(torch.int8))
    assert torch.equal(output_s, s_from_norm)


@pytest.mark.parametrize(
    "arch_major,cuda_version,expected",
    [
        (10, (12, 9), True),
        (10, (12, 8), False),
        (9, (13, 0), False),
        (12, (13, 0), False),
    ],
)
def test_blackwell_runtime_gate(
    monkeypatch: pytest.MonkeyPatch,
    arch_major: int,
    cuda_version: tuple[int, int],
    expected: bool,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(fused_rmsnorm_quant, "is_hip_runtime", lambda: False)
    monkeypatch.setattr(
        fused_rmsnorm_quant,
        "get_jit_cuda_arch",
        lambda: SimpleNamespace(major=arch_major),
    )
    monkeypatch.setattr(fused_rmsnorm_quant, "get_cuda_version", lambda: cuda_version)

    assert (
        fused_rmsnorm_quant.can_use_rmsnorm_per_token_group_quant_fp8(
            DTYPE, TRACE_HIDDEN_SIZE
        )
        is expected
    )


def test_runtime_capability_result_is_not_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = iter((False, True))
    monkeypatch.setattr(
        fused_rmsnorm_quant,
        "_is_supported_blackwell_runtime",
        lambda: next(results),
    )

    assert not fused_rmsnorm_quant.can_use_rmsnorm_per_token_group_quant_fp8(
        DTYPE, TRACE_HIDDEN_SIZE
    )
    assert fused_rmsnorm_quant.can_use_rmsnorm_per_token_group_quant_fp8(
        DTYPE, TRACE_HIDDEN_SIZE
    )


def test_runtime_gate_precedes_jit_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = _make_input(2)
    weight = torch.randn(TRACE_HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    output_q, output_s, output_norm = _alloc_outputs(2)

    monkeypatch.setattr(
        fused_rmsnorm_quant, "_is_supported_blackwell_runtime", lambda: False
    )

    def fail_jit_lookup(*args, **kwargs):
        pytest.fail("unsupported hardware reached JIT compilation")

    monkeypatch.setattr(fused_rmsnorm_quant, "_jit_module", fail_jit_lookup)
    monkeypatch.setattr(fused_rmsnorm_quant, "is_arch_support_pdl", fail_jit_lookup)
    with pytest.raises(RuntimeError, match="SM10x Blackwell.*CUDA 12.9"):
        fused_rmsnorm_quant.rmsnorm_per_token_group_quant_fp8_out(
            x, weight, output_q, output_s, output_norm, EPS
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
