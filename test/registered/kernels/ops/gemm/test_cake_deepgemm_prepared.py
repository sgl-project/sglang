"""DeepGEMM-family Cake prepared plans through sglang.kernels.

Checks registry resolution of the five prepared-plan ops and, on an exported
device (SM100a with 148 SMs or SM103a with 152 SMs), that each facade plan is
bitwise identical to FlashInfer's own plan and produces the analytically
expected value for constant E4M3 / E2M1 operands with unit UE8M0 scales. Skips
(with the reason) when FlashInfer lacks the modules, the device/SM count has no
exported route, or (FP8 1D1D) the PTX cannot be assembled.
"""

import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import gemm_deepgemm as cake
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.gemm.cake import (
    cake_prepare_fp4_gemm,
    cake_prepare_fp4_k_grouped_gemm,
    cake_prepare_fp8_batched_gemm,
    cake_prepare_fp8_fp4_gemm,
    cake_prepare_fp8_gemm_1d1d,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OPS = (
    "gemm.prepare_fp8_batched_gemm",
    "gemm.prepare_fp4_gemm",
    "gemm.prepare_fp8_gemm_1d1d",
    "gemm.prepare_fp4_k_grouped_gemm",
    "gemm.prepare_fp8_fp4_gemm",
)
UE8M0_ONE = 0x7F7F7F7F  # four exponent-127 bytes = scale 1.0 per word
E2M1_ONE_PAIR = 0x22  # two packed E2M1 1.0 values
E4M3_ONE = 0x38


@pytest.mark.parametrize("op", OPS)
def test_registry_resolves_flashinfer_backend(op):
    spec = select_kernel(op, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.gemm_deepgemm:")


def _skip_unless_device(*modules):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(*modules):
        pytest.skip(f"installed FlashInfer lacks {modules}")
    cc = torch.cuda.get_device_capability()
    if cc not in cake.ARCHS:
        pytest.skip(
            f"DeepGEMM-family Cake plans are exported for sm_100a/103a, device is {cc}"
        )
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    if cake.EXPORTED_SM_COUNTS.get(cc) != sms:
        pytest.skip(
            f"exported routes pin {cake.EXPORTED_SM_COUNTS.get(cc)} SMs, device has {sms}"
        )


def _exact(actual, expected):
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "tokens,fp8,alpha", [(4, True, None), (4, False, None), (128, False, 0.5)]
)
def test_fp8_batched_gemm_plan(tokens, fp8, alpha):
    _skip_unless_device(cake.FI_BATCHED_MODULE, cake.FI_BATCHED_RUNTIME)
    device = torch.device("cuda")
    heads, inner, width = 8, 4096, 1024
    aq = torch.ones((tokens, heads, inner), dtype=torch.float8_e4m3fn, device=device)
    bq = torch.full(
        (heads, width, inner), 1 / 64, dtype=torch.float8_e4m3fn, device=device
    )
    asf = torch.ones((tokens, heads, inner // 128), dtype=torch.float32, device=device)
    bsf = torch.ones(
        (heads, width // 128, inner // 128), dtype=torch.float32, device=device
    )
    if not cake.supports_fp8_batched_gemm(
        (aq, asf), (bq, bsf), output_fp8=fp8, alpha=alpha
    ):
        pytest.skip("no exported batched FP8 route for this configuration")
    plan = cake_prepare_fp8_batched_gemm(
        (aq, asf), (bq, bsf), output_fp8=fp8, alpha=alpha
    )
    assert isinstance(plan, cake.get_batched_gemm_plan_class())
    result = plan.run()
    from flashinfer.fp8_batched_gemm import prepare_fp8_batched_gemm

    plan_fi = prepare_fp8_batched_gemm(
        (aq, asf), (bq, bsf), output_fp8=fp8, alpha=alpha
    )
    result_fi = plan_fi.run()
    torch.cuda.synchronize()
    if fp8:
        values, scales = result
        values_fi, scales_fi = result_fi
        assert torch.equal(values.view(torch.uint8), values_fi.view(torch.uint8))
        assert torch.equal(scales, scales_fi)
        _exact(values.view(torch.uint8), torch.full_like(values, 256).view(torch.uint8))
        _exact(scales, torch.full_like(scales, 125 * 0x01010101))
    else:
        assert torch.equal(result, result_fi)
        expected = inner / 64 * (1 if alpha is None else alpha)
        torch.testing.assert_close(
            result, torch.full_like(result, expected), atol=0.01, rtol=0.01
        )


@pytest.mark.parametrize(
    "m,n,k,num_stages,alpha", [(256, 128, 256, None, 1.0), (256, 128, 2048, 7, 0.5)]
)
def test_fp4_gemm_plan(m, n, k, num_stages, alpha):
    _skip_unless_device(cake.FI_FP4_MODULE, cake.FI_FP4_RUNTIME)
    device = torch.device("cuda")
    a = torch.full((m, k // 2), E2M1_ONE_PAIR, dtype=torch.uint8, device=device)
    b = torch.full((n, k // 2), E2M1_ONE_PAIR, dtype=torch.uint8, device=device)
    sfa = torch.full((k // 128, m), UE8M0_ONE, dtype=torch.int32, device=device)
    sfb = torch.full((k // 128, n), UE8M0_ONE, dtype=torch.int32, device=device)
    if not cake.supports_fp4_gemm(a, b, sfa, sfb, m=m, num_stages=num_stages):
        pytest.skip("no exported native FP4 route for this shape")
    plan = cake_prepare_fp4_gemm(
        a, b, sfa, sfb, m=m, alpha=alpha, num_stages=num_stages
    )
    assert isinstance(plan, cake.get_fp4_gemm_plan_class())
    out = plan.run()
    from flashinfer.fp4_gemm import prepare_fp4_gemm

    out_fi = prepare_fp4_gemm(
        a, b, sfa, sfb, m=m, alpha=alpha, num_stages=num_stages
    ).run()
    torch.cuda.synchronize()
    assert torch.equal(out, out_fi)
    _exact(out, torch.full_like(out, k * alpha))


@pytest.mark.parametrize(
    "m,n,k,variant,block_n,gran_k_a",
    [(256, 256, 256, None, 128, 32), (256, 128, 256, "bk256_s4", 128, 128)],
)
def test_fp8_fp4_gemm_plan(m, n, k, variant, block_n, gran_k_a):
    _skip_unless_device(cake.FI_MIXED_MODULE, cake.FI_MIXED_RUNTIME)
    from flashinfer.experimental.deepgemm_mixed_gemm import mixed_gemm as runtime

    device = torch.device("cuda")
    arch = runtime.device_arch(device)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    key = runtime.route_key(
        dict(
            M=m,
            N=n,
            K=k,
            num_sms=sms,
            variant=variant,
            block_n=block_n,
            gran_k_a=gran_k_a,
        )
    )
    routes = runtime._catalog()["arches"][arch]["routes"]
    if key not in routes:
        pytest.skip("no exported mixed FP8 x FP4 route for this shape")
    cfg = routes[key]["config"]
    a = torch.full(
        (cfg["input_m"], k), E4M3_ONE, dtype=torch.uint8, device=device
    ).view(torch.float8_e4m3fn)
    b = torch.full((n, k // 2), E2M1_ONE_PAIR, dtype=torch.uint8, device=device)
    sfa = torch.full(
        (cfg["sfa_words"], cfg["sfa_mn"]), UE8M0_ONE, dtype=torch.int32, device=device
    )
    sfb = torch.full(
        (cfg["sfb_words"], cfg["sfb_mn"]), UE8M0_ONE, dtype=torch.int32, device=device
    )
    assert cake.supports_fp8_fp4_gemm(
        a, b, sfa, sfb, m=m, block_n=block_n, gran_k_a=gran_k_a, variant=variant
    )
    plan = cake_prepare_fp8_fp4_gemm(
        a, b, sfa, sfb, m=m, variant=variant, block_n=block_n, gran_k_a=gran_k_a
    )
    assert isinstance(plan, cake.get_mixed_gemm_plan_class())
    out = plan.run()
    from flashinfer.fp8_fp4_gemm import prepare_fp8_fp4_gemm

    out_fi = prepare_fp8_fp4_gemm(
        a, b, sfa, sfb, m=m, variant=variant, block_n=block_n, gran_k_a=gran_k_a
    ).run()
    torch.cuda.synchronize()
    assert tuple(out.shape) == (m, n)
    assert torch.equal(out, out_fi)
    _exact(out, torch.full_like(out, k))


def test_fp4_k_grouped_gemm_plan():
    _skip_unless_device(cake.FI_KGROUP_MODULE, cake.FI_KGROUP_RUNTIME)
    device = torch.device("cuda")
    m, n, group_ks, k_alignment = 256, 128, [257, 0, 511], 256
    padded = [(k + k_alignment - 1) // k_alignment * k_alignment for k in group_ks]
    total_k = sum(padded)
    # Every logical element is E2M1 1.0; padding nibbles must hold zero.
    row = torch.zeros(total_k // 2, dtype=torch.uint8, device=device)
    cursor = 0
    for logical, pad in zip(group_ks, padded):
        full_bytes = logical // 2
        row[cursor // 2 : cursor // 2 + full_bytes] = E2M1_ONE_PAIR
        if logical % 2:
            row[cursor // 2 + full_bytes] = (
                0x02  # low nibble valid, high nibble padding
            )
        cursor += pad
    a = row.expand(m, -1).contiguous()
    b = row.expand(n, -1).contiguous()
    sfa = torch.full((total_k // 128, m), UE8M0_ONE, dtype=torch.int32, device=device)
    sfb = torch.full((total_k // 128, n), UE8M0_ONE, dtype=torch.int32, device=device)
    if not cake.supports_fp4_k_grouped_gemm(a, b, sfa, sfb, m=m, group_ks=group_ks):
        pytest.skip("no exported K-grouped FP4 route for this shape")
    plan = cake_prepare_fp4_k_grouped_gemm(a, b, sfa, sfb, m=m, group_ks=group_ks)
    assert isinstance(plan, cake.get_grouped_fp4_plan_class())
    out = plan.run()
    from flashinfer.fp4_k_grouped_gemm import prepare_fp4_k_grouped_gemm

    out_fi = prepare_fp4_k_grouped_gemm(a, b, sfa, sfb, m=m, group_ks=group_ks).run()
    torch.cuda.synchronize()
    assert tuple(out.shape) == (len(group_ks), m, n)
    assert torch.equal(out, out_fi)
    for g, logical in enumerate(group_ks):
        _exact(out[g], torch.full_like(out[g], logical))


@pytest.mark.parametrize("accumulate", [False, True])
def test_fp8_gemm_1d1d_plan(tmp_path, accumulate):
    _skip_unless_device(cake.FI_FP8_1D1D_MODULE, cake.FI_FP8_1D1D_RUNTIME)
    device = torch.device("cuda")
    m, n, k = cake.FP8_1D1D_M, cake.FP8_1D1D_N, cake.FP8_1D1D_K
    a = torch.full((m, k), E4M3_ONE, dtype=torch.uint8, device=device)
    b = torch.full((n, k), E4M3_ONE, dtype=torch.uint8, device=device)
    sfa = torch.full((k // 512, m), UE8M0_ONE, dtype=torch.uint32, device=device)
    sfb = torch.full((k // 512, n), UE8M0_ONE, dtype=torch.uint32, device=device)
    init = 0.25 if accumulate else 0.0
    out = torch.full(
        (m, n),
        init,
        dtype=torch.float32 if accumulate else torch.bfloat16,
        device=device,
    )
    if not cake.supports_fp8_gemm_1d1d(a, b, sfa, sfb, out, accumulate=accumulate):
        pytest.skip("no exported FP8 1D1D route for this device")
    try:
        plan = cake_prepare_fp8_gemm_1d1d(
            a,
            b,
            sfa,
            sfb,
            out,
            accumulate=accumulate,
            cache_dir=str(tmp_path / "facade"),
        )
    except (FileNotFoundError, ImportError, RuntimeError) as error:
        pytest.skip(f"exported PTX cannot be assembled here: {error}")
    assert isinstance(plan, cake.get_fp8_gemm_plan_class())
    plan.run()
    torch.cuda.synchronize()
    _exact(out, torch.full_like(out, k + init))
    from flashinfer.experimental.deepgemm_fp8_gemm import prepare_fp8_gemm_1d1d

    out_fi = torch.full_like(out, init)
    prepare_fp8_gemm_1d1d(
        a,
        b,
        sfa,
        sfb,
        out_fi,
        accumulate=accumulate,
        cache_dir=str(tmp_path / "direct"),
    ).run()
    torch.cuda.synchronize()
    assert torch.equal(out, out_fi)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
