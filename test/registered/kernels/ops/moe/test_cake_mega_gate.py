"""Cake (DeepGEMM-port) fused routing GEMM + mapping + normalized top-k through sglang.kernels.

Checks that the registry resolves the explicit FlashInfer backend, that
``supports_mega_gate`` admits the tested configurations and refuses the
unsupported ones, that the facade plan outputs are bitwise identical to a plan
prepared directly through FlashInfer, and that routing ids / weights match the
analytically known answer for a constant-score problem (the same construction
FlashInfer's own test uses: ``x = 1``, ``weight = 0`` so every expert scores
equally and the bias alone orders them). The programs take the token count at
runtime, so exported and held-out ``M`` are both covered. Skips (with the
reason) when FlashInfer lacks the module or the device has no generated
program / template.
"""

import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import moe_mega_gate as adapter
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.moe.cake import cake_prepare_mega_gate
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OP = "moe.prepare_mega_gate"
K, E, TOPK = adapter.K, adapter.NUM_EXPERTS, adapter.TOP_K
# (M, deterministic, physical map): exported rows plus one held-out token count.
ROWS = [
    (1, False, True),
    (7, False, True),
    (16, False, True),
    (128, False, True),
    (512, False, True),
    (16, True, True),
    (16, False, False),
]


def test_registry_resolves_flashinfer_backend():
    spec = select_kernel(OP, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.moe_mega_gate:")


def _skip_unless_supported(M, *, deterministic, physical):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(adapter.FI_MODULE, adapter.FI_JIT_MODULE):
        pytest.skip("installed FlashInfer lacks experimental.deepgemm_mega_gate")
    from flashinfer.experimental.deepgemm_mega_gate import mega_gate as runtime

    try:
        _arch, sms = runtime.device_facts(torch.cuda.current_device())
    except RuntimeError as error:
        pytest.skip(str(error))
    try:
        runtime.select_template(
            M,
            sms,
            has_physical_map=physical,
            unmapped_output=deterministic or not physical,
            deterministic=deterministic,
        )
    except (ValueError, NotImplementedError) as error:
        pytest.skip(f"no Mega Gate template for M={M} on {sms} SMs: {error}")
    return torch.device("cuda", torch.cuda.current_device())


def _problem(M, device, *, physical):
    x = torch.ones((M, K), dtype=torch.bfloat16, device=device)
    weight = torch.zeros((E, K), dtype=torch.bfloat16, device=device)
    bias = torch.arange(E, dtype=torch.float32, device=device)
    if physical:
        counts = torch.full((E,), 2, dtype=torch.int32, device=device)
        mapping = torch.stack(
            (
                torch.arange(E, device=device, dtype=torch.int32),
                torch.arange(E, device=device, dtype=torch.int32) + E,
            ),
            1,
        )
    else:
        counts = mapping = None
    return x, weight, bias, mapping, counts


@pytest.mark.parametrize("M,deterministic,physical", ROWS)
def test_matches_flashinfer_and_reference(M, deterministic, physical):
    device = _skip_unless_supported(M, deterministic=deterministic, physical=physical)
    x, weight, bias, mapping, counts = _problem(M, device, physical=physical)
    ep_rank = 7 if deterministic else 0

    def unmapped():
        if deterministic or not physical:
            return torch.empty((M, TOPK), dtype=torch.int64, device=device)
        return None

    def kwargs(unmapped_topk_idx):
        return dict(
            bias=bias,
            to_physical_map=mapping,
            logical_count=counts,
            unmapped_topk_idx=unmapped_topk_idx,
            ep_rank=ep_rank,
            deterministic=deterministic,
        )

    unmapped_facade = unmapped()
    assert adapter.supports_mega_gate(x, weight, TOPK, **kwargs(unmapped_facade))

    plan = cake_prepare_mega_gate(x, weight, TOPK, **kwargs(unmapped_facade))
    ids, weights = plan.run()
    ids, weights = ids.clone(), weights.clone()

    from flashinfer.mega_gate import prepare_mega_gate as fi_prepare

    unmapped_fi = unmapped()
    plan_fi = fi_prepare(x, weight, TOPK, **kwargs(unmapped_fi))
    ids_fi, weights_fi = plan_fi.run()
    torch.cuda.synchronize()
    assert plan.route == plan_fi.route
    assert torch.equal(ids, ids_fi)
    assert torch.equal(weights, weights_fi)
    if unmapped_facade is not None:
        assert torch.equal(unmapped_facade, unmapped_fi)

    # Equal scores -> the bias (arange) picks experts E-1 .. E-6; the physical map
    # duplicates every logical expert twice, the kernel alternates duplicates per
    # token (offset by ep_rank).
    logical = torch.arange(E - 1, E - TOPK - 1, -1, device=device)
    expected = logical[None].expand(M, -1)
    if physical:
        duplicate = ((ep_rank + torch.arange(M, device=device) * 23333) % 2)[:, None]
        expected = expected + duplicate * E
    assert torch.equal(ids, expected)
    if unmapped_facade is not None:
        assert torch.equal(unmapped_facade, logical[None].expand(M, -1))
    # Six equal routed weights normalized then scaled by routed_scaling_factor 1.5.
    torch.testing.assert_close(
        weights, torch.full_like(weights, 0.25), atol=1e-5, rtol=1e-5
    )


def test_supports_rejects_unsupported_configurations():
    device = _skip_unless_supported(16, deterministic=False, physical=True)
    x, weight, bias, mapping, counts = _problem(16, device, physical=True)
    good = dict(bias=bias, to_physical_map=mapping, logical_count=counts)
    assert adapter.supports_mega_gate(x, weight, TOPK, **good)
    # The exported programs fix top-k, the scoring function and need a bias.
    assert not adapter.supports_mega_gate(x, weight, 8, **good)
    assert not adapter.supports_mega_gate(
        x, weight, TOPK, scoring_func="softmax", **good
    )
    assert not adapter.supports_mega_gate(
        x, weight, TOPK, to_physical_map=mapping, logical_count=counts
    )
    # Mapping and counts come together; no descriptor workspace; no image bias.
    assert not adapter.supports_mega_gate(
        x, weight, TOPK, bias=bias, to_physical_map=mapping
    )
    assert not adapter.supports_mega_gate(
        x,
        weight,
        TOPK,
        **good,
        descriptor_workspace=torch.empty(1024, dtype=torch.uint8, device=device),
    )
    assert not adapter.supports_mega_gate(
        x, weight, TOPK, **good, image_bias=torch.zeros_like(bias)
    )
    # Other problem constants (here K) have no programs.
    assert not adapter.supports_mega_gate(
        x[:, : K // 2].contiguous(), weight[:, : K // 2].contiguous(), TOPK, **good
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
