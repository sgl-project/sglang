"""Cake (DeepGEMM-port) fused routing GEMM + mapping + normalized top-k through sglang.kernels.

Checks that the registry resolves the explicit FlashInfer backend, that the
facade plan outputs are bitwise identical to a plan prepared directly through
FlashInfer, and that routing ids / weights match the analytically known
answer for a constant-score problem (the same construction FlashInfer's own
test uses: ``x = 1``, ``weight = 0`` so every expert scores equally and the
bias alone orders them). Skips (with the reason) when FlashInfer lacks the
module or the GPU / SM count has no exported route.
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


def test_registry_resolves_flashinfer_backend():
    spec = select_kernel(OP, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.moe_mega_gate:")


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(adapter.FI_MODULE, adapter.FI_JIT_MODULE):
        pytest.skip("installed FlashInfer lacks experimental.deepgemm_mega_gate")
    from flashinfer.experimental.deepgemm_mega_gate import mega_gate as runtime

    try:
        arch = runtime.device_arch(torch.device("cuda"))
    except RuntimeError as error:
        pytest.skip(str(error))
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    if sms not in runtime.supported_num_sms(arch):
        pytest.skip(
            f"exported {arch} routes cover {runtime.supported_num_sms(arch)} SMs, device has {sms}"
        )
    return torch.device("cuda", 0)


@pytest.mark.parametrize("M", [1, 16, 128, 512])
def test_matches_flashinfer_and_reference(M):
    device = _skip_unless_supported()
    x = torch.ones((M, K), dtype=torch.bfloat16, device=device)
    weight = torch.zeros((E, K), dtype=torch.bfloat16, device=device)
    bias = torch.arange(E, dtype=torch.float32, device=device)
    counts = torch.full((E,), 2, dtype=torch.int32, device=device)
    mapping = torch.stack(
        (
            torch.arange(E, device=device, dtype=torch.int32),
            torch.arange(E, device=device, dtype=torch.int32) + E,
        ),
        1,
    )
    assert adapter.supports_mega_gate(x, weight, TOPK)
    kwargs = dict(bias=bias, to_physical_map=mapping, logical_count=counts)

    plan = cake_prepare_mega_gate(x, weight, TOPK, **kwargs)
    ids, weights = plan.run()
    ids, weights = ids.clone(), weights.clone()

    from flashinfer.mega_gate import prepare_mega_gate as fi_prepare

    ids_fi, weights_fi = fi_prepare(x, weight, TOPK, **kwargs).run()
    torch.cuda.synchronize()
    assert torch.equal(ids, ids_fi)
    assert torch.equal(weights, weights_fi)

    # Equal scores -> the bias (arange) picks experts E-1 .. E-6; the physical map
    # duplicates every logical expert twice, the kernel alternates duplicates per token.
    logical = torch.arange(E - 1, E - TOPK - 1, -1, device=device)
    expected = logical[None].expand(M, -1)
    duplicate = ((torch.arange(M, device=device) * 23333) % 2)[:, None]
    assert torch.equal(ids, expected + duplicate * E)
    # Six equal routed weights normalized then scaled by routed_scaling_factor 1.5.
    torch.testing.assert_close(
        weights, torch.full_like(weights, 0.25), atol=1e-5, rtol=1e-5
    )


def test_supports_rejects_unexported_shapes():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda", 0)
    x = torch.ones((7, K), dtype=torch.bfloat16, device=device)
    weight = torch.zeros((E, K), dtype=torch.bfloat16, device=device)
    assert not adapter.supports_mega_gate(x, weight, TOPK)  # M=7 has no route
    assert not adapter.supports_mega_gate(x[:16], weight, 8)  # top-8 not exported
    assert not adapter.supports_mega_gate(x[:16], weight, TOPK, scoring_func="softmax")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
