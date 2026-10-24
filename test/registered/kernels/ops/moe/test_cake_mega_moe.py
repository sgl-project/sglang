"""Cake (DeepGEMM-port) MegaMoE v3 / source MegaMoE prepared plans through sglang.kernels.

Checks that the registry resolves the explicit FlashInfer backend for the six
MegaMoE entries, that the device admission probes mirror FlashInfer's catalog
(compute capability AND physical SM count), and that the facade's prepared
objects are FlashInfer's plan classes. The end-to-end parity test needs the
synthetic packed-FP4 model inputs FlashInfer ships only under
``examples/experimental/mega_moe_inputs.py`` (about 7 GiB of packed weights),
which the wheel does not install; it skips with that reason unless the module
is importable.
"""

import importlib.util
import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import moe_mega_moe as adapter
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.moe.cake import (
    cake_prepare_mega_moe_pipeline,
    cake_prepare_source_mega_moe,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OPS = (
    "moe.prepare_mega_moe_pipeline",
    "moe.prepare_mega_moe_grouped_l2",
    "moe.prepare_mega_moe_grouped_fused",
    "moe.prepare_mega_moe_grouped_l1",
    "moe.bind_mega_moe_prepared",
    "moe.prepare_source_mega_moe",
)


@pytest.mark.parametrize("op", OPS)
def test_registry_resolves_flashinfer_backend(op):
    spec = select_kernel(op, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.moe_mega_moe:")


def test_supports_never_raises_without_cuda_or_flashinfer():
    # Admission must be a plain bool on every host.
    assert adapter.supports_mega_moe_v3(torch.device("cpu")) is False
    assert adapter.supports_source_mega_moe(torch.device("cpu")) is False


def _skip_unless_catalogued(supports, runtime_module):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(adapter.FI_MODULE, adapter.FI_JIT_MODULE):
        pytest.skip("installed FlashInfer lacks mega_moe_v3 / source_mega_moe")
    device = torch.device("cuda", 0)
    cc = torch.cuda.get_device_capability(device)
    if cc not in adapter.ARCHS:
        pytest.skip(f"MegaMoE is exported for sm_100a/sm_103a, device is {cc}")
    runtime = importlib.import_module(runtime_module)
    arch = adapter.ARCH_NAMES[cc]
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    if sms not in runtime.supported_num_sms(arch):
        pytest.skip(
            f"exported {arch} routes cover {runtime.supported_num_sms(arch)} SMs, device has {sms}"
        )
    assert supports(device)
    return device


def test_admission_matches_catalog():
    _skip_unless_catalogued(adapter.supports_mega_moe_v3, adapter.FI_JIT_MODULE)
    _skip_unless_catalogued(
        adapter.supports_source_mega_moe, adapter.FI_SOURCE_JIT_MODULE
    )
    from flashinfer.mega_moe_v3 import V3Plan
    from flashinfer.source_mega_moe import MegaMoEPlan

    assert adapter.get_mega_moe_v3_plan_class() is V3Plan
    assert adapter.get_source_mega_moe_plan_class() is MegaMoEPlan


@pytest.mark.parametrize("family", ["v3", "source"])
def test_pipeline_matches_flashinfer_and_reference(family):
    _skip_unless_catalogued(adapter.supports_mega_moe_v3, adapter.FI_JIT_MODULE)
    if importlib.util.find_spec("mega_moe_inputs") is None:
        pytest.skip(
            "needs FlashInfer examples/experimental/mega_moe_inputs.py (synthetic packed "
            "FP4 inputs + torch reference, ~7 GiB of packed weights); not shipped in the wheel"
        )
    inputs_mod = importlib.import_module("mega_moe_inputs")
    precision, num_tokens = "fp4", 16
    inputs, x_scales = inputs_mod.model_inputs(precision, num_tokens=num_tokens, seed=0)
    if family == "source":
        from flashinfer.source_mega_moe import prepare_mega_moe as fi_prepare

        shared = inputs_mod.shared_expert(0)
        weights = inputs_mod.source_weights(inputs, shared)
        kwargs = dict(
            weights=weights,
            num_experts=inputs["num_experts"],
            intermediate=inputs["intermediate"],
            routed_weight_dtype=precision,
            num_shared_experts=1,
            activation_clamp=10.0,
            fast_math=True,
        )
        args = (
            inputs["x_fp8_packed"],
            inputs["x_sf_packed"],
            inputs["topk_idx"],
            inputs["topk_weights"],
        )
        plan = cake_prepare_source_mega_moe(*args, **kwargs)
        plan_fi = fi_prepare(*args, **kwargs)
    else:
        from flashinfer.mega_moe_v3 import prepare_pipeline as fi_prepare

        shared = None
        plan = cake_prepare_mega_moe_pipeline(inputs)
        plan_fi = fi_prepare(inputs)
    out = plan.run().clone()
    out_fi = plan_fi.run().clone()
    torch.cuda.synchronize()
    assert torch.equal(out, out_fi)
    expected = inputs_mod.model_reference(inputs, x_scales, shared)
    torch.testing.assert_close(out.float(), expected.float(), atol=1.0, rtol=0.1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
