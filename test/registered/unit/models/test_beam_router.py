"""Beam router dispatch and prepared-weight refresh under CUDA graphs."""

import sys
from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch

from sglang.srt.configs.beam import BeamConfig
from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.models.beam import BeamMoE
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytest.importorskip("cuda.tile")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] not in (9, 10),
    reason="Hopper or Blackwell required",
)


def _router(backend, nvfp4=False):
    # Test real router initialization without allocating the expert weights.
    experts = torch.nn.Module()
    experts.should_fuse_routed_scaling_factor_in_topk = False
    quant = (
        ModelOptFp4Config.from_config({"quant_algo": "NVFP4", "group_size": 16})
        if nvfp4
        else None
    )
    runner = MoeRunnerBackend.FLASHINFER_TRTLLM if nvfp4 else MoeRunnerBackend.TRITON
    with (
        (
            envs.SGLANG_BEAM_ROUTER_BACKEND.override(backend)
            if backend is not None
            else nullcontext()
        ),
        get_context().override_server_args(
            moe_runner_backend=runner.value,
            moe_a2a_backend="none",
        ),
        get_flags().moe.override(
            runner_backend=runner,
            a2a_backend=MoeA2ABackend.NONE,
        ),
        get_parallel().override(
            tp_size=1,
            tp_rank=0,
            moe_tp_size=1,
            moe_tp_rank=0,
            moe_ep_size=1,
            moe_ep_rank=0,
        ),
        patch(
            "sglang.srt.models.beam.get_moe_impl_class",
            return_value=lambda **kw: experts,
        ),
        torch.device("cuda"),
    ):
        router = BeamMoE(
            BeamConfig(hidden_size=5120, num_experts=256), 3, quant_config=quant
        )
    with torch.no_grad():
        router.gate.weight.normal_(std=0.02)
    DefaultModelLoader.postprocess_weights(router, torch.device("cuda"))
    return router


@pytest.mark.parametrize("nvfp4", [False, True])
@pytest.mark.parametrize("m", [1, 16, 17, 512])
def test_router_refresh_and_dispatch(m, nvfp4):
    router = _router("cutile", nvfp4)
    assert router.use_cutile_router and router.use_fp32_router
    assert "bf16x3_weight" not in router.gate.state_dict()
    parts_ptr = router.gate.bf16x3_weight.data_ptr()
    weight_ptr = router.gate.weight.data_ptr()
    x = torch.randn(m, 5120, device="cuda", dtype=torch.bfloat16)
    with patch.object(
        router, "router_gemm_bf16x3_cutile", wraps=router.router_gemm_bf16x3_cutile
    ) as gemm:
        router._router_logits(x)
        assert gemm.call_count == int(m > 16)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = router._router_logits(x)
    for _ in range(2):
        with torch.no_grad():
            router.gate.weight.normal_(std=0.02)
            x.normal_()
        DefaultModelLoader.postprocess_weights(router, torch.device("cuda"))
        assert router.gate.bf16x3_weight.data_ptr() == parts_ptr
        assert router.gate.weight.data_ptr() == weight_ptr
        graph.replay()
        ref = x.double() @ router.gate.weight.double().T
        torch.testing.assert_close(out.double(), ref, atol=8e-6, rtol=3e-6)
        assert (
            (out.double() - ref).square().mean() / ref.square().mean()
        ).sqrt() < 1e-6


@pytest.mark.parametrize("backend", [None, "cublas", "cutile"])
@pytest.mark.parametrize("nvfp4", [False, True])
def test_backend_dispatch(backend, nvfp4, monkeypatch):
    monkeypatch.delenv("SGLANG_BEAM_ROUTER_BACKEND", raising=False)
    expected_backend = backend or "cutile"
    x = torch.randn(64, 5120, device="cuda", dtype=torch.bfloat16)
    with (
        patch(
            "sglang.kernels.ops.gemm.router_gemm_cutile.router_gemm_bf16x3_cutile",
        ) as cutile,
        patch("sglang.srt.models.beam.router_gemm_bf16x3_cublas") as cublas,
    ):
        router = _router(backend, nvfp4)
        assert router.use_cutile_router == (expected_backend == "cutile")
        router._router_logits(x)
    assert cutile.call_count == int(expected_backend == "cutile")
    assert cublas.call_count == int(expected_backend == "cublas")


@pytest.mark.parametrize("m", [1, 32])
def test_compiled_router(m):
    router = _router("cutile")
    x = torch.randn(m, 5120, device="cuda", dtype=torch.bfloat16)
    compiled = torch.compile(router._router_logits, fullgraph=True)
    out = compiled(x)
    ref = x.double() @ router.gate.weight.double().T
    torch.testing.assert_close(out.double(), ref, atol=8e-6, rtol=3e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
