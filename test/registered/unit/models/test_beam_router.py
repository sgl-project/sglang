"""Beam router dispatch and prepared-weight refresh under CUDA graphs."""

import sys
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.gemm.router_gemv import router_gemm_bf16x3
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
    with (
        envs.SGLANG_BEAM_ROUTER_BACKEND.override(backend),
        get_context().override_server_args(
            moe_runner_backend="flashinfer_trtllm" if nvfp4 else "triton",
            moe_a2a_backend="none",
        ),
        get_flags().moe.override(
            runner_backend=(
                MoeRunnerBackend.FLASHINFER_TRTLLM if nvfp4 else MoeRunnerBackend.TRITON
            ),
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
        router, "cutile_router_gemm", wraps=router.cutile_router_gemm
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


@pytest.mark.parametrize("backend", ["cublas", "cutile"])
def test_backend_dispatch(backend):
    """Selecting cutile must run cuTile, not fall back to the cuBLAS BF16x3 path."""
    router = _router(backend)
    assert router.use_cutile_router == (backend == "cutile")
    x = torch.randn(64, 5120, device="cuda", dtype=torch.bfloat16)
    with patch(
        "sglang.srt.models.beam.router_gemm_bf16x3", wraps=router_gemm_bf16x3
    ) as cublas:
        if backend == "cutile":
            with patch.object(
                router, "cutile_router_gemm", wraps=router.cutile_router_gemm
            ) as cutile:
                router._router_logits(x)
            assert cutile.call_count == 1
        else:
            router._router_logits(x)
    assert cublas.call_count == int(backend == "cublas")


@pytest.mark.parametrize("backend", ["torch", "cublas", "cutile"])
def test_fp32_input_fallback(backend):
    router = _router(backend)
    x = torch.randn(17, 5120, device="cuda")
    torch.testing.assert_close(router._router_logits(x), x @ router.gate.weight.T)
    assert hasattr(router.gate, "bf16x3_weight") == (backend == "cutile")


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
