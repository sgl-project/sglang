from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.models.nemotron_h import NemotronHMoE
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


class _LatentProjection(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.bias = None
        nn.init.normal_(self.weight, std=input_size**-0.5)

    def forward(self, hidden_states):
        return F.linear(hidden_states, self.weight), None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_nemotron_h_fused_latent_projection_shared_add_and_graph():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    projection = _LatentProjection(2048, 8192).cuda()
    moe = SimpleNamespace(
        _fuse_latent_projection_shared_add=True,
        use_latent_moe=True,
        fc2_latent_proj=projection,
    )
    routed = torch.randn(16, 2048, device="cuda", dtype=torch.bfloat16)
    shared = torch.randn(16, 8192, device="cuda", dtype=torch.bfloat16)

    projected, _ = projection(routed)
    reference = projected + shared
    candidate, remaining_shared = NemotronHMoE._apply_latent_projection(
        moe, routed, shared.clone()
    )

    assert remaining_shared is None
    torch.testing.assert_close(candidate, reference, rtol=1e-2, atol=3.125e-2)

    # Warm the BLAS workspace and allocator before capture.
    NemotronHMoE._apply_latent_projection(moe, routed, shared.clone())
    torch.cuda.synchronize()
    graph_shared_input = shared.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # In the model, the shared MLP rewrites this buffer on each replay.
        graph_shared_output = graph_shared_input.clone()
        graph_output, graph_shared = NemotronHMoE._apply_latent_projection(
            moe, routed, graph_shared_output
        )
    graph.replay()
    torch.cuda.synchronize()

    assert graph_shared is None
    torch.testing.assert_close(graph_output, reference, rtol=1e-2, atol=3.125e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_nemotron_h_latent_projection_fallback_is_unchanged():
    projection = _LatentProjection(64, 128).cuda()
    moe = SimpleNamespace(
        _fuse_latent_projection_shared_add=False,
        use_latent_moe=True,
        fc2_latent_proj=projection,
    )
    routed = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    shared = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)

    projected, remaining_shared = NemotronHMoE._apply_latent_projection(
        moe, routed, shared
    )
    reference, _ = projection(routed)

    torch.testing.assert_close(projected, reference, rtol=0, atol=0)
    assert remaining_shared is shared


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_nemotron_h_latent_projection_falls_back_for_mixed_dtype():
    projection = _LatentProjection(64, 128).cuda()
    moe = SimpleNamespace(
        _fuse_latent_projection_shared_add=True,
        use_latent_moe=True,
        fc2_latent_proj=projection,
    )
    routed = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    shared = torch.randn(4, 128, device="cuda", dtype=torch.float32)

    projected, remaining_shared = NemotronHMoE._apply_latent_projection(
        moe, routed, shared
    )
    reference, _ = projection(routed)

    torch.testing.assert_close(projected, reference, rtol=0, atol=0)
    assert remaining_shared is shared
