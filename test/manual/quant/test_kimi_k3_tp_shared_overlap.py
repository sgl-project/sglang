"""GPU regression for the single-collective MoE branch fork/join."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.models import kimi_k3 as k3

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires CUDA graph replay"
)


@pytest.mark.parametrize("rows", [1, 4, 32, 64, 65])
def test_shared_compute_is_joined_before_collective(rows, monkeypatch):
    torch.manual_seed(42)
    width = 128
    x = torch.randn(rows, width, device="cuda")
    front = torch.randn(3 * width, width, device="cuda")
    shared_w = torch.randn(width, width, device="cuda")
    routed_w = torch.randn(width, width, device="cuda")
    alt = torch.cuda.Stream()
    calls = []

    def shared(gate_up, out):
        calls.append(torch.cuda.current_stream() == alt)
        # Make a missing join observable: the main branch is deliberately short.
        torch.cuda._sleep(500_000)
        torch.mm(gate_up, shared_w, out=out)

    def routed(hidden, logits, inputs, out):
        torch.mm(inputs, routed_w, out=out)

    layer = SimpleNamespace(
        _front_w=front,
        _front_fp32=False,
        _front_sizes=[width, width, width],
        _moe_front_needs_dense_bf16=False,
        moe_hidden_size=width,
        tp_size=8,
        alt_stream=alt,
        experts=SimpleNamespace(
            moe_ep_size=1,
            quant_method=SimpleNamespace(use_flashinfer_trtllm_weight_layout=True),
        ),
        _forward_shared=shared,
        _forward_routed=routed,
        fuse_ar_norm=False,
        # Clone stands in for a consumer collective: it must see BOTH writes.
        _reduce_latent_and_shared=lambda buf: buf.clone(),
        _latent_norm=lambda a: a,
        routed_expert_up_proj=lambda a: (a, None),
    )

    def gemm(a, w, out_dtype=None):
        return torch.nn.functional.linear(a, w)

    with (
        patch.object(k3, "_k3_bf16_gemm", side_effect=gemm),
        patch.object(k3.k3_ar_fusion, "enabled", return_value=False),
        patch.object(k3, "use_symmetric_memory", return_value=nullcontext()),
        patch.object(k3, "is_allocation_symmetric", return_value=False),
        patch.object(k3, "get_parallel", return_value=SimpleNamespace(tp_group=None)),
        patch.object(
            k3,
            "get_moe_runner_backend",
            return_value=SimpleNamespace(is_flashinfer_trtllm=lambda: True),
        ),
        patch.object(k3, "_add3", side_effect=lambda a, b, c, **kw: a + b),
    ):
        monkeypatch.setenv("SGLANG_K3_NVFP4_CAKE", "0")
        monkeypatch.setenv("SGLANG_K3_NVFP4_TP_SHARED_OVERLAP", "1")

        def run():
            return k3.KimiK3MoE._forward_fused(layer, x, prefix_sum=None)

        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(warm)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run()
        assert all(calls) if rows <= 64 else not any(calls)
        for _ in range(3):
            x.copy_(torch.randn_like(x))
            graph.replay()
            fused = torch.nn.functional.linear(x, front)
            expected = fused[:, :width] @ shared_w + fused[:, 2 * width :] @ routed_w
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-3)
