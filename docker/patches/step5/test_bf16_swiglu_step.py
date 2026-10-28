import json
import unittest

import torch
from flashinfer.fused_moe import trtllm_bf16_routed_moe
from flashinfer.fused_moe.core import (
    _maybe_get_cached_w3_w1_permute_indices,
    convert_to_block_layout,
)
from flashinfer.tllm_enums import ActivationType

torch.manual_seed(123)


def prepared(w, shuffled):
    cache = {}
    result = []
    for a in w:
        if shuffled:
            idx = _maybe_get_cached_w3_w1_permute_indices(
                cache, a.view(torch.uint8), 128, is_gated_act_gemm=False
            )
            a = a.view(torch.uint8)[idx].view(torch.bfloat16)
        result.append(
            convert_to_block_layout(a.view(torch.uint8), 128).view(torch.bfloat16)
        )
    return torch.stack(result)


def case(
    num_tokens=32,
    hidden_size=1024,
    intermediate_size=512,
    num_experts=8,
    offset=0,
    local=8,
    shuffled=True,
    pdl=True,
    boundary=False,
):
    top_k = 2
    x = torch.randn(num_tokens, hidden_size, device="cuda").bfloat16()
    a = (
        torch.randn(local, 2 * intermediate_size, hidden_size, device="cuda") * 0.08
    ).bfloat16()
    b = (
        torch.randn(local, hidden_size, intermediate_size, device="cuda") * 0.04
    ).bfloat16()
    limits = torch.linspace(3, 7, local, device="cuda")
    if boundary:
        x.zero_()
        x[:, 0] = 1
        a.zero_()
        b.zero_()
        limits.fill_(7)
        gates = torch.tensor(
            [-20.0, -7.0, -1.0, 0.0, 1.0, 6.96875, 7.0, 7.03125, 8.0, 20.0],
            device="cuda",
        ).bfloat16()
        for c in range(intermediate_size):
            a[:, c, 0] = -10 if c % 2 else 10
            a[:, intermediate_size + c, 0] = gates[c % len(gates)]
            b[:, c, c] = 1
    ids = torch.stack(
        [torch.randperm(num_experts, device="cuda")[:top_k] for _ in range(num_tokens)]
    ).int()
    weights = torch.rand(num_tokens, top_k, device="cuda")
    weights = weights / weights.sum(-1, keepdim=True) * 3
    aa, bb = prepared(a, shuffled), prepared(b, shuffled)

    def run():
        return trtllm_bf16_routed_moe(
            (ids, weights),
            x,
            aa,
            bb,
            num_experts,
            top_k,
            None,
            None,
            intermediate_size,
            offset,
            local,
            routing_method_type=5,
            use_shuffled_weight=shuffled,
            activation_type=ActivationType.SwigluStep,
            gemm1_clamp_limit=limits,
            enable_pdl=pdl,
        )

    y = run()
    torch.cuda.synchronize()
    ref = torch.zeros(num_tokens, hidden_size, device="cuda")
    for e in range(local):
        ti, ki = torch.where(ids == e + offset)
        if ti.numel() == 0:
            continue
        z = torch.nn.functional.linear(x[ti], a[e]).float()
        u, g = z.chunk(2, -1)
        z = (
            torch.nn.functional.silu(g).clamp(max=limits[e])
            * u.clamp(-limits[e], limits[e])
        ).bfloat16()
        v = torch.nn.functional.linear(z, b[e]).float()
        ref.index_add_(0, ti, v * weights[ti, ki, None])
    ref = ref.bfloat16()
    error = (y.float() - ref.float()).abs()
    metrics = dict(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        offset=offset,
        local=local,
        shuffled=shuffled,
        pdl=pdl,
        boundary=boundary,
        max_abs=error.max().item(),
        rmse=error.square().mean().sqrt().item(),
        cos=torch.nn.functional.cosine_similarity(
            y.float().flatten(), ref.float().flatten(), dim=0
        ).item(),
    )
    print(json.dumps(metrics), flush=True)
    torch.testing.assert_close(
        y, ref, atol=1e-8 if boundary else 0.125, rtol=0 if boundary else 0.02
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        gy = run()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(gy, y, atol=0, rtol=0)
    print("PASS_GRAPH", flush=True)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "Blackwell CUDA device is required",
)
class TestTrtllmBf16SwigluStep(unittest.TestCase):
    def test_formula_and_graph_replay(self):
        cases = [
            dict(num_tokens=1),
            dict(num_tokens=129),
            dict(offset=4, local=4),
            dict(num_tokens=32, hidden_size=256, intermediate_size=128, boundary=True),
            dict(
                num_tokens=32,
                hidden_size=4096,
                intermediate_size=1536,
                num_experts=8,
                local=8,
            ),
            dict(num_tokens=32, shuffled=True, pdl=False),
        ]
        for kwargs in cases:
            with self.subTest(**kwargs):
                case(**kwargs)


if __name__ == "__main__":
    unittest.main()
