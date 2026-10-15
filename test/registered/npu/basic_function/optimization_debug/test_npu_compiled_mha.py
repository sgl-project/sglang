"""Run with pytest on one idle NPU; no checkpoint or network is needed."""

import sys
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import split_qkv_rmsnorm_rope

from sglang.srt.hardware_backend.npu.attention.compiled_mha import (
    npu_graph_mha_attention,
)
from sglang.srt.hardware_backend.npu.qkv import npu_split_qkv_rmsnorm_rope
from sglang.srt.utils import get_compiler_backend
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=120, suite="per-commit-1-npu-a2")


@pytest.fixture(autouse=True)
def reset_compiler():
    torch.npu.set_device(0)
    torch._dynamo.reset()
    yield
    torch.npu.synchronize()
    torch._dynamo.reset()


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_compiled_mha_replays_live_lengths_and_values(batch_size):
    q = torch.zeros((batch_size, 1, 128), device="npu", dtype=torch.bfloat16)
    k = torch.zeros((2, 128, 128), device="npu", dtype=q.dtype)
    v = torch.zeros_like(k)
    v[:, 0].fill_(2)
    v[:, 1].fill_(6)
    blocks = torch.zeros((batch_size, 1), device="npu", dtype=torch.int32)

    def forward(q, k, v, blocks):
        return npu_graph_mha_attention(
            q,
            k,
            v,
            blocks,
            block_size=128,
            num_heads=1,
            num_key_value_heads=1,
            scale=128**-0.5,
            actual_seq_lengths_kv=[1] * batch_size,
            v_head_dim=128,
        )

    compiled = torch.compile(
        forward,
        backend=get_compiler_backend("npugraph_ex"),
        fullgraph=True,
        dynamic=False,
    )
    for _ in range(2):
        compiled(q, k, v, blocks)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with (
        torch.compiler.set_stance(skip_guard_eval_unsafe=True),
        torch.npu.graph(graph, auto_dispatch_capture=True),
    ):
        output = compiled(q, k, v, blocks)

    with ThreadPoolExecutor(1, initializer=torch.npu.set_device, initargs=(0,)) as pool:
        # Include unequal lengths within a batch and a change to live V storage.
        for lengths, first, second in [
            ([1] * batch_size, 2, 6),
            ([2] * batch_size, 2, 6),
            ([1] * batch_size, 2, 6),
            ([1 + i % 2 for i in range(batch_size)], 8, 2),
        ]:
            v[:, 0].fill_(first)
            v[:, 1].fill_(second)
            torch.npu.synchronize()
            update = pool.submit(
                graph.update, cpu_update_input=[{"actual_seq_lengths_kv": lengths}]
            )
            graph.replay()
            update.result(timeout=30)
            torch.npu.synchronize()
            expected = (
                torch.tensor(
                    [first if n == 1 else (first + second) / 2 for n in lengths],
                    dtype=torch.float32,
                    device="cpu",
                )
                .view(batch_size, 1, 1)
                .expand(batch_size, 1, 128)
            )
            torch.testing.assert_close(output.cpu().float(), expected, rtol=0, atol=0)
        # Rebind page indices in the live device block table as well as lengths.
        v[1, 0].fill_(10)
        v[1, 1].fill_(14)
        blocks.fill_(1)
        torch.npu.synchronize()
        update = pool.submit(
            graph.update,
            cpu_update_input=[{"actual_seq_lengths_kv": [2] * batch_size}],
        )
        graph.replay()
        update.result(timeout=30)
        torch.npu.synchronize()
        actual = output.cpu().float()
        torch.testing.assert_close(actual, torch.full_like(actual, 12), rtol=0, atol=0)


@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_compiled_qkv_matches_original_kernel(batch_size):
    x = torch.randn((batch_size, 4096), device="npu", dtype=torch.bfloat16)
    angles = torch.randn((batch_size, 64), device="npu", dtype=x.dtype)
    sin = angles.sin().repeat(1, 2)
    cos = angles.cos().repeat(1, 2)
    weight = torch.ones(128, device="npu", dtype=x.dtype)

    def forward(x, sin, cos, weight):
        return npu_split_qkv_rmsnorm_rope(
            x,
            sin,
            cos,
            2048,
            1024,
            128,
            eps=1e-6,
            q_weight=weight,
            k_weight=weight,
        )

    compiled = torch.compile(
        forward, backend=get_compiler_backend("npugraph_ex"), fullgraph=True
    )
    for scale in (1, 2):
        weight.fill_(scale)
        expected = split_qkv_rmsnorm_rope(
            x,
            sin,
            cos,
            2048,
            1024,
            128,
            eps=1e-6,
            q_weight=weight,
            k_weight=weight,
        )
        actual = compiled(x, sin, cos, weight)
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", *sys.argv[1:]]))
