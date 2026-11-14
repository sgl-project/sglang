"""Regression tests for optional strided KDA preparation, including state writes."""

import pytest
import torch

from sglang.kernels.ops.attention.fla import kda
from sglang.kernels.ops.attention.fla.kda_qkv_pack import (
    kda_qkv_pack,
    supports_kda_qkv_pack,
)
from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("length", [1, 65, 4237, 32768])
def test_strided_pack_bit_parity_and_graph(length):
    torch.manual_seed(45)
    x = torch.randn((1, length, 6144), device="cuda", dtype=torch.bfloat16)
    q, k, v = [z.unflatten(-1, (16, 128)) for z in x.chunk(3, dim=-1)]

    def reference():
        return l2norm_fwd(q.contiguous()), l2norm_fwd(k.contiguous()), v.contiguous()

    actual = kda_qkv_pack(q, k, v)
    for a, b in zip(actual, reference()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay = kda_qkv_pack(q, k, v)
    x.mul_(0.5)
    graph.replay()
    for a, b in zip(replay, reference()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize(
    "lengths,dim", [([65], 128), ([65, 67], 128), ([4237], 128), ([65], 64)]
)
def test_complete_kda_state_and_track_parity(monkeypatch, lengths, dim):
    torch.manual_seed(46)
    total = sum(lengths)
    mixed = (
        torch.randn((1, total, 3 * 16 * dim), device="cuda", dtype=torch.bfloat16) * 0.1
    )
    q, k, v = [z.unflatten(-1, (16, dim)) for z in mixed.chunk(3, dim=-1)]
    gate = -torch.rand((1, total, 16, dim), device="cuda", dtype=torch.float32) * 0.02
    beta = torch.randn((1, total, 16), device="cuda", dtype=torch.float32)
    cu = torch.tensor(
        [0] + list(__import__("itertools").accumulate(lengths)),
        device="cuda",
        dtype=torch.int64,
    )
    pool = (
        torch.randn((4, 2, 16, dim, dim), device="cuda", dtype=torch.bfloat16) * 0.001
    )
    indices = torch.tensor([3, 1][: len(lengths)], device="cuda", dtype=torch.int64)
    snapshots = torch.tensor([0] * len(lengths), device="cuda", dtype=torch.int32)

    def run(enabled):
        state = pool.clone()[:, 0]
        track = torch.zeros(
            (len(lengths), 16, dim, dim), device="cuda", dtype=torch.float32
        )
        monkeypatch.setattr(kda, "_FUSE_KDA_QKV_PACK", enabled)
        output = kda.chunk_kda(
            q,
            k,
            v,
            gate,
            beta,
            initial_state=state,
            initial_state_indices=indices,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu,
            beta_is_raw=True,
            track_state=track,
            track_chunk_idx=snapshots,
        )
        return output, state, track

    before = run(False)
    after = run(True)
    for a, b in zip(after, before):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_unsupported_layout_guard():
    cpu = torch.zeros((1, 1, 16, 128), dtype=torch.bfloat16)
    assert not supports_kda_qkv_pack(cpu, cpu, cpu)
    x = torch.zeros((1, 1, 16, 64), device="cuda", dtype=torch.bfloat16)
    assert not supports_kda_qkv_pack(x, x, x)
