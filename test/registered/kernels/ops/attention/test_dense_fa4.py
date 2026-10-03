# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.dense_kv import _pack_kv
from sglang.kernels.ops.attention.flash_attn.cute.testing import attention_ref
from sglang.srt.layers.attention.flashattention_dense_backend import (
    FlashAttentionDenseBackend,
)
from sglang.srt.layers.attention.graph_variants import DLLM_FULL_WINDOW
from sglang.srt.model_executor.runner_utils.capture_mode import (
    _set_capture_attention_variant,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="SM100 family required",
)


@pytest.mark.parametrize("prefix", [False, True])
@torch.inference_mode()
def test_pack_kv_offsets_beyond_int32(prefix):
    heads, dim = 8, 256
    far = 2**31 // (heads * dim)
    k = torch.empty(far + 1, heads, dim, device="cuda", dtype=torch.bfloat16)
    v, kd, vd = torch.empty_like(k), torch.empty_like(k), torch.empty_like(k)
    k[far].fill_(3)
    v[far].fill_(4)
    qo = torch.tensor(
        [0, far, far + 1] if not prefix else [0, 0], device="cuda", dtype=torch.int32
    )
    ki = torch.tensor(
        [0, 0, 0] if not prefix else [0, 1], device="cuda", dtype=torch.int32
    )
    ids = torch.tensor([far], device="cuda", dtype=torch.int32)
    cu = torch.empty_like(qo)
    _pack_kv[(qo.numel() - 1, 2)](
        k,
        v,
        k,
        v,
        qo,
        ki,
        ids,
        kd,
        vd,
        cu,
        heads,
        dim,
        *k.stride()[:2],
        *v.stride()[:2],
        *k.stride()[:2],
        *v.stride()[:2],
        TILE=1024,
    )
    index = 0 if prefix else far
    torch.testing.assert_close(kd[index], k[far], atol=0, rtol=0)
    torch.testing.assert_close(vd[index], v[far], atol=0, rtol=0)
    torch.testing.assert_close(cu, qo + ki)


@pytest.mark.parametrize(
    "dim,causal,batch,fixed",
    [
        (256, False, 2, False),
        (256, True, 2, False),
        (512, False, 2, False),
        (512, True, 2, False),
        (256, False, 8, True),
    ],
)
@torch.inference_mode()
def test_dense_attention_replay(dim, causal, batch, fixed, monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    torch.manual_seed(41)
    heads, kv_heads = 16, 8 if dim == 256 else 2
    prefix = 1023 if dim == 256 else 8193
    query_offsets = list(range(0, (batch + 1) * 256, 256)) if fixed else [0, 17, 273]
    prefix_offsets = (
        [i * prefix for i in range(batch + 1)] if fixed else [0, 1, prefix + 1]
    )
    qkv = torch.randn(
        query_offsets[-1],
        heads + 2 * kv_heads,
        dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q, k, v = qkv.split((heads, kv_heads, kv_heads), dim=1)
    q.mul_(0.1)
    kb = torch.randn(
        prefix_offsets[-1] + 64, kv_heads, dim, device="cuda", dtype=q.dtype
    )
    vb = torch.randn_like(kb)
    qo = torch.tensor(query_offsets, device="cuda", dtype=torch.int32)
    ki = torch.tensor(prefix_offsets, device="cuda", dtype=torch.int32)
    ids = torch.randperm(kb.shape[0], device="cuda")[: prefix_offsets[-1]]
    out = torch.empty_like(q)
    backend = SimpleNamespace(
        _dense_workspaces={},
        _dense_graph_slots=batch,
        _dense_graph_tokens=q.shape[0],
        sliding_window_size=1023,
        max_context_len=9000,
    )
    layer = SimpleNamespace(sliding_window_size=1023 if dim == 256 else None)
    window = 1023 if dim == 256 and causal else -1

    def run():
        FlashAttentionDenseBackend._forward_extend_kernel(
            backend,
            layer,
            q,
            k,
            v,
            out,
            kb,
            vb,
            qo,
            ki,
            ids,
            None,
            causal,
            None,
            256,
            1.0,
            1.0,
            sm_scale=dim**-0.5,
            sliding_window_size=window,
        )

    def check():
        for row in range(batch):
            qs, qe = qo[row : row + 2].tolist()
            ps, pe = ki[row : row + 2].tolist()
            keys = torch.cat((kb[ids[ps:pe]], k[qs:qe]))
            values = torch.cat((vb[ids[ps:pe]], v[qs:qe]))
            expected, _ = attention_ref(
                q[qs:qe][None].float(),
                keys[None].float(),
                values[None].float(),
                causal=causal,
                window_size=(window if window > 0 else None, None),
            )
            torch.testing.assert_close(
                out[qs:qe].float(), expected[0], atol=3e-3, rtol=2e-2
            )

    run()
    check()
    try:
        _set_capture_attention_variant(DLLM_FULL_WINDOW if fixed else None)
        run()
        check()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
    finally:
        _set_capture_attention_variant(None)
    q.add_(0.01)
    k.mul_(0.95)
    v.add_(0.03)
    ids.copy_(ids.flip(0))
    if not fixed:
        qo[1].add_(3)
        ki[1].add_(31)
    graph.replay()
    check()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
