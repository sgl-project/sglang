"""Main QKV must retain token positions across indexer compression/RoPE."""

import os
from types import SimpleNamespace

import pytest

if os.getenv("SGLANG_TEST_CANN930_QSA_NATIVE") != "1":
    pytest.skip("Requires an idle NPU die", allow_module_level=True)

from test_qsa_va_norm_npu import make_case, runtime_context  # noqa: F401
import torch
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.models.qwen3_5 import Qwen3_5AttentionDecoderLayer


@pytest.mark.parametrize("use_va", [False, True])
@pytest.mark.parametrize("rows,axes", [(4, 1), (128, 3)])
def test_two_main_layers_with_indexer_between(monkeypatch, use_va, rows, axes):
    from sglang.srt.layers.attention.qsa import qsa_indexer

    monkeypatch.setattr(qsa_indexer, "_use_va_norm_rope", use_va)
    c = make_case(rows, axes, torch.bfloat16)
    c.obj.rotary_emb.head_size = 256
    c.positions.copy_(torch.arange(rows, device="npu").expand_as(c.positions) + 4096)
    if axes == 3:
        c.positions[1].add_(7)
        c.positions[2].add_(31)
    c.key_positions.copy_(c.positions // 4 * 4)
    qkv = torch.randn(rows, 2048, device="npu", dtype=torch.bfloat16)
    norms = [GemmaRMSNorm(256, eps=1e-6).to(device="npu", dtype=torch.bfloat16) for _ in range(2)]
    obj = SimpleNamespace(qkv_proj=lambda _: (qkv, None), rotary_emb=c.obj.rotary_emb,
                          refreshes_rope_cos_sin=True, q_size=768, kv_size=256,
                          head_dim=256, partial_rotary_factor=0.25,
                          q_norm=norms[0], k_norm=norms[1])
    prepare = Qwen3_5AttentionDecoderLayer.forward_prepare_npu

    def chain():
        obj.refreshes_rope_cos_sin = True
        first = prepare(obj, c.positions, None, None)
        before = (obj.rotary_emb.position_cos, obj.rotary_emb.position_sin)
        c.obj.project_qk(None, c.positions, pool=c.pool)
        c.obj.normalize_compressed_keys(c.keys, c.key_positions)
        assert obj.rotary_emb.position_cos is before[0]
        assert obj.rotary_emb.position_sin is before[1]
        obj.refreshes_rope_cos_sin = False
        return first, prepare(obj, c.positions, None, None)

    with torch.no_grad():
        first, later = chain()
        for a, b in zip(first, later):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, auto_dispatch_capture=False):
            first, later = chain()
        for step in range(3):
            c.positions.add_(5)
            c.key_positions.copy_(c.positions // 4 * 4)
            c.obj.rotary_emb.cos_sin_cache.mul_(0.984375)
            c.obj.q_layernorm.weight.add_(0.03125)
            qkv.mul_(-0.875)
            graph.replay()
            torch.npu.synchronize()
            for a, b in zip(first, later):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
