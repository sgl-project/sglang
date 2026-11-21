"""Reused VA prep through real QSAIndexer methods, checked against retained Q04."""

import ast
import copy
import json
import os
import textwrap
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

if os.getenv("SGLANG_TEST_CANN930_QSA_NATIVE") != "1":
    pytest.skip("Requires an idle NPU die", allow_module_level=True)

import sgl_kernel_npu  # noqa: F401
import torch
from safetensors import safe_open
from sglang.srt.hardware_backend.npu.utils import init_npu_backend

init_npu_backend()

from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs


@pytest.fixture(autouse=True)
def runtime_context():
    torch.npu.set_device(0)
    publish(ServerArgs(model_path="dummy", device="npu", attention_backend="ascend"), role="test")
    try:
        yield
    finally:
        reset_context()


def legacy_methods():
    root = Path(os.environ["RUN_ROOT"])
    source = (root / "framework_q01_q03_q04/python/sglang/srt/layers/attention/qsa/qsa_indexer.py").read_text()
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "QSAIndexer")
    from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb
    ns = {"torch": torch, "apply_rotary_emb": apply_rotary_emb}
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("project_qk", "normalize_compressed_keys", "apply_rope"):
            exec("from __future__ import annotations\n" + textwrap.dedent(ast.get_source_segment(source, node)), ns)
    return ns


def make_case(rows, axes, cache_dtype, neox=True):
    torch.manual_seed(931)
    rotary = MRotaryEmbedding(
        head_size=128, rotary_dim=64, max_position_embeddings=65536,
        base=1000000, is_neox_style=neox, dtype=cache_dtype,
        mrope_section=[11, 11, 10], mrope_interleaved=True,
    ).to(device="npu")
    rotary.cos_sin_cache = rotary.cos_sin_cache.to(cache_dtype)
    norms = [GemmaRMSNorm(128, eps=1e-6).to(device="npu", dtype=torch.bfloat16) for _ in range(2)]
    model = Path(os.environ["MODEL_PATH"])
    mapping = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    for label, norm in zip(("q", "k"), norms):
        key = f"model.language_model.layers.3.self_attn.indexer.{label}_layernorm.weight"
        with safe_open(model / mapping[key], framework="pt", device="cpu") as handle:
            norm.weight.data.copy_(handle.get_tensor(key))
        norm.requires_grad_(False)
    qk = torch.randn(rows, 640, device="npu", dtype=torch.bfloat16)
    nkeys = rows // 4 if rows >= 1024 else rows
    keys = torch.randn(nkeys, 1, 128, device="npu", dtype=torch.bfloat16)
    positions = torch.randint(0, 65536, (3, rows) if axes == 3 else (rows,), dtype=torch.int64).npu()
    key_positions = positions[..., :nkeys].clone()
    pool = SimpleNamespace(qsa_compressed_dtype=torch.bfloat16)
    obj = SimpleNamespace(index_n_heads=4, index_kv_heads=1, index_head_dim=128,
                          rotary_emb=rotary, q_layernorm=norms[0], k_layernorm=norms[1],
                          index_qk_proj=lambda _: (qk, None))
    for name in ("_use_fused_prep", "_normalize_and_rope", "apply_rope", "project_qk", "normalize_compressed_keys"):
        setattr(obj, name, MethodType(getattr(QSAIndexer, name), obj))
    return SimpleNamespace(obj=obj, qk=qk, keys=keys, positions=positions,
                           key_positions=key_positions, pool=pool)


@pytest.mark.parametrize("rows,axes,cache_dtype", [
    (1, 3, torch.bfloat16), (32, 3, torch.bfloat16), (128, 3, torch.bfloat16),
    (4096, 3, torch.bfloat16), (128, 1, torch.bfloat16), (128, 3, torch.float32),
])
def test_real_methods_weights_and_graph(rows, axes, cache_dtype):
    from sglang.srt.layers.attention.qsa import qsa_indexer as module

    assert module._use_va_norm_rope
    c = make_case(rows, axes, cache_dtype)
    old = legacy_methods()
    reference = copy.copy(c.obj)
    reference.rotary_emb = copy.deepcopy(c.obj.rotary_emb)
    reference.rotary_emb.cos_sin_cache = c.obj.rotary_emb.cos_sin_cache
    reference.apply_rope = MethodType(old["apply_rope"], reference)

    def native():
        q, k, stored = old["project_qk"](reference, None, c.positions, pool=c.pool)
        assert not stored
        compressed = old["normalize_compressed_keys"](reference, c.keys, c.key_positions)
        return q, k, compressed

    def candidate():
        c.obj.rotary_emb.get_cos_sin_with_position(c.positions)
        before = (c.obj.rotary_emb.position_cos, c.obj.rotary_emb.position_sin)
        q, k, stored = c.obj.project_qk(None, c.positions, pool=c.pool)
        assert not stored
        compressed = c.obj.normalize_compressed_keys(c.keys, c.key_positions)
        assert c.obj.rotary_emb.position_cos is before[0]
        assert c.obj.rotary_emb.position_sin is before[1]
        return q, k, compressed

    graphs, outputs, shared = {}, {}, {}
    with torch.no_grad():
        for name, fn in (("native", native), ("va", candidate)):
            for _ in range(2):
                fn()
                torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, auto_dispatch_capture=False):
                outputs[name] = fn()
                shared[name] = (c.obj.rotary_emb.position_cos, c.obj.rotary_emb.position_sin)
            graphs[name] = graph
        for step in range(3):
            if step:
                c.qk.mul_(-0.875)
                c.keys.mul_(-0.75)
                c.positions.add_(3).remainder_(65536)
                c.key_positions.add_(7).remainder_(65536)
                c.obj.q_layernorm.weight.add_(0.03125)
                c.obj.k_layernorm.weight.sub_(0.015625)
                c.obj.rotary_emb.cos_sin_cache.mul_(0.9375)
            for graph in graphs.values():
                graph.replay()
            torch.npu.synchronize()
            for i, (a, b) in enumerate(zip(outputs["va"], outputs["native"])):
                torch.testing.assert_close(a, b, atol=0 if i == 1 else 0.02, rtol=0 if i == 1 else 0.02)
            reference.rotary_emb.get_cos_sin_with_position(c.positions)
            expected_main = (reference.rotary_emb.position_cos, reference.rotary_emb.position_sin)
            for a, b in zip(shared["va"], expected_main):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
                assert a.dtype == b.dtype == cache_dtype


def test_unsupported_rotary_keeps_native_path():
    c = make_case(33, 3, torch.bfloat16, neox=False)
    old = legacy_methods()
    with torch.no_grad():
        a = c.obj.project_qk(None, c.positions, pool=c.pool)
        b = old["project_qk"](c.obj, None, c.positions, pool=c.pool)
        torch.testing.assert_close(a[0], b[0], atol=0, rtol=0)
        torch.testing.assert_close(a[1], b[1], atol=0, rtol=0)
        assert a[2] is b[2] is False
