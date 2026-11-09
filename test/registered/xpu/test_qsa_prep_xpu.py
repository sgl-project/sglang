"""Native XPU QSA indexer preparation against SGLang's unfused path."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.qsa.kernel import average_pool_qsa_keys
from sglang.srt.layers.attention.qsa.metadata import build_rope_position_matrix
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=30, suite="stage-b-test-1-gpu-xpu")

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available(), reason="XPU required"
)

HEAD_DIM = 128
HEADS = 4
RATIO = 4


@pytest.fixture(autouse=True)
def runtime_context():
    publish(ServerArgs(model_path="dummy"), role="test")
    try:
        yield
    finally:
        reset_context()


def make_indexer(interleaved):
    rotary = MRotaryEmbedding(
        head_size=HEAD_DIM,
        rotary_dim=64,
        max_position_embeddings=256,
        base=1000000,
        is_neox_style=True,
        dtype=torch.bfloat16,
        mrope_section=[11, 11, 10] if interleaved else None,
        mrope_interleaved=interleaved,
    )
    config = SimpleNamespace(
        hidden_size=64,
        indexer_n_heads=HEADS,
        indexer_kv_heads=1,
        indexer_head_dim=HEAD_DIM,
        indexer_budget=2048,
        indexer_compress_ratio=RATIO,
        rms_norm_eps=1e-6,
    )
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        indexer = QSAIndexer(config, layer_id=0, rotary_emb=rotary).to("xpu")
    finally:
        torch.set_default_dtype(previous_dtype)
    with torch.no_grad():
        indexer.index_qk_proj.weight.normal_(0, 0.02)
    return indexer


class Pool:
    qsa_compressed_dtype = torch.bfloat16
    index_state_dtype = torch.bfloat16

    def __init__(self, compressed_dtype=torch.bfloat16):
        self.qsa_compressed_dtype = compressed_dtype
        self.keys = torch.zeros(32, 1, HEAD_DIM, device="xpu", dtype=torch.bfloat16)
        self.qsa_rope_position_buffer = torch.zeros(
            32, 3, device="xpu", dtype=torch.int64
        )
        self.compressed = torch.zeros(
            16, 1, HEAD_DIM, device="xpu", dtype=compressed_dtype
        )

    def get_qsa_key_state_buffer(self, layer_id):
        return self.keys

    def get_qsa_compressed_k_buffer(self, layer_id):
        return self.compressed


@pytest.mark.parametrize("interleaved", [False, True])
def test_xpu_query_prep_matches_unfused_and_stores_state(monkeypatch, interleaved):
    import sgl_kernel

    native = sgl_kernel.qsa_index_q_norm_rope_store
    calls = []

    def traced(*args):
        calls.append(True)
        return native(*args)

    monkeypatch.setattr(sgl_kernel, "qsa_index_q_norm_rope_store", traced)
    torch.manual_seed(23)
    indexer = make_indexer(interleaved)
    pool = Pool()
    hidden = torch.randn(8, 64, device="xpu", dtype=torch.bfloat16)
    positions = torch.randint(0, 128, (3, 8), device="xpu", dtype=torch.int64)
    if not interleaved:
        positions = positions[0]
    slots = torch.arange(1, 9, device="xpu", dtype=torch.int64)

    expected, raw_keys, _ = indexer.project_qk(hidden, positions)
    actual, _, stored = indexer.project_qk(
        hidden, positions, pool=pool, cache_loc=slots
    )

    assert stored and calls == [True]
    torch.testing.assert_close(actual, expected, rtol=0.015, atol=0.02)
    torch.testing.assert_close(pool.keys[slots], raw_keys)
    expected_positions = (
        positions.T if interleaved else positions[:, None].expand(-1, 3)
    )
    torch.testing.assert_close(pool.qsa_rope_position_buffer[slots], expected_positions)


@pytest.mark.parametrize("interleaved", [False, True])
def test_xpu_key_compression_matches_unfused(monkeypatch, interleaved):
    import sgl_kernel

    native = sgl_kernel.qsa_index_k_compress_store
    calls = []

    def traced(*args):
        calls.append(True)
        return native(*args)

    monkeypatch.setattr(sgl_kernel, "qsa_index_k_compress_store", traced)
    torch.manual_seed(24)
    indexer = make_indexer(interleaved)
    pool = Pool()
    pool.keys.normal_()
    pool.qsa_rope_position_buffer.random_(0, 128)
    groups = torch.tensor([[3, 0, 1, 2], [8, 9, 10, 11]], device="xpu")
    writes = torch.tensor([2, 5], device="xpu", dtype=torch.int32)
    pooled = average_pool_qsa_keys(pool.keys[groups])
    positions = indexer._rope_from_matrix(pool.qsa_rope_position_buffer[groups[:, 0]])
    expected = indexer.normalize_compressed_keys(pooled, positions)

    assert indexer._use_fused_compress(pool)
    indexer._fused_compress_store(pool, groups, writes)

    assert calls == [True]
    torch.testing.assert_close(
        pool.compressed[writes.long()], expected, rtol=0.015, atol=0.02
    )


@pytest.mark.parametrize("extend", [False, True])
def test_xpu_compression_dispatches_for_prefill_and_ring(extend):
    torch.manual_seed(25)
    indexer = make_indexer(True)
    pool = Pool()
    token_k = torch.randn(8, 1, HEAD_DIM, device="xpu", dtype=torch.bfloat16)
    rope_positions = torch.randint(0, 128, (3, 8), device="xpu")
    pool.keys[:8] = token_k
    pool.qsa_rope_position_buffer[:8] = rope_positions.T
    starts = torch.tensor([0, 4], device="xpu", dtype=torch.int64)
    writes = torch.tensor([2, 5], device="xpu", dtype=torch.int64)
    expected = indexer.normalize_compressed_keys(
        average_pool_qsa_keys(token_k.reshape(2, RATIO, 1, HEAD_DIM)),
        rope_positions[:, starts],
    )
    metadata = SimpleNamespace(
        token_to_kv_pool=pool,
        compress_member_rows=starts if extend else None,
        is_cuda_graph=False,
        write_locs=writes,
        compress_group_positions=torch.tensor([3, 7], device="xpu"),
        extend_rope_matrix=build_rope_position_matrix(rope_positions, 8),
        compress_group_ring_locs=torch.arange(
            8, device="xpu", dtype=torch.int64
        ).reshape(2, RATIO),
    )

    indexer.update_key_state_and_compress(
        token_k,
        logical_positions=torch.arange(8, device="xpu"),
        rope_positions=rope_positions,
        metadata=metadata,
        state_stored=True,
    )

    torch.testing.assert_close(pool.compressed[writes], expected, rtol=0.015, atol=0.02)


def test_xpu_fp8_prep_and_compress_dispatch_to_native(monkeypatch):
    import sgl_kernel

    torch.manual_seed(26)
    indexer = make_indexer(False)
    pool = Pool(torch.float8_e4m3fn)
    hidden = torch.randn(8, 64, device="xpu", dtype=torch.bfloat16)
    positions = torch.arange(8, device="xpu")
    slots = torch.arange(1, 9, device="xpu")
    reference_q, raw_keys, _ = indexer.project_qk(hidden, positions)
    reference_k = indexer.normalize_compressed_keys(
        average_pool_qsa_keys(raw_keys.reshape(2, RATIO, 1, HEAD_DIM)),
        positions[::RATIO],
    )
    calls = []
    original_q = sgl_kernel.qsa_index_q_norm_rope_store
    original_k = sgl_kernel.qsa_index_k_compress_store

    def traced_q(*args, **kwargs):
        calls.append("q")
        return original_q(*args, **kwargs)

    def traced_k(*args, **kwargs):
        calls.append("k")
        return original_k(*args, **kwargs)

    monkeypatch.setattr(sgl_kernel, "qsa_index_q_norm_rope_store", traced_q)
    monkeypatch.setattr(sgl_kernel, "qsa_index_k_compress_store", traced_k)
    q, _, stored = indexer.project_qk(hidden, positions, pool=pool, cache_loc=slots)
    assert stored and calls == ["q"]
    assert q.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(pool.keys[slots], raw_keys)
    expected_q = reference_q.to(torch.float8_e4m3fn)
    assert torch.all(
        (q.view(torch.uint8).int() - expected_q.view(torch.uint8).int()).abs() <= 1
    )
    groups = slots.reshape(2, RATIO).to(torch.int32)
    writes = torch.tensor([2, 5], device="xpu", dtype=torch.int32)
    assert indexer._use_fused_compress(pool)
    indexer._fused_compress_store(pool, groups, writes)
    assert calls == ["q", "k"]
    actual = pool.compressed[writes.long()]
    expected_k = reference_k.to(torch.float8_e4m3fn)
    assert torch.all(
        (actual.view(torch.uint8).int() - expected_k.view(torch.uint8).int()).abs() <= 1
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


@pytest.mark.parametrize("keep_fp32", [True, False])
def test_mrope_forward_xpu_cache_dtype(keep_fp32):
    rope = MRotaryEmbedding(
        head_size=HEAD_DIM,
        rotary_dim=HEAD_DIM,
        max_position_embeddings=256,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
        mrope_section=[16, 24, 24],
    ).to("xpu")
    rope.keep_cos_sin_cache_fp32 = keep_fp32
    query = torch.randn(3, HEADS * HEAD_DIM, dtype=torch.bfloat16, device="xpu")
    key = torch.randn_like(query)
    positions = torch.arange(3, device="xpu").repeat(3, 1)
    rope.forward_xpu(positions, query, key)
    expected = torch.float32 if keep_fp32 else torch.bfloat16
    assert rope.cos_sin_cache.dtype == expected


def test_fp16_norm_weight_keeps_the_unfused_compression():
    indexer = make_indexer(interleaved=False)
    pool = Pool()
    assert indexer._use_fused_compress(pool)
    indexer.k_layernorm.weight.data = indexer.k_layernorm.weight.data.half()
    assert not indexer._use_fused_compress(pool)
