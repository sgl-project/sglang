"""Small Ascend 950 DSA cache tests; no server, model or distributed launch."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu
import sgl_kernel_npu

from sglang.srt.hardware_backend.npu import memory_pool_npu as mp


def pool(fp8=True, layers=(0, 2), size=512):
    with patch.object(mp, "get_parallel", return_value=SimpleNamespace(attn_dcp_size=1, attn_dcp_rank=0)):
        return mp.NPUMLATokenToKVPool(
            size=size, page_size=128,
            dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16,
            kv_lora_rank=128, qk_rope_head_dim=64, layer_num=3,
            device="npu", enable_memory_saver=False, index_head_dim=128,
            indexer_layer_ids=layers, kv_cache_dim=260 if fp8 else None,
        )


def values(n):
    x = torch.randn(n, 128, dtype=torch.bfloat16, device="npu")
    k, s = torch_npu.npu_dynamic_quant(x, dst_type=torch.float8_e4m3fn)
    return k, s.float().reshape(-1)


def same(a, b):
    torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)


def test_allocation_and_native_page_descriptor():
    p = pool()
    assert p.index_k_buffer.shape == (2, 5, 128 * 132)
    assert p.index_k_buffer.dtype == torch.uint8
    assert p.index_k_scale_buffer is None
    native = p._mla_native_buffers()
    assert native.index_k.shape == (2, 5, 128, 1, 132)
    assert native.index_k.data_ptr() == p.index_k_buffer.data_ptr()
    assert native.index_scales is None
    assert native.index_width == 132
    assert p.get_state_layer_ids() == [0, 2]
    empty = pool(layers=())
    assert empty.num_indexer_layers == 0 and empty.index_k_buffer.numel() == 0
    bf = pool(fp8=False)
    assert bf.index_k_buffer.shape == (2, 5, 128, 1, 128)
    loc = torch.tensor([128, 129], device="npu")
    k = torch.randn(2, 1, 128, dtype=torch.bfloat16, device="npu")
    bf.set_index_k_buffer(2, loc, k)
    same(bf.get_index_k_buffer(2).view(-1, 1, 128)[loc], k)


def test_write_padding_boundaries_and_overwrite():
    p = pool()
    p.index_k_buffer.fill_(165)
    loc = torch.tensor([0, 127, 128, 129, 255, 256, 0], device="npu")
    expected = p.get_index_k_with_scale_buffer(2).cpu()
    for _ in range(2):
        k, s = values(len(loc))
        p.set_index_k_scale_buffer(2, loc, k, s)
        for i, slot in enumerate(loc.cpu().tolist()):
            if slot:
                page, offset = divmod(slot, 128)
                expected[page, offset * 128 : (offset + 1) * 128] = k.view(torch.uint8).cpu()[i]
                expected[page, 128 * 128 + offset * 4 : 128 * 128 + (offset + 1) * 4] = s.view(torch.uint8).cpu()[i * 4 : (i + 1) * 4]
        actual = p.get_index_k_with_scale_buffer(2).cpu()
        # Duplicate padding writes may change only slot 0's K and scale.
        actual[0, :128] = expected[0, :128]
        actual[0, 16384:16388] = expected[0, 16384:16388]
        same(actual, expected)
    assert torch.all(p.get_index_k_with_scale_buffer(0).cpu() == 165)


@pytest.mark.parametrize("loc_dtype", [torch.int32, torch.int64])
def test_writer_graph_replay_with_padding(loc_dtype):
    p = pool()
    loc = torch.tensor([128, 129, 0, 0], device="npu", dtype=loc_dtype)
    k, s = values(4)
    # Warm up Triton compilation before recording the static writer graph.
    p.set_index_k_scale_buffer(0, loc, k, s)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        p.set_index_k_scale_buffer(0, loc, k, s)
    for positions in ([255, 256, 0, 0], [384, 0, 0, 0]):
        loc.copy_(torch.tensor(positions, device="npu"))
        new_k, new_s = values(4)
        k.copy_(new_k)
        s.copy_(new_s)
        before = p.index_k_buffer.clone()
        graph.replay()
        packed = p.get_index_k_with_scale_buffer(0)
        for i, pos in enumerate(positions):
            if pos:
                page, slot = divmod(pos, 128)
                same(packed[page, slot*128:(slot+1)*128], k.view(torch.uint8)[i])
                same(packed[page, 16384+slot*4:16384+(slot+1)*4], s.view(torch.uint8)[i*4:(i+1)*4])
        page_zero = p.index_k_buffer[0, 0].cpu()
        before_zero = before[0, 0].cpu()
        same(page_zero[128:16384], before_zero[128:16384])
        same(page_zero[16388:], before_zero[16388:])
        same(p.index_k_buffer[1], before[1])


@pytest.mark.parametrize("qlen", [1, 4])
def test_pool_writer_to_packed_reader(qlen):
    p = pool()
    k, s = values(512)
    loc = torch.arange(128, 640, device="npu")
    p.set_index_k_scale_buffer(0, loc, k, s)
    # Dummy writes to page 0 must not affect scores for real pages 1..4.
    p.set_index_k_scale_buffer(
        0, torch.zeros(3, dtype=torch.int32, device="npu"), *values(3)
    )
    q, qs = values(qlen * 64)
    q, qs = q.reshape(qlen, 64, 128), qs.reshape(qlen, 64)
    w = torch.rand(qlen, 64, device="npu", dtype=torch.bfloat16)
    sq = torch.tensor([qlen], device="npu", dtype=torch.int32)
    sk = torch.tensor([511], device="npu", dtype=torch.int32)
    table = torch.tensor([[1, 2, 3, 4]], device="npu", dtype=torch.int32)
    actual = sgl_kernel_npu.quant_lightning_indexer_packed(q, p.get_index_k_with_scale_buffer(0), w, qs, sq, sk, table, 256, 3)
    separate_k = torch.zeros(5, 128, 1, 128, device="npu", dtype=torch.uint8)
    separate_k[1:].copy_(k.view(torch.uint8).reshape(4, 128, 1, 128))
    separate_s = torch.ones(5, 128, 1, device="npu")
    separate_s[1:].copy_(s.reshape(4, 128, 1))
    expected = torch.ops.npu.npu_quant_lightning_indexer(
        q, separate_k.view(torch.float8_e4m3fn), w, qs, separate_s, 0, 0,
        actual_seq_lengths_query=sq, actual_seq_lengths_key=sk,
        block_table=table, layout_query="TND", layout_key="PA_BSND",
        sparse_count=256, sparse_mode=3)
    same(actual, expected[0] if isinstance(expected, (list, tuple)) else expected)


@pytest.mark.parametrize("prefill", [False, True])
def test_actual_indexer_forward(prefill):
    """Run the real forward method with tiny projection stubs, not a model."""
    from sglang.srt.layers.attention.dsa import dsa_npu_indexer as mod

    p = pool()
    p.set_index_k_scale_buffer(0, torch.arange(128, 640, device="npu"), *values(512))
    n = 4 if prefill else 1
    q = torch.randn(n, 64 * 128, dtype=torch.bfloat16, device="npu")
    x = torch.randn(n, 128, dtype=torch.bfloat16, device="npu")
    weights = torch.rand(n, 64, dtype=torch.bfloat16, device="npu")

    class Rotary:
        is_neox_style = False
        cos_sin_cache = torch.zeros(512, 128, device="npu")

        def __call__(self, positions, q, k):
            return q, k

    indexer = SimpleNamespace(
        n_heads=64, head_dim=128, rope_head_dim=64, hidden_size=128,
        index_topk=256, dsa_enable_prefill_cp=False, alt_stream=None,
        rotary_emb=Rotary(), wq_b=lambda z: (z,), wk=lambda z: (z,),
        k_norm=lambda z: z, weights_proj=lambda z: (weights,),
    )
    sq = torch.tensor([n], dtype=torch.int32, device="npu")
    sk = torch.tensor([512], dtype=torch.int32, device="npu")
    table = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32, device="npu")
    backend = SimpleNamespace(forward_metadata=SimpleNamespace(
        seq_lens_cpu_int=None, seq_lens=sk, actual_seq_lengths_q=sq,
        block_tables=table), is_draft_worker=False)
    mode = SimpleNamespace(is_extend=lambda: prefill,
        is_draft_extend_v2=lambda: False, is_target_verify=lambda: False)
    batch = SimpleNamespace(forward_mode=mode, seq_lens=sk, extend_seq_lens=sq,
        out_cache_loc=torch.arange(640-n, 640, device="npu"), attn_cp_metadata=None)
    with patch.object(mod, "get_token_to_kv_pool", return_value=p), \
         patch.object(mod, "get_attn_backend", return_value=backend), \
         patch.object(mod, "get_parallel", return_value=SimpleNamespace(dcp_enabled=False)), \
         patch.object(mod, "_use_ag_after_qlora", False), \
         patch.object(mod, "envs", SimpleNamespace(SGLANG_NPU_USE_MULTI_STREAM=SimpleNamespace(get=lambda: False))), \
         patch.object(sgl_kernel_npu, "quant_lightning_indexer_packed", wraps=sgl_kernel_npu.quant_lightning_indexer_packed) as reader:
        actual = mod.DSANPUIndexerMixin.forward_npu(indexer, x, q, torch.arange(n, device="npu"), batch, 0)
    reader.assert_called_once()
    args = reader.call_args.args
    assert args[1].data_ptr() == p.get_index_k_with_scale_buffer(0).data_ptr()
    cache = args[1].cpu()
    separate_k = cache[:, :16384].contiguous().view(torch.float8_e4m3fn).reshape(5, 128, 1, 128).npu()
    separate_s = cache[:, 16384:].contiguous().view(torch.float32).reshape(5, 128, 1).npu()
    expected = torch.ops.npu.npu_quant_lightning_indexer(
        args[0], separate_k, args[2], args[3], separate_s, 0, 0,
        actual_seq_lengths_query=sq, actual_seq_lengths_key=sk, block_table=table,
        layout_query="TND", layout_key="PA_BSND", sparse_count=256, sparse_mode=3)
    if isinstance(expected, (list, tuple)):
        expected = expected[0]
    same(actual, expected.squeeze(1))
