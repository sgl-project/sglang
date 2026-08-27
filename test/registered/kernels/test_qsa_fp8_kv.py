import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention import qwen_sparse_attn_backend as qsa_backend_module
from sglang.srt.layers.attention.qsa.kernel import qsa_sparse_attention
from sglang.srt.layers.attention.qsa.sparse_attn import (
    qwen_sparse_kv_extraction_compact_triton,
    sparse_gqa_fwd_interface_triton,
    sparse_gqa_fwd_interface_triton_ck,
)
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _quantize_fp8(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    return (tensor / scale).to(torch.float8_e4m3fn)


def test_qsa_sparse_attention_reference_applies_fp8_kv_descales():
    torch.manual_seed(23)
    q = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    k_scale, v_scale = 0.25, 0.5
    k = _quantize_fp8(torch.randn(7, 2, 16, dtype=torch.bfloat16), k_scale)
    v = _quantize_fp8(torch.randn(7, 2, 16, dtype=torch.bfloat16), v_scale)
    slots = torch.tensor([[0, 2, 4, 6], [1, 3, 5, -1]], dtype=torch.int32)

    actual = qsa_sparse_attention(q, k, v, slots, k_scale=k_scale, v_scale=v_scale)
    expected = qsa_sparse_attention(
        q,
        k.float() * k_scale,
        v.float() * v_scale,
        slots,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_qsa_fp8_prefill_matches_dequantized_reference():
    if not torch.cuda.is_available():
        return

    torch.manual_seed(27)
    device = "cuda"
    q = torch.randn(3, 4, 128, dtype=torch.bfloat16, device=device)
    k_scale, v_scale = 0.25, 0.5
    k = _quantize_fp8(
        torch.randn(3, 1, 128, dtype=torch.bfloat16, device=device), k_scale
    )
    v = _quantize_fp8(
        torch.randn(3, 1, 128, dtype=torch.bfloat16, device=device), v_scale
    )
    indices = torch.tensor(
        [[0, -1, -1], [0, 1, -1], [0, 1, 2]],
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens = torch.tensor([0, 3], dtype=torch.int32, device=device)
    softmax_scale = 128**-0.5

    actual = sparse_gqa_fwd_interface_triton(
        q,
        k,
        v,
        max_seqlen_k=3,
        indices=indices,
        cu_seqlens=cu_seqlens,
        scale=softmax_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    expected = qsa_sparse_attention(
        q,
        k,
        v,
        indices,
        softmax_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


def test_qsa_fp8_chunk_prefill_matches_dequantized_reference():
    if not torch.cuda.is_available():
        return

    torch.manual_seed(29)
    device = "cuda"
    q = torch.randn(3, 4, 128, dtype=torch.bfloat16, device=device)
    k_scale, v_scale = 0.25, 0.5
    k = _quantize_fp8(
        torch.randn(6, 1, 128, dtype=torch.bfloat16, device=device), k_scale
    )
    v = _quantize_fp8(
        torch.randn(6, 1, 128, dtype=torch.bfloat16, device=device), v_scale
    )
    indices = torch.tensor(
        [[0, 1, 2, 3], [0, 2, 3, 4], [1, 3, 4, 5]],
        dtype=torch.int32,
        device=device,
    )
    cu_q = torch.tensor([0, 3], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 6], dtype=torch.int32, device=device)
    kv_lens = torch.tensor([6], dtype=torch.int32, device=device)
    softmax_scale = 128**-0.5

    actual = sparse_gqa_fwd_interface_triton_ck(
        q,
        k,
        v,
        indices,
        cu_q,
        cu_k,
        kv_lens,
        softmax_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    expected = qsa_sparse_attention(
        q,
        k,
        v,
        indices,
        softmax_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


def test_qsa_bf16_chunk_prefill_regression():
    if not torch.cuda.is_available():
        return

    torch.manual_seed(30)
    device = "cuda"
    q = torch.randn(3, 4, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(6, 1, 128, dtype=torch.bfloat16, device=device)
    v = torch.randn(6, 1, 128, dtype=torch.bfloat16, device=device)
    indices = torch.tensor(
        [[0, 1, 2, 3], [0, 2, 3, 4], [1, 3, 4, 5]],
        dtype=torch.int32,
        device=device,
    )
    cu_q = torch.tensor([0, 3], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 6], dtype=torch.int32, device=device)
    kv_lens = torch.tensor([6], dtype=torch.int32, device=device)
    softmax_scale = 128**-0.5

    actual = sparse_gqa_fwd_interface_triton_ck(
        q, k, v, indices, cu_q, cu_k, kv_lens, softmax_scale
    )
    expected = qsa_sparse_attention(q, k, v, indices, softmax_scale)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_qsa_fp8_compact_gather_dequantizes_for_flash_attention():
    if not torch.cuda.is_available():
        return

    torch.manual_seed(31)
    device = "cuda"
    k_scale, v_scale = 0.25, 0.5
    k = _quantize_fp8(
        torch.randn(16, 1, 16, dtype=torch.bfloat16, device=device), k_scale
    )
    v = _quantize_fp8(
        torch.randn(16, 1, 16, dtype=torch.bfloat16, device=device), v_scale
    )
    req_to_token = torch.tensor(
        [[3, 5, 7, 9, 11, 13], [2, 4, 6, 8, 10, 12]],
        dtype=torch.int32,
        device=device,
    )
    req_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    indices = torch.tensor(
        [[0, 3, 5, -1], [1, 4, -1, -1]], dtype=torch.int32, device=device
    )
    seq_lens = torch.tensor([6, 5], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    out_k = torch.empty(8, 1, 16, dtype=torch.bfloat16, device=device)
    out_v = torch.empty_like(out_k)

    qwen_sparse_kv_extraction_compact_triton(
        k,
        v,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        cu_k,
        out_k,
        out_v,
        batch=2,
        topk=4,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    selected_slots = torch.tensor([3, 9, 13, 4, 10], device=device)
    torch.testing.assert_close(
        out_k[:5].float(),
        k[selected_slots].float() * k_scale,
        rtol=0,
        atol=2e-2,
    )
    torch.testing.assert_close(
        out_v[:5].float(),
        v[selected_slots].float() * v_scale,
        rtol=0,
        atol=2e-2,
    )

    # TRTLLM consumes FP8 scratch directly and applies the descales in BMM1/BMM2.
    out_k_fp8 = torch.empty(8, 1, 16, dtype=torch.float8_e4m3fn, device=device)
    out_v_fp8 = torch.empty_like(out_k_fp8)
    qwen_sparse_kv_extraction_compact_triton(
        k,
        v,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        cu_k,
        out_k_fp8,
        out_v_fp8,
        batch=2,
        topk=4,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch.testing.assert_close(
        out_k_fp8[:5].float(), k[selected_slots].float(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        out_v_fp8[:5].float(), v[selected_slots].float(), rtol=0, atol=0
    )


def test_qsa_fp8_cache_write_preserves_prefill_kv():
    class MutatingPool:
        dtype = torch.float8_e4m3fn

        def set_kv_buffer(
            self, layer, loc, cache_k, cache_v, k_scale=None, v_scale=None
        ):
            self.args = (cache_k, cache_v, k_scale, v_scale)
            if k_scale is not None:
                cache_k.div_(k_scale)
                cache_v.div_(v_scale)

    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend.token_to_kv_pool = MutatingPool()
    layer = SimpleNamespace(layer_id=0, k_scale_float=0.25, v_scale_float=0.5)
    k = torch.randn(3, 1, 16, dtype=torch.bfloat16)
    v = torch.randn(3, 1, 16, dtype=torch.bfloat16)
    expected_k, expected_v = k.clone(), v.clone()

    backend._store_kv(layer, torch.arange(3, dtype=torch.int32), k, v)

    written_k, written_v, written_k_scale, written_v_scale = (
        backend.token_to_kv_pool.args
    )
    assert written_k is not k and written_v is not v
    assert written_k_scale == 0.25 and written_v_scale == 0.5
    torch.testing.assert_close(k, expected_k)
    torch.testing.assert_close(v, expected_v)


def test_qsa_trtllm_decode_receives_fp8_kv_descales(monkeypatch):
    seen = {}

    def fake_valid_counts(seq_lens, indices, counts, batch, topk):
        counts.fill_(topk)

    def fake_compact(*args, **kwargs):
        return None

    def fake_decode(**kwargs):
        seen.update(kwargs)
        return torch.zeros_like(kwargs["query"])

    monkeypatch.setattr(
        qsa_backend_module, "qwen_sparse_valid_counts_triton", fake_valid_counts
    )
    monkeypatch.setattr(
        qsa_backend_module,
        "qwen_sparse_kv_extraction_compact_triton",
        fake_compact,
    )
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(8, dtype=torch.int32).reshape(1, 8)
    )
    backend._fa2_scratch = {}
    backend._trtllm_sparse_tables = {}
    backend._trtllm_workspace = torch.empty(1, dtype=torch.uint8)
    backend._cuda_graph_max_tokens = 0
    layer = SimpleNamespace(scaling=0.125, k_scale_float=0.25, v_scale_float=0.5)
    q = torch.randn(1, 4, 16, dtype=torch.bfloat16)
    k = torch.empty(8, 1, 16, dtype=torch.float8_e4m3fn)
    v = torch.empty_like(k)
    topk_indices = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    forward_batch = SimpleNamespace(req_pool_indices=torch.tensor([0]))
    metadata = SimpleNamespace(
        sequence_lengths=torch.tensor([8], dtype=torch.int32),
        row_req_pool_indices=None,
        is_cuda_graph=False,
    )

    output = backend._forward_trtllm_sparse(
        q, k, v, layer, forward_batch, metadata, topk_indices, fake_decode
    )

    assert output.shape == (1, 64)
    assert seen["kv_cache"][0].dtype == torch.float8_e4m3fn
    assert seen["bmm1_scale"] == 0.125 * 0.25
    assert seen["bmm2_scale"] == 0.5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
