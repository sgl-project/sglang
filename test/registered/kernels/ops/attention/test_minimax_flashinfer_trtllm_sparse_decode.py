import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.minimax_sparse_backend import (
    MiniMaxSparseAttnBackend,
    _positive_python_scale,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@pytest.mark.parametrize("fp8", [False, True])
def test_trtllm_sparse_decode_dispatch_contract(fp8: bool):
    backend = object.__new__(MiniMaxSparseAttnBackend)
    backend.page_size = 128
    backend.topk_blocks = 16
    backend.block_size_k = 128
    backend.model_dtype = torch.bfloat16
    backend.fp8_attn_gemm = fp8
    backend._trtllm_sparse_workspace = torch.empty(32, dtype=torch.uint8)
    backend._trtllm_sparse_multi_ctas_kv_counter_buffer = torch.zeros(
        16, dtype=torch.uint8
    )

    calls = []

    def fake_decode(**kwargs):
        calls.append(kwargs)
        return torch.empty_like(kwargs["query"], dtype=kwargs["out_dtype"])

    backend._trtllm_sparse_decode_fn = fake_decode

    batch_size, num_q_heads, head_dim = 2, 16, 128
    q_dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    kv_dtype = q_dtype
    # Match MiniMax's split QKV projection: q is a strided view in BF16 mode.
    q_storage = torch.empty((batch_size, num_q_heads, head_dim + 8), dtype=q_dtype)
    q = q_storage[..., :head_dim]
    assert not q.is_contiguous()
    k_cache = torch.empty((4 * 128, 1, head_dim), dtype=kv_dtype)
    v_cache = torch.empty_like(k_cache)
    page_table = torch.arange(batch_size * 16, dtype=torch.int32).view(batch_size, 16)
    seq_lens = torch.tensor([2048, 1937], dtype=torch.int32)
    layer = SimpleNamespace(
        scaling=head_dim**-0.5,
        q_scale_float=0.5 if fp8 else None,
        k_scale_float=0.25 if fp8 else None,
        v_scale_float=0.75 if fp8 else None,
    )

    out = backend._trtllm_sparse_main_decode(
        q, page_table, seq_lens, k_cache, v_cache, layer
    )

    assert out.shape == q.shape
    assert out.dtype == torch.bfloat16
    assert len(calls) == 1
    call = calls[0]
    assert call["query"].data_ptr() == q.data_ptr()
    assert call["query"].stride() == q.stride()
    assert call["enable_block_sparse_attention"] is True
    assert call["backend"] == "trtllm-gen"
    assert call["kv_layout"] == "HND"
    assert call["q_len_per_req"] == 1
    assert call["max_seq_len"] == 2048
    assert call["block_tables"].shape == (1, batch_size, 16)
    assert call["seq_lens"].shape == (1, batch_size)
    assert call["block_tables"].data_ptr() == page_table.data_ptr()
    assert call["seq_lens"].data_ptr() == seq_lens.data_ptr()
    assert call["multi_ctas_kv_counter_buffer"] is (
        backend._trtllm_sparse_multi_ctas_kv_counter_buffer
    )
    assert call["workspace_buffer"] is backend._trtllm_sparse_workspace

    k_hnd, v_hnd = call["kv_cache"]
    assert k_hnd.shape == (4, 1, 128, 128)
    assert v_hnd.shape == k_hnd.shape
    assert k_hnd.stride() == (128 * 128, 128 * 128, 128, 1)
    assert v_hnd.stride() == k_hnd.stride()
    if fp8:
        assert call["bmm1_scale"] == pytest.approx(0.5 * 0.25 * layer.scaling)
        assert call["bmm2_scale"] == pytest.approx(0.75)
    else:
        assert call["bmm1_scale"] == pytest.approx(layer.scaling)
        assert call["bmm2_scale"] == 1.0


def test_trtllm_sparse_decode_kernel_matches_reference():
    flashinfer = pytest.importorskip("flashinfer")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("TRTLLM-GEN block-sparse decode requires SM100 or SM103")

    from sglang.srt.layers.attention.trtllm_mla_backend import (
        make_persistent_multi_ctas_kv_counter_buffer,
    )

    device = torch.device("cuda")
    batch_size, num_q_heads, head_dim = 2, 16, 128
    page_size, topk_blocks, num_pages = 128, 4, 8
    torch.manual_seed(42)

    backend = object.__new__(MiniMaxSparseAttnBackend)
    backend.page_size = page_size
    backend.topk_blocks = topk_blocks
    backend.block_size_k = page_size
    backend.model_dtype = torch.bfloat16
    backend.fp8_attn_gemm = False
    backend._trtllm_sparse_decode_fn = (
        flashinfer.decode.trtllm_batch_decode_with_kv_cache
    )
    backend._trtllm_sparse_workspace = torch.empty(
        256 * 1024 * 1024, dtype=torch.uint8, device=device
    )
    backend._trtllm_sparse_multi_ctas_kv_counter_buffer = (
        make_persistent_multi_ctas_kv_counter_buffer(
            device, num_q_heads=num_q_heads, max_batch_size=batch_size
        )
    )

    q_storage = torch.randn(
        batch_size,
        num_q_heads,
        head_dim + 8,
        dtype=torch.bfloat16,
        device=device,
    )
    q = q_storage[..., :head_dim]
    k_cache = torch.randn(
        num_pages * page_size, 1, head_dim, dtype=torch.bfloat16, device=device
    )
    v_cache = torch.randn_like(k_cache)
    page_table = torch.tensor(
        [[0, 2, 4, 6], [1, 3, 5, 7]], dtype=torch.int32, device=device
    )
    seq_lens = torch.tensor([512, 397], dtype=torch.int32, device=device)
    layer = SimpleNamespace(
        scaling=head_dim**-0.5,
        q_scale_float=None,
        k_scale_float=None,
        v_scale_float=None,
    )

    out = backend._trtllm_sparse_main_decode(
        q, page_table, seq_lens, k_cache, v_cache, layer
    )

    paged_k = k_cache.view(num_pages, page_size, head_dim)
    paged_v = v_cache.view(num_pages, page_size, head_dim)
    refs = []
    for batch_idx in range(batch_size):
        length = int(seq_lens[batch_idx])
        k = paged_k[page_table[batch_idx].long()].reshape(-1, head_dim)[:length]
        v = paged_v[page_table[batch_idx].long()].reshape(-1, head_dim)[:length]
        scores = torch.matmul(q[batch_idx].float(), k.float().transpose(0, 1))
        probs = torch.softmax(scores * layer.scaling, dim=-1)
        refs.append(torch.matmul(probs, v.float()))
    ref = torch.stack(refs)

    cosine = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.flatten(), dim=0
    ).item()
    assert cosine > 0.999, f"cosine similarity: {cosine}"


def test_positive_python_scale_is_graph_stable():
    assert _positive_python_scale(None, "scale") == 1.0
    assert _positive_python_scale(-1.0, "scale") == 1.0
    assert _positive_python_scale(0.5, "scale") == 0.5
    with pytest.raises(TypeError, match="Python scalar"):
        _positive_python_scale(torch.tensor(0.5), "scale")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
