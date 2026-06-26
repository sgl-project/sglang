import pytest
import torch

from sglang.srt.layers.attention import dsa_backend as dsa_backend_module
from sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute import (
    DeepseekSparseAttnBackendMTPPrecomputeMixin,
)
from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    DSAMetadata,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="base-b-kernel-unit-1-gpu-large")


class _DummyDSAPrecomputeBackend(DeepseekSparseAttnBackendMTPPrecomputeMixin):
    def __init__(self, req_to_token, real_page_size, topk, next_n):
        self.req_to_token = req_to_token
        self.real_page_size = real_page_size
        self.dsa_index_topk = topk
        self.speculative_num_draft_tokens = next_n
        self.device = req_to_token.device
        self.dsa_decode_impl = "none"

    def _transform_table_1_to_real(self, page_table):
        if self.real_page_size == 1:
            return page_table
        return (
            page_table[
                :,
                torch.arange(
                    0,
                    page_table.shape[1],
                    self.real_page_size,
                    device=page_table.device,
                ),
            ]
            // self.real_page_size
        )


@pytest.mark.parametrize("real_page_size", [1, 4])
@pytest.mark.parametrize("seq_dtype", [torch.int32, torch.int64])
def test_fused_dsa_decode_metadata(real_page_size, seq_dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from sglang.srt.layers.attention.triton_ops.dsa_metadata import (
        fused_dsa_decode_metadata,
    )

    device = "cuda"
    bs = 7
    req_to_token_cols = 80
    max_len = 37
    topk = 23

    seq_lens = torch.tensor([37, 12, 23, 4, 31, 18, 9], dtype=seq_dtype, device=device)
    req_pool_indices = torch.tensor(
        [3, 0, 5, 2, 7, 1, 6], dtype=torch.int64, device=device
    )
    req_to_token = (
        torch.arange(8 * req_to_token_cols, dtype=torch.int32, device=device)
        .view(8, req_to_token_cols)
        .mul_(3)
    )

    cache_seqlens = torch.empty(bs, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.empty(bs + 1, dtype=torch.int32, device=device)
    page_table_1 = torch.full((bs, max_len + 11), -1, dtype=torch.int32, device=device)
    dsa_cache_seqlens = torch.empty(bs, dtype=torch.int32, device=device)
    dsa_cu_seqlens_k = torch.empty(bs + 1, dtype=torch.int32, device=device)

    if real_page_size > 1:
        real_cols = (max_len + real_page_size - 1) // real_page_size
        real_page_table = torch.full(
            (bs, real_cols + 3), -1, dtype=torch.int32, device=device
        )
    else:
        real_page_table = page_table_1

    fused_dsa_decode_metadata(
        seq_lens=seq_lens,
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        cache_seqlens=cache_seqlens,
        cu_seqlens_k=cu_seqlens_k,
        page_table_1=page_table_1,
        dsa_cache_seqlens=dsa_cache_seqlens,
        dsa_cu_seqlens_k=dsa_cu_seqlens_k,
        real_page_table=real_page_table,
        bs=bs,
        max_len=max_len,
        dsa_index_topk=topk,
        real_page_size=real_page_size,
    )
    torch.cuda.synchronize()

    ref_cache = seq_lens.to(torch.int32)
    ref_dsa = ref_cache.clamp(max=topk)
    ref_page = req_to_token[req_pool_indices, :max_len]

    assert torch.equal(cache_seqlens, ref_cache)
    assert torch.equal(
        cu_seqlens_k, torch.nn.functional.pad(torch.cumsum(ref_cache, 0), (1, 0))
    )
    assert torch.equal(dsa_cache_seqlens, ref_dsa)
    assert torch.equal(
        dsa_cu_seqlens_k, torch.nn.functional.pad(torch.cumsum(ref_dsa, 0), (1, 0))
    )
    assert torch.equal(page_table_1[:, :max_len], ref_page)
    assert torch.all(page_table_1[:, max_len:] == -1)

    if real_page_size > 1:
        ref_real = ref_page[:, torch.arange(0, max_len, real_page_size, device=device)]
        ref_real = ref_real // real_page_size
        assert torch.equal(real_page_table[:, : ref_real.shape[1]], ref_real)
        assert torch.all(real_page_table[:, ref_real.shape[1] :] == -1)


@pytest.mark.parametrize("real_page_size", [1, 4])
@pytest.mark.parametrize("seq_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("next_n", [1, 3])
def test_fused_dsa_target_verify_metadata(real_page_size, seq_dtype, next_n):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from sglang.srt.layers.attention.triton_ops.dsa_metadata import (
        fused_dsa_target_verify_metadata,
    )

    device = "cuda"
    bs = 5
    req_to_token_cols = 80
    max_seq_len = 37
    max_seqlen_k = max_seq_len + next_n
    expanded_size = bs * next_n
    topk = 23

    seq_lens = torch.tensor([37, 12, 23, 4, 31], dtype=seq_dtype, device=device)
    req_pool_indices = torch.tensor([3, 0, 5, 2, 7], dtype=torch.int64, device=device)
    req_to_token = (
        torch.arange(8 * req_to_token_cols, dtype=torch.int32, device=device)
        .view(8, req_to_token_cols)
        .mul_(3)
    )

    cache_seqlens = torch.empty(bs, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.empty(bs + 1, dtype=torch.int32, device=device)
    page_table_1 = torch.full(
        (expanded_size, max_seqlen_k + 11), -1, dtype=torch.int32, device=device
    )
    seqlens_expanded = torch.empty(expanded_size, dtype=torch.int32, device=device)
    dsa_cache_seqlens = torch.empty(expanded_size, dtype=torch.int32, device=device)
    dsa_cu_seqlens_k = torch.empty(expanded_size + 1, dtype=torch.int32, device=device)

    if real_page_size > 1:
        real_cols = (max_seqlen_k + real_page_size - 1) // real_page_size
        real_page_table = torch.full(
            (expanded_size, real_cols + 3), -1, dtype=torch.int32, device=device
        )
    else:
        real_page_table = page_table_1

    fused_dsa_target_verify_metadata(
        seq_lens=seq_lens,
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        cache_seqlens=cache_seqlens,
        cu_seqlens_k=cu_seqlens_k,
        page_table_1=page_table_1,
        seqlens_expanded=seqlens_expanded,
        dsa_cache_seqlens=dsa_cache_seqlens,
        dsa_cu_seqlens_k=dsa_cu_seqlens_k,
        real_page_table=real_page_table,
        bs=bs,
        max_seqlen_k=max_seqlen_k,
        dsa_index_topk=topk,
        real_page_size=real_page_size,
        next_n=next_n,
    )
    torch.cuda.synchronize()

    ref_cache = seq_lens.to(torch.int32) + next_n
    ref_page = req_to_token[req_pool_indices, :max_seqlen_k]
    ref_page = torch.repeat_interleave(ref_page, repeats=next_n, dim=0)
    ref_expanded = (
        seq_lens.to(torch.int32).view(bs, 1)
        + torch.arange(1, next_n + 1, dtype=torch.int32, device=device).view(1, next_n)
    ).reshape(-1)
    ref_dsa = ref_expanded.clamp(max=topk)

    assert torch.equal(cache_seqlens, ref_cache)
    assert torch.equal(
        cu_seqlens_k, torch.nn.functional.pad(torch.cumsum(ref_cache, 0), (1, 0))
    )
    assert torch.equal(seqlens_expanded, ref_expanded)
    assert torch.equal(dsa_cache_seqlens, ref_dsa)
    assert torch.equal(
        dsa_cu_seqlens_k, torch.nn.functional.pad(torch.cumsum(ref_dsa, 0), (1, 0))
    )
    assert torch.equal(page_table_1[:, :max_seqlen_k], ref_page)
    assert torch.all(page_table_1[:, max_seqlen_k:] == -1)

    if real_page_size > 1:
        ref_real = ref_page[
            :, torch.arange(0, max_seqlen_k, real_page_size, device=device)
        ]
        ref_real = ref_real // real_page_size
        assert torch.equal(real_page_table[:, : ref_real.shape[1]], ref_real)
        assert torch.all(real_page_table[:, ref_real.shape[1] :] == -1)


@pytest.mark.parametrize("real_page_size", [1, 4])
@pytest.mark.parametrize("seq_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("accept_lengths", [[1, 2, 3, 1, 2], [3, 1, 4, 2, 1]])
def test_fused_dsa_draft_extend_metadata(real_page_size, seq_dtype, accept_lengths):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from sglang.srt.layers.attention.triton_ops.dsa_metadata import (
        fused_dsa_draft_extend_metadata,
    )

    device = "cuda"
    bs = len(accept_lengths)
    req_to_token_cols = 80
    max_seqlen_k = 37
    total_len = sum(accept_lengths)
    next_n = max(accept_lengths)
    topk = 23

    seq_lens = torch.tensor([37, 12, 23, 4, 31], dtype=seq_dtype, device=device)
    extend_seq_lens = torch.tensor(accept_lengths, dtype=torch.int32, device=device)
    req_pool_indices = torch.tensor([3, 0, 5, 2, 7], dtype=torch.int64, device=device)
    req_to_token = (
        torch.arange(8 * req_to_token_cols, dtype=torch.int32, device=device)
        .view(8, req_to_token_cols)
        .mul_(3)
    )

    cache_seqlens = torch.empty(bs, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.empty(bs + 1, dtype=torch.int32, device=device)
    page_table_1 = torch.full(
        (total_len, max_seqlen_k + 11), -1, dtype=torch.int32, device=device
    )
    seqlens_expanded = torch.empty(total_len, dtype=torch.int32, device=device)
    dsa_cache_seqlens = torch.empty(total_len, dtype=torch.int32, device=device)
    dsa_cu_seqlens_k = torch.empty(total_len + 1, dtype=torch.int32, device=device)

    if real_page_size > 1:
        real_cols = (max_seqlen_k + real_page_size - 1) // real_page_size
        real_page_table = torch.full(
            (total_len, real_cols + 3), -1, dtype=torch.int32, device=device
        )
    else:
        real_page_table = page_table_1

    fused_dsa_draft_extend_metadata(
        seq_lens=seq_lens,
        extend_seq_lens=extend_seq_lens,
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        cache_seqlens=cache_seqlens,
        cu_seqlens_k=cu_seqlens_k,
        page_table_1=page_table_1,
        seqlens_expanded=seqlens_expanded,
        dsa_cache_seqlens=dsa_cache_seqlens,
        dsa_cu_seqlens_k=dsa_cu_seqlens_k,
        real_page_table=real_page_table,
        bs=bs,
        total_len=total_len,
        max_seqlen_k=max_seqlen_k,
        dsa_index_topk=topk,
        real_page_size=real_page_size,
        max_total_len=bs * next_n,
    )
    torch.cuda.synchronize()

    ref_cache = seq_lens.to(torch.int32)
    ref_page_base = req_to_token[req_pool_indices, :max_seqlen_k]
    ref_page = torch.repeat_interleave(ref_page_base, repeats=extend_seq_lens, dim=0)
    ref_expanded_parts = []
    for qo_len, kv_len in zip(accept_lengths, seq_lens.tolist(), strict=True):
        ref_expanded_parts.append(
            torch.arange(
                kv_len - qo_len + 1,
                kv_len + 1,
                dtype=torch.int32,
                device=device,
            )
        )
    ref_expanded = torch.cat(ref_expanded_parts)
    ref_dsa = ref_expanded.clamp(max=topk)

    assert torch.equal(cache_seqlens, ref_cache)
    assert torch.equal(
        cu_seqlens_k, torch.nn.functional.pad(torch.cumsum(ref_cache, 0), (1, 0))
    )
    assert torch.equal(seqlens_expanded, ref_expanded)
    assert torch.equal(dsa_cache_seqlens, ref_dsa)
    assert torch.equal(
        dsa_cu_seqlens_k, torch.nn.functional.pad(torch.cumsum(ref_dsa, 0), (1, 0))
    )
    assert torch.equal(page_table_1[:, :max_seqlen_k], ref_page)
    assert torch.all(page_table_1[:, max_seqlen_k:] == -1)

    if real_page_size > 1:
        ref_real = ref_page[
            :, torch.arange(0, max_seqlen_k, real_page_size, device=device)
        ]
        ref_real = ref_real // real_page_size
        assert torch.equal(real_page_table[:, : ref_real.shape[1]], ref_real)
        assert torch.all(real_page_table[:, ref_real.shape[1] :] == -1)


@pytest.mark.parametrize("real_page_size", [1, 4])
def test_fused_precompute_decode_metadata(real_page_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    bs = 6
    max_len = 37
    topk = 23
    req_to_token = (
        torch.arange(8 * 80, dtype=torch.int32, device=device).view(8, 80).mul_(3)
    )
    req_pool_indices = torch.tensor(
        [3, 0, 5, 2, 7, 1], dtype=torch.int64, device=device
    )
    seq_lens = torch.tensor([37, 12, 23, 4, 31, 18], dtype=torch.int64, device=device)
    seq_lens_cpu = seq_lens.cpu()

    backend = _DummyDSAPrecomputeBackend(req_to_token, real_page_size, topk, next_n=3)
    metadata = backend._precompute_decode_mode(
        bs, req_pool_indices, seq_lens, seq_lens_cpu
    )

    ref_cache = seq_lens.to(torch.int32)
    ref_page = req_to_token[req_pool_indices, :max_len]
    ref_dsa = ref_cache.clamp(max=topk)

    assert torch.equal(metadata.cache_seqlens, ref_cache)
    assert torch.equal(
        metadata.cu_seqlens_k,
        torch.nn.functional.pad(torch.cumsum(ref_cache, 0), (1, 0)),
    )
    assert torch.equal(metadata.page_indices, ref_page)
    assert metadata.seqlens_expanded is metadata.cache_seqlens
    assert torch.equal(metadata.dsa_cache_seqlens, ref_dsa)
    assert torch.equal(
        metadata.dsa_cu_seqlens_k,
        torch.nn.functional.pad(torch.cumsum(ref_dsa, 0), (1, 0)),
    )
    assert metadata.seqlens_expanded_size == bs
    assert metadata.max_len == max_len
    assert metadata.max_seqlen_k == max_len

    if real_page_size > 1:
        ref_real = ref_page[:, torch.arange(0, max_len, real_page_size, device=device)]
        ref_real = ref_real // real_page_size
        assert torch.equal(metadata.real_page_table, ref_real)
    else:
        assert metadata.real_page_table is None


@pytest.mark.parametrize("real_page_size", [1, 4])
@pytest.mark.parametrize("next_n", [1, 3])
def test_fused_precompute_target_verify_metadata(real_page_size, next_n):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    bs = 5
    topk = 23
    req_to_token = (
        torch.arange(8 * 80, dtype=torch.int32, device=device).view(8, 80).mul_(3)
    )
    req_pool_indices = torch.tensor([3, 0, 5, 2, 7], dtype=torch.int64, device=device)
    seq_lens = torch.tensor([37, 12, 23, 4, 31], dtype=torch.int64, device=device)
    seq_lens_cpu = seq_lens.cpu()
    max_seqlen_k = int(seq_lens_cpu.max().item() + next_n)
    expanded_size = bs * next_n

    backend = _DummyDSAPrecomputeBackend(req_to_token, real_page_size, topk, next_n)
    metadata = backend._precompute_target_verify_mode(
        bs, req_pool_indices, seq_lens, seq_lens_cpu
    )

    ref_cache = seq_lens.to(torch.int32) + next_n
    ref_page = req_to_token[req_pool_indices, :max_seqlen_k]
    ref_page = torch.repeat_interleave(ref_page, repeats=next_n, dim=0)
    ref_expanded = (
        seq_lens.to(torch.int32).view(bs, 1)
        + torch.arange(1, next_n + 1, dtype=torch.int32, device=device).view(1, next_n)
    ).reshape(-1)
    ref_dsa = ref_expanded.clamp(max=topk)

    assert torch.equal(metadata.cache_seqlens, ref_cache)
    assert torch.equal(
        metadata.cu_seqlens_k,
        torch.nn.functional.pad(torch.cumsum(ref_cache, 0), (1, 0)),
    )
    assert torch.equal(metadata.page_indices, ref_page)
    assert torch.equal(metadata.seqlens_expanded, ref_expanded)
    assert torch.equal(metadata.dsa_cache_seqlens, ref_dsa)
    assert torch.equal(
        metadata.dsa_cu_seqlens_k,
        torch.nn.functional.pad(torch.cumsum(ref_dsa, 0), (1, 0)),
    )
    assert metadata.seqlens_expanded_size == expanded_size
    assert metadata.max_len == -1
    assert metadata.max_seqlen_k == max_seqlen_k

    if real_page_size > 1:
        ref_real = ref_page[
            :, torch.arange(0, max_seqlen_k, real_page_size, device=device)
        ]
        ref_real = ref_real // real_page_size
        assert torch.equal(metadata.real_page_table, ref_real)
    else:
        assert metadata.real_page_table is None


def _compiled_kernel_cache_size(kernel):
    return sum(len(entry[0]) for entry in kernel.device_caches.values())


def test_apply_cuda_graph_metadata_decode_fused_wiring(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    bs = 5
    max_len = 37
    topk = 23
    real_page_size = 4
    req_to_token = (
        torch.arange(8 * 80, dtype=torch.int32, device=device).view(8, 80).mul_(3)
    )
    req_pool_indices = torch.tensor([3, 0, 5, 2, 7], dtype=torch.int64, device=device)
    seq_lens = torch.tensor([37, 12, 23, 4, 31], dtype=torch.int64, device=device)
    seq_lens_cpu = seq_lens.cpu()
    real_cols = (max_len + real_page_size - 1) // real_page_size

    metadata = DSAMetadata(
        page_size=real_page_size,
        cache_seqlens_int32=torch.empty(bs, dtype=torch.int32, device=device),
        max_seq_len_q=1,
        max_seq_len_k=max_len,
        cu_seqlens_q=torch.arange(bs + 1, dtype=torch.int32, device=device),
        cu_seqlens_k=torch.empty(bs + 1, dtype=torch.int32, device=device),
        page_table_1=torch.full((bs, max_len), -1, dtype=torch.int32, device=device),
        real_page_table=torch.full(
            (bs, real_cols), -1, dtype=torch.int32, device=device
        ),
        dsa_cache_seqlens_int32=torch.empty(bs, dtype=torch.int32, device=device),
        dsa_cu_seqlens_q=torch.arange(bs + 1, dtype=torch.int32, device=device),
        dsa_cu_seqlens_k=torch.empty(bs + 1, dtype=torch.int32, device=device),
        dsa_extend_seq_lens_list=[],
        dsa_seqlens_expanded=torch.empty(bs, dtype=torch.int32, device=device),
    )

    backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
    backend.decode_cuda_graph_metadata = {bs: metadata}
    backend.req_to_token = req_to_token
    backend.real_page_size = real_page_size
    backend.dsa_index_topk = topk
    backend.dsa_decode_impl = "none"
    backend.device = torch.device(device)
    backend.speculative_num_draft_tokens = 3
    backend.set_dsa_prefill_impl = lambda forward_batch: None

    class _FakeDeepGemm:
        @staticmethod
        def get_num_sms():
            return 1

        @staticmethod
        def get_paged_mqa_logits_metadata(context_lens, *_args):
            return context_lens.to(torch.int32).contiguous()

    monkeypatch.setattr(dsa_backend_module, "deep_gemm", _FakeDeepGemm)
    monkeypatch.setattr(dsa_backend_module, "_USE_FUSED_METADATA_GENERATION", True)

    backend._apply_cuda_graph_metadata(
        bs=bs,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        forward_mode=ForwardMode.DECODE,
        spec_info=None,
    )
    torch.cuda.synchronize()

    ref_cache = seq_lens.to(torch.int32)
    ref_dsa = ref_cache.clamp(max=topk)
    ref_page = req_to_token[req_pool_indices, :max_len]
    ref_real = ref_page[:, torch.arange(0, max_len, real_page_size, device=device)]
    ref_real = ref_real // real_page_size

    assert torch.equal(metadata.cache_seqlens_int32, ref_cache)
    assert torch.equal(
        metadata.cu_seqlens_k,
        torch.nn.functional.pad(torch.cumsum(ref_cache, 0), (1, 0)),
    )
    assert torch.equal(metadata.page_table_1, ref_page)
    assert torch.equal(metadata.dsa_cache_seqlens_int32, ref_dsa)
    assert torch.equal(
        metadata.dsa_cu_seqlens_k,
        torch.nn.functional.pad(torch.cumsum(ref_dsa, 0), (1, 0)),
    )
    assert torch.equal(metadata.real_page_table, ref_real)
    assert torch.equal(metadata.paged_mqa_ctx_lens_2d, ref_cache.view(-1, 1))
    assert torch.equal(metadata.paged_mqa_schedule_metadata, ref_cache.view(-1, 1))


def test_refresh_paged_mqa_schedule_metadata_uses_deepgemm_out(monkeypatch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    schedule = torch.full((5, 2), -1, dtype=torch.int32, device=device)
    seqlens_2d = torch.tensor([[4, 5], [6, 7]], dtype=torch.int32, device=device)
    metadata = DSAMetadata(
        page_size=1,
        cache_seqlens_int32=torch.empty(2, dtype=torch.int32, device=device),
        max_seq_len_q=1,
        max_seq_len_k=8,
        cu_seqlens_q=torch.empty(3, dtype=torch.int32, device=device),
        cu_seqlens_k=torch.empty(3, dtype=torch.int32, device=device),
        page_table_1=torch.empty((2, 8), dtype=torch.int32, device=device),
        real_page_table=torch.empty((2, 8), dtype=torch.int32, device=device),
        dsa_cache_seqlens_int32=torch.empty(2, dtype=torch.int32, device=device),
        dsa_cu_seqlens_q=torch.empty(3, dtype=torch.int32, device=device),
        dsa_cu_seqlens_k=torch.empty(3, dtype=torch.int32, device=device),
        dsa_extend_seq_lens_list=[],
        dsa_seqlens_expanded=torch.empty(2, dtype=torch.int32, device=device),
        paged_mqa_schedule_metadata=schedule,
    )
    calls = []

    class _FakeDeepGemm:
        @staticmethod
        def get_num_sms():
            return 4

        @staticmethod
        def get_paged_mqa_logits_metadata_out(
            context_lens, schedule_metadata, block_kv, num_sms, indices=None
        ):
            calls.append((context_lens, block_kv, num_sms, indices))
            schedule_metadata.copy_(
                torch.arange(
                    schedule_metadata.numel(),
                    dtype=schedule_metadata.dtype,
                    device=schedule_metadata.device,
                ).view_as(schedule_metadata)
            )

        @staticmethod
        def get_paged_mqa_logits_metadata(*_args, **_kwargs):
            raise AssertionError("allocating DeepGEMM metadata path should not run")

    monkeypatch.setattr(dsa_backend_module, "deep_gemm", _FakeDeepGemm)

    DeepseekSparseAttnBackend._refresh_paged_mqa_schedule_metadata(
        DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend),
        metadata,
        seqlens_2d,
    )

    assert calls == [(seqlens_2d, 64, 4, None)]
    assert torch.equal(
        schedule,
        torch.arange(schedule.numel(), dtype=schedule.dtype, device=device).view_as(
            schedule
        ),
    )


def test_refresh_paged_mqa_schedule_metadata_copies_without_deepgemm_out(monkeypatch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    schedule = torch.full((5, 2), -1, dtype=torch.int32, device=device)
    new_schedule = torch.arange(
        schedule.numel(), dtype=schedule.dtype, device=device
    ).view_as(schedule)
    seqlens_2d = torch.tensor([[4, 5], [6, 7]], dtype=torch.int32, device=device)
    metadata = DSAMetadata(
        page_size=1,
        cache_seqlens_int32=torch.empty(2, dtype=torch.int32, device=device),
        max_seq_len_q=1,
        max_seq_len_k=8,
        cu_seqlens_q=torch.empty(3, dtype=torch.int32, device=device),
        cu_seqlens_k=torch.empty(3, dtype=torch.int32, device=device),
        page_table_1=torch.empty((2, 8), dtype=torch.int32, device=device),
        real_page_table=torch.empty((2, 8), dtype=torch.int32, device=device),
        dsa_cache_seqlens_int32=torch.empty(2, dtype=torch.int32, device=device),
        dsa_cu_seqlens_q=torch.empty(3, dtype=torch.int32, device=device),
        dsa_cu_seqlens_k=torch.empty(3, dtype=torch.int32, device=device),
        dsa_extend_seq_lens_list=[],
        dsa_seqlens_expanded=torch.empty(2, dtype=torch.int32, device=device),
        paged_mqa_schedule_metadata=schedule,
    )
    calls = []

    class _FakeDeepGemm:
        @staticmethod
        def get_num_sms():
            return 4

        @staticmethod
        def get_paged_mqa_logits_metadata(
            context_lens, block_kv, num_sms, indices=None
        ):
            calls.append((context_lens, block_kv, num_sms, indices))
            return new_schedule

    monkeypatch.setattr(dsa_backend_module, "deep_gemm", _FakeDeepGemm)

    DeepseekSparseAttnBackend._refresh_paged_mqa_schedule_metadata(
        DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend),
        metadata,
        seqlens_2d,
    )

    assert calls == [(seqlens_2d, 64, 4, None)]
    assert metadata.paged_mqa_schedule_metadata is schedule
    assert torch.equal(schedule, new_schedule)


def test_prepare_paged_mqa_ctx_lens_target_verify_fills_existing(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    bs = 3
    next_n = 4
    cache_seqlens = torch.tensor([11, 23, 37], dtype=torch.int32, device=device)
    ctx_lens = torch.full((bs, next_n), -1, dtype=torch.int32, device=device)
    metadata = DSAMetadata(
        page_size=1,
        cache_seqlens_int32=cache_seqlens,
        max_seq_len_q=1,
        max_seq_len_k=64,
        cu_seqlens_q=torch.empty(bs + 1, dtype=torch.int32, device=device),
        cu_seqlens_k=torch.empty(bs + 1, dtype=torch.int32, device=device),
        page_table_1=torch.empty((bs * next_n, 64), dtype=torch.int32, device=device),
        real_page_table=torch.empty(
            (bs * next_n, 64), dtype=torch.int32, device=device
        ),
        dsa_cache_seqlens_int32=torch.empty(
            bs * next_n, dtype=torch.int32, device=device
        ),
        dsa_cu_seqlens_q=torch.empty(bs * next_n + 1, dtype=torch.int32, device=device),
        dsa_cu_seqlens_k=torch.empty(bs * next_n + 1, dtype=torch.int32, device=device),
        dsa_extend_seq_lens_list=[],
        dsa_seqlens_expanded=torch.empty(bs * next_n, dtype=torch.int32, device=device),
        paged_mqa_ctx_lens_2d=ctx_lens,
    )
    backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
    backend.speculative_num_draft_tokens = next_n
    monkeypatch.setattr(dsa_backend_module, "is_sm100_supported", lambda: True)

    returned = DeepseekSparseAttnBackend._prepare_paged_mqa_ctx_lens_2d(
        backend,
        metadata,
        ForwardMode.TARGET_VERIFY,
        cache_seqlens,
        metadata.dsa_seqlens_expanded,
        bs,
    )
    torch.cuda.synchronize()

    expected = cache_seqlens.view(bs, 1).expand(bs, next_n).contiguous()
    assert returned is ctx_lens
    assert metadata.paged_mqa_ctx_lens_2d is ctx_lens
    assert torch.equal(ctx_lens, expected)


def test_fused_metadata_runtime_lengths_do_not_recompile():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from sglang.srt.layers.attention.triton_ops.dsa_metadata import (
        _fill_paged_mqa_ctx_lens_target_verify_kernel,
        _fused_dsa_decode_metadata_kernel,
        _fused_dsa_draft_extend_metadata_kernel,
        _fused_dsa_target_verify_metadata_kernel,
        fill_paged_mqa_ctx_lens_target_verify,
        fused_dsa_decode_metadata,
        fused_dsa_draft_extend_metadata,
        fused_dsa_target_verify_metadata,
    )

    device = "cuda"
    bs = 5
    next_n = 3
    max_runtime_len = 260
    topk = 64
    req_to_token = torch.randint(
        0, 100000, (16, max_runtime_len), dtype=torch.int32, device=device
    )
    req_pool_indices = torch.tensor([3, 0, 5, 2, 7], dtype=torch.int64, device=device)
    seq_lens = torch.tensor([101, 97, 88, 105, 93], dtype=torch.int64, device=device)
    extend_seq_lens = torch.tensor([1, 3, 2, 1, 3], dtype=torch.int32, device=device)
    extend_seq_lens_short = torch.ones(bs, dtype=torch.int32, device=device)
    extend_seq_lens_full = torch.full((bs,), next_n, dtype=torch.int32, device=device)
    max_total_len = bs * next_n
    total_len = int(extend_seq_lens.cpu().sum().item())

    def decode_call(max_len):
        fused_dsa_decode_metadata(
            seq_lens=seq_lens,
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            cache_seqlens=torch.empty(bs, dtype=torch.int32, device=device),
            cu_seqlens_k=torch.empty(bs + 1, dtype=torch.int32, device=device),
            page_table_1=torch.empty((bs, max_len), dtype=torch.int32, device=device),
            dsa_cache_seqlens=torch.empty(bs, dtype=torch.int32, device=device),
            dsa_cu_seqlens_k=torch.empty(bs + 1, dtype=torch.int32, device=device),
            real_page_table=torch.empty(
                (bs, max_len), dtype=torch.int32, device=device
            ),
            bs=bs,
            max_len=max_len,
            dsa_index_topk=topk,
            real_page_size=1,
        )

    def target_call(max_seqlen_k):
        expanded_size = bs * next_n
        fused_dsa_target_verify_metadata(
            seq_lens=seq_lens,
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            cache_seqlens=torch.empty(bs, dtype=torch.int32, device=device),
            cu_seqlens_k=torch.empty(bs + 1, dtype=torch.int32, device=device),
            page_table_1=torch.empty(
                (expanded_size, max_seqlen_k), dtype=torch.int32, device=device
            ),
            seqlens_expanded=torch.empty(
                expanded_size, dtype=torch.int32, device=device
            ),
            dsa_cache_seqlens=torch.empty(
                expanded_size, dtype=torch.int32, device=device
            ),
            dsa_cu_seqlens_k=torch.empty(
                expanded_size + 1, dtype=torch.int32, device=device
            ),
            real_page_table=torch.empty(
                (expanded_size, max_seqlen_k), dtype=torch.int32, device=device
            ),
            bs=bs,
            max_seqlen_k=max_seqlen_k,
            dsa_index_topk=topk,
            real_page_size=1,
            next_n=next_n,
        )

    def draft_call(extend_lens, total, max_seqlen_k):
        fused_dsa_draft_extend_metadata(
            seq_lens=seq_lens,
            extend_seq_lens=extend_lens,
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            cache_seqlens=torch.empty(bs, dtype=torch.int32, device=device),
            cu_seqlens_k=torch.empty(bs + 1, dtype=torch.int32, device=device),
            page_table_1=torch.empty(
                (total, max_seqlen_k), dtype=torch.int32, device=device
            ),
            seqlens_expanded=torch.empty(total, dtype=torch.int32, device=device),
            dsa_cache_seqlens=torch.empty(total, dtype=torch.int32, device=device),
            dsa_cu_seqlens_k=torch.empty(total + 1, dtype=torch.int32, device=device),
            real_page_table=torch.empty(
                (total, max_seqlen_k), dtype=torch.int32, device=device
            ),
            bs=bs,
            total_len=total,
            max_seqlen_k=max_seqlen_k,
            dsa_index_topk=topk,
            real_page_size=1,
            max_total_len=max_total_len,
        )

    def fill_ctx_call(fill_bs):
        fill_paged_mqa_ctx_lens_target_verify(
            cache_seqlens=torch.arange(fill_bs, dtype=torch.int32, device=device),
            ctx_lens_2d=torch.empty(
                (fill_bs, next_n), dtype=torch.int32, device=device
            ),
            bs=fill_bs,
            next_n=next_n,
        )

    decode_call(101)
    target_call(101)
    draft_call(extend_seq_lens, total_len, 101)
    fill_ctx_call(5)
    torch.cuda.synchronize()

    decode_cache_size = _compiled_kernel_cache_size(_fused_dsa_decode_metadata_kernel)
    target_cache_size = _compiled_kernel_cache_size(
        _fused_dsa_target_verify_metadata_kernel
    )
    draft_cache_size = _compiled_kernel_cache_size(
        _fused_dsa_draft_extend_metadata_kernel
    )
    fill_ctx_cache_size = _compiled_kernel_cache_size(
        _fill_paged_mqa_ctx_lens_target_verify_kernel
    )

    for max_len in [102, 127, 128, 129, 257]:
        decode_call(max_len)
    for max_seqlen_k in [102, 127, 128, 129, 257]:
        target_call(max_seqlen_k)
        draft_call(extend_seq_lens, total_len, max_seqlen_k)
    draft_call(extend_seq_lens_short, bs, 257)
    draft_call(extend_seq_lens_full, max_total_len, 257)
    for fill_bs in [1, 2, 3, 4, 5, 8, 16]:
        fill_ctx_call(fill_bs)
    torch.cuda.synchronize()

    assert (
        _compiled_kernel_cache_size(_fused_dsa_decode_metadata_kernel)
        == decode_cache_size
    )
    assert (
        _compiled_kernel_cache_size(_fused_dsa_target_verify_metadata_kernel)
        == target_cache_size
    )
    assert (
        _compiled_kernel_cache_size(_fused_dsa_draft_extend_metadata_kernel)
        == draft_cache_size
    )
    assert (
        _compiled_kernel_cache_size(_fill_paged_mqa_ctx_lens_target_verify_kernel)
        == fill_ctx_cache_size
    )
