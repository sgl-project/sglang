import inspect
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.qsa import sparse_attn as sparse_attn_module
from sglang.srt.layers.attention.qsa.kernel import qsa_sparse_attention
from sglang.srt.layers.attention.qsa.sparse_attn import (
    qsa_sparse_decode_triton,
    qwen_sparse_fa2_cu_seqlens_triton,
    qwen_sparse_kv_extraction_compact_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# RadixArk/Qwen3.8-Flash-Next-NVFP4 text_config full-attention profile.
NUM_Q_HEADS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 256
SOFTMAX_SCALE = HEAD_DIM**-0.5
COMPRESS_RATIO = 4
TOKEN_TOPK = 2048
FINAL_TOPK = TOKEN_TOPK + COMPRESS_RATIO - 1


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("QSA Triton decode requires CUDA")


def _make_case(bs, topk, *, single_count=None, padded=True):
    torch.manual_seed(20260827 + bs + topk + int(single_count or 0))
    device = torch.device("cuda")
    padded_rows = bs + (3 if padded else 0)
    max_seq_len = topk + 31
    num_requests = bs + 1  # request zero is the graph-padding sink

    req_to_token = torch.randperm(
        num_requests * max_seq_len, dtype=torch.int32, device=device
    ).reshape(num_requests, max_seq_len)
    req_indices = torch.cat(
        [
            torch.arange(1, bs + 1, dtype=torch.int32, device=device),
            torch.zeros(padded_rows - bs, dtype=torch.int32, device=device),
        ]
    )
    seq_lens = torch.ones(padded_rows, dtype=torch.int32, device=device)
    seq_lens[:bs] = max_seq_len - torch.arange(bs, device=device) % 17

    counts = torch.randint(1, topk, (bs,), device=device)
    if single_count is not None:
        counts.fill_(single_count)
    else:
        counts[0] = 0
        if bs > 1:
            counts[1] = topk
        if bs > 2:
            counts[2] = min(65, topk)
    indices = torch.full((padded_rows, topk), -1, dtype=torch.int32, device=device)
    for row in range(bs):
        count = int(counts[row])
        if count:
            indices[row, :count] = torch.randperm(
                int(seq_lens[row]), dtype=torch.int32, device=device
            )[:count]
    if padded_rows > bs:
        # Positive but sequence-invalid: the -1 convention does not cover this
        # value, so the kernel has to clamp validity before dereferencing
        # req_to_token.
        indices[bs, 0] = max_seq_len + 7

    num_cache_slots = num_requests * max_seq_len
    q = torch.randn(
        padded_rows,
        NUM_Q_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    k = torch.randn(
        num_cache_slots,
        NUM_KV_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    v = torch.randn_like(k)
    return q, k, v, req_to_token, req_indices, indices, seq_lens


def _physical_slots(req_to_token, req_indices, indices, seq_lens):
    valid = (indices >= 0) & (indices < seq_lens[:, None])
    safe = indices.clamp(min=0, max=req_to_token.shape[1] - 1).long()
    slots = req_to_token[req_indices.long()[:, None], safe]
    return torch.where(valid, slots, torch.full_like(slots, -1)).to(torch.int32)


def _run_and_compare(case):
    q, k, v, req_to_token, req_indices, indices, seq_lens = case
    actual = qsa_sparse_decode_triton(
        q,
        k,
        v,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        SOFTMAX_SCALE,
    )
    expected = qsa_sparse_attention(
        q,
        k,
        v,
        _physical_slots(req_to_token, req_indices, indices, seq_lens),
        SOFTMAX_SCALE,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all()
    max_abs_err = (actual.float() - expected.float()).abs().max().item()
    assert max_abs_err <= 2e-2, f"max abs error {max_abs_err:.6f}"
    repeated = qsa_sparse_decode_triton(
        q,
        k,
        v,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        SOFTMAX_SCALE,
    )
    torch.cuda.synchronize()
    assert torch.equal(actual, repeated)
    return actual


@pytest.mark.parametrize("bs", [1, 4, 16, 64])
@pytest.mark.parametrize("topk", [512, FINAL_TOPK])
def test_qsa_sparse_decode_triton_matches_torch(bs, topk):
    _require_cuda()
    if bs == 1:
        # A single row cannot simultaneously contain zero, random, and full
        # valid counts, so cover all three in separate launches.
        for count in (0, topk // 3, topk):
            output = _run_and_compare(
                _make_case(bs, topk, single_count=count, padded=True)
            )
            torch.testing.assert_close(output[bs:], torch.zeros_like(output[bs:]))
    else:
        output = _run_and_compare(_make_case(bs, topk, padded=True))
        torch.testing.assert_close(output[bs:], torch.zeros_like(output[bs:]))


def _resolve_flash_attn():
    try:
        from flash_attn import flash_attn_varlen_func

        return flash_attn_varlen_func
    except ImportError:
        pass
    try:
        from flash_attn.cute.interface import flash_attn_varlen_func

        return flash_attn_varlen_func
    except ImportError:
        return None


def _make_flash_call(case, flash_attn_varlen_func):
    q, k, v, req_to_token, req_indices, indices, seq_lens = case
    rows, topk = indices.shape
    valid_counts = torch.empty(rows, dtype=torch.int32, device=q.device)
    cu_k = torch.empty(rows + 1, dtype=torch.int32, device=q.device)
    cu_q = torch.arange(rows + 1, dtype=torch.int32, device=q.device)
    packed_k = torch.empty(
        (rows * topk, NUM_KV_HEADS, HEAD_DIM), dtype=k.dtype, device=k.device
    )
    packed_v = torch.empty_like(packed_k)

    def call():
        qwen_sparse_fa2_cu_seqlens_triton(
            seq_lens, indices, valid_counts, cu_k, rows, topk
        )
        qwen_sparse_kv_extraction_compact_triton(
            k,
            v,
            req_to_token,
            req_indices,
            indices,
            seq_lens,
            cu_k,
            packed_k,
            packed_v,
            rows,
            topk,
        )
        output = flash_attn_varlen_func(
            q=q,
            k=packed_k,
            v=packed_v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=1,
            max_seqlen_k=topk,
            softmax_scale=SOFTMAX_SCALE,
            causal=True,
        )
        return output[0] if isinstance(output, tuple) else output

    return call


def test_qsa_sparse_decode_triton_matches_flash_attn():
    _require_cuda()
    flash_attn_varlen_func = _resolve_flash_attn()
    if flash_attn_varlen_func is None:
        pytest.skip("flash_attn varlen is not importable")
    case = _make_case(4, FINAL_TOPK, padded=True)
    triton_output = qsa_sparse_decode_triton(*case, SOFTMAX_SCALE)
    flash_output = _make_flash_call(case, flash_attn_varlen_func)()
    torch.cuda.synchronize()
    assert torch.isfinite(flash_output).all()
    max_abs_err = (triton_output.float() - flash_output.float()).abs().max().item()
    assert max_abs_err <= 2e-2, f"max abs error {max_abs_err:.6f}"


@pytest.mark.parametrize("bs", [1, 16, 64])
def test_qsa_sparse_decode_triton_cuda_graph_replay(bs):
    _require_cuda()
    case = _make_case(bs, FINAL_TOPK, single_count=FINAL_TOPK, padded=True)
    q = case[0]
    for _ in range(3):
        qsa_sparse_decode_triton(*case, SOFTMAX_SCALE)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = qsa_sparse_decode_triton(*case, SOFTMAX_SCALE)
    graph.replay()
    torch.cuda.synchronize()
    first = output.clone()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(first, output)
    q.mul_(0.5)
    graph.replay()
    torch.cuda.synchronize()

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[bs:], torch.zeros_like(output[bs:]))
    assert not torch.equal(first, output)


def test_qsa_sparse_decode_triton_graph_survives_larger_counter_shape(monkeypatch):
    _require_cuda()
    small = _make_case(1, FINAL_TOPK, single_count=FINAL_TOPK, padded=True)
    for _ in range(3):
        qsa_sparse_decode_triton(*small, SOFTMAX_SCALE)
    torch.cuda.synchronize()

    counter_sizes = []
    torch_zeros = torch.zeros

    def record_zeros(*args, **kwargs):
        tensor = torch_zeros(*args, **kwargs)
        if tensor.dtype == torch.int32 and tensor.numel() in {
            4 * NUM_KV_HEADS,
            11 * NUM_KV_HEADS,
        }:
            counter_sizes.append(tensor.numel())
        return tensor

    monkeypatch.setattr(sparse_attn_module.torch, "zeros", record_zeros)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = qsa_sparse_decode_triton(*small, SOFTMAX_SCALE)
    graph.replay()
    torch.cuda.synchronize()
    expected = output.clone()

    # Eleven padded rows still use split-K and need a larger counter tensor
    # than the four-row captured call.
    large = _make_case(8, FINAL_TOPK, single_count=FINAL_TOPK, padded=True)
    qsa_sparse_decode_triton(*large, SOFTMAX_SCALE)
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    assert counter_sizes == [4 * NUM_KV_HEADS, 11 * NUM_KV_HEADS]
    assert torch.equal(output, expected)


def test_qsa_sparse_decode_graph_resets_dirty_counter_on_replay(monkeypatch):
    _require_cuda()
    case = _make_case(1, FINAL_TOPK, single_count=FINAL_TOPK, padded=False)
    expected = _run_and_compare(case)

    counters = []
    torch_zeros = torch.zeros

    def record_zeros(*args, **kwargs):
        tensor = torch_zeros(*args, **kwargs)
        if tensor.dtype == torch.int32 and tensor.numel() == NUM_KV_HEADS:
            counters.append(tensor)
        return tensor

    monkeypatch.setattr(sparse_attn_module.torch, "zeros", record_zeros)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = qsa_sparse_decode_triton(*case, SOFTMAX_SCALE)
    torch.cuda.synchronize()
    assert len(counters) == 1

    # A replay must execute the captured memset before the split-K kernel. If
    # it does not, no CTA observes ticket NUM_SPLITS - 1 and output stays dirty.
    counters[0].fill_(16)
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    torch.testing.assert_close(counters[0], torch.zeros_like(counters[0]))


def test_qsa_sparse_decode_splitk_merge_loads_bypass_l1():
    _require_cuda()
    case = _make_case(1, 64, single_count=64, padded=False)
    _run_and_compare(case)

    compiled_kernels = []
    device_caches = sparse_attn_module._qsa_sparse_decode_splitk.device_caches.values()
    for device_cache in device_caches:
        compiled_kernels.extend(device_cache[0].values())
    ptx = next(
        compiled.asm["ptx"]
        for compiled in compiled_kernels
        if "atom.global.gpu.acq_rel.add" in compiled.asm.get("ptx", "")
    )
    atomic = ptx.index("atom.global.gpu.acq_rel.add")
    last_partial_store = ptx.rfind("st.global", 0, atomic)
    assert ptx.count("ld.global.cg") >= 3
    assert last_partial_store >= 0
    assert ptx.find("bar.sync", last_partial_store, atomic) >= 0


def test_qsa_sparse_decode_splitk_requests_publication_barrier():
    source = inspect.getsource(sparse_attn_module._qsa_sparse_decode_splitk.fn)
    stores = source.index("partial_out + partial_offset")
    barrier = source.index("tl.debug_barrier()", stores)
    atomic = source.index("tl.atomic_add(counter", barrier)
    assert stores < barrier < atomic


def test_qsa_sparse_decode_split_count_is_static_from_rows(monkeypatch):
    _require_cuda()
    expected = {1: 16, 2: 8, 8: 8, 9: 4, 16: 4, 17: 1}
    assert {
        rows: sparse_attn_module._qsa_sparse_decode_num_splits(rows)
        for rows in expected
    } == expected

    launches = []

    class SplitKRecorder:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((grid, kwargs["NUM_SPLITS"]))

            return launch

    monkeypatch.setattr(
        sparse_attn_module, "_qsa_sparse_decode_splitk", SplitKRecorder()
    )
    case = list(_make_case(8, 512, single_count=512, padded=False))
    qsa_sparse_decode_triton(*case, SOFTMAX_SCALE)
    case[5].fill_(-1)
    case[6].fill_(1)
    qsa_sparse_decode_triton(*case, SOFTMAX_SCALE)
    assert launches == [((8, NUM_KV_HEADS, 8), 8)] * 2


def test_qsa_sparse_decode_splitk_cross_cta_publication():
    _require_cuda()
    topk = 1024
    case = list(_make_case(1, topk, single_count=topk, padded=False))
    case[0].zero_()
    case[1].zero_()
    case[2].zero_()
    case[3][1].copy_(
        torch.arange(case[3].shape[1], dtype=torch.int32, device=case[0].device)
    )
    case[5][0].copy_(torch.arange(topk, dtype=torch.int32, device=case[0].device))
    split_values = (
        torch.arange(16, dtype=torch.bfloat16, device=case[0].device) / 16
    ).repeat_interleave(topk // 16)
    case[2][:topk].copy_(
        split_values[:, None, None].expand(topk, NUM_KV_HEADS, HEAD_DIM)
    )
    expected = torch.full_like(case[0], split_values.float().mean())

    outputs = [qsa_sparse_decode_triton(*case, SOFTMAX_SCALE) for _ in range(100)]
    torch.cuda.synchronize()
    for output in outputs:
        torch.testing.assert_close(output, expected, atol=0, rtol=0)


def test_qsa_sparse_decode_triton_speculative_rows_share_requests():
    _require_cuda()
    case = list(_make_case(4, 512, padded=True))
    # Target-verify/draft-extend metadata fans several query rows out from one
    # request-table row while retaining a distinct visible length per step.
    case[4][:4].copy_(
        torch.tensor([1, 1, 2, 2], dtype=torch.int32, device=case[0].device)
    )
    output = _run_and_compare(tuple(case))
    torch.testing.assert_close(output[4:], torch.zeros_like(output[4:]))


def test_qsa_decode_backend_override(monkeypatch):
    import sglang.srt.utils as utils
    from sglang.srt.layers.attention import qwen_sparse_attn_backend as backend_module
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
        _qsa_decode_backend,
    )

    try:
        for backend in ("triton", "flash_attn", "trtllm"):
            monkeypatch.setenv("SGLANG_QSA_DECODE_BACKEND", backend)
            monkeypatch.setattr(
                backend_module, "_resolve_trtllm_sparse_decode", lambda: object()
            )
            _qsa_decode_backend.cache_clear()
            assert _qsa_decode_backend() == backend
        monkeypatch.setenv("SGLANG_QSA_DECODE_BACKEND", "invalid")
        _qsa_decode_backend.cache_clear()
        with pytest.raises(ValueError, match="SGLANG_QSA_DECODE_BACKEND"):
            _qsa_decode_backend()
        monkeypatch.delenv("SGLANG_QSA_DECODE_BACKEND")
        monkeypatch.setattr(utils, "is_sm120_supported", lambda: True)
        _qsa_decode_backend.cache_clear()
        assert _qsa_decode_backend() == "triton"
        monkeypatch.setattr(utils, "is_sm120_supported", lambda: False)
        _qsa_decode_backend.cache_clear()
        assert _qsa_decode_backend() == "trtllm"
        monkeypatch.setattr(
            backend_module, "_resolve_trtllm_sparse_decode", lambda: None
        )
        _qsa_decode_backend.cache_clear()
        assert _qsa_decode_backend() == "flash_attn"
        monkeypatch.setenv("SGLANG_QSA_DECODE_BACKEND", "trtllm")
        _qsa_decode_backend.cache_clear()
        with pytest.raises(RuntimeError, match="trtllm.*unavailable"):
            _qsa_decode_backend()
    finally:
        _qsa_decode_backend.cache_clear()


def test_qsa_explicit_triton_bypasses_available_trtllm(monkeypatch):
    _require_cuda()
    from sglang.srt.layers.attention import qwen_sparse_attn_backend as backend_module

    q = torch.zeros(1, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    metadata = SimpleNamespace(
        row_req_pool_indices=None,
        sequence_lengths=torch.ones(1, dtype=torch.int32, device=q.device),
    )

    class Pool:
        def get_key_buffer(self, layer_id):
            return q

        def get_value_buffer(self, layer_id):
            return q

    backend = object.__new__(backend_module.QwenSparseAttnBackend)
    backend.token_to_kv_pool = Pool()
    backend.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.zeros(1, 1, dtype=torch.int32, device=q.device)
    )
    backend._resolve_metadata = lambda forward_batch: metadata
    marker = torch.ones_like(q)

    monkeypatch.setenv("SGLANG_QSA_DECODE_BACKEND", "triton")
    monkeypatch.setattr(
        backend_module, "_resolve_trtllm_sparse_decode", lambda: object()
    )
    monkeypatch.setattr(
        backend_module,
        "qsa_sparse_decode_triton",
        lambda *args, **kwargs: marker,
    )
    monkeypatch.setattr(
        backend,
        "_forward_trtllm_sparse",
        lambda *args, **kwargs: pytest.fail("explicit triton request used trtllm"),
    )
    backend_module._qsa_decode_backend.cache_clear()
    try:
        output = backend._forward_paged_attention(
            q,
            SimpleNamespace(layer_id=0, scaling=1.0),
            SimpleNamespace(
                req_pool_indices=torch.zeros(1, dtype=torch.int32, device=q.device)
            ),
            torch.zeros(1, 1, dtype=torch.int32, device=q.device),
        )
        assert torch.equal(output, marker.reshape(1, -1))
    finally:
        backend_module._qsa_decode_backend.cache_clear()


def _time_us(fn, *, warmup=10, iterations=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def run_microbenchmark():
    _require_cuda()
    flash_attn_varlen_func = _resolve_flash_attn()
    print("| batch | topk | Triton direct (us) | flash_attn packed (us) |")
    print("|---:|---:|---:|---:|")
    for bs in (1, 16, 64):
        case = _make_case(bs, TOKEN_TOPK, single_count=TOKEN_TOPK, padded=False)

        def triton_call():
            return qsa_sparse_decode_triton(*case, SOFTMAX_SCALE)

        triton_us = _time_us(triton_call)
        if flash_attn_varlen_func is None:
            flash_text = "not installed"
        else:
            flash_us = _time_us(_make_flash_call(case, flash_attn_varlen_func))
            flash_text = f"{flash_us:.1f}"
        print(f"| {bs} | {TOKEN_TOPK} | {triton_us:.1f} | {flash_text} |")


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        run_microbenchmark()
    else:
        sys.exit(pytest.main([__file__, "-v"]))
