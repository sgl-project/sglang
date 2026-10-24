"""Custom DSA metadata must observe the current SRT metadata on every forward."""

import sys
from types import MethodType, ModuleType, SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention import dsa_backend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

DSA = dsa_backend.DeepseekSparseAttnBackend


@pytest.mark.parametrize("in_capture", [False, True])
def test_graph_metadata_hook_runs_after_refresh_on_each_batch(in_capture):
    seen = []
    backend = SimpleNamespace(forward_metadata=None)

    def refresh(**kwargs):
        backend.forward_metadata = kwargs

    backend._apply_cuda_graph_metadata = refresh
    backend.init_kernel_metadata = lambda batch: seen.append(
        (batch, backend.forward_metadata)
    )
    for batch_size in (3, 1, 0):
        seq_lens = torch.arange(batch_size, dtype=torch.int32) + 4
        batch = SimpleNamespace(
            batch_size=batch_size,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.clone(),
            req_pool_indices=torch.arange(batch_size),
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
            out_cache_loc=torch.arange(batch_size),
        )
        DSA.init_forward_metadata_out_graph(backend, batch, in_capture=in_capture)
        assert seen[-1][0] is batch
        assert seen[-1][1] is backend.forward_metadata
        assert seen[-1][1]["bs"] == batch_size
        assert seen[-1][1]["out_cache_loc"] is batch.out_cache_loc
        torch.testing.assert_close(seen[-1][1]["seq_lens_cpu"], seq_lens)
    assert len(seen) == 3
    assert len({id(metadata) for _, metadata in seen}) == 3


def test_eager_metadata_hook_sees_current_request_page_table(monkeypatch):
    seen = []
    page_table = torch.arange(32, dtype=torch.int32).reshape(4, 8)
    backend = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=page_table),
        _kv_shard_pool=None,
        set_dsa_prefill_impl=lambda batch: None,
        dsa_decode_impl="trtllm",
        use_mha=False,
        get_topk_transform_method=lambda mode: None,
        dsa_index_kpool=1,
        dsa_index_topk=4,
        physical_page_size=1,
        get_device_int32_arange=lambda n: torch.arange(n, dtype=torch.int32),
        _cal_indexer_k_start_end=lambda batch, indices: (None, None),
        _transform_table_1_to_real=lambda table: table,
        _build_topk_v2_plan=lambda lengths: None,
        _init_kpool_metadata=lambda metadata, batch, **kwargs: metadata,
    )
    backend.init_kernel_metadata = lambda batch: seen.append(
        (batch, backend.forward_metadata)
    )
    monkeypatch.setattr(dsa_backend, "is_cuda", lambda: False)
    monkeypatch.setattr(
        dsa_backend, "pad_dsa_cache_seqlens", lambda batch, lengths: lengths
    )
    for requests in ([2, 0], [1]):
        seq_lens = torch.tensor([3] * len(requests), dtype=torch.int32)
        batch = SimpleNamespace(
            batch_size=len(requests),
            forward_mode=ForwardMode.DECODE,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
            req_pool_indices=torch.tensor(requests),
            seq_lens_sum=int(seq_lens.sum()),
        )
        DSA.init_forward_metadata(backend, batch)
        assert seen[-1][0] is batch
        metadata = seen[-1][1]
        assert metadata is backend.forward_metadata
        torch.testing.assert_close(metadata.page_table_1, page_table[requests, :3])
    assert len(seen) == 2
    assert seen[0][1] is not seen[1][1]


@pytest.mark.parametrize("path", ["capture", "loop", "kpool", "fused", "fallback"])
def test_multistep_refreshes_custom_metadata_for_every_child(monkeypatch, path):
    multi = dsa_backend.DeepseekSparseAttnMultiStepBackend
    children = []
    seen = []
    calls = []
    count = 4 if path in ("fused", "fallback") else 2

    def metadata():
        return SimpleNamespace(
            **{
                name: None
                for name in (
                    "cache_seqlens_int32",
                    "cu_seqlens_k",
                    "page_table_1",
                    "dsa_cache_seqlens_int32",
                    "dsa_cu_seqlens_k",
                    "real_page_table",
                )
            }
        )

    for index in range(count):
        child = SimpleNamespace(
            experimental_kpool_metadata_fusion=path == "kpool",
            decode_cuda_graph_metadata={size: metadata() for size in (3, 1, 0)},
            set_dsa_prefill_impl=lambda **kwargs: None,
        )

        def update(md, precomputed, mode, child=child):
            calls.append("fused")
            md.requests = precomputed.requests

        def replay(*, bs, precomputed, forward_mode, child=child):
            calls.append("replay")
            child.forward_metadata = SimpleNamespace(requests=precomputed.requests)

        def sibling(*, src_backend, child=child, **kwargs):
            calls.append("sibling")
            child.forward_metadata = SimpleNamespace(
                requests=src_backend.forward_metadata.requests
            )

        def capture(*, req_pool_indices, child=child, **kwargs):
            child.forward_metadata = SimpleNamespace(requests=req_pool_indices)

        def hook(batch, child=child, index=index):
            seen.append((index, batch, child.forward_metadata.requests.clone()))

        child._update_kpool_metadata_from_precomputed = update
        child.init_forward_metadata_replay_cuda_graph_from_precomputed = replay
        child._copy_replay_metadata_from_sibling = sibling
        child._apply_cuda_graph_metadata = capture
        child.init_forward_metadata_out_graph = MethodType(
            DSA.init_forward_metadata_out_graph, child
        )
        child.init_kernel_metadata = hook
        children.append(child)

    def precompute(*, req_pool_indices, **kwargs):
        return SimpleNamespace(
            requests=req_pool_indices,
            **{
                name: None
                for name in (
                    "flashmla_metadata",
                    "cache_seqlens",
                    "cu_seqlens_k",
                    "page_indices",
                    "dsa_cache_seqlens",
                    "dsa_cu_seqlens_k",
                    "real_page_table",
                    "max_len",
                    "seqlens_expanded_size",
                )
            },
        )

    children[0]._precompute_replay_metadata = precompute
    module = ModuleType("sglang.kernels.ops.attention.dsa.fused_metadata_copy")

    def fused_copy(*args):
        if path == "fallback":
            raise RuntimeError("simulated unavailable fused copy")

    module.fused_metadata_copy_multi_cuda = fused_copy
    monkeypatch.setitem(sys.modules, module.__name__, module)
    backend = multi.__new__(multi)
    backend.attn_backends = children
    backend.speculative_num_steps = count + 1
    for requests in ([2, 0, 1], [1], []):
        size = len(requests)
        lengths = torch.full((size,), 4, dtype=torch.int32)
        batch = SimpleNamespace(
            batch_size=size,
            forward_mode=ForwardMode.TARGET_VERIFY,
            req_pool_indices=torch.tensor(requests, dtype=torch.int64),
            seq_lens=lengths,
            seq_lens_cpu=lengths,
            seq_lens_sum=4 * size,
            spec_info=None,
            out_cache_loc=torch.arange(size),
        )
        start = len(seen)
        multi.init_forward_metadata_out_graph(
            backend, batch, in_capture=path == "capture"
        )
        assert len(seen) == start + count
        for index, (observed_index, view, observed_requests) in enumerate(seen[start:]):
            assert observed_index == index
            assert view.forward_mode == ForwardMode.DECODE
            assert view.actual_forward_mode == batch.forward_mode
            assert view.out_cache_loc is batch.out_cache_loc
            torch.testing.assert_close(observed_requests, batch.req_pool_indices)
    if path == "kpool":
        assert calls.count("sibling") == 3
    elif path == "fused":
        assert calls.count("fused") == 9
        assert calls.count("replay") == 3
    elif path != "capture":
        assert calls.count("replay") == 3 * count

    def fail_hook(batch):
        raise ValueError("custom metadata failed")

    children[0].init_kernel_metadata = fail_hook
    with pytest.raises(ValueError, match="custom metadata failed"):
        multi.init_forward_metadata_out_graph(
            backend, batch, in_capture=path == "capture"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
