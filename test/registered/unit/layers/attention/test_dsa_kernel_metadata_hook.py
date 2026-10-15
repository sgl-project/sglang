"""Custom DSA metadata must observe the current SRT metadata on every forward."""

import sys
from types import SimpleNamespace

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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
