import sys

import pytest
import sgl_kernel.flash_mla as flash_mla
import torch

from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sparse_flashmla_sink_padding_refreshes_reused_buffer(monkeypatch):
    backend = object.__new__(DeepseekSparseAttnBackend)
    backend.device_sm_major = 10
    backend.dsa_index_topk = 2
    backend._sink_pad_cache = {}
    captured_sinks = []

    def capture_flash_mla_sparse_fwd(**kwargs):
        captured_sinks.append(kwargs["attn_sink"].clone())
        q = kwargs["q"]
        return q.new_zeros((*q.shape[:2], kwargs["d_v"])), None, None

    monkeypatch.setattr(flash_mla, "flash_mla_sparse_fwd", capture_flash_mla_sparse_fwd)

    q = torch.zeros((1, 64, 8), device="cuda")
    kv_cache = torch.zeros((1, 1, 8), device="cuda")
    page_table = torch.zeros((1, 2), dtype=torch.int32, device="cuda")
    sink = torch.arange(64, dtype=torch.float32, device="cuda")

    backend._forward_flashmla_sparse(q, kv_cache, 8, page_table, 1.0, attn_sink=sink)
    cached_sink = next(iter(backend._sink_pad_cache.values()))
    cached_ptr = cached_sink.data_ptr()

    sink.add_(100)
    backend._forward_flashmla_sparse(q, kv_cache, 8, page_table, 1.0, attn_sink=sink)

    assert next(iter(backend._sink_pad_cache.values())).data_ptr() == cached_ptr
    torch.testing.assert_close(captured_sinks[0][:64], sink - 100)
    torch.testing.assert_close(captured_sinks[1][:64], sink)
    torch.testing.assert_close(captured_sinks[1][64:], torch.zeros_like(sink))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
