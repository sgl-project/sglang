import sys
from unittest.mock import patch

import pytest
import torch

import sglang.srt.layers.attention.aiter_backend as aiter_backend
from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeKernel:
    def __getitem__(self, _grid):
        return lambda *_args, **_kwargs: None


def _make_backend(*, unified: bool) -> AiterAttnBackend:
    backend = object.__new__(AiterAttnBackend)
    backend.use_mla = False
    backend.use_triton_unified_attention = unified
    backend.use_sliding_window_kv_pool = False
    backend.max_context_len = 8
    backend.page_size = 1
    backend.device = torch.device("cpu")
    backend.req_to_token = torch.tensor(
        [
            [10, 11, 12, 13, 14, 15, 16, 17],
            [20, 21, 22, 23, 24, 25, 26, 27],
        ],
        dtype=torch.int32,
    )
    backend.kv_indptr = torch.zeros(5, dtype=torch.int32)
    backend.cuda_graph_kv_indices = torch.zeros(32, dtype=torch.int32)
    backend.cuda_graph_page_table = torch.full((4, 8), -1, dtype=torch.int32)
    backend.qo_indptr_unified_decode = torch.arange(5, dtype=torch.int32)
    return backend


def _apply_decode_metadata(
    backend: AiterAttnBackend,
    *,
    seq_lens: torch.Tensor,
) -> None:
    backend._apply_cuda_graph_metadata(
        bs=2,
        req_pool_indices=torch.tensor([1, 0], dtype=torch.int64),
        seq_lens=seq_lens,
        seq_lens_sum=5,
        forward_mode=ForwardMode.DECODE,
        spec_info=None,
        seq_lens_cpu=torch.tensor([3, 2], dtype=torch.int32),
        verify_tokens_per_req=None,
    )


def test_decode_graph_indptr_ignores_padded_rows():
    backend = _make_backend(unified=False)
    seq_lens = torch.tensor([3, 2, 99, 99], dtype=torch.int32)

    with patch.object(
        aiter_backend,
        "create_flashinfer_kv_indices_triton",
        _FakeKernel(),
    ):
        _apply_decode_metadata(backend, seq_lens=seq_lens)

    torch.testing.assert_close(
        backend.forward_metadata.kv_indptr,
        torch.tensor([0, 3, 5], dtype=torch.int32),
    )


def test_page_size_one_decode_graph_refreshes_page_table():
    backend = _make_backend(unified=True)
    seq_lens = torch.tensor([3, 2, 99, 99], dtype=torch.int32)

    _apply_decode_metadata(backend, seq_lens=seq_lens)

    expected = torch.tensor(
        [
            [20, 21, 22],
            [10, 11, 12],
        ],
        dtype=torch.int32,
    )
    torch.testing.assert_close(
        backend.forward_metadata.kv_indices[:2, :3],
        expected,
    )
    assert torch.all(backend.forward_metadata.kv_indices[:2, 3:] == -1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
