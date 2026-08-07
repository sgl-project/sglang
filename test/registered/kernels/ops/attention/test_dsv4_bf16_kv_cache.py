import torch
from unittest.mock import patch

from sglang.srt.layers.attention.dsv4.bf16_kv_cache import (
    BF16_KV_HEAD_DIM,
    build_bf16_sparse_decode_inputs,
)


def _paged_cache(num_pages: int, page_size: int, offset: int) -> torch.Tensor:
    values = torch.arange(
        offset,
        offset + num_pages * page_size * BF16_KV_HEAD_DIM,
        dtype=torch.float32,
    )
    return values.view(num_pages, page_size, 1, BF16_KV_HEAD_DIM).to(
        torch.bfloat16
    )


def test_bf16_sparse_decode_preserves_historical_slot_layout():
    swa_cache = _paged_cache(num_pages=2, page_size=4, offset=0)
    extra_cache = _paged_cache(num_pages=2, page_size=2, offset=100000)

    swa_indices = torch.tensor([[1, 4, 7], [0, 3, 6]], dtype=torch.int32)
    swa_lengths = torch.tensor([3, 2], dtype=torch.int32)
    extra_indices = torch.tensor([[0, 3], [1, 2]], dtype=torch.int32)
    extra_lengths = torch.tensor([2, 1], dtype=torch.int32)

    workspace, combined_indices, combined_lengths = build_bf16_sparse_decode_inputs(
        swa_cache=swa_cache,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        swa_page_size=4,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        extra_lengths=extra_lengths,
        extra_page_size=2,
    )

    assert workspace.shape == (10, 1, BF16_KV_HEAD_DIM)
    assert combined_indices.shape == (2, 1, 128)
    assert combined_lengths.tolist() == [128, 128]

    # Historical layout stores all extra rows first, then all SWA rows. The
    # complete aligned length is intentional: FlashMLA must scan past the -1
    # hole in row 1 to reach its valid SWA slots.
    assert combined_indices[0, 0, :5].tolist() == [0, 1, 4, 5, 6]
    assert combined_indices[1, 0, :5].tolist() == [2, -1, 7, 8, -1]
    assert (combined_indices[:, 0, 5:] == -1).all()

    flat_extra = extra_cache.reshape(-1, 1, BF16_KV_HEAD_DIM)
    flat_swa = swa_cache.reshape(-1, 1, BF16_KV_HEAD_DIM)
    torch.testing.assert_close(workspace[0], flat_extra[0])
    torch.testing.assert_close(workspace[1], flat_extra[3])
    torch.testing.assert_close(workspace[2], flat_extra[1])
    torch.testing.assert_close(workspace[4], flat_swa[1])
    torch.testing.assert_close(workspace[5], flat_swa[4])
    torch.testing.assert_close(workspace[6], flat_swa[7])
    torch.testing.assert_close(workspace[7], flat_swa[0])
    torch.testing.assert_close(workspace[8], flat_swa[3])


def test_bf16_sparse_decode_supports_swa_only():
    swa_cache = _paged_cache(num_pages=1, page_size=4, offset=0)
    swa_indices = torch.tensor([[0, 2, 3]], dtype=torch.int32)
    swa_lengths = torch.tensor([2], dtype=torch.int32)

    workspace, combined_indices, combined_lengths = build_bf16_sparse_decode_inputs(
        swa_cache=swa_cache,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        swa_page_size=4,
    )

    assert combined_lengths.tolist() == [128]
    assert combined_indices[0, 0, :2].tolist() == [0, 1]
    assert (combined_indices[0, 0, 2:] == -1).all()
    expected = swa_cache.reshape(-1, 1, BF16_KV_HEAD_DIM)[[0, 2]]
    torch.testing.assert_close(workspace[:2], expected)


def test_bf16_sparse_decode_skips_python_bool_checks_during_cuda_graph_capture():
    swa_cache = _paged_cache(num_pages=1, page_size=4, offset=0)
    extra_cache = _paged_cache(num_pages=1, page_size=2, offset=100000)

    # CUDA Graph capture rejects converting a CUDA scalar tensor to a Python
    # bool. Simulate that restriction on CPU so this regression test is
    # runnable without a CUDA device.
    with patch.object(torch.cuda, "is_available", return_value=True):
        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=True
        ):
            with patch.object(
                torch.Tensor,
                "__bool__",
                side_effect=AssertionError("unexpected Tensor-to-bool conversion"),
            ):
                workspace, indices, lengths = build_bf16_sparse_decode_inputs(
                    swa_cache=swa_cache,
                    swa_indices=torch.tensor([[0, 2]], dtype=torch.int32),
                    swa_lengths=torch.tensor([2], dtype=torch.int32),
                    swa_page_size=4,
                    extra_cache=extra_cache,
                    extra_indices=torch.tensor([[0]], dtype=torch.int32),
                    extra_lengths=torch.tensor([1], dtype=torch.int32),
                    extra_page_size=2,
                )

    assert workspace.shape == (3, 1, BF16_KV_HEAD_DIM)
    assert indices[0, 0, :3].tolist() == [0, 1, 2]
    assert lengths.tolist() == [128]
