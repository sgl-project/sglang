"""Check the Aiter paged-indexer row contract without launching GPU kernels."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.layers.attention.dsa import dsa_indexer
from sglang.srt.layers.attention.dsa.paged_mqa_logits_backend import (
    DSAPagedMQALogitsBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.mark.parametrize("preshuffle", [False, True])
@pytest.mark.parametrize(
    "mode,real_rows,physical_rows",
    [
        (ForwardMode.DECODE, 3, 5),
        (ForwardMode.DECODE, 3, 3),
        (ForwardMode.DECODE, 0, 4),
        (ForwardMode.DECODE, 0, 0),
        (ForwardMode.TARGET_VERIFY, 6, 8),
        (ForwardMode.DRAFT_EXTEND_V2, 6, 8),
    ],
)
def test_aiter_paged_rows_and_restored_padding(
    preshuffle, mode, real_rows, physical_rows
):
    page_size = 64 if preshuffle else 1
    q = torch.arange(physical_rows * 2 * 128, dtype=torch.float32).reshape(
        physical_rows, 2, 128
    )
    weights = torch.arange(physical_rows * 2, dtype=torch.float32).reshape(
        physical_rows, 2, 1
    )
    lengths = torch.arange(real_rows, dtype=torch.int32) + 2
    expanded = mode.is_target_verify() or mode.is_draft_extend_v2()
    base_lengths = lengths[::3] if expanded else lengths
    extend_lengths = [3] * len(base_lengths) if expanded else [1] * real_rows
    page_table = torch.zeros((real_rows, 8), dtype=torch.int32)
    kv = torch.zeros((1, page_size * 132), dtype=torch.uint8)
    pool = SimpleNamespace(
        index_page_size=page_size,
        get_index_k_with_scale_buffer=lambda layer_id: kv,
    )

    def kernel(query, cache, w, seq_lens, blocks, max_len, **kwargs):
        assert real_rows > 0, "Aiter must not be launched for an empty query"
        # Model the kernel's real-row contract, including the contents of each
        # row. Merely matching q and weight shapes would miss wrong slices.
        assert query.shape[0] == w.shape[0] == seq_lens.shape[0] == real_rows
        torch.testing.assert_close(query, q[:real_rows])
        torch.testing.assert_close(w, weights[:real_rows, :, 0])
        torch.testing.assert_close(seq_lens, lengths)
        assert blocks is page_table
        assert cache.shape == (1, page_size, 1, 132)
        assert kwargs == {"preshuffle": preshuffle, "kv_block_size": page_size}
        return torch.arange(max_len, dtype=torch.float32).repeat(real_rows, 1)

    def topk_transform(logits, topk):
        assert real_rows > 0, "An empty query must not launch top-k either"
        assert logits.shape[0] == lengths.shape[0]
        # Mask the unused context before top-k, as the real transform does.
        logits.masked_fill_(
            torch.arange(logits.shape[1])[None, :] >= lengths[:, None],
            float("-inf"),
        )
        return logits.topk(topk, dim=-1).indices.to(torch.int32)

    metadata = SimpleNamespace(
        get_page_table_1=lambda: page_table,
        get_page_table_64=lambda: page_table,
        get_seqlens_int32=lambda: base_lengths,
        get_seqlens_expanded=lambda: lengths,
        get_dsa_extend_len_cpu=lambda: extend_lengths,
        topk_transform=topk_transform,
    )
    indexer = dsa_indexer.Indexer.__new__(dsa_indexer.Indexer)
    torch.nn.Module.__init__(indexer)
    indexer.paged_mqa_logits_backend = DSAPagedMQALogitsBackend.AITER
    indexer.index_topk = 2
    indexer.num_init_tokens = 0
    indexer.num_local_tokens = 0
    with (
        mock.patch.object(dsa_indexer, "_is_hip", True),
        mock.patch.object(dsa_indexer, "_is_cuda", False),
        mock.patch.object(dsa_indexer, "_is_xpu", False),
        mock.patch.object(dsa_indexer, "_use_aiter_preshuffle", preshuffle),
        mock.patch.object(dsa_indexer, "get_token_to_kv_pool", return_value=pool),
        mock.patch.object(
            dsa_indexer, "aiter_paged_mqa_logits", side_effect=kernel
        ) as kernel_call,
    ):
        output = indexer._get_topk_paged(
            SimpleNamespace(forward_mode=mode), 0, q, weights, metadata
        )

    if real_rows == 0:
        kernel_call.assert_not_called()
    else:
        kernel_call.assert_called_once()

    assert output.shape == (physical_rows, 2)
    assert output.dtype == torch.int32
    expected = torch.stack((lengths - 1, lengths - 2), dim=-1)
    torch.testing.assert_close(output[:real_rows], expected)
    assert torch.all(output[real_rows:] == -1)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
