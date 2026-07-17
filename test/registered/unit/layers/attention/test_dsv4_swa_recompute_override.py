import pytest
import torch

from sglang.srt.layers.attention.deepseek_v4_backend import (
    SWA_WINDOW,
    DeepseekV4AttnBackend,
)
from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
    build_swa_token_ids,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def test_apply_swa_recompute_override_reindexes_only_current_extend() -> None:
    backend = object.__new__(DeepseekV4AttnBackend)
    backend.cuda_int32_kwargs = {"device": "cpu", "dtype": torch.int32}

    num_real_tokens = 5
    num_padded_tokens = 7
    width = SWA_WINDOW + 2
    original = torch.arange(num_padded_tokens * width, dtype=torch.int32).reshape(
        num_padded_tokens, width
    )
    actual = original.clone()
    override = torch.tensor([1000, 1001, 1002, 2000, 2001], dtype=torch.int32)

    backend._apply_swa_recompute_override(
        swa_page_indices=actual,
        seq_lens_casual=torch.tensor([3, 4, 5, 6, 7, 0, 0], dtype=torch.int32),
        seq_lens=torch.tensor([5, 7], dtype=torch.int32),
        extend_seq_lens=torch.tensor([3, 2], dtype=torch.int32),
        extend_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        swa_out_cache_loc_override=override,
    )

    expected = original.clone()
    expected[0, :1] = torch.tensor([1000])
    expected[1, :2] = torch.tensor([1001, 1000])
    expected[2, :3] = torch.tensor([1002, 1001, 1000])
    expected[3, :1] = torch.tensor([2000])
    expected[4, :2] = torch.tensor([2001, 2000])

    assert torch.equal(actual, expected)
    assert torch.equal(actual[num_real_tokens:], original[num_real_tokens:])
    assert torch.equal(actual[:, SWA_WINDOW:], original[:, SWA_WINDOW:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton requires CUDA")
def test_sparse_prefill_override_reindexes_each_request_extend() -> None:
    device = torch.device("cuda")
    seq_lens = torch.tensor([6, 7], dtype=torch.int32, device=device)
    extend_seq_lens = torch.tensor([2, 3], dtype=torch.int32, device=device)
    req_pool_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    req_to_token = torch.arange(16, dtype=torch.int32, device=device).reshape(2, 8)
    full_to_swa = 100 + torch.arange(16, dtype=torch.int64, device=device)
    override = torch.tensor(
        [1000, 1001, 2000, 2001, 2002], dtype=torch.int32, device=device
    )

    token_ids, first_pos, gather_lens, offsets = build_swa_token_ids(
        seq_lens=seq_lens,
        extend_seq_lens=extend_seq_lens,
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        full_to_swa=full_to_swa,
        swa_window=4,
        swa_out_cache_loc_override=override,
        extend_start_loc=torch.tensor([0, 2], dtype=torch.int32, device=device),
    )

    assert token_ids.tolist() == [
        101,
        102,
        103,
        1000,
        1001,
        109,
        110,
        111,
        2000,
        2001,
        2002,
    ]
    assert first_pos.tolist() == [1, 1]
    assert gather_lens.tolist() == [5, 6]
    assert offsets.tolist() == [0, 5, 11]
