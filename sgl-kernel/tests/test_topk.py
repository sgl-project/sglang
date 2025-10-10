import pytest
import torch
from sgl_kernel import fast_topk_transform_fused, fast_topk_v2


def _ref_torch_impl(score: torch.Tensor, seq_len: int, topk: int) -> torch.Tensor:
    assert score.dim() == 2
    return torch.topk(score[:, :seq_len], topk, dim=-1, sorted=False).indices


def _ref_torch_transform_decode_impl(
    score: torch.Tensor,
    seq_len: int,
    src_page_table: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    batch_size, _ = score.shape
    assert score.shape[0] == src_page_table.shape[0]
    assert seq_len >= topk
    indices = _ref_torch_impl(score, seq_len, topk)
    topk_indices = torch.empty(
        (batch_size, topk), dtype=torch.int32, device=score.device
    )
    for i in range(batch_size):
        topk_indices[i] = src_page_table[i, indices[i]]
    return topk_indices


MAX_SEQ_LEN = 131072
MAX_PERMIT_ERROR = 0


def assert_equal(
    score: torch.Tensor,
    indices_ref: torch.Tensor,
    indices_our: torch.Tensor,
    bs: int,
    k: int,
    seq_len: int,
):
    indices_our_cpu = indices_our.cpu().tolist()
    indices_ref_cpu = indices_ref.cpu().tolist()
    for i in range(bs):
        indices_ref_set_i = set(indices_ref_cpu[i])
        indices_our_set_i = set(indices_our_cpu[i])
        more = indices_our_set_i - indices_ref_set_i
        less = indices_ref_set_i - indices_our_set_i
        if len(more) > MAX_PERMIT_ERROR or len(less) > MAX_PERMIT_ERROR:
            # check whether more values are the same with less values
            # if so, either one is acceptable, since their values are the same
            more_values = sorted(score[i, idx].item() for idx in more)
            less_values = sorted(score[i, idx].item() for idx in less)
            assert (
                more_values == less_values
            ), f"{bs=}, {k=}, {seq_len=}, {i=}, {more=}, {less=} failed, with {more_values=}, {less_values=}"


@pytest.mark.parametrize("bs", [1, 132, 256, 4096])
@pytest.mark.parametrize("k", [2048])  # we only support 2048 now
@pytest.mark.parametrize("seq_len", [2048, 4096, 16384, 65536])
@torch.inference_mode()
def test_topk_kernel(bs: int, k: int, seq_len: int) -> None:
    torch.manual_seed(42)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device="cuda")
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")

    indices_ref = _ref_torch_impl(score, seq_len, k)
    indices_our = fast_topk_v2(score, lengths, k)

    # sort and compare
    indices_ref = torch.sort(indices_ref, dim=-1).values
    indices_our = torch.sort(indices_our, dim=-1).values

    assert_equal(score, indices_ref, indices_our, bs, k, seq_len)


@pytest.mark.parametrize("bs", [1, 132, 256, 4096])
@pytest.mark.parametrize("k", [2048])  # we only support 2048 now
@pytest.mark.parametrize("seq_len", [2048, 4096, 16384, 65536])
@torch.inference_mode()
def test_topk_transform_kernel(bs: int, k: int, seq_len: int) -> None:
    # TODO(dark): test prefill kernel, though nothing special
    MAX_PERMIT_ERROR = 1
    torch.manual_seed(42)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device="cuda")
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")
    src_page_table = torch.arange(0, seq_len, dtype=torch.int32, device="cuda")
    src_page_table = src_page_table.unsqueeze(0).expand(bs, -1)
    # NOTE: for decode, cumulative seqlens_q is just 0..=bs
    # NOTE: since page table is arange, they equal topk indices
    cu_seqlens_q = torch.arange(0, bs + 1, dtype=torch.int32, device="cuda")
    dst_page_table_ref = _ref_torch_transform_decode_impl(
        score=score,
        seq_len=seq_len,
        src_page_table=src_page_table,
        topk=k,
    )
    dst_page_table_our = fast_topk_transform_fused(
        score=score,
        lengths=lengths,
        page_table_size_1=src_page_table,
        cu_seqlens_q=cu_seqlens_q,
        topk=k,
    )

    # sort and compare
    dst_page_table_our = torch.sort(dst_page_table_our, dim=-1).values
    dst_page_table_ref = torch.sort(dst_page_table_ref, dim=-1).values

    assert_equal(score, dst_page_table_ref, dst_page_table_our, bs, k, seq_len)


if __name__ == "__main__":
    pytest.main([__file__])
