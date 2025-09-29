import torch

from sglang.srt.utils import align

# NOTE(dark): flashmla P requires `params.topk % (2*B_TOPK) == 0`,
# where `B_TOPK=64`. So we align to 128 by default.

_TOPK_ALIGNMENT = 128


# TODO(dark): maybe this torch_op can support torch.compile
def _fast_topk_torch(
    input: torch.Tensor, seq_lens: torch.Tensor, topk: int, alignment: int
) -> torch.Tensor:
    # Fallback to torch.topk
    bs, max_seq_len = input.shape
    assert len(seq_lens) == bs
    # set those out-of-bound input to -inf
    padded_max_seq_len = align(max_seq_len, alignment)
    positions = torch.arange(
        padded_max_seq_len, device=input.device, dtype=seq_lens.dtype
    )
    positions = positions.unsqueeze(0).expand(bs, -1)
    mask = positions >= seq_lens.unsqueeze(1)

    # NOTE(dark): just return all valid indices as an optimization
    if padded_max_seq_len <= topk:
        return positions.masked_fill(mask, -1)

    assert topk % alignment == 0

    # in-place operation: mask invalid inputs to -inf
    input = input.masked_fill_(mask[:, :max_seq_len], float("-inf"))
    result = input.topk(topk, dim=-1, sorted=True)
    return result.indices.masked_fill_(mask[:, :topk], -1)


def fast_topk_impl(
    input: torch.Tensor,
    seq_lens: torch.Tensor,
    topk: int,
    alignment: int = _TOPK_ALIGNMENT,
) -> torch.Tensor:
    return _fast_topk_torch(input, seq_lens, topk, alignment)


def fast_topk_transform_fused_cuda(
    input: torch.Tensor,
    seq_lens: torch.Tensor,
    topk: int,
    dst_page_table: torch.Tensor,
    src_page_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    alignment: int = _TOPK_ALIGNMENT,
) -> torch.Tensor:
    from sglang.srt.layers.attention.nsa.cuda import fast_topk_transform

    assert topk == 2048 and topk % alignment == 0
    return fast_topk_transform(
        score=input,
        lengths=seq_lens,
        dst_page_table=dst_page_table,
        src_page_table=src_page_table,
        cu_seqlens=cu_seqlens_q,
    )
