"""Token remapping and shard mask for Ascend vocabulary-parallel embeddings."""

import torch
import triton
import triton.language as tl


@triton.jit
def _masked_input_kernel(
    tokens,
    mapped,
    mask,
    n_tokens: tl.constexpr,
    org_start: tl.constexpr,
    org_end: tl.constexpr,
    org_padding: tl.constexpr,
    added_start: tl.constexpr,
    added_end: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = tl.load(tokens + offsets, offsets < n_tokens, other=0).to(tl.int64)
    in_org = (token >= org_start) & (token < org_end)
    in_added = (token >= added_start) & (token < added_end)
    added_offset = added_start - (org_end - org_start) - org_padding
    # Keep the sum of both offsets, including the reference's behavior if
    # ranges overlap. Padding occupies local rows before the added vocabulary.
    valid_offset = tl.where(in_org, org_start, 0).to(tl.int64) + tl.where(
        in_added, added_offset, 0
    ).to(tl.int64)
    valid = in_org | in_added
    local_token = tl.where(valid, token - valid_offset, 0)
    tl.store(mapped + offsets, local_token, offsets < n_tokens)
    tl.store(mask + offsets, ~valid, offsets < n_tokens)


def get_masked_input_and_mask_npu(
    input_: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
):
    """Return int64 local ids and a bool out-of-shard mask in one NPU launch.

    The eager reference promotes even int32 tokens to int64 through its
    integer-scalar multiplication of boolean masks. Empty inputs allocate
    matching empty outputs without launching a kernel. Launch on the current
    stream; callers own any dependencies on other streams.
    """
    assert input_.device.type == "npu"
    assert input_.is_contiguous()
    assert input_.dtype in (torch.int32, torch.int64)
    mapped = torch.empty(input_.shape, dtype=torch.int64, device=input_.device)
    mask = torch.empty(input_.shape, dtype=torch.bool, device=input_.device)
    n_tokens = input_.numel()
    if n_tokens:
        block = min(1024, triton.next_power_of_2(n_tokens))
        _masked_input_kernel[(triton.cdiv(n_tokens, block),)](
            input_,
            mapped,
            mask,
            n_tokens,
            org_vocab_start_index,
            org_vocab_end_index,
            num_org_vocab_padding,
            added_vocab_start_index,
            added_vocab_end_index,
            BLOCK=block,
        )
    return mapped, mask
