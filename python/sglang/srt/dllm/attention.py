from typing import Optional, Sequence

import torch


def build_dllm_prefill_blockwise_mask(
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
    block_size: int,
    device: torch.device,
    *,
    include_prefix: bool,
) -> Optional[torch.Tensor]:
    """Build FlashInfer's flattened per-request dLLM prefill mask.

    Tokens in the same dLLM block are mutually visible, while a query may only
    attend to its own block and earlier blocks. With ``include_prefix=False``
    the key dimension covers only the current ragged extend chunk. With
    ``include_prefix=True`` it covers the paged ``prefix + extend`` sequence.

    Returns ``None`` when no request's extend range crosses a block boundary,
    for which non-causal attention already has the desired visibility.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if len(prefix_lens) != len(extend_lens):
        raise ValueError(
            "prefix_lens and extend_lens must have the same length: "
            f"{len(prefix_lens)} != {len(extend_lens)}"
        )
    if any(
        prefix_len < 0 or extend_len < 0
        for prefix_len, extend_len in zip(prefix_lens, extend_lens)
    ):
        raise ValueError("prefix and extend lengths must be non-negative")
    needs_mask = any(
        extend_len > 0
        and prefix_len // block_size
        != (prefix_len + extend_len - 1) // block_size
        for prefix_len, extend_len in zip(prefix_lens, extend_lens)
    )
    if not needs_mask:
        return None

    mask_parts = []
    for prefix_len, extend_len in zip(prefix_lens, extend_lens):
        query_positions = prefix_len + torch.arange(
            extend_len, device=device, dtype=torch.int64
        )
        if include_prefix:
            key_positions = torch.arange(
                prefix_len + extend_len, device=device, dtype=torch.int64
            )
        else:
            key_positions = prefix_len + torch.arange(
                extend_len, device=device, dtype=torch.int64
            )

        query_blocks = torch.div(
            query_positions, block_size, rounding_mode="floor"
        )
        key_blocks = torch.div(key_positions, block_size, rounding_mode="floor")
        mask_parts.append((key_blocks[None, :] <= query_blocks[:, None]).flatten())

    if not mask_parts:
        return None
    return torch.cat(mask_parts)
