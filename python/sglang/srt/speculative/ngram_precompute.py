from collections.abc import Sequence

import torch


def extract_local_accept_path_nodes(
    accept_index: torch.Tensor,
    accept_lens: torch.Tensor,
    draft_token_num: int,
) -> torch.Tensor:
    """Convert sampler-global accept indices to one local tree node per request."""
    bs = accept_lens.shape[0]
    device = accept_index.device
    accept_index_2d = accept_index.reshape(bs, draft_token_num).to(torch.long)
    slot_indices = torch.arange(draft_token_num, device=device)
    valid_slots = (slot_indices[None, :] < accept_lens[:, None]) & (
        accept_index_2d >= 0
    )
    path_slots = (
        torch.where(
            valid_slots,
            slot_indices[None, :],
            torch.full_like(accept_index_2d, -1),
        )
        .max(dim=1)
        .values
    )
    safe_path_slots = path_slots.clamp(min=0, max=draft_token_num - 1)
    row_indices = torch.arange(bs, device=device)
    path_nodes = (
        accept_index_2d[row_indices, safe_path_slots] - row_indices * draft_token_num
    )
    valid_path_nodes = (
        (path_slots >= 0) & (path_nodes >= 0) & (path_nodes < draft_token_num)
    )
    return torch.where(valid_path_nodes, path_nodes, -1)


def select_precomputed_drafts(
    cache_rows: torch.Tensor,
    accept_tokens: torch.Tensor,
    accept_lens: torch.Tensor,
    accept_path_nodes: torch.Tensor,
    cached_bonus_tokens: torch.Tensor,
    cached_draft_tokens: torch.Tensor,
    cached_tree_masks: torch.Tensor,
    fallback_tree_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select dense precomputed trees entirely on device.

    A miss returns a legal shallow tree whose root is the verified bonus token.
    No result-dependent value is copied to or inspected by the CPU.
    """
    bs = accept_lens.shape[0]
    draft_token_num = cached_draft_tokens.shape[-1]
    device = accept_tokens.device

    has_precomputed_row = cache_rows >= 0
    safe_cache_rows = cache_rows.clamp(min=0)
    accept_lens_long = accept_lens.to(torch.long)
    last_slots = (accept_lens_long - 1).clamp(min=0, max=draft_token_num - 1)
    row_indices = torch.arange(bs, device=device)
    accept_tokens_2d = accept_tokens.reshape(bs, draft_token_num)
    bonus_tokens = accept_tokens_2d[row_indices, last_slots].to(torch.int32)

    path_nodes = accept_path_nodes.to(torch.long)
    valid_path = (
        has_precomputed_row
        & (accept_lens_long > 0)
        & (accept_lens_long <= draft_token_num)
        & (path_nodes >= 0)
        & (path_nodes < draft_token_num)
    )
    safe_path_nodes = path_nodes.clamp(min=0, max=draft_token_num - 1)

    bonus_candidates = cached_bonus_tokens[safe_cache_rows, safe_path_nodes]
    slot_matches = (bonus_candidates == bonus_tokens[:, None]) & valid_path[:, None]
    cache_hits = slot_matches.any(dim=1)
    bonus_slots = slot_matches.to(torch.int32).argmax(dim=1).to(torch.long)

    cached_drafts = cached_draft_tokens[safe_cache_rows, safe_path_nodes, bonus_slots]
    cached_masks = cached_tree_masks[safe_cache_rows, safe_path_nodes, bonus_slots]

    fallback_drafts = torch.zeros_like(cached_drafts)
    fallback_drafts[:, 0] = bonus_tokens.to(fallback_drafts.dtype)
    fallback_masks = fallback_tree_mask.expand(bs, -1, -1)

    selected_drafts = torch.where(cache_hits[:, None], cached_drafts, fallback_drafts)
    selected_masks = torch.where(
        cache_hits[:, None, None], cached_masks, fallback_masks
    )
    return selected_drafts, selected_masks, cache_hits


def select_precomputed_drafts_for_rows(
    cache_rows: Sequence[int],
    accept_tokens: torch.Tensor,
    accept_lens: torch.Tensor,
    accept_path_nodes: torch.Tensor,
    cached_bonus_tokens: torch.Tensor,
    cached_draft_tokens: torch.Tensor,
    cached_tree_masks: torch.Tensor,
    fallback_tree_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Production select entry point, including the request-row H2D staging."""
    cache_rows_cpu = torch.tensor(
        cache_rows,
        dtype=torch.long,
        device="cpu",
        pin_memory=accept_tokens.is_cuda,
    )
    cache_rows_device = cache_rows_cpu.to(
        accept_tokens.device, non_blocking=accept_tokens.is_cuda
    )
    return select_precomputed_drafts(
        cache_rows_device,
        accept_tokens,
        accept_lens,
        accept_path_nodes,
        cached_bonus_tokens,
        cached_draft_tokens,
        cached_tree_masks,
        fallback_tree_mask,
    )


def apply_precomputed_drafts_for_rows(
    cache_rows: Sequence[int],
    accept_tokens: torch.Tensor,
    accept_lens: torch.Tensor,
    accept_path_nodes: torch.Tensor,
    cached_bonus_tokens: torch.Tensor,
    cached_draft_tokens: torch.Tensor,
    cached_tree_masks: torch.Tensor,
    fallback_tree_mask: torch.Tensor,
    draft_tokens: torch.Tensor,
    tree_mask: torch.Tensor,
) -> torch.Tensor:
    """Run production selection and install the result in worker-owned buffers."""
    selected_drafts, selected_masks, cache_hits = select_precomputed_drafts_for_rows(
        cache_rows,
        accept_tokens,
        accept_lens,
        accept_path_nodes,
        cached_bonus_tokens,
        cached_draft_tokens,
        cached_tree_masks,
        fallback_tree_mask,
    )
    draft_tokens.copy_(selected_drafts.reshape(-1), non_blocking=True)
    tree_mask.copy_(selected_masks.reshape(-1), non_blocking=True)
    return cache_hits
