from dataclasses import dataclass

import torch


FLOAT32_EXACT_INT_LIMIT = 1 << 24


@dataclass(frozen=True)
class VocabState:
    max_values: torch.Tensor
    argmax_ids: torch.Tensor
    logsumexp: torch.Tensor
    max_probs: torch.Tensor


def local_vocab_state_from_logits(
    *,
    local_logits: torch.Tensor,
    vocab_start: int,
    valid_vocab_size: int | None = None,
    penalized_token_ids: torch.Tensor | None = None,
    penalty_lambda: float = 0.0,
) -> VocabState:
    if valid_vocab_size is not None:
        local_logits = local_logits[:, : int(valid_vocab_size)]
    local_logits = local_logits.float()
    if penalized_token_ids is not None and penalty_lambda > 0:
        local_logits = local_logits.clone()
        local_token_ids = penalized_token_ids.to(device=local_logits.device).long() - int(
            vocab_start
        )
        valid_rows = (local_token_ids >= 0) & (local_token_ids < local_logits.shape[-1])
        if valid_rows.any():
            row_ids = torch.arange(local_logits.shape[0], device=local_logits.device)
            local_logits[row_ids[valid_rows], local_token_ids[valid_rows]] -= float(
                penalty_lambda
            )
    max_values, local_argmax_ids = torch.max(local_logits, dim=-1)
    logsumexp = torch.logsumexp(local_logits, dim=-1)
    return VocabState(
        max_values=max_values,
        argmax_ids=local_argmax_ids.long() + int(vocab_start),
        logsumexp=logsumexp,
        max_probs=torch.exp(max_values - logsumexp),
    )


def can_pack_vocab_ids_as_float32(max_vocab_id_inclusive: int) -> bool:
    max_vocab_id_inclusive = int(max_vocab_id_inclusive)
    return 0 <= max_vocab_id_inclusive <= FLOAT32_EXACT_INT_LIMIT


def pack_vocab_state_for_tp_gather(state: VocabState) -> torch.Tensor:
    if state.max_values.dtype != torch.float32:
        raise ValueError("VocabState.max_values must be torch.float32")
    if state.logsumexp.dtype != torch.float32:
        raise ValueError("VocabState.logsumexp must be torch.float32")

    return torch.stack(
        [
            state.max_values,
            state.logsumexp,
            state.argmax_ids.to(dtype=torch.float32),
        ],
        dim=-1,
    ).contiguous()


def merge_gathered_packed_vocab_state(gathered: torch.Tensor) -> VocabState:
    if gathered.ndim != 3 or gathered.shape[-1] != 3:
        raise ValueError("gathered packed vocab state must have shape [tp, rows, 3]")
    if gathered.shape[0] == 0:
        raise ValueError("merge_gathered_packed_vocab_state requires at least one rank")

    max_values_by_rank = gathered[:, :, 0]
    logsumexp_by_rank = gathered[:, :, 1]
    argmax_ids_by_rank = gathered[:, :, 2].long()

    best_values = max_values_by_rank[0]
    best_ids = argmax_ids_by_rank[0]
    for rank in range(1, max_values_by_rank.shape[0]):
        rank_values = max_values_by_rank[rank]
        rank_ids = argmax_ids_by_rank[rank]
        use_rank = (rank_values > best_values) | (
            (rank_values == best_values) & (rank_ids < best_ids)
        )
        best_values = torch.where(use_rank, rank_values, best_values)
        best_ids = torch.where(use_rank, rank_ids, best_ids)

    merged_logsumexp = torch.logsumexp(logsumexp_by_rank, dim=0)
    return VocabState(
        max_values=best_values,
        argmax_ids=best_ids.long(),
        logsumexp=merged_logsumexp,
        max_probs=torch.exp(best_values - merged_logsumexp),
    )


def merge_vocab_states(states: list[VocabState]) -> VocabState:
    if not states:
        raise ValueError("merge_vocab_states requires at least one local state")

    max_values_by_rank = torch.stack([state.max_values for state in states], dim=0)
    argmax_ids_by_rank = torch.stack([state.argmax_ids for state in states], dim=0)
    logsumexp_by_rank = torch.stack([state.logsumexp for state in states], dim=0)

    best_values = max_values_by_rank[0]
    best_ids = argmax_ids_by_rank[0]
    for rank in range(1, max_values_by_rank.shape[0]):
        rank_values = max_values_by_rank[rank]
        rank_ids = argmax_ids_by_rank[rank]
        use_rank = (rank_values > best_values) | (
            (rank_values == best_values) & (rank_ids < best_ids)
        )
        best_values = torch.where(use_rank, rank_values, best_values)
        best_ids = torch.where(use_rank, rank_ids, best_ids)

    merged_logsumexp = torch.logsumexp(logsumexp_by_rank, dim=0)
    return VocabState(
        max_values=best_values,
        argmax_ids=best_ids.long(),
        logsumexp=merged_logsumexp,
        max_probs=torch.exp(best_values - merged_logsumexp),
    )


def argmax_max_prob_from_logits_output(
    logits_output,
    *,
    start: int,
    end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    vocab_state = getattr(logits_output, "dllm_vocab_state", None)
    if vocab_state is not None:
        return vocab_state.argmax_ids[start:end], vocab_state.max_probs[start:end]

    logits = logits_output.full_logits[start:end].float()
    max_values, argmax_ids = torch.max(logits, dim=-1)
    max_probs = torch.exp(max_values - torch.logsumexp(logits, dim=-1))
    return argmax_ids.long(), max_probs


def low_confidence_transfer_mask(
    *,
    input_ids: torch.Tensor,
    argmax_ids: torch.Tensor,
    max_probs: torch.Tensor,
    mask_id: int,
    threshold: float,
) -> torch.Tensor:
    del argmax_ids
    mask_index = input_ids == int(mask_id)
    transfer_index = torch.zeros_like(mask_index, dtype=torch.bool)
    if not mask_index.any():
        return transfer_index

    confidence = torch.where(
        mask_index,
        max_probs,
        torch.full_like(max_probs, -torch.inf),
    )
    transfer_index = confidence > float(threshold)
    if not transfer_index.any():
        select_index = torch.topk(confidence, k=1).indices
        transfer_index[select_index] = True
    return transfer_index
