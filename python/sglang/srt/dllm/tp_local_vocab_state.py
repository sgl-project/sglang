from dataclasses import dataclass

import torch

FLOAT32_EXACT_INT_LIMIT = 1 << 24
LOCAL_VOCAB_STATE_TRITON_MAX_BLOCK_VOCAB = 131072


@dataclass(frozen=True)
class VocabState:
    max_values: torch.Tensor
    argmax_ids: torch.Tensor
    logsumexp: torch.Tensor
    max_probs: torch.Tensor

    def slice_rows(self, end: int) -> "VocabState":
        """Return the leading rows while preserving the state contract."""
        return VocabState(
            max_values=self.max_values[:end],
            argmax_ids=self.argmax_ids[:end],
            logsumexp=self.logsumexp[:end],
            max_probs=self.max_probs[:end],
        )


def local_vocab_state_from_logits(
    *,
    local_logits: torch.Tensor,
    vocab_start: int,
    valid_vocab_size: int | None = None,
) -> VocabState:
    if local_logits.dim() != 2:
        raise ValueError("local_logits must be a 2D tensor")
    valid_vocab_size = (
        local_logits.shape[-1] if valid_vocab_size is None else int(valid_vocab_size)
    )
    if valid_vocab_size <= 0 or valid_vocab_size > local_logits.shape[-1]:
        raise ValueError("valid_vocab_size must be in [1, local_logits.shape[-1]]")

    local_logits = local_logits[:, :valid_vocab_size]
    local_logits = local_logits.float()
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


def can_use_local_vocab_state_triton(valid_vocab_size: int) -> bool:
    if valid_vocab_size <= 0:
        return False
    block_vocab = 1 << (int(valid_vocab_size) - 1).bit_length()
    return block_vocab <= LOCAL_VOCAB_STATE_TRITON_MAX_BLOCK_VOCAB


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


def _merge_rank_vocab_state(
    max_values_by_rank: torch.Tensor,
    argmax_ids_by_rank: torch.Tensor,
    logsumexp_by_rank: torch.Tensor,
) -> VocabState:
    if max_values_by_rank.ndim != 2:
        raise ValueError("rank vocab state must have shape [tp, rows]")
    if max_values_by_rank.shape[0] == 0:
        raise ValueError("at least one rank is required")

    best_values = torch.max(max_values_by_rank, dim=0).values
    max_id = torch.iinfo(argmax_ids_by_rank.dtype).max
    candidate_ids = torch.where(
        max_values_by_rank == best_values.unsqueeze(0),
        argmax_ids_by_rank,
        max_id,
    )
    best_ids = torch.min(candidate_ids, dim=0).values
    merged_logsumexp = torch.logsumexp(logsumexp_by_rank, dim=0)
    return VocabState(
        max_values=best_values,
        argmax_ids=best_ids.long(),
        logsumexp=merged_logsumexp,
        max_probs=torch.exp(best_values - merged_logsumexp),
    )


def merge_gathered_packed_vocab_state(gathered: torch.Tensor) -> VocabState:
    if gathered.ndim != 3 or gathered.shape[-1] != 3:
        raise ValueError("gathered packed vocab state must have shape [tp, rows, 3]")
    return _merge_rank_vocab_state(
        max_values_by_rank=gathered[:, :, 0],
        argmax_ids_by_rank=gathered[:, :, 2].long(),
        logsumexp_by_rank=gathered[:, :, 1],
    )


def merge_vocab_states(states: list[VocabState]) -> VocabState:
    if not states:
        raise ValueError("merge_vocab_states requires at least one local state")

    max_values_by_rank = torch.stack([state.max_values for state in states], dim=0)
    argmax_ids_by_rank = torch.stack([state.argmax_ids for state in states], dim=0)
    logsumexp_by_rank = torch.stack([state.logsumexp for state in states], dim=0)

    return _merge_rank_vocab_state(
        max_values_by_rank=max_values_by_rank,
        argmax_ids_by_rank=argmax_ids_by_rank,
        logsumexp_by_rank=logsumexp_by_rank,
    )


def gather_vocab_state_across_tp(
    *,
    local_state: VocabState,
    tp_group,
    tp_size: int,
    packed: bool,
) -> VocabState:
    """Gather rank-local statistics and reconstruct global argmax/confidence."""
    if tp_size <= 1:
        return local_state

    num_rows = local_state.max_values.shape[0]
    if packed:
        local_packed = pack_vocab_state_for_tp_gather(local_state)
        gathered = torch.empty(
            (tp_size * num_rows, 3),
            device=local_packed.device,
            dtype=local_packed.dtype,
        )
        tp_group.all_gather_into_tensor(gathered, local_packed)
        return merge_gathered_packed_vocab_state(gathered.view(tp_size, num_rows, 3))

    def gather(tensor: torch.Tensor) -> torch.Tensor:
        output = torch.empty(
            (tp_size * num_rows,), device=tensor.device, dtype=tensor.dtype
        )
        tp_group.all_gather_into_tensor(output, tensor.contiguous())
        return output.view(tp_size, num_rows)

    return _merge_rank_vocab_state(
        max_values_by_rank=gather(local_state.max_values),
        argmax_ids_by_rank=gather(local_state.argmax_ids),
        logsumexp_by_rank=gather(local_state.logsumexp),
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

    if logits_output.full_logits is None:
        raise ValueError("dLLM output contains neither vocab state nor full logits")
    logits = logits_output.full_logits[start:end].float()
    argmax_ids = torch.argmax(logits, dim=-1)
    max_probs = torch.gather(
        torch.softmax(logits, dim=-1), dim=-1, index=argmax_ids.unsqueeze(-1)
    ).squeeze(-1)
    return argmax_ids.long(), max_probs
