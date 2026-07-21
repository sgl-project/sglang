from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

_DOMINO_CANDIDATE_POOL_SIZE = 2048
# This is a throughput policy for the logical gathered tensor, not a peak-memory cap.
_DOMINO_TP_FULL_BASE_LOGITS_MAX_BYTES = 96 * 1024 * 1024


def _domino_gru_cell(
    prefix_gru: nn.GRU, input: torch.Tensor, hidden: torch.Tensor
) -> torch.Tensor:
    """Run one feedback step without cuDNN's per-call RNN weight packing."""
    return torch.ops.aten.gru_cell.default(
        input,
        hidden,
        prefix_gru.weight_ih_l0,
        prefix_gru.weight_hh_l0,
        prefix_gru.bias_ih_l0 if prefix_gru.bias else None,
        prefix_gru.bias_hh_l0 if prefix_gru.bias else None,
    )


def _domino_tp_first_ids(
    local_logits: torch.Tensor,
    *,
    org_vocab_start: int,
    num_org: int,
    tp_group,
) -> torch.Tensor:
    """Select the full-vocab first argmax from contiguous vocab shards."""
    local_max, local_arg = torch.max(local_logits[:, :num_org], dim=-1)
    local_ids = local_arg.to(torch.int64) + int(org_vocab_start)
    tp_size = int(tp_group.world_size)
    batch_size = int(local_logits.shape[0])
    gathered_max = torch.empty(
        (tp_size * batch_size,), dtype=local_max.dtype, device=local_max.device
    )
    gathered_ids = torch.empty(
        (tp_size * batch_size,), dtype=torch.int64, device=local_max.device
    )
    tp_group.all_gather_into_tensor(gathered_max, local_max.contiguous())
    tp_group.all_gather_into_tensor(gathered_ids, local_ids.contiguous())
    gathered_max = gathered_max.view(tp_size, batch_size)
    gathered_ids = gathered_ids.view(tp_size, batch_size)
    best_rank = torch.argmax(gathered_max, dim=0, keepdim=True)
    return torch.gather(gathered_ids, 0, best_rank).squeeze(0)


def _domino_tp_candidate_state(
    local_feedback_logits: torch.Tensor,
    *,
    candidate_pool_size: int,
    org_vocab_start: int,
    num_org: int,
    tp_group,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one strict global candidate pool from vocab-sharded base logits."""
    num_steps, batch_size, local_vocab_size = local_feedback_logits.shape
    tp_size = int(tp_group.world_size)
    local_k = min(int(candidate_pool_size), int(local_vocab_size))
    if int(candidate_pool_size) > tp_size * local_k:
        raise ValueError("Domino TP shards do not cover the requested candidate pool.")

    pool_scores = local_feedback_logits.amax(dim=0)
    if num_org < local_vocab_size:
        pool_scores[:, num_org:].fill_(float("-inf"))
    local_scores, local_positions = torch.topk(
        pool_scores, k=local_k, dim=-1, sorted=False
    )
    local_ids = local_positions.to(torch.int64) + int(org_vocab_start)

    gathered_scores = torch.empty(
        (tp_size * batch_size, local_k),
        dtype=local_scores.dtype,
        device=local_scores.device,
    )
    gathered_ids = torch.empty(
        (tp_size * batch_size, local_k),
        dtype=torch.int64,
        device=local_scores.device,
    )
    tp_group.all_gather_into_tensor(gathered_scores, local_scores.contiguous())
    tp_group.all_gather_into_tensor(gathered_ids, local_ids.contiguous())
    gathered_scores = (
        gathered_scores.view(tp_size, batch_size, local_k)
        .permute(1, 0, 2)
        .reshape(batch_size, tp_size * local_k)
    )
    gathered_ids = (
        gathered_ids.view(tp_size, batch_size, local_k)
        .permute(1, 0, 2)
        .reshape(batch_size, tp_size * local_k)
    )
    global_positions = torch.topk(
        gathered_scores,
        k=int(candidate_pool_size),
        dim=-1,
        sorted=False,
    ).indices
    candidate_ids = torch.gather(gathered_ids, 1, global_positions).contiguous()

    owned = (candidate_ids >= int(org_vocab_start)) & (
        candidate_ids < int(org_vocab_start + num_org)
    )
    local_positions = (candidate_ids - int(org_vocab_start)).clamp(
        0, max(num_org - 1, 0)
    )
    candidate_base = torch.gather(
        local_feedback_logits.transpose(0, 1),
        2,
        local_positions[:, None, :].expand(-1, num_steps, -1),
    )
    # Each global candidate is owned by exactly one vocab shard, so SUM
    # reconstructs its base logit without gathering the full vocabulary.
    candidate_base.masked_fill_(~owned[:, None, :], 0)
    candidate_base = tp_group.all_reduce(candidate_base.contiguous())
    return candidate_ids, candidate_base.transpose(0, 1)


def validate_domino_runtime(
    *,
    device: torch.device,
    tp_size: int,
    tp_rank: int,
    target_vocab_size: int,
    draft_vocab_size: int,
    hidden_size: int,
    target_embedding: nn.Module,
    lm_head: nn.Module,
    prefix_gru: nn.GRU,
    embed_proj: nn.Sequential,
) -> None:
    """Validate the deliberately narrow correctness-first Domino runtime."""
    if device.type != "cuda":
        raise ValueError(f"DFLASH Domino currently requires CUDA, got {device}.")
    tp_size = int(tp_size)
    if tp_size < 1:
        raise ValueError(f"DFLASH Domino requires TP>=1, got TP={tp_size}.")
    tp_rank = int(tp_rank)
    if not 0 <= tp_rank < tp_size:
        raise ValueError(
            f"DFLASH Domino requires 0<=TP rank<TP size, got rank={tp_rank}, size={tp_size}."
        )
    if int(target_vocab_size) != int(draft_vocab_size):
        raise ValueError(
            "DFLASH Domino requires identical target and draft vocab sizes, "
            f"got target={target_vocab_size}, draft={draft_vocab_size}."
        )

    embedding_weight = getattr(target_embedding, "weight", None)
    lm_head_weight = getattr(lm_head, "weight", None)
    if embedding_weight is None or lm_head_weight is None:
        raise ValueError(
            "DFLASH Domino requires target embedding and lm_head weight tensors."
        )

    lm_head_shard = getattr(lm_head, "shard_indices", None)
    if lm_head_shard is not None:
        if (
            int(getattr(lm_head, "num_added_embeddings", 0)) != 0
            or int(lm_head_shard.num_added_elements) != 0
        ):
            raise ValueError(
                "DFLASH Domino does not support added-vocab lm_head shards."
            )
        if int(getattr(lm_head, "org_vocab_size", target_vocab_size)) != int(
            target_vocab_size
        ):
            raise ValueError(
                "DFLASH Domino lm_head original vocab size does not match the target."
            )
        if int(getattr(lm_head, "tp_size", tp_size)) != tp_size:
            raise ValueError(
                "DFLASH Domino lm_head TP size does not match the runtime TP size."
            )
        required_shard_fields = (
            "org_vocab_start_index",
            "org_vocab_end_index",
            "num_org_elements",
            "num_org_elements_padded",
        )
        missing_shard_fields = [
            name for name in required_shard_fields if not hasattr(lm_head_shard, name)
        ]
        if missing_shard_fields:
            raise ValueError(
                "DFLASH Domino lm_head shard metadata is missing: "
                + ", ".join(missing_shard_fields)
            )
        org_vocab_start = int(lm_head_shard.org_vocab_start_index)
        org_vocab_end = int(lm_head_shard.org_vocab_end_index)
        num_org = int(lm_head_shard.num_org_elements)
        num_org_padded = int(lm_head_shard.num_org_elements_padded)
        if (
            num_org <= 0
            or org_vocab_start < 0
            or org_vocab_end != org_vocab_start + num_org
            or org_vocab_end > int(target_vocab_size)
            or num_org_padded < num_org
        ):
            raise ValueError("DFLASH Domino lm_head original-vocab shard is invalid.")
        if int(lm_head_weight.shape[0]) < num_org_padded:
            raise ValueError(
                "DFLASH Domino lm_head weight is smaller than its padded vocab shard."
            )
        expected_start = tp_rank * num_org_padded
        expected_end = min(expected_start + num_org_padded, int(target_vocab_size))
        if (
            org_vocab_start != expected_start
            or org_vocab_end != expected_end
            or num_org != expected_end - expected_start
        ):
            raise ValueError(
                "DFLASH Domino lm_head vocab shard does not match its TP rank."
            )
    else:
        if tp_size != 1:
            raise ValueError("DFLASH Domino requires lm_head shard metadata for TP>1.")
        if int(lm_head_weight.shape[0]) != int(target_vocab_size):
            raise ValueError(
                "DFLASH Domino lm_head row count must equal the target vocab size, "
                f"got rows={int(lm_head_weight.shape[0])}, vocab={target_vocab_size}."
            )

    embedding_shard = getattr(target_embedding, "shard_indices", None)
    if embedding_shard is not None:
        if (
            int(getattr(target_embedding, "num_added_embeddings", 0)) != 0
            or int(embedding_shard.num_added_elements) != 0
        ):
            raise ValueError(
                "DFLASH Domino does not support added-vocab embedding shards."
            )
        if int(getattr(target_embedding, "org_vocab_size", target_vocab_size)) != int(
            target_vocab_size
        ):
            raise ValueError(
                "DFLASH Domino embedding original vocab size does not match the target."
            )
        embedding_tp_size = int(getattr(target_embedding, "tp_size", tp_size))
        if embedding_tp_size not in (1, tp_size):
            raise ValueError(
                "DFLASH Domino embedding TP size does not match the runtime TP size."
            )
        required_embedding_rows = (
            int(target_vocab_size)
            if embedding_tp_size == 1
            else int(embedding_shard.num_org_elements_padded)
        )
        if int(embedding_weight.shape[0]) < required_embedding_rows:
            raise ValueError(
                "DFLASH Domino embedding weight is smaller than its padded vocab shard."
            )
    elif int(embedding_weight.shape[0]) < int(target_vocab_size):
        if tp_size != 1:
            raise ValueError(
                "DFLASH Domino requires embedding shard metadata for TP>1."
            )
        raise ValueError(
            "DFLASH Domino target embedding has fewer rows than the target vocab "
            f"size: rows={int(embedding_weight.shape[0])}, vocab={target_vocab_size}."
        )

    if int(embedding_weight.shape[-1]) != int(hidden_size) or int(
        lm_head_weight.shape[-1]
    ) != int(hidden_size):
        raise ValueError(
            "DFLASH Domino target embedding/lm_head hidden size does not match "
            f"the draft hidden size {hidden_size}."
        )
    if int(prefix_gru.input_size) != int(hidden_size):
        raise ValueError(
            "DFLASH Domino GRU input size does not match the draft hidden size."
        )
    if int(embed_proj[0].in_features) != int(hidden_size + prefix_gru.hidden_size):
        raise ValueError("DFLASH Domino projector input shape is inconsistent.")
    if int(embed_proj[2].out_features) != int(target_vocab_size):
        raise ValueError("DFLASH Domino projector output vocab size is inconsistent.")

    weights = (
        embedding_weight,
        lm_head_weight,
        prefix_gru.weight_ih_l0,
        prefix_gru.weight_hh_l0,
        embed_proj[0].weight,
        embed_proj[2].weight,
    )
    non_bf16 = [
        str(weight.dtype) for weight in weights if weight.dtype != torch.bfloat16
    ]
    if non_bf16:
        raise ValueError(
            "DFLASH Domino currently requires BF16 target and projector weights; "
            f"found {non_bf16}."
        )


@torch.no_grad()
def domino_greedy_rollout(
    *,
    draft_hidden: torch.Tensor,
    verified_ids: torch.Tensor,
    target_embedding: nn.Module,
    lm_head_weight: torch.Tensor,
    prefix_gru: nn.GRU,
    embed_proj: nn.Sequential,
    vocab_size: int,
    shift_label: bool,
    candidate_pool_size: int = _DOMINO_CANDIDATE_POOL_SIZE,
    tp_group=None,
    lm_head_org_vocab_start: int = 0,
    lm_head_num_org: int | None = None,
    lm_head_num_org_padded: int | None = None,
) -> torch.Tensor:
    """Generate a Domino chain using one block-shared base-logit candidate pool."""
    if draft_hidden.ndim != 3:
        raise ValueError(
            f"draft_hidden must have shape [batch, block, hidden], got {tuple(draft_hidden.shape)}."
        )
    batch_size, block_size, hidden_size = draft_hidden.shape
    if verified_ids.shape != (batch_size,):
        raise ValueError(
            f"verified_ids must have shape ({batch_size},), got {tuple(verified_ids.shape)}."
        )

    num_proposals = int(block_size) - 1
    if num_proposals < 1:
        raise ValueError(f"Domino requires block_size > 1, got {block_size}.")
    candidate_pool_size = int(candidate_pool_size)
    if candidate_pool_size < 0:
        raise ValueError(
            "Domino candidate_pool_size must be non-negative, "
            f"got {candidate_pool_size}."
        )
    candidate_pool_size = min(candidate_pool_size, int(vocab_size))
    start = 0 if shift_label else 1
    z = draft_hidden[:, start : start + num_proposals, :]
    if int(z.shape[1]) != num_proposals:
        raise ValueError(
            "Domino draft hidden states do not contain enough proposal positions."
        )

    tp_size = int(tp_group.world_size) if tp_group is not None else 1
    if tp_size > 1 and (lm_head_num_org is None or lm_head_num_org_padded is None):
        raise ValueError(
            "Domino TP rollout requires local lm_head vocab shard metadata."
        )
    local_vocab_size = int(lm_head_num_org_padded) if tp_size > 1 else int(vocab_size)
    if tp_size > 1:
        num_org = int(lm_head_num_org)
        org_vocab_start = int(lm_head_org_vocab_start)
        if (
            num_org <= 0
            or num_org > local_vocab_size
            or org_vocab_start < 0
            or org_vocab_start + num_org > int(vocab_size)
        ):
            raise ValueError(
                "Domino TP rollout received an invalid lm_head vocab shard."
            )
    if int(lm_head_weight.shape[0]) < local_vocab_size:
        raise ValueError(
            "Domino lm_head weight is smaller than its padded vocab shard."
        )
    weight = lm_head_weight[:local_vocab_size]
    z_for_logits = z.to(weight.dtype) if z.dtype != weight.dtype else z
    logits_input = (
        z_for_logits.transpose(0, 1)
        .contiguous()
        .view(num_proposals * batch_size, hidden_size)
    )
    local_logits = F.linear(logits_input, weight).view(
        num_proposals, batch_size, local_vocab_size
    )
    full_base_logits_bytes = (
        num_proposals * batch_size * int(vocab_size) * local_logits.element_size()
    )
    use_tp_candidate_pool = (
        tp_size > 1
        and 0 < candidate_pool_size < int(vocab_size)
        and full_base_logits_bytes > _DOMINO_TP_FULL_BASE_LOGITS_MAX_BYTES
    )
    first_ids = None
    candidate_ids = None
    candidate_base = None
    if tp_size == 1:
        base_logits = local_logits[:, :, : int(vocab_size)]
    elif use_tp_candidate_pool:
        first_ids = _domino_tp_first_ids(
            local_logits[0],
            org_vocab_start=int(lm_head_org_vocab_start),
            num_org=int(lm_head_num_org),
            tp_group=tp_group,
        )
        if num_proposals > 1:
            candidate_ids, candidate_base = _domino_tp_candidate_state(
                local_logits[1:],
                candidate_pool_size=candidate_pool_size,
                org_vocab_start=int(lm_head_org_vocab_start),
                num_org=int(lm_head_num_org),
                tp_group=tp_group,
            )
        base_logits = None
    else:
        local_logits_t = local_logits.view(
            num_proposals * batch_size, local_vocab_size
        ).T.contiguous()
        gathered_logits = torch.empty(
            (tp_size * local_vocab_size, num_proposals * batch_size),
            dtype=local_logits.dtype,
            device=local_logits.device,
        )
        tp_group.all_gather_into_tensor(gathered_logits, local_logits_t)
        base_logits = (
            gathered_logits.T[:, : int(vocab_size)]
            .contiguous()
            .view(num_proposals, batch_size, int(vocab_size))
        )

    if first_ids is None:
        first_ids = torch.argmax(base_logits[0], dim=-1).to(torch.long)
    proposals = [first_ids]
    if num_proposals == 1:
        return first_ids[:, None]

    candidate_weight = None
    if candidate_ids is not None:
        candidate_weight = F.embedding(candidate_ids, embed_proj[2].weight)
    elif 0 < candidate_pool_size < int(vocab_size):
        feedback_logits = base_logits[1:]
        candidate_ids = torch.topk(
            feedback_logits.amax(dim=0),
            k=candidate_pool_size,
            dim=-1,
            sorted=False,
        ).indices.contiguous()
        candidate_base = torch.gather(
            feedback_logits.transpose(0, 1),
            2,
            candidate_ids[:, None, :].expand(-1, num_proposals - 1, -1),
        ).transpose(0, 1)
        candidate_weight = F.embedding(candidate_ids, embed_proj[2].weight)

    prefix_ids = torch.stack((verified_ids, first_ids), dim=1)
    _, gru_hidden = prefix_gru(target_embedding(prefix_ids))

    for index in range(1, num_proposals):
        step_hidden = z[:, index, :]
        correction_hidden = embed_proj[1](
            embed_proj[0](torch.cat((step_hidden, gru_hidden[0]), dim=-1))
        )
        if candidate_ids is None:
            correction = embed_proj[2](correction_hidden)
            next_ids = torch.argmax(base_logits[index] + correction, dim=-1).to(
                torch.long
            )
        else:
            correction = torch.bmm(
                candidate_weight, correction_hidden.unsqueeze(-1)
            ).squeeze(-1)
            candidate_position = torch.argmax(
                candidate_base[index - 1] + correction, dim=-1
            )
            next_ids = torch.gather(
                candidate_ids, 1, candidate_position[:, None]
            ).squeeze(1)
        proposals.append(next_ids)
        if index + 1 < num_proposals:
            gru_hidden = _domino_gru_cell(
                prefix_gru, target_embedding(next_ids), gru_hidden[0]
            )[None]

    return torch.stack(proposals, dim=1)
