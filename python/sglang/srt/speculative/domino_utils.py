from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

_DOMINO_CANDIDATE_POOL_SIZE = 2048


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


def validate_domino_runtime(
    *,
    device: torch.device,
    tp_size: int,
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
    if int(tp_size) != 1:
        raise ValueError(f"DFLASH Domino currently requires TP=1, got TP={tp_size}.")
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

    shard = getattr(lm_head, "shard_indices", None)
    if shard is not None:
        if int(shard.num_added_elements) != 0:
            raise ValueError(
                "DFLASH Domino does not support added-vocab lm_head shards."
            )
        if int(shard.org_vocab_start_index) != 0 or int(shard.num_org_elements) != int(
            target_vocab_size
        ):
            raise ValueError(
                "DFLASH Domino requires the complete target vocabulary on TP=1."
            )
    elif int(lm_head_weight.shape[0]) != int(target_vocab_size):
        raise ValueError(
            "DFLASH Domino lm_head row count must equal the target vocab size, "
            f"got rows={int(lm_head_weight.shape[0])}, vocab={target_vocab_size}."
        )

    if int(embedding_weight.shape[0]) < int(target_vocab_size):
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

    weight = lm_head_weight[: int(vocab_size)]
    z_for_logits = z.to(weight.dtype) if z.dtype != weight.dtype else z
    logits_input = (
        z_for_logits.transpose(0, 1)
        .contiguous()
        .view(num_proposals * batch_size, hidden_size)
    )
    base_logits = F.linear(logits_input, weight).view(num_proposals, batch_size, -1)

    first_ids = torch.argmax(base_logits[0], dim=-1).to(torch.long)
    proposals = [first_ids]
    if num_proposals == 1:
        return first_ids[:, None]

    candidate_ids = None
    candidate_base = None
    candidate_weight = None
    if 0 < candidate_pool_size < int(vocab_size):
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
