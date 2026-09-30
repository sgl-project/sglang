from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import msgspec
import torch

from sglang.kernels.ops.speculative.lilicorr import MAX_FUSED_CANDIDATE_TOPK
from sglang.srt.environ import envs
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.models.dflash import candidate_topk
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.speculative.dflash_utils import _get_dflash_config
from sglang.srt.speculative.dspark_components.dspark_draft import resolve_greedy_mask

logger = logging.getLogger(__name__)


def resolve_sampling_enabled(*, device_supported: bool) -> bool:
    requested = envs.SGLANG_ENABLE_LILICORR_SAMPLING.get()
    if envs.SGLANG_LILICORR_REQUIRE_SAMPLING.get() and not requested:
        raise RuntimeError(
            "SGLANG_LILICORR_REQUIRE_SAMPLING is set but SGLANG_ENABLE_LILICORR_SAMPLING "
            "is not; this process would serve the greedy commit under a sampled arm's name."
        )
    logger.info("LiLiCorr draft commit: %s", "sampled" if requested else "greedy")
    return requested and device_supported


class LiLiCorrConfig(msgspec.Struct, frozen=True):
    candidate_topk: int
    hidden_size: int
    num_layers: int
    num_heads: int
    mlp_ratio: float
    factor_dim: int
    vector_eps: float
    logit_scale: float

    def resolve_hidden_size(self, *, model_hidden_size: int) -> int:
        # 0 means "as wide as the draft", recorded for a head with no token_proj.
        return int(self.hidden_size) if self.hidden_size else int(model_hidden_size)


def _parse_lilicorr_config(dflash_cfg: dict) -> Optional[LiLiCorrConfig]:
    if not any(key.startswith("lilicorr_") for key in dflash_cfg):
        return None
    enabled = dflash_cfg.get("lilicorr_enabled")
    if enabled is not None and not bool(enabled):
        return None

    def required(key: str, cast, *, positive: bool = True):
        # No defaults: logit_scale and vector_eps change no shape, so load cannot catch them.
        full_key = f"lilicorr_{key}"
        if full_key not in dflash_cfg:
            raise ValueError(
                f"DFLASH dflash_config.{full_key} is required to rebuild the LiLiCorr "
                "head, and the checkpoint does not carry it."
            )
        try:
            value = cast(dflash_cfg[full_key])
        except Exception as e:
            raise ValueError(
                f"Invalid dflash_config.{full_key}={dflash_cfg[full_key]!r}."
            ) from e
        if positive and value <= 0:
            raise ValueError(f"dflash_config.{full_key} must be positive, got {value}.")
        return value

    candidate_topk = required("candidate_topk", int)
    if candidate_topk & (candidate_topk - 1):
        raise ValueError(
            f"dflash_config.lilicorr_candidate_topk must be a power of two, got "
            f"{candidate_topk}: tl.arange requires a power-of-two extent."
        )
    if candidate_topk > MAX_FUSED_CANDIDATE_TOPK:
        raise ValueError(
            f"dflash_config.lilicorr_candidate_topk={candidate_topk} exceeds the fused "
            f"greedy commit's width of {MAX_FUSED_CANDIDATE_TOPK}, one Triton lane "
            "group; a wider head would serve on the reference path."
        )

    return LiLiCorrConfig(
        candidate_topk=candidate_topk,
        hidden_size=required("hidden_size", int, positive=False),
        num_layers=required("num_layers", int),
        num_heads=required("num_heads", int),
        mlp_ratio=required("mlp_ratio", float),
        factor_dim=required("factor_dim", int),
        vector_eps=required("vector_eps", float),
        logit_scale=required("logit_scale", float),
    )


def parse_lilicorr_draft_config(*, draft_hf_config: Any) -> LiLiCorrConfig:
    config = _parse_lilicorr_config(_get_dflash_config(draft_hf_config))
    if config is None:
        raise ValueError(
            "LiLiCorr requires the lilicorr_* geometry fields in dflash_config; a "
            'checkpoint declaring architectures=["LiLiCorrDraftModel"] without them '
            "cannot be rebuilt."
        )
    return config


def reject_added_vocab(lm_head) -> None:
    if not isinstance(lm_head, VocabParallelEmbedding):
        return
    if int(lm_head.shard_indices.num_added_elements) != 0:
        raise NotImplementedError(
            "LiLiCorr's candidate head does not support added vocabulary: those rows "
            "sit past the padded base shard, so a contiguous top-k would skip them."
        )


def target_input_embeddings(target_model):
    # Not the worker's _resolve_dflash_embedding_module, which returns the draft's own
    # table for Nemotron-3.5: the head must embed with the table it was trained against.
    embed = target_model.get_input_embeddings()
    if embed is None:
        raise RuntimeError("DFLASH target model exposes no input embeddings.")
    return embed


def lilicorr_candidates(
    *, hidden_states: torch.Tensor, lm_head, topk: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    # log_softmax(logits).topk(topk) over the FULL vocabulary: the head was trained on it.
    ids, vals, lse = candidate_topk(
        hidden_states, lm_head, int(topk), with_partition=True
    )
    return vals.float() - lse.unsqueeze(-1), ids


def per_request_last_row(
    *,
    num_rows: int,
    extend_lens: Optional[torch.Tensor],
    commit_lens: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    # Verify's commit_lens index a [bs, block_size]-padded buffer (constant stride);
    # prefill/extend's extend_lens index a packed ragged one.
    if commit_lens is not None:
        bs = int(commit_lens.shape[0])
        if bs == 0 or num_rows % bs != 0:
            return None
        stride = num_rows // bs
        base = torch.arange(bs, device=commit_lens.device, dtype=torch.int64) * stride
        return (base + commit_lens.to(torch.int64) - 1).clamp_min(0)
    if extend_lens is None or extend_lens.numel() == 0:
        return None
    lens = extend_lens.to(torch.int64).flatten()
    return (torch.cumsum(lens, dim=0) - 1).clamp_min(0)


def publish_anchor(
    *,
    draft_sampler,
    ctx_hidden: torch.Tensor,
    extend_lens: Optional[torch.Tensor] = None,
    commit_lens: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    ends = per_request_last_row(
        num_rows=int(ctx_hidden.shape[0]),
        extend_lens=extend_lens,
        commit_lens=commit_lens,
    )
    anchor = None if ends is None else ctx_hidden.index_select(0, ends)
    if draft_sampler is not None:
        draft_sampler.set_anchor(anchor, 0 if anchor is None else int(anchor.shape[0]))
    return anchor


def propose_lilicorr_block(
    *,
    head,
    draft_hidden: torch.Tensor,
    lm_head,
    embed_tokens,
    anchor: Optional[torch.Tensor],
    sampling_info=None,
    sampling_enabled: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    bs, block_size, hidden_size = draft_hidden.shape
    slots = block_size - 1
    pass_hidden = draft_hidden[:, 1:, :]

    log_probs, candidate_tokens = lilicorr_candidates(
        hidden_states=pass_hidden.reshape(bs * slots, hidden_size),
        lm_head=lm_head,
        topk=int(head.candidate_topk),
    )
    candidate_tokens = candidate_tokens.view(bs, slots, int(head.candidate_topk))

    feat = int(head.context_proj.input_size)
    if anchor is None or int(anchor.shape[0]) != bs:
        # The batch resized since the anchor write; score invalid rather than misaligned.
        anchor_hidden = torch.zeros(
            (bs, feat), device=draft_hidden.device, dtype=draft_hidden.dtype
        )
        anchor_valid = torch.zeros((bs,), dtype=torch.bool, device=draft_hidden.device)
    else:
        anchor_hidden = anchor.view(bs, feat)
        anchor_valid = torch.ones((bs,), dtype=torch.bool, device=draft_hidden.device)

    # Outside the head because on the target model it may be a TP-sharded collective.
    common = dict(
        token_embeddings=embed_tokens(candidate_tokens).detach(),
        candidate_tokens=candidate_tokens,
        candidate_log_probs=log_probs.view(bs, slots, int(head.candidate_topk)),
        pass_hidden=pass_hidden,
        anchor_hidden=anchor_hidden,
        anchor_valid=anchor_valid,
    )
    device = draft_hidden.device
    if sampling_enabled and sampling_info is not None:
        # A greedy row's temperature may be 0, and it divides before greedy_mask applies.
        temperatures = sampling_info.temperatures.view(-1)[:bs].float().clamp_min(1e-5)
    else:
        temperatures = torch.ones(bs, dtype=torch.float32, device=device)
    greedy_mask = (
        resolve_greedy_mask(bs=bs, sampling_info=sampling_info, device=device)
        if sampling_enabled
        else torch.ones(bs, dtype=torch.bool, device=device)
    )
    selected, q_rows = head.select_with_proposal(
        uniforms=torch.rand(bs, slots, dtype=torch.float32, device=device),
        temperatures=temperatures,
        greedy_mask=greedy_mask,
        **common,
    )
    if not sampling_enabled:
        return selected.to(torch.long), None, None
    return selected.to(torch.long), candidate_tokens, q_rows


class LiLiCorrDraftSampler:
    def __init__(
        self,
        *,
        head,
        embed_tokens,
        lm_head,
        block_size: int,
        max_bs: int,
        anchor_features: int,
        sampling_enabled: bool,
    ) -> None:
        self.head = head
        self.sampling_enabled = bool(sampling_enabled)
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head
        self.block_size = int(block_size)
        self.slots = self.block_size - 1
        self.topk = int(head.candidate_topk)
        self.max_bs = int(max_bs)

        # From the head, not from lm_head.weight: a quantized head's weight is packed,
        # so its dtype is not the dtype these buffers carry.
        anchor_row = head.slot_embedding
        device, dtype = anchor_row.device, anchor_row.dtype
        max_rows = self.max_bs * self.slots

        self.out = torch.empty((max_rows,), dtype=torch.int64, device=device)
        self.anchor = torch.zeros(
            (self.max_bs, int(anchor_features)), dtype=dtype, device=device
        )
        self.anchor_valid = torch.zeros((self.max_bs,), dtype=torch.bool, device=device)
        self.token_table = head.build_token_table(embed_tokens)

        self.temperatures = torch.ones(
            (self.max_bs,), dtype=torch.float32, device=device
        )
        self.greedy_mask = torch.ones((self.max_bs,), dtype=torch.bool, device=device)
        self.uniforms = torch.zeros(
            (self.max_bs, self.slots), dtype=torch.float32, device=device
        )
        self.q_out = torch.empty(
            (self.max_bs, self.slots, self.topk), dtype=torch.float32, device=device
        )
        self.candidate_out = torch.empty(
            (self.max_bs, self.slots, self.topk), dtype=torch.int64, device=device
        )

    def set_anchor(self, rows: Optional[torch.Tensor], bs: int) -> None:
        # Rows past bs are zeroed, not left stale: the graph runs at the bucket size.
        # Filter/merge can hand a row a neighbour's anchor, costing acceptance only.
        count = int(bs)
        if rows is None or int(rows.shape[0]) != count or count > self.max_bs:
            count = 0
        if count:
            self.anchor[:count].copy_(rows)
            self.anchor_valid[:count].fill_(True)
        if count < self.max_bs:
            self.anchor[count:].zero_()
            self.anchor_valid[count:].fill_(False)

    def stage_sampling_params(self, *, bs: int, sampling_info) -> None:
        if not self.sampling_enabled:
            return
        if sampling_info is None:
            self.temperatures[:bs].fill_(1.0)
            self.greedy_mask[:bs].fill_(True)
            return
        torch.clamp(
            sampling_info.temperatures.view(-1)[:bs].to(torch.float32),
            min=1e-5,
            out=self.temperatures[:bs],
        )
        self.greedy_mask[:bs].copy_(
            resolve_greedy_mask(
                bs=bs, sampling_info=sampling_info, device=self.greedy_mask.device
            )
        )

    def __call__(self, hidden_states: torch.Tensor, input_ids=None) -> None:
        del input_ids
        bs = hidden_states.shape[0] // self.block_size
        rows = bs * self.slots

        if bs > self.max_bs:
            raise RuntimeError(
                f"LiLiCorrDraftSampler was built for max_bs={self.max_bs} but the draft "
                f"graph replayed at bs={bs}; its static buffers are undersized."
            )
        hidden_size = hidden_states.shape[-1]
        pass_hidden = hidden_states.view(bs, self.block_size, hidden_size)[:, 1:, :]
        log_probs, candidate_tokens = lilicorr_candidates(
            hidden_states=pass_hidden.reshape(rows, hidden_size),
            lm_head=self.lm_head,
            topk=self.topk,
        )
        candidate_tokens = candidate_tokens.view(bs, self.slots, self.topk)

        pre_projected = self.token_table is not None
        token_embeddings = (
            self.token_table[candidate_tokens]
            if pre_projected
            else self.embed_tokens(candidate_tokens).detach()
        )
        common = dict(
            token_embeddings=token_embeddings,
            candidate_tokens=candidate_tokens,
            candidate_log_probs=log_probs.view(bs, self.slots, self.topk),
            pass_hidden=pass_hidden,
            anchor_hidden=self.anchor[:bs],
            anchor_valid=self.anchor_valid[:bs],
            already_projected=pre_projected,
        )
        if self.sampling_enabled:
            self.uniforms[:bs].uniform_()
        selected, q_rows = self.head.select_with_proposal(
            uniforms=self.uniforms[:bs],
            temperatures=self.temperatures[:bs],
            greedy_mask=self.greedy_mask[:bs],
            **common,
        )
        if self.sampling_enabled:
            self.q_out[:bs].copy_(q_rows)
            self.candidate_out[:bs].copy_(candidate_tokens)
        self.out[:rows].copy_(selected.reshape(-1).to(torch.int64))


def draft_graph_batch_sizes() -> list[int]:
    return sorted(
        {int(bs) for bs in get_exec().graph.cuda_graph_config.decode.bs if bs > 0}
    )


def build_lilicorr_draft_sampler(
    *,
    head,
    draft_model,
    embed_tokens,
    lm_head,
    block_size: int,
    sampling_enabled: bool = False,
) -> Optional[LiLiCorrDraftSampler]:
    def eager(reason: str) -> None:
        logger.warning(
            "LiLiCorr head kept eager (reason=%s): a bring-up path, not a serving "
            "configuration. Expect a large throughput regression and numbers that are "
            "not comparable to any published row.",
            reason,
        )
        return None

    tp_group = get_parallel().tp_group
    if int(tp_group.world_size) != 1:
        return eager("tp>1")
    batch_sizes = draft_graph_batch_sizes()
    if not batch_sizes:
        return eager("no draft graph batch sizes to size the static buffers from")
    max_bs = batch_sizes[-1]

    reject_added_vocab(lm_head)
    parameter = head.slot_embedding
    head.materialize_inference_buffers(parameter.device, parameter.dtype)

    sampler = LiLiCorrDraftSampler(
        head=head,
        embed_tokens=embed_tokens,
        lm_head=lm_head,
        block_size=int(block_size),
        max_bs=int(max_bs),
        anchor_features=int(draft_model.fc.output_size),
        sampling_enabled=sampling_enabled,
    )
    logger.info(
        "LiLiCorr select folded into the draft cuda graph: max_bs=%d block_size=%d K=%d.",
        max_bs,
        int(block_size),
        sampler.topk,
    )
    return sampler
