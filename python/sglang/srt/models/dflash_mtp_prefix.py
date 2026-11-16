"""Double-anchor T1 residual only, with optional native DFlash2 conv/selector."""

import copy

import torch

from sglang.srt.models.dflash import (
    CandidateSelector,
    DFlash2DraftModel,
    DFlashDraftModel,
)
from sglang.srt.speculative.dflash2_contract import checked_weights, validate_dflash2
from sglang.srt.speculative.mtp_prefix_contract import (
    MTPPrefixConditioner,
    normalize_residual_config,
    validate_config,
)


class DFlashMTPPrefixDraftModel(DFlashDraftModel):
    def __init__(self, config, quant_config=None, prefix=""):
        config = normalize_residual_config(config)
        validate_config(config)
        if quant_config is not None:
            raise ValueError("DMTP T1 residual supports unquantized weights only")
        selector_rank = config.dflash_config.get("selector_rank", 0)
        if selector_rank:
            validate_dflash2(config)
        runtime = copy.deepcopy(config)
        runtime.layer_types = ["full_attention"] * config.num_hidden_layers
        runtime.sliding_window = None
        runtime.use_sliding_window = False
        super().__init__(runtime, quant_config=quant_config, prefix=prefix)
        self.config = config
        cfg = config.dflash_config
        self.mtp_prefix_conditioner = MTPPrefixConditioner(
            config.hidden_size, cfg["mtp_prefix_rank"], cfg["mtp_prefix_eps"]
        )
        if selector_rank:
            self.candidate_selector = CandidateSelector(
                hidden_size=config.hidden_size,
                vocab_size=config.vocab_size,
                state_rank=self.draft_config.selector_rank,
                top_k=self.draft_config.selector_top_k,
            )
        self.lm_head = None

    _transform_unary_logits = DFlash2DraftModel._transform_unary_logits
    compute_candidates = DFlash2DraftModel.compute_candidates

    def get_attention_sliding_window_size(self):
        return None

    def load_weights(self, weights):
        super().load_weights(checked_weights(self, weights))

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        get_embedding=False,
        pp_proxy_tensors=None,
    ):
        if getattr(forward_batch, "mtp_prefix_preconditioned", False) and getattr(
            forward_batch, "mtp_prefix_scheduler_ready", False
        ):
            return super().forward(
                input_ids,
                positions,
                forward_batch,
                input_embeds,
                get_embedding,
                pp_proxy_tensors,
            )
        payload = forward_batch
        if not getattr(payload, "mtp_prefix_scheduler_ready", False):
            payload = getattr(forward_batch, "spec_info", None)
        if not getattr(payload, "mtp_prefix_scheduler_ready", False):
            raise RuntimeError("DMTP T1 residual requires the MTP-first scheduler")
        hidden = getattr(payload, "mtp_t1_hidden", None)
        t1 = getattr(payload, "mtp_t1_ids", None)
        if input_embeds is None or hidden is None or t1 is None:
            raise ValueError("Missing actual T1 embeddings, IDs or MTP hidden")
        blocks = input_embeds.reshape(-1, 8, self.config.hidden_size)
        tokens = input_ids.reshape(-1, 8)
        if t1.shape != tokens[:, 1].shape or not torch.equal(tokens[:, 1], t1):
            raise ValueError("Slot 1 must be the actual MTP T1")
        conditioned = self.mtp_prefix_conditioner(blocks, hidden).reshape_as(
            input_embeds
        )
        return super().forward(
            input_ids,
            positions,
            forward_batch,
            conditioned,
            get_embedding,
            pp_proxy_tensors,
        )


class DFlashMTPAblationDraftModel(DFlashMTPPrefixDraftModel):
    """HF architecture alias retained for the existing T1-residual export only."""


class DFlash2T1ResidualDraftModel(DFlashMTPPrefixDraftModel):
    """HF architecture name for a matching DFlash2 T1-residual export."""


EntryClass = [
    DFlashMTPPrefixDraftModel,
    DFlashMTPAblationDraftModel,
    DFlash2T1ResidualDraftModel,
]
