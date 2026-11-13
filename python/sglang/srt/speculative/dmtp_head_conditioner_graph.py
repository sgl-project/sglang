"""One graph for sampled T1 support/q, TP embedding and T1-only residual."""

import logging

import torch
from torch.nn import functional as F

from sglang.srt.layers.layer_boundary import get_attn_tp_context
from sglang.srt.speculative.dmtp_selector_graph import (
    DMTPSelectorGraphs,
    _tensor_signature,
    native_proposal,
)

logger = logging.getLogger(__name__)


def _call_signature(method):
    return (
        id(getattr(method, "__self__", None)),
        id(getattr(method, "__func__", method)),
    )


def conditioned_embeddings(worker, embedding, condition, ids, hidden):
    if worker._full_embed_gpu is not None:
        blocks = F.embedding(ids, worker._full_embed_gpu)
    else:
        blocks = embedding(ids)
    if worker._noise_embed_scale != 1.0:
        blocks = blocks * worker._noise_embed_scale
    return condition(blocks, hidden)


class SampledHeadConditionerGraphs(DMTPSelectorGraphs):
    """Retain separate paths for fallback/reference, but submit one actual graph."""

    @torch.no_grad()
    def __init__(self, controller, *, enable_graph):
        self.worker = controller.worker
        target = self.worker.target_worker.model_runner.model
        self.embedding = target.get_input_embeddings()
        self.conditioner = self.worker.draft_model.mtp_prefix_conditioner
        self.condition = controller.conditioner_graphs.original
        maximum = self.worker.server_args.max_running_requests
        weight = target.lm_head.weight
        self.block_ids = torch.zeros(
            (maximum, 8), dtype=torch.int64, device=weight.device
        )
        self.pending_ids = None
        self.hits = self.fallbacks = 0
        super().__init__(controller, t1=True, enable_graph=enable_graph)
        logger.info(
            "DMTP_SAMPLED_COMBINED_CAPTURE batches=%s top_k=%d",
            sorted(self.graphs),
            self.top_k,
        )

    def signature(self):
        target = self.worker.target_worker.model_runner.model
        modules = (
            target.get_input_embeddings(),
            self.worker.draft_model.mtp_prefix_conditioner,
        )
        embedding, conditioner = modules
        shard = getattr(embedding, "shard_indices", None)
        full = self.worker._full_embed_gpu
        return (
            super().signature(),
            tuple((id(m), m.training) for m in modules),
            tuple(
                (name, _tensor_signature(p))
                for m in modules
                for name, p in m.named_parameters()
            ),
            id(self.controller.conditioner_graphs.original),
            tuple(_call_signature(m.forward) for m in modules),
            tuple(
                getattr(embedding, name, None)
                for name in (
                    "enable_tp",
                    "tp_size",
                    "use_attn_tp_group",
                    "output_dtype",
                )
            ),
            tuple(
                getattr(shard, name, None)
                for name in (
                    "num_org_elements",
                    "num_org_elements_padded",
                    "num_added_elements",
                    "org_vocab_start_index",
                    "added_vocab_start_index",
                )
            ),
            conditioner.eps,
            _tensor_signature(full) if full is not None else None,
            self.worker._noise_embed_scale,
            get_attn_tp_context().input_scattered,
        )

    def forward(self, bs):
        tokens, ids, q = native_proposal(
            self.model,
            self.hidden[:bs],
            self.anchors[:bs],
            self.temperatures[:bs],
            self.greedy_mask[:bs],
            self.coins[:bs],
            t1=True,
            top_k=self.top_k,
        )
        self.block_ids[:bs, 1].copy_(tokens[:, 0])
        embeds = conditioned_embeddings(
            self.worker,
            self.embedding,
            self.condition,
            self.block_ids[:bs],
            self.hidden[:bs, 0],
        )
        return tokens, ids, q, embeds

    def _stage_inputs(self, bs, hidden, anchors, temperatures, mask, coins):
        super()._stage_inputs(bs, hidden, anchors, temperatures, mask, coins)
        self.block_ids[:bs].copy_(self.pending_ids)

    def _reference(self, hidden, anchors, temperatures, mask, coins):
        # Same coins and original native proposal arithmetic. The original
        # conditioner graph retains its own independent eager first-B check.
        tokens, ids, q = super()._reference(hidden, anchors, temperatures, mask, coins)
        reference_ids = self.pending_ids.clone()
        reference_ids[:, 1].copy_(tokens[:, 0])
        embeds = conditioned_embeddings(
            self.worker,
            self.embedding,
            self.controller.conditioner_graphs.run,
            reference_ids,
            hidden[:, 0],
        )
        return tokens, ids, q, embeds

    @torch.no_grad()
    def run(self, hidden, block_ids, info):
        if info is None:
            raise ValueError("Combined T1 requires current SamplingBatchInfo")
        bs = block_ids.shape[0]
        valid = (
            bs in self.graphs
            and block_ids.shape == (bs, 8)
            and hidden.shape == (bs, self.hidden.shape[-1])
            and block_ids.device == self.block_ids.device
            and block_ids.dtype == self.block_ids.dtype
            and hidden.device == self.hidden.device
            and hidden.dtype == self.hidden.dtype
            and self.signature() == self.initial_signature
        )
        if not valid:
            self.last_used = False
            self.fallbacks += 1
            return None
        self.pending_ids = block_ids
        key = (bs, bool(info.is_all_greedy))
        first = key not in self.checked
        result = super().run(hidden[:, None], block_ids[:, 0], info)
        if first:
            # The reference above checked all four outputs, including sparse
            # support/q and the complete eight-row conditioned embedding.
            logger.info(
                "DMTP_SAMPLED_COMBINED_STRICT_PASS B=%d greedy=%s graph=%s "
                "support_q_exact=True embeds_exact=True",
                bs,
                key[1],
                self.last_used,
            )
        self.pending_ids = None
        self.hits += 1
        return result
