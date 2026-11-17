"""Operator tails recorded in the MTP and DF parent graphs, not child replays."""

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.speculative.dmtp_head_conditioner_graph import conditioned_embeddings
from sglang.srt.speculative.dmtp_selector_graph import (
    DMTPSelectorGraphs,
    _tensor_signature,
    native_proposal,
)
from sglang.srt.speculative.dspark_components.dspark_draft import resolve_greedy_mask


class DMTPGraphSamplingTail:
    def __init__(self, controller, *, t1, scored_start=1):
        self.controller = controller
        self.worker = controller.worker
        self.model = self.worker.draft_model
        self.model.lm_head = controller.head
        self.t1, self.scored_start = t1, scored_start
        self.top_k = getattr(self.model.candidate_selector, "top_k", 16)
        self.maximum = self.worker.server_args.max_running_requests
        weight = controller.head.weight
        self.temperatures = torch.ones(self.maximum, device=weight.device)
        self.greedy_mask = torch.ones(
            self.maximum, dtype=torch.bool, device=weight.device
        )
        self.coins = torch.zeros(self.maximum, 1 if t1 else 6, device=weight.device)
        self.block_ids = torch.full(
            (self.maximum, 8),
            int(self.worker._mask_token_id),
            dtype=torch.int64,
            device=weight.device,
        )
        self.outputs = {}
        self.captured_outputs = None
        self.embedding = (
            self.worker.target_worker.model_runner.model.get_input_embeddings()
        )
        self.conditioner = self.model.mtp_prefix_conditioner
        self.condition = self.conditioner.forward
        self.initial_signature = self.signature()

    def signature(self):
        embedding = self.worker.target_worker.model_runner.model.get_input_embeddings()
        conditioner = self.worker.draft_model.mtp_prefix_conditioner
        return (
            DMTPSelectorGraphs.signature(self),
            id(self.worker.draft_model),
            tuple(
                (
                    id(m),
                    m.training,
                    tuple((n, _tensor_signature(p)) for n, p in m.named_parameters()),
                )
                for m in (embedding, conditioner)
            ),
            getattr(conditioner, "eps", None),
            (
                _tensor_signature(self.worker._full_embed_gpu)
                if self.worker._full_embed_gpu is not None
                else None
            ),
            self.worker._noise_embed_scale,
            getattr(conditioner.forward, "__func__", conditioner.forward),
            getattr(embedding.forward, "__func__", embedding.forward),
            tuple(
                getattr(embedding, key, None)
                for key in ("enable_tp", "tp_size", "use_attn_tp_group", "output_dtype")
            ),
        )

    def stage(self, anchors, sampling_info, indices=None):
        if self.signature() != self.initial_signature:
            raise RuntimeError(
                "DMTP graph-tail weights/layout changed; restart required"
            )
        bs = anchors.numel()
        if not 1 <= bs <= self.maximum or sampling_info is None:
            raise ValueError("DMTP graph-tail sampling batch is invalid")
        temperatures = sampling_info.temperatures.reshape(-1).float().clamp_min(1e-5)
        mask = resolve_greedy_mask(
            bs=temperatures.numel(), sampling_info=sampling_info, device=anchors.device
        )
        if indices is not None:
            temperatures = temperatures.index_select(0, indices)
            mask = mask.index_select(0, indices)
        if temperatures.numel() != bs:
            raise ValueError("DMTP graph-tail request/sampling alignment mismatch")
        self.temperatures[:bs].copy_(temperatures)
        self.greedy_mask[:bs].copy_(mask)
        self.block_ids[:bs].fill_(int(self.worker._mask_token_id))
        self.block_ids[:bs, 0].copy_(anchors)
        coins = torch.rand_like(self.coins[:bs])
        get_tp_group().broadcast(coins, src=0)
        self.coins[:bs].copy_(coins)

    def stage_sampling_params(self, *, bs, sampling_info):
        # DF anchors come from the live block ids inside the parent graph.
        self.stage(self.block_ids[:bs, 0].clone(), sampling_info)

    def forward(self, hidden, input_ids=None):
        if self.t1:
            bs = hidden.shape[0]
            scored, anchors = hidden[:, None], self.block_ids[:bs, 0]
        else:
            bs = hidden.shape[0] // 8
            scored = hidden.view(bs, 8, -1)[
                :, self.scored_start : self.scored_start + 6
            ]
            anchors = input_ids.view(bs, 8)[:, 1]
        tokens, ids, q = native_proposal(
            self.model,
            scored,
            anchors,
            self.temperatures[:bs],
            self.greedy_mask[:bs],
            self.coins[:bs],
            t1=self.t1,
            top_k=self.top_k,
        )
        if self.t1:
            self.block_ids[:bs, 1].copy_(tokens[:, 0])
            embeds = conditioned_embeddings(
                self.worker, self.embedding, self.condition, self.block_ids[:bs], hidden
            )
            result = (tokens, ids, q, embeds)
        else:
            result = (tokens, ids, q)
        self.outputs[bs] = result
        return result

    __call__ = forward

    def read(self, bs):
        outputs = (
            self.outputs if self.captured_outputs is None else self.captured_outputs
        )
        return tuple(value.clone() for value in outputs[bs])

    def freeze_outputs(self):
        self.captured_outputs = dict(self.outputs)

    def check(self, actual, expected):
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
        q = actual[2]
        torch._assert_async(
            (
                torch.isfinite(q).all()
                & (q >= 0).all()
                & ((q.sum(-1) - 1).abs() < 1e-5).all()
            ),
            "Invalid DMTP graph-tail sparse q",
        )
