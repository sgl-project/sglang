"""DMTP top-k T1 and six-tail proposals using the upstream DFlash2 selector."""

import logging
from types import SimpleNamespace

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.models.dflash import CandidateSelector, candidate_topk
from sglang.srt.speculative.dspark_components.dspark_draft import resolve_greedy_mask

logger = logging.getLogger(__name__)


def native_proposal(
    model, hidden, anchors, temperatures, greedy_mask, coins, *, t1, top_k=16
):
    bs, positions, width = hidden.shape
    selector = model.candidate_selector
    top_k = selector.top_k if selector is not None else top_k
    ids, unary, _ = candidate_topk(hidden.reshape(-1, width), model.lm_head, top_k)
    ids = ids.view(bs, positions, -1)
    if t1 or selector is None:
        # MTP T1 has no learned predecessor edge. Do not apply the DF selector
        # projection or DF unary transforms to the Target's MTP head logits.
        if not t1:
            unary = model._transform_unary_logits(unary)
        scores = (
            unary.float()
            .view(bs, positions, 1, -1)
            .expand(bs, positions, top_k, top_k)
            .contiguous()
        )
    else:
        scores = selector.build_lattice(
            candidate_ids=ids,
            unary_logits=model._transform_unary_logits(unary).view(bs, positions, -1),
            hidden_states=hidden,
            anchor_token_ids=anchors,
        )
    tokens, q = CandidateSelector.sample_path(
        selector if selector is not None else SimpleNamespace(top_k=top_k),
        candidate_ids=ids,
        scores=scores,
        uniforms=coins,
        temperatures=temperatures,
        greedy_mask=greedy_mask,
    )
    return tokens, ids, q


def _tensor_signature(value):
    return (
        id(value),
        value.data_ptr(),
        value._version,
        tuple(value.shape),
        tuple(value.stride()),
        value.dtype,
        value.device,
    )


class DMTPSelectorGraphs:
    """Serial submissions with exact-B capture and protected per-round sparse q."""

    @torch.no_grad()
    def __init__(self, controller, *, t1, enable_graph):
        self.controller = controller
        self.model = controller.worker.draft_model
        self.t1 = t1
        self.positions = 1 if t1 else 6
        head = controller.head
        self.model.lm_head = head
        k = getattr(self.model.candidate_selector, "top_k", 16)
        self.top_k = k
        if k < 1 or k & (k - 1):
            raise ValueError("DMTP native selector requires power-of-two top-k")
        maximum = controller.worker.server_args.max_running_requests
        device, dtype = head.weight.device, head.weight.dtype
        self.hidden = torch.zeros(
            (maximum, self.positions, head.weight.shape[1]), device=device, dtype=dtype
        )
        self.anchors = torch.zeros(maximum, device=device, dtype=torch.int64)
        self.temperatures = torch.ones(maximum, device=device, dtype=torch.float32)
        self.greedy_mask = torch.ones(maximum, device=device, dtype=torch.bool)
        self.coins = torch.zeros(
            (maximum, self.positions), device=device, dtype=torch.float32
        )
        self.graphs, self.outputs, self.checked = {}, {}, set()
        self.last_used = False
        self.initial_signature = self.signature()
        self.done = torch.cuda.Event()
        self.done.record()
        if not enable_graph:
            return
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        pool = torch.cuda.graph_pool_handle()
        with get_tp_group().graph_capture(stream=stream):
            for bs in range(maximum, 0, -1):
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self.forward(bs)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, stream=stream):
                    result = self.forward(bs)
                self.graphs[bs], self.outputs[bs] = graph, result
        torch.cuda.current_stream(device).wait_stream(stream)
        self.done.record(stream)

    def signature(self):
        head = self.controller.worker.target_worker.model_runner.model.lm_head
        selector = self.model.candidate_selector
        shard = head.shard_indices
        return (
            id(head),
            head.training,
            _tensor_signature(head.weight),
            id(selector),
            getattr(selector, "training", False),
            self.model.training,
            getattr(selector, "top_k", self.top_k),
            (
                tuple(
                    (name, _tensor_signature(p))
                    for name, p in selector.named_parameters()
                )
                if selector is not None
                else ()
            ),
            id(get_tp_group()),
            get_tp_group().world_size,
            tuple(
                getattr(shard, name)
                for name in (
                    "num_org_elements",
                    "num_org_elements_padded",
                    "num_added_elements",
                    "org_vocab_start_index",
                    "added_vocab_start_index",
                )
            ),
            self.model.draft_config.output_multiplier,
            self.model.draft_config.final_logit_softcapping,
        )

    def forward(self, bs):
        return native_proposal(
            self.model,
            self.hidden[:bs],
            self.anchors[:bs],
            self.temperatures[:bs],
            self.greedy_mask[:bs],
            self.coins[:bs],
            t1=self.t1,
            top_k=self.top_k,
        )

    def _stage_inputs(self, bs, hidden, anchors, temperatures, mask, coins):
        self.hidden[:bs].copy_(hidden)
        self.anchors[:bs].copy_(anchors)
        self.temperatures[:bs].copy_(temperatures)
        self.greedy_mask[:bs].copy_(mask)
        self.coins[:bs].copy_(coins)

    def _reference(self, hidden, anchors, temperatures, mask, coins):
        return native_proposal(
            self.model,
            hidden,
            anchors,
            temperatures,
            mask,
            coins,
            t1=self.t1,
            top_k=self.top_k,
        )

    @torch.no_grad()
    def run(self, hidden, anchors, info):
        bs = hidden.shape[0]
        if hidden.shape != (
            bs,
            self.positions,
            self.hidden.shape[-1],
        ) or anchors.shape != (bs,):
            raise ValueError("DMTP selector proposal geometry mismatch")
        if info is None:
            raise ValueError("DMTP selector requires current SamplingBatchInfo")
        self.model.lm_head = (
            self.controller.worker.target_worker.model_runner.model.lm_head
        )
        temperatures = info.temperatures.reshape(-1)[:bs].float().clamp_min(1e-5)
        mask = resolve_greedy_mask(bs=bs, sampling_info=info, device=hidden.device)
        # Broadcast a fresh draw before replay: both TP ranks and the first-use
        # eager reference must consume exactly the same random numbers.
        coins = torch.rand((bs, self.positions), device=hidden.device)
        get_tp_group().broadcast(coins, src=0)
        self.last_used = (
            bs in self.graphs
            and self.signature() == self.initial_signature
            and hidden.dtype == self.hidden.dtype
            and hidden.device == self.hidden.device
            and anchors.dtype == self.anchors.dtype
            and anchors.device == self.anchors.device
        )
        if self.last_used:
            stream = torch.cuda.current_stream(hidden.device)
            stream.wait_event(self.done)
            self._stage_inputs(bs, hidden, anchors, temperatures, mask, coins)
            self.graphs[bs].replay()
            result = tuple(value.clone() for value in self.outputs[bs])
            self.done.record(stream)
        else:
            result = self._reference(hidden, anchors, temperatures, mask, coins)
        key = (bs, bool(info.is_all_greedy))
        if key not in self.checked:
            reference = self._reference(hidden, anchors, temperatures, mask, coins)
            for actual, expected in zip(result, reference):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            q = result[2]
            torch._assert_async(
                (
                    torch.isfinite(q).all()
                    & (q >= 0).all()
                    & ((q.sum(-1) - 1).abs() < 1e-5).all()
                ),
                "Invalid DMTP sparse q",
            )
            self.checked.add(key)
            logger.info(
                "DMTP_NATIVE_SELECTOR_FIRST_USE t1=%s B=%d greedy=%s graph=%s",
                self.t1,
                bs,
                key[1],
                self.last_used,
            )
        return result
