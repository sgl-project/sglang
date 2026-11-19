"""Pre-copy of the next MoE layer's predicted experts.

A decode step pages a layer's missing experts in only after the layer has routed, so every page-in
sits on the critical path. With pre-copy, after a layer's gather a side stream applies the next MoE
layer's router to this layer's input (one GEMV), and ``paged_experts_precopy_plan`` gives the next
layer's best predicted experts that are in none of its slots (margin over the (top_k + 1)-th above
``MARGIN``, at most ``MAX_COPIES``) that layer's least recently used slots outside the prediction,
moving its device residency as its decide would; the gather copies them from the pinned store at a
narrow grid (``copy_blocks``), beside this layer's compute. The next layer joins the
side stream before it routes (``join``), so its decide sees the copied experts as resident. A right
prediction takes a page-in off the critical path; a wrong one costs its copy and its victim slot.

Installed for every pair of consecutive paged MoE layers whose second one routes by plain softmax
top-k (the routers it was measured on: Qwen3-Next, Qwen3-30B); other layers are skipped and logged.
Runs in full CUDA-graph decode captures and eager steps; not inside breakable (prefill) graphs, whose
segments could split the fork from its join.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

import torch

from sglang.srt.utils.device import (
    create_device_stream,
    device_stream_context,
    get_device_core_count,
    is_device_stream_capturing,
)

logger = logging.getLogger(__name__)

# Measured on an RTX 5070 Ti (70 SMs): softmax margin 0, at most 3 copies per layer. A copy at 32
# blocks (about half the SMs) overlaps a Marlin layer's compute at -1.6 % of its bandwidth; 256
# blocks slow the layer 1.7-2x.
MARGIN, MAX_COPIES, MAX_BLOCKS = 0.0, 3, 32


def copy_blocks(core_count: int) -> int:
    """The pre-copy's grid: half the device's SMs, at most ``MAX_BLOCKS``, so the copy beside a
    layer's compute takes no larger share of a smaller GPU than of the one it was measured on."""
    return max(1, min(MAX_BLOCKS, core_count // 2))


LOG_EVERY_S = 30.0
_LOG = {"next": 0.0}


class Precopy:
    """Pre-copy into one layer (the target), launched by the layer before it (``launch``). Every
    buffer is made here, outside any capture (a tensor made in a capture re-runs its init with
    every replay)."""

    def __init__(
        self,
        method,
        weight: torch.Tensor,
        top_k: int,
        stream,
        margin: float = MARGIN,
        max_copies: int = MAX_COPIES,
        blocks: Optional[int] = None,
    ):
        executor = method.device_executor
        r = executor.residency
        k, e = r.slot_expert.numel(), r.expert_slot.numel()
        device = r.step.device
        i32 = dict(dtype=torch.int32, device=device)
        self.method = method
        self.residency = r
        self.store = executor.store
        self.weight = weight  # the target's router [E, H]
        self.top_k = top_k
        self.stream = stream
        if blocks is None:
            blocks = copy_blocks(get_device_core_count(device.index or 0))
        self.margin, self.blocks = margin, blocks
        self.max_copies = min(max_copies, max(0, k - top_k))
        # The plan reads bf16 logits; a router in another 16-bit type (fp16 GPTQ models) writes its
        # GEMV into ``router_out`` and is converted.
        self.logits = torch.zeros((1, e), dtype=torch.bfloat16, device=device)
        self.router_out = (
            self.logits
            if weight.dtype == torch.bfloat16
            else torch.zeros((1, e), dtype=weight.dtype, device=device)
        )
        self.src = torch.zeros(k, **i32)
        self.dst = torch.zeros(k, **i32)
        self.count = torch.zeros(1, **i32)
        self.last = torch.full((k,), -1, **i32)
        self.last_stamp = torch.zeros(1, **i32)
        self.counts = torch.zeros(4, **i32)
        # Inputs a captured side stream reads: kept as long as the graphs (the caching allocator
        # tracks a block's uses on its own stream only).
        self.keep: List[torch.Tensor] = []
        self.side_sleep = (
            0  # tests: GPU cycles the side stream spins between plan and copy
        )

    def launch(self, hidden_states: torch.Tensor) -> None:
        """After the predicting layer's gather: predict, plan and copy on the side stream; the
        target's next ``join`` waits for it."""
        from sglang.kernels.ops.moe.paged_experts import paged_experts_precopy_plan

        r = self.residency
        dev = torch.get_device_module(self.weight.device)
        main = dev.current_stream()
        self.stream.wait_stream(main)
        with device_stream_context(self.stream):
            x = hidden_states[:1]
            if x.dtype != self.weight.dtype:
                x = x.to(self.weight.dtype)
            torch.mm(x, self.weight.t(), out=self.router_out)
            if self.router_out is not self.logits:
                self.logits.copy_(self.router_out)
            paged_experts_precopy_plan(
                self.logits[0],
                self.top_k,
                self.margin,
                self.max_copies,
                r.step,
                r.slot_expert,
                r.expert_slot,
                r.slot_lastuse,
                self.src,
                self.dst,
                self.count,
                self.last,
                self.last_stamp,
                self.counts,
            )
            if self.side_sleep:
                dev._sleep(self.side_sleep)
            self.store.gather(
                src=self.src, dst=self.dst, count=self.count, blocks=self.blocks
            )
            done = dev.Event()
            done.record(self.stream)
        if is_device_stream_capturing(self.weight.device):
            self.keep.append(hidden_states)
        else:
            hidden_states.record_stream(self.stream)
        r.precopy_join = done


def join(residency) -> None:
    """Before a layer routes: wait for the copies a pre-copy started into it."""
    done = getattr(residency, "precopy_join", None)
    if done is not None:
        residency.precopy_join = None
        torch.get_device_module(residency.step.device).current_stream().wait_event(done)


# The pre-copies of the installed model, for the counters.
precopies: List[Precopy] = []


def counts() -> dict:
    """Pre-copy counters summed over the layers (a host sync): planned, used, wasted (of the plans
    tallied so far: a plan is tallied at the next one into its layer), skipped (predicted experts
    below the margin or already in a slot)."""
    if not precopies:
        return {}
    total = torch.stack([p.counts for p in precopies]).sum(0).tolist()
    return dict(zip(("planned", "used", "wasted", "skipped"), total))


def maybe_log() -> None:
    """From an eager step (prefill), at most every ``LOG_EVERY_S`` seconds: log the counters
    (debug level)."""
    if not precopies or is_device_stream_capturing(precopies[0].weight.device):
        return
    now = time.monotonic()
    if now < _LOG["next"]:
        return
    _LOG["next"] = now + LOG_EVERY_S
    logger.debug("Paged experts: pre-copy %s", counts())


def _routing_ok(moe) -> Optional[str]:
    """Why the plan kernel cannot reproduce this layer's routing, or None."""
    tk = getattr(getattr(moe, "topk", None), "topk_config", None)
    if tk is None:
        return "no TopK config"
    if tk.scoring_func != "softmax":
        return f"scoring {tk.scoring_func!r}"
    if tk.use_grouped_topk or tk.custom_routing_function is not None:
        return "grouped or custom routing"
    if getattr(tk, "correction_bias", None) is not None:
        return "a correction bias"
    return None


def _copies_ahead(store) -> bool:
    """Whether pre-copy may call this store's ``gather`` beside compute: the host store's own
    copy (each planned expert from the pinned store into its slot, nothing else). A store that
    replaces ``gather``, e.g. one that also plans reads of its own, is skipped until it supports
    being copied into ahead."""
    from sglang.srt.layers.moe.paged_experts.store import HostExpertStore

    return type(store).gather is HostExpertStore.gather


def install(model) -> int:
    """Before the decode graphs are captured (the executors are final): link each paged MoE layer
    with a device executor to the next one whose routing the plan can predict; return how many
    pre-copies there are."""
    from sglang.srt.layers.moe.paged_experts.method import PagedExpertsMoEMethod

    layers = getattr(getattr(model, "model", None), "layers", None) or []
    moes = []
    for layer in layers:
        m = getattr(layer, "mlp", None)
        method = getattr(getattr(m, "experts", None), "quant_method", None)
        if (
            isinstance(method, PagedExpertsMoEMethod)
            and method.device_executor is not None
            and hasattr(getattr(m, "gate", None), "weight")
        ):
            moes.append((m, method))
    if moes and any(m.device_executor.precopy_next is not None for _, m in moes):
        return len(
            precopies
        )  # linked by an earlier capture (recapture keeps the links)
    del precopies[:]
    if len(moes) < 2:
        return 0
    stream = None
    skipped = {}
    for (_, method), (nxt, nxt_method) in zip(moes, moes[1:]):
        why = _routing_ok(nxt)
        if why is None and not _copies_ahead(nxt_method.device_executor.store):
            why = "a store that plans its own copies"
        weight = nxt.gate.weight
        experts = nxt_method.device_executor.residency.expert_slot.numel()
        # Each layer's own routing width (some models set it per layer).
        top_k = int(nxt.topk.topk_config.top_k or 0) if why is None else 0
        if why is None and weight.dtype not in (torch.bfloat16, torch.float16):
            why = f"a {weight.dtype} router"
        if why is None and weight.shape[0] != experts:
            why = "a router width other than the expert count (redundant experts)"
        if why is None and experts > 1024:
            why = f"{experts} experts (at most 1024)"
        if why is None and not 1 <= top_k <= min(32, experts - 1):
            why = f"top-{top_k} routing (1 to 32)"
        if why is None and nxt_method.num_resident <= top_k:
            why = "no slot beyond the routed ones"
        if why is not None:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        if stream is None:
            stream = create_device_stream(weight.device)
        pre = Precopy(nxt_method, weight.data, top_k, stream)
        method.device_executor.precopy_next = pre
        precopies.append(pre)
    if skipped:
        logger.info(
            "Paged experts: no pre-copy into %s",
            ", ".join(f"{n} layer(s) with {why}" for why, n in skipped.items()),
        )
    if not precopies:
        return 0
    # cuBLAS sets up its workspace for a stream on the first GEMM there: do it before any capture.
    with device_stream_context(stream):
        p = precopies[0]
        x = torch.zeros(
            (1, p.weight.shape[1]), dtype=p.weight.dtype, device=p.weight.device
        )
        torch.mm(x, p.weight.t(), out=p.router_out)
    stream.synchronize()
    logger.info(
        "Paged experts: pre-copy of predicted experts into %d of %d MoE layers",
        len(precopies),
        len(moes),
    )
    return len(precopies)
