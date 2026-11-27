"""Force recomputed positions to the experts they first ran with.

A retracted request re-prefills the tokens it had already generated. With the
rows kept by routed_experts_retract.py, the gates of that re-prefill take the
kept experts instead of their own top-k, so the KV rebuilt for those tokens is
the KV their successors were sampled against.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Iterator, Optional, Sequence

import msgspec
import torch
import torch.nn.functional as F

from sglang.srt.runtime_context import get_forward, get_resources

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import TopKConfig
    from sglang.srt.state_capturer.routed_experts import RoutedExpertsCapturer


class ForcingBatch(msgspec.Struct, frozen=True, kw_only=True, eq=False):
    """An extend forward's forced rows: mask[t] forces token row t.

    routes is [tokens, layers, top_k] logical expert ids indexed by layer id,
    zero where the mask is off.
    """

    mask: torch.Tensor
    routes: torch.Tensor

    def padded_to(self, num_tokens: int) -> ForcingBatch:
        """The batch with unforced rows appended (a mixed batch's decode rows)."""
        extra = num_tokens - self.mask.shape[0]
        if extra < 0:
            raise ValueError(f"cannot pad {self.mask.shape[0]} rows to {num_tokens}")
        if extra == 0:
            return self
        pad = self.routes.new_zeros((extra, *self.routes.shape[1:]))
        return ForcingBatch(
            mask=torch.cat([self.mask, self.mask.new_zeros(extra)]),
            routes=torch.cat([self.routes, pad]),
        )

    def to_device(self, device) -> ForcingBatch:
        return ForcingBatch(
            mask=self.mask.to(device, non_blocking=True),
            routes=self.routes.to(device, non_blocking=True),
        )


def build_forcing_batch(
    *,
    kept_rows: Sequence[Optional[torch.Tensor]],
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
    pin_memory: bool = False,
) -> Optional[ForcingBatch]:
    """The forced rows of an extend forward, or None if it recomputes no kept row.

    Request i computes positions from prefix_lens[i] at forward rows from
    sum(extend_lens[:i]); each one below its kept rows' end is forced.
    """
    num_tokens = sum(extend_lens)
    mask = routes = None
    offset = 0
    for rows, prefix_len, extend_len in zip(
        kept_rows, prefix_lens, extend_lens, strict=True
    ):
        stop = min(prefix_len + extend_len, 0 if rows is None else rows.shape[0])
        if stop > prefix_len:
            if mask is None:
                mask = torch.zeros(num_tokens, dtype=torch.bool, pin_memory=pin_memory)
                routes = torch.zeros(
                    (num_tokens, *rows.shape[1:]),
                    dtype=torch.int32,
                    pin_memory=pin_memory,
                )
            n = stop - prefix_len
            mask[offset : offset + n] = True
            routes[offset : offset + n] = rows[prefix_len:stop]
        offset += extend_len
    return None if mask is None else ForcingBatch(mask=mask, routes=routes)


class RoutedExpertsForcing(msgspec.Struct, frozen=True, kw_only=True):
    """The layers whose gates every forced forward must reach."""

    gate_layer_ids: frozenset[int]

    @staticmethod
    def create(
        *, model: torch.nn.Module, capturer: RoutedExpertsCapturer
    ) -> RoutedExpertsForcing:
        """Raises ValueError if the loaded model's gates cannot be forced."""
        # Imported here so that importing this module does not load the MoE layers.
        from sglang.srt.layers.moe.topk import TopK

        gates = [
            m
            for m in model.modules()
            if isinstance(m, TopK) and m.topk_config.allow_routed_experts_capture
        ]
        problems = _gate_problems(
            [g.topk_config for g in gates], top_k=capturer.topk_size
        )
        if not capturer.has_host_cache:
            problems.append("ranks without the routed-experts host cache")
        if problems:
            raise ValueError(
                "SGLANG_ENABLE_ROUTED_EXPERTS_RETRACT_FORCING does not support "
                + "; ".join(problems)
            )
        return RoutedExpertsForcing(gate_layer_ids=frozenset(g.layer_id for g in gates))


@contextlib.contextmanager
def forcing_forward(batch: ForcingBatch, *, device) -> Iterator[None]:
    """Expose one extend forward's forced rows to every MoE gate it runs."""
    forced = _ForcedForward(
        batch=batch.to_device(device),
        gate_layer_ids=get_resources().routed_experts_forcing.gate_layer_ids,
    )
    with get_forward().scoped(routed_experts_forced=forced):
        yield
    forced.check_every_layer_forced()


class _ForcedForward:
    """One extend forward's forced rows on the device, and the gates that took them."""

    def __init__(self, *, batch: ForcingBatch, gate_layer_ids: frozenset[int]):
        self._batch = batch
        self._gate_layer_ids = gate_layer_ids
        self._forced_layers: set[int] = set()

    def force_routing(
        self,
        *,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        router_logits: torch.Tensor,
        layer_id: Optional[int],
        topk_config: TopKConfig,
        packed: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not topk_config.allow_routed_experts_capture:
            return topk_ids, topk_weights
        if layer_id not in self._gate_layer_ids or packed:
            raise RuntimeError(
                f"routed-experts forcing cannot force the gate of layer {layer_id}: "
                "it is not a captured TopK layer, or it packs its top-k"
            )
        self._forced_layers.add(layer_id)
        return apply_forced_routing(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            router_logits=router_logits,
            forced_ids=self._batch.routes[:, layer_id],
            mask=self._batch.mask,
            scoring_func=topk_config.scoring_func,
            renormalize=topk_config.renormalize,
            routed_scaling_factor=topk_config.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=topk_config.apply_routed_scaling_factor_on_output,
        )

    def check_every_layer_forced(self) -> None:
        # A gate that routes outside select_experts would recompute kept rows freely.
        missing = self._gate_layer_ids - self._forced_layers
        if missing:
            raise RuntimeError(
                "routed-experts forcing never reached the gates of MoE layers "
                f"{sorted(missing)}"
            )


def _gate_problems(configs: Sequence[TopKConfig], *, top_k: int) -> list[str]:
    problems = set()
    for config in configs:
        num_routed = config.top_k - config.num_shared_topk_slots
        if num_routed != top_k:
            problems.add(
                f"gates whose routed top-k ({num_routed}) differs from the "
                f"captured top-k ({top_k})"
            )
        elif config.sqrtsoftplus_log1p:
            problems.add("sqrtsoftplus gates with log1p")
        elif problem := gate_mirror_problem(
            scoring_func=config.scoring_func,
            has_correction_bias=config.correction_bias is not None,
            has_custom_routing=config.custom_routing_function is not None,
            renormalize=config.renormalize,
            apply_routed_scaling_factor_on_output=config.apply_routed_scaling_factor_on_output,
            num_shared_topk_slots=config.num_shared_topk_slots,
        ):
            problems.add(problem)
    if not configs:
        problems.add("models without TopK MoE gates")
    return sorted(problems)


def gate_mirror_problem(
    *,
    scoring_func: str,
    has_correction_bias: bool,
    has_custom_routing: bool,
    renormalize: bool,
    apply_routed_scaling_factor_on_output: bool,
    num_shared_topk_slots: int,
) -> Optional[str]:
    """Why _forced_routing_weights cannot reproduce this gate's weights, or None."""
    if has_custom_routing:
        return "custom routing functions"
    if scoring_func == "softmax":
        # The fused gate weighs with the bias; flashinfer_trtllm_routed's raw-logit
        # gate never scales, and weighs raw logits unless it renormalizes.
        if has_correction_bias or apply_routed_scaling_factor_on_output:
            return "softmax gates with a correction bias or routed scaling on output"
        if not renormalize:
            return "softmax gates without renormalize"
    elif scoring_func not in ("sigmoid", "sqrtsoftplus"):
        return f"scoring_func={scoring_func!r}"
    if not renormalize and (
        apply_routed_scaling_factor_on_output or num_shared_topk_slots
    ):
        # Unrenormalized, only the fused gate scales, and a shared slot's
        # weight follows the routed experts.
        return "routed scaling or shared top-k slots without renormalize"
    return None


def _forced_routing_weights(
    router_logits: torch.Tensor,
    forced_ids: torch.Tensor,
    *,
    scoring_func: str,
    renormalize: bool,
    routed_scaling_factor: Optional[float],
    apply_routed_scaling_factor_on_output: bool,
) -> torch.Tensor:
    """The gate's fp32 weights for forced_ids.

    A correction bias affects only which experts are selected, not their weights.
    """
    logits = router_logits.float()
    if scoring_func == "softmax":
        scores = torch.softmax(logits, dim=-1)
    elif scoring_func == "sigmoid":
        scores = torch.sigmoid(logits)
    elif scoring_func == "sqrtsoftplus":
        scores = F.softplus(logits).sqrt()
    else:
        raise ValueError(f"forcing does not mirror scoring_func={scoring_func!r}")
    weights = scores.gather(1, forced_ids.long())
    if renormalize:
        total = weights.sum(dim=-1, keepdim=True)
        weights = weights / torch.where(total > 0, total, torch.ones_like(total))
        if apply_routed_scaling_factor_on_output and routed_scaling_factor is not None:
            weights = weights * routed_scaling_factor
    return weights


def apply_forced_routing(
    *,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    router_logits: torch.Tensor,
    forced_ids: torch.Tensor,
    mask: torch.Tensor,
    scoring_func: str,
    renormalize: bool,
    routed_scaling_factor: Optional[float],
    apply_routed_scaling_factor_on_output: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route masked rows to forced_ids.

    A row whose forced set equals the gate's keeps the gate's ids and weights;
    shared top-k columns are untouched.
    """
    if topk_ids.shape[0] != mask.shape[0]:
        raise ValueError(f"{topk_ids.shape[0]} rows but {mask.shape[0]} forcing rows")
    num_routed = forced_ids.shape[1]
    routed_ids, routed_weights = topk_ids[:, :num_routed], topk_weights[:, :num_routed]
    forced = forced_ids.to(routed_ids.dtype)
    same_set = (forced.sort(dim=-1).values == routed_ids.sort(dim=-1).values).all(-1)
    replace = (mask & ~same_set)[:, None]
    # Unmasked rows carry zeros, a valid id, so the gather below stays in range.
    weights = _forced_routing_weights(
        router_logits,
        forced,
        scoring_func=scoring_func,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    ).to(routed_weights.dtype)
    ids = torch.where(replace, forced, routed_ids)
    weights = torch.where(replace, weights, routed_weights)
    return (
        torch.cat([ids, topk_ids[:, num_routed:]], dim=1),
        torch.cat([weights, topk_weights[:, num_routed:]], dim=1),
    )
