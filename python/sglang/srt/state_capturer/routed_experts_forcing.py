"""Routed-experts forcing.

For a request with ``routed_experts_session_key``, every token below
``routed_experts_start_len`` that the engine recomputes reuses the experts the
engine first computed for it; a request whose routes the engine no longer holds
is refused before it runs. This keeps R3 (rollout routing replay) exact when
a session's tokens are recomputed: the trainer replays the experts that sampled
each token. Held routes live in routed_experts_store.py.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Iterator, Optional, Sequence

import msgspec
import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.runtime_context import (
    get_exec,
    get_forward,
    get_parallel,
    get_resources,
)
from sglang.srt.state_capturer.routed_experts_store import (
    Admission,
    ForcingBatch,
    HeldRoutes,
    HeldRoutesStore,
    RouteLayout,
    RouteSegment,
    build_forcing_batch,
    unforced_recompute_tokens,
)
from sglang.srt.utils.common import is_pin_memory_available

if TYPE_CHECKING:
    from sglang.srt.eplb.expert_location import ExpertLocationMetadata
    from sglang.srt.layers.moe.topk import TopKConfig
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.state_capturer.routed_experts import RoutedExpertsCapturer

logger = logging.getLogger(__name__)

_CAPABILITY_VERSION = 1
# Refusal reason codes: a refusal's message is "<code>: <detail>", and
# meta_info["routed_experts_forcing"]["refused"] carries the code.
REFUSED_DISABLED = "routed_experts_forcing_disabled"
REFUSED_NO_RETURN_ROUTED_EXPERTS = (
    "routed_experts_forcing_requires_return_routed_experts"
)
REFUSED_ROUTES_MISSING = "routed_experts_missing"


class ForcingState(msgspec.Struct, kw_only=True):
    """A request's held rows and forcing counters, kept across retracts.

    ``admission`` is None for a request refused before the store was consulted.
    """

    key: str
    force_len: int
    held: HeldRoutes
    admission: Optional[Admission] = None
    forced_tokens: int = 0
    unforced_tokens: int = 0
    retract_snapshots: int = 0
    refused: Optional[str] = None

    def meta_info(self) -> dict:
        info = {
            "held_tokens": 0 if self.admission is None else self.admission.rows.end,
            "forced_tokens": self.forced_tokens,
            "unforced_tokens": self.unforced_tokens,
            "retract_snapshots": self.retract_snapshots,
        }
        if self.refused is not None:
            info["refused"] = self.refused
        return info


class RoutedExpertsForcing:
    """The process's held-routes store and the scheduler steps that fill and read it.

    Every TP rank runs the same admits, snapshots and commits in the same order.
    """

    def __init__(
        self,
        *,
        layout: RouteLayout,
        max_tokens: int,
        host_table: torch.Tensor,
        pin_memory: bool,
    ):
        self.layout = layout
        self.store = HeldRoutesStore(max_tokens)
        self.layer_ordinals = {
            layer_id: i for i, layer_id in enumerate(layout.moe_layer_ids)
        }
        self.refused_requests = 0
        self.dropped_commits = 0
        self._host_table = host_table
        self._pin_memory = pin_memory

    @staticmethod
    def create(
        *,
        model: torch.nn.Module,
        capturer: RoutedExpertsCapturer,
        expert_location: Optional[ExpertLocationMetadata],
    ) -> RoutedExpertsForcing:
        """Build forcing for the loaded model.

        Raises ValueError if the model cannot be forced.
        """
        # Imported here so that importing this module (schedule_batch does)
        # does not load the MoE layers.
        from sglang.srt.layers.moe.topk import TopK

        gates = [
            m
            for m in model.modules()
            if isinstance(m, TopK) and m.topk_config.allow_routed_experts_capture
        ]
        problems = _gate_problems(
            [g.topk_config for g in gates], top_k=capturer.topk_size
        )
        if expert_location is None:
            problems.append("models without expert location metadata")
        if problems:
            raise ValueError(
                "--enable-routed-experts-forcing does not support "
                + "; ".join(problems)
            )
        layout = RouteLayout(
            moe_layer_ids=tuple(sorted({g.layer_id for g in gates})),
            top_k=capturer.topk_size,
            num_logical_experts=expert_location.num_logical_experts,
        )
        return RoutedExpertsForcing(
            layout=layout,
            max_tokens=get_exec().features.routed_experts_forcing_max_tokens,
            host_table=capturer.host_cache.buffer,
            pin_memory=is_pin_memory_available(),
        )

    def capability(self) -> dict:
        return {
            "version": _CAPABILITY_VERSION,
            "max_tokens": self.store.max_tokens,
        }

    def admit(self, req: Req, *, key: str) -> Optional[str]:
        """Attach the request's held rows, or refuse it with the returned message."""
        force_len = req.routed_experts_start_len
        if not req.return_routed_experts:
            req.routed_experts_forcing_state = ForcingState(
                key=key, force_len=force_len, held=HeldRoutes()
            )
            return self._refuse(
                req, REFUSED_NO_RETURN_ROUTED_EXPERTS, "set return_routed_experts"
            )
        admission = self.store.admit(
            key, token_ids=req.origin_input_ids, force_len=force_len
        )
        req.routed_experts_forcing_state = ForcingState(
            key=key, force_len=force_len, held=admission.rows, admission=admission
        )
        if admission.rows.end < force_len:
            return self._refuse(
                req,
                REFUSED_ROUTES_MISSING,
                f"held={admission.rows.end} requested={force_len}",
            )
        # A radix hit on another session's KV would attend KV routed for that session.
        namespace = f"routed_experts_session:{key}"
        req.extra_key = f"{req.extra_key}|{namespace}" if req.extra_key else namespace
        return None

    def build_batch(
        self,
        *,
        reqs: Sequence[Req],
        prefix_lens: Sequence[int],
        extend_lens: Sequence[int],
    ) -> Optional[ForcingBatch]:
        """An extend batch's forced rows; counts each request's recomputed rows."""
        held = []
        for req, prefix_len, extend_len in zip(
            reqs, prefix_lens, extend_lens, strict=True
        ):
            state = req.routed_experts_forcing_state
            if state is None:
                held.append(HeldRoutes())
                continue
            held.append(state.held)
            end = state.held.end
            state.forced_tokens += max(
                0, min(prefix_len + extend_len, end) - prefix_len
            )
            state.unforced_tokens += unforced_recompute_tokens(
                prefix_len=prefix_len,
                extend_len=extend_len,
                held_end=end,
                force_len=state.force_len,
            )
        return build_forcing_batch(
            held=held,
            prefix_lens=prefix_lens,
            extend_lens=extend_lens,
            layout=self.layout,
            pin_memory=self._pin_memory,
        )

    def snapshot_on_retract(self, req: Req, *, req_to_token: torch.Tensor) -> None:
        """Hold the rows a retracted request computed past its held rows.

        Runs before the request releases its KV slots, which index the capture table.
        """
        state = req.routed_experts_forcing_state
        if state is None or not req.kv.holds_kv or not req.output_ids:
            return
        # Skip the last kept token's row: under overlap scheduling its forward may
        # still be in flight, with its capture not written yet.
        end = min(
            req.kv.kv_committed_len, len(req.origin_input_ids) + len(req.output_ids) - 1
        )
        start = state.held.end
        if end <= start:
            return
        slots = req_to_token[req.kv.req_pool_idx, start:end].to("cpu", torch.int64)
        rows = self.layout.narrow(self._host_table[slots].numpy())
        state.held = state.held.extended(RouteSegment(start=start, routes=rows))
        state.retract_snapshots += 1

    def commit_on_finish(self, req: Req) -> None:
        """Commit the session's rows: held rows below the force length, then this
        request's collected routed experts.

        A commit that cannot apply drops the key's rows instead of raising in the
        result loop, so the session's next turn is refused.
        """
        state = req.routed_experts_forcing_state
        if state is None:
            return
        tokens = np.concatenate(
            [
                np.asarray(req.origin_input_ids, dtype=np.int64),
                np.asarray(req.output_ids_through_stop, dtype=np.int64),
            ]
        )
        if len(tokens) - 1 < state.force_len:
            return
        try:
            if req.routed_experts is None:
                raise ValueError("no routed experts were collected")
            own = RouteSegment(
                start=state.force_len,
                routes=self.layout.narrow(req.routed_experts.numpy()),
            )
            rows = state.held.truncated(state.force_len).extended(own)
            if self.store.commit(
                state.key, token_ids=tokens, rows=rows, admission=state.admission
            ):
                return
            reason = "the key's rows changed after it was admitted"
        except ValueError as error:
            self.store.invalidate(state.key)
            reason = str(error)
        self.dropped_commits += 1
        if get_parallel().tp_rank == 0:
            logger.warning(
                "Routed-experts forcing dropped the rows of key %s at the commit of "
                "request %s (%s); the key's next turn is refused (%d dropped).",
                state.key,
                req.rid,
                reason,
                self.dropped_commits,
            )

    def _refuse(self, req: Req, code: str, detail: str) -> str:
        """Record the refusal's reason code on the request; returns its message."""
        req.routed_experts_forcing_state.refused = code
        message = f"{code}: {detail}"
        self.refused_requests += 1
        if get_parallel().tp_rank == 0:
            logger.warning(
                "Routed-experts forcing refused request %s for key %s: %s "
                "(%d refused).",
                req.rid,
                req.routed_experts_forcing_state.key,
                message,
                self.refused_requests,
            )
        return message


def get_global_routed_experts_forcing() -> Optional[RoutedExpertsForcing]:
    return get_resources().routed_experts_forcing


def set_global_routed_experts_forcing(
    forcing: Optional[RoutedExpertsForcing],
) -> None:
    get_resources().routed_experts_forcing = forcing


def admit_forcing_request(req: Req, *, key: str) -> Optional[str]:
    """Attach a routed_experts_session_key request's held rows, or return why it
    must not run."""
    if (forcing := get_global_routed_experts_forcing()) is None:
        return (
            f"{REFUSED_DISABLED}: start the server with --enable-routed-experts-forcing"
        )
    return forcing.admit(req, key=key)


@contextlib.contextmanager
def forcing_forward(batch: ForcingBatch, *, device) -> Iterator[None]:
    """Expose one extend forward's forced rows to every MoE gate it runs."""
    forced = _ForcedForward(
        batch=batch.to_device(device),
        layer_ordinals=get_global_routed_experts_forcing().layer_ordinals,
    )
    with get_forward().scoped(routed_experts_forced=forced):
        yield
    forced.check_every_layer_forced()


class _ForcedForward:
    """One extend forward's forced rows on the device, and the gates that took them."""

    def __init__(self, *, batch: ForcingBatch, layer_ordinals: dict[int, int]):
        self._batch = batch
        self._layer_ordinals = layer_ordinals
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
        ordinal = self._layer_ordinals.get(layer_id)
        if ordinal is None or packed:
            raise RuntimeError(
                f"routed-experts forcing cannot force the gate of layer {layer_id}: "
                "it is not a captured TopK layer, or it packs its top-k"
            )
        self._forced_layers.add(layer_id)
        return apply_forced_routing(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            router_logits=router_logits,
            forced_ids=self._batch.routes[:, ordinal],
            mask=self._batch.mask,
            scoring_func=topk_config.scoring_func,
            renormalize=topk_config.renormalize,
            routed_scaling_factor=topk_config.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=topk_config.apply_routed_scaling_factor_on_output,
        )

    def check_every_layer_forced(self) -> None:
        # A gate that routes outside select_experts would recompute held rows freely.
        missing = self._layer_ordinals.keys() - self._forced_layers
        if missing:
            raise RuntimeError(
                "routed-experts forcing never reached the gates of MoE layers "
                f"{sorted(missing)}"
            )


def _gate_problems(configs: Sequence[TopKConfig], *, top_k: int) -> list[str]:
    problems = set()
    for config in configs:
        num_routed = config.top_k - config.num_fused_shared_experts
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
            num_fused_shared_experts=config.num_fused_shared_experts,
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
    num_fused_shared_experts: int,
) -> Optional[str]:
    """Why ``_forced_routing_weights`` cannot reproduce this gate's weights, or None."""
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
        apply_routed_scaling_factor_on_output or num_fused_shared_experts
    ):
        # Unrenormalized, only the fused gate scales, and a fused shared expert's
        # weight follows the routed experts.
        return "routed scaling or fused shared experts without renormalize"
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
    """The gate's fp32 weights for ``forced_ids``.

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
    """Route masked rows to ``forced_ids``.

    A row whose forced set equals the gate's keeps the gate's ids and weights;
    fused shared-expert columns are untouched.
    """
    if topk_ids.shape[0] != mask.shape[0]:
        raise ValueError(f"{topk_ids.shape[0]} rows but {mask.shape[0]} forcing rows")
    num_routed = forced_ids.shape[1]
    routed_ids, routed_weights = topk_ids[:, :num_routed], topk_weights[:, :num_routed]
    forced = forced_ids.to(routed_ids.dtype)
    same_set = (forced.sort(dim=-1).values == routed_ids.sort(dim=-1).values).all(-1)
    replace = (mask & ~same_set)[:, None]
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
