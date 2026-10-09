"""Select a cohort before TP broadcast and carry its identity through PP ingress."""

from __future__ import annotations

import math
import random
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Literal

import msgspec
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.training_capture.cohort_service import CaptureHandle, CaptureTicket
from sglang.srt.training_capture.protocol import (
    ContractError,
    Identifier,
    StrictStruct,
    canonical_bytes,
    digest_bytes,
)

_MAX_TICKET_BYTES = 2048
_MAX_REQUEST_BYTES = 1 << 20


class CaptureRequestTicket(StrictStruct):
    version: Literal[1]
    nonce: Identifier
    cohort: CaptureTicket


@dataclass(eq=False)
class CaptureRequestRoute:
    ticket: CaptureRequestTicket
    attempted: bool = False
    handle: CaptureHandle | None = None
    execution_sha256: str | None = None


def _sampling_contract(params):
    # Include new SamplingParams fields by default; an allowlist would silently
    # omit future sampling controls. Only stop_token_ids needs set normalization.
    values = {name: getattr(params, name) for name in params.__struct_fields__}
    values["stop_token_ids"] = sorted(values["stop_token_ids"] or ())
    return values


def _request_digest(
    *, nonce, rid, tokens, params, require_reasoning, extra_key, token_type_ids
):
    data = canonical_bytes(
        {
            "version": 1,
            "nonce": nonce,
            "rid": rid,
            "tokens": list(tokens),
            "sampling": _sampling_contract(params),
            "require_reasoning": require_reasoning,
            "extra_key": extra_key,
            "token_type_ids": token_type_ids,
        }
    )
    if len(data) > _MAX_REQUEST_BYTES:
        raise ContractError("capture request exceeds its identity budget")
    return digest_bytes(data)


def _ingress_digest(req, nonce):
    return _request_digest(
        nonce=nonce,
        rid=req.rid,
        tokens=req.input_ids,
        params=req.sampling_params,
        require_reasoning=req.require_reasoning,
        extra_key=req.extra_key,
        token_type_ids=req.token_type_ids,
    )


class CaptureRequestRouter:
    """Scheduler-thread adapter for a ready CaptureCohortService.

    prepare() runs once on the ingress rank, before existing TP/PP request
    communication. attach() verifies that wire decision on each local Req.
    The coordinator calls bind() before its first capture forward and later
    owns finish(handle). No method here performs a collective or Catalog call.
    """

    def __init__(self, service, *, sample_ratio=None):
        self.service = service
        self.config = service.allocator.config
        if self.config.adaptive is not None and sample_ratio is None:
            raise ContractError("adaptive capture requires the admission controller")
        self.sample_ratio = sample_ratio or (lambda: self.config.sample_ratio)
        self.rng = random.Random(self.config.sample_seed)
        self.counters = Counter()

    def _eligible(self, req):
        if not isinstance(req.rid, str) or req.rid.startswith(HEALTH_CHECK_RID_PREFIX):
            return False
        if any(
            (
                req.no_logs,
                req.lora_id is not None,
                req.mm_inputs is not None,
                req.mm_data_mooncake is not None,
                req.input_embeds is not None,
                req.positional_embed_overrides is not None,
                req.session_id is not None,
                req.session_params is not None,
                req.custom_logit_processor is not None,
                req.sampling_params.custom_params is not None,
            )
        ):
            return False
        count = req.sampling_params.max_new_tokens
        return (
            req.input_ids is not None
            and len(req.input_ids) > 0
            and count is not None
            and count > 0
            and len(req.input_ids) + count <= self.config.max_sample_tokens
        )

    def prepare(self, requests):
        from sglang.srt.managers.io_struct import (
            BatchTokenizedGenerateReqInput,
            TokenizedGenerateReqInput,
        )

        if self.service.allocator.rank != 0:
            raise ContractError("only the capture ingress rank may select requests")
        for request in requests or ():
            batch = (
                request
                if isinstance(request, BatchTokenizedGenerateReqInput)
                else (request,)
            )
            for req in batch:
                if not isinstance(req, TokenizedGenerateReqInput):
                    continue
                # A client/tokenizer-supplied ticket never chooses server resources.
                req.training_capture_ticket = None
                ticket = None
                try:
                    self.counters["considered"] += 1
                    if not self._eligible(req):
                        self.counters["excluded"] += 1
                        continue
                    ratio = self.sample_ratio()
                    if (
                        not math.isfinite(ratio)
                        or not 0 <= ratio <= self.config.sample_ratio
                    ):
                        raise ContractError("invalid capture admission ratio")
                    if self.rng.random() >= ratio:
                        self.counters["sampled_out"] += 1
                        continue
                    nonce = uuid.uuid4().hex
                    ticket = self.service.claim(_ingress_digest(req, nonce))
                    if ticket is None:
                        self.counters["backpressure"] += 1
                        continue
                    wire = msgspec.json.encode(
                        CaptureRequestTicket(version=1, nonce=nonce, cohort=ticket)
                    )
                    if len(wire) > _MAX_TICKET_BYTES:
                        raise ContractError("capture ticket exceeds its wire budget")
                    req.training_capture_ticket = wire
                    self.counters["selected"] += 1
                except Exception:  # noqa: BLE001 - capture must not reject inference
                    if ticket is not None:
                        self.service.cancel(ticket, "request_ticket_failed")
                    self.counters["selection_failed"] += 1

    def attach(self, incoming, req):
        """Validate ingress before Req normalization; do not bind a Host slot yet."""
        wire = incoming.training_capture_ticket
        if wire is None:
            return
        ticket = None
        try:
            if not isinstance(wire, bytes) or len(wire) > _MAX_TICKET_BYTES:
                raise ContractError("invalid capture ticket encoding")
            ticket = msgspec.json.decode(wire, type=CaptureRequestTicket)
            if (
                not self._eligible(incoming)
                or _ingress_digest(incoming, ticket.nonce)
                != ticket.cohort.request_sha256
            ):
                raise ContractError("capture ticket belongs to a different request")
            if req.training_capture_route is not None:
                raise ContractError("request already has a capture route")
            req.training_capture_route = CaptureRequestRoute(ticket)
            req.training_capture_cancel = self.cancel
            self.counters["attached"] += 1
        except Exception:  # noqa: BLE001 - malformed capture metadata fails closed
            if ticket is not None:
                self.service.cancel(ticket.cohort, "request_identity_mismatch")
            self.counters["attachment_failed"] += 1

    def bind(self, req) -> CaptureRequestRoute | None:
        route = req.training_capture_route
        if route is None or route.attempted:
            return None
        route.attempted = True
        handle = None
        try:
            if (
                req.finished()
                or req.to_finish is not None
                or req.is_retracted
                or req.output_ids
            ):
                raise ContractError("capture request already finished or advanced")
            requested = req.sampling_params.max_new_tokens
            if (
                requested is None
                or requested < 1
                or len(req.origin_input_ids) + requested > self.config.max_sample_tokens
            ):
                raise ContractError("effective request exceeds capture capacity")
            execution = _request_digest(
                nonce=route.ticket.nonce,
                rid=req.rid,
                tokens=req.origin_input_ids,
                params=req.sampling_params,
                require_reasoning=req.require_reasoning,
                extra_key=req.extra_key,
                token_type_ids=req.token_type_ids,
            )
            handle = self.service.bind(
                route.ticket.cohort, route.ticket.cohort.request_sha256
            )
            if handle is None:
                raise ContractError("request cohort is no longer bindable")
            route.handle, route.execution_sha256 = handle, execution
            self.counters["bound"] += 1
            return route
        except Exception:  # noqa: BLE001 - inference continues without capture
            self.service.cancel(route.ticket.cohort, "request_binding_failed")
            if handle is not None:
                # Ownership was never returned to the actor; no copy can have
                # started, so even post-bind bookkeeping failure can drain it.
                self.service.finish(handle, outcome="failed", transfer_complete=True)
                route.handle = route.execution_sha256 = None
            self.counters["binding_failed"] += 1
            return None

    def cancel(self, req, reason):
        route = req.training_capture_route
        if route is not None:
            route.attempted = True
            self.service.cancel(route.ticket.cohort, reason)
            self.counters["cancelled"] += 1
