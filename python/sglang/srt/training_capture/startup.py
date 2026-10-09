"""Bounded metadata exchange and all-rank agreement for capture startup."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import timedelta

import msgspec
import torch
import torch.distributed as dist
from sglang.srt.training_capture.identity import (
    RankTargetContract,
    assemble_target_contract,
)
from sglang.srt.training_capture.protocol import (
    ContractError,
    Nonnegative,
    Positive,
    StrictStruct,
    canonical_bytes,
    digest_bytes,
)

PROTOCOL_VERSION = 3
MAX_RECORD_BYTES = 1 << 20
MAX_EXCHANGE_BYTES = 64 << 20
_PHASES = (
    "binding",
    "allocation",
    "validation",
    "identity_ready",
    "policy",
    "resources",
    "resources_ready",
    "cleanup",
    "activation",
    "activation_ready",
)


class CaptureStartupError(ContractError):
    def __init__(self, phase: str, failed_ranks=()):
        self.phase = phase
        self.failed_ranks = tuple(failed_ranks)
        super().__init__(
            f"capture startup {phase} failed; group ranks={self.failed_ranks}"
        )


class _StartupRecord(StrictStruct):
    tp_size: Positive
    pp_size: Positive
    dp_rank: Nonnegative
    aux_tp_rank: Nonnegative
    contract: RankTargetContract


def _fingerprint(value, *, max_bytes=MAX_RECORD_BYTES):
    payload = canonical_bytes(value)
    if len(payload) > max_bytes:
        raise ContractError("startup agreement metadata exceeds its byte limit")
    raw_digest = bytes.fromhex(digest_bytes(payload))
    if len(raw_digest) != 32:
        raise ContractError("startup agreement requires a SHA-256 digest")
    return tuple(
        int.from_bytes(raw_digest[offset : offset + 8], "little", signed=True)
        for offset in range(0, 32, 8)
    )


class _StartupCollectives:
    def __init__(self, group, timeout_seconds):
        if (
            not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
            or str(dist.get_backend(group)) != "gloo"
        ):
            raise ContractError("capture startup requires Gloo and a finite timeout")
        self.group = group
        self.world_size = dist.get_world_size(group)
        if self.world_size < 1:
            raise ContractError("capture startup caller is outside its process group")
        self.timeout = timedelta(seconds=timeout_seconds)
        # Reserve control buffers before callbacks can exhaust payload memory.
        self.header = torch.zeros(8, dtype=torch.int64, device="cpu")
        self.headers = [torch.empty_like(self.header) for _ in range(self.world_size)]

    def gather(self, outputs, value):
        try:
            work = dist.all_gather(outputs, value, group=self.group, async_op=True)
            if not work.wait(self.timeout):
                raise TimeoutError("startup collective did not complete")
        except Exception as error:
            raise CaptureStartupError("transport") from error

    def vote(self, phase, error, *, length=0, fingerprint=()):
        self.header.zero_()
        self.header[0], self.header[1], self.header[2], self.header[7] = (
            PROTOCOL_VERSION,
            int(error is not None),
            length,
            _PHASES.index(phase),
        )
        for index, word in enumerate(fingerprint, start=3):
            self.header[index] = word
        self.gather(self.headers, self.header)
        values = [item.tolist() for item in self.headers]
        if any(
            item[0] != PROTOCOL_VERSION or item[7] != _PHASES.index(phase)
            for item in values
        ):
            raise CaptureStartupError("protocol", range(self.world_size)) from error
        failures = [rank for rank, item in enumerate(values) if item[1] != 0]
        if failures:
            raise CaptureStartupError(phase, failures) from error
        return values


def _agree_policy(channel, build_policy):
    error, fingerprint = None, ()
    try:
        fingerprint = _fingerprint(build_policy())
    except Exception as cause:  # noqa: BLE001 - every peer must reach the policy vote
        error = cause
    votes = channel.vote("policy", error, fingerprint=fingerprint)
    if len({tuple(item[3:7]) for item in votes}) != 1:
        raise CaptureStartupError("policy_agreement", range(channel.world_size))


def coordinate_policy_startup(*, group, build_policy, timeout_seconds=120.0):
    """Agree on immutable subsystem metadata before subsequent collectives."""
    channel = _StartupCollectives(group, timeout_seconds)
    _agree_policy(channel, build_policy)
    channel.vote("identity_ready", None)


def coordinate_resource_startup(
    *, group, build_policy: Callable, prepare_local: Callable, timeout_seconds=120.0
):
    """Prepare closeable, passive resources only after common policy agreement.

    Preparation must clean its own partial failures. Returned resources must not
    issue Catalog requests or transfers until the caller activates them after
    this function returns. A failed vote closes every successfully prepared rank;
    cleanup failures are also voted unless transport has already failed. Local
    callbacks and close barriers still require the worker supervisor's watchdog.
    """
    channel = _StartupCollectives(group, timeout_seconds)
    _agree_policy(channel, build_policy)

    error, resource = None, None
    try:
        resource = prepare_local()
        if resource is None or not callable(resource.close):
            raise ContractError("startup resource must provide a close barrier")
    except Exception as cause:  # noqa: BLE001 - resource failures need all-rank cleanup
        error = cause
    try:
        channel.vote("resources", error)
        # A timed-out all_gather can still complete for a late peer. Require
        # acknowledgement that peers returned from preparation and its vote.
        channel.vote("resources_ready", None)
    except CaptureStartupError as failure:
        cleanup_error = None
        try:
            if resource is not None:
                resource.close()
        except Exception as cause:  # noqa: BLE001 - communicate cleanup failures too
            cleanup_error = cause
        if failure.phase != "transport":
            channel.vote("cleanup", cleanup_error)
        elif cleanup_error is not None:
            raise CaptureStartupError("cleanup") from cleanup_error
        raise
    return resource


def coordinate_capture_activation(*, group, coordinator, timeout_seconds=120.0):
    """Start every local actor behind a gate before admitting capture work.

    The startup group is distinct from the coordinator's control group. No
    background Catalog/Store operation or control collective starts until both
    activation votes succeed. Failure closes passive actors on every rank.
    """
    channel = _StartupCollectives(group, timeout_seconds)
    error = None
    try:
        coordinator.activate(defer=True)
    except Exception as cause:  # noqa: BLE001 - peers must roll back together
        error = cause
    try:
        channel.vote("activation", error)
        channel.vote("activation_ready", None)
    except CaptureStartupError as failure:
        cleanup_error = None
        try:
            if not coordinator.close():
                raise ContractError("capture activation retained live resources")
        except Exception as cause:  # noqa: BLE001 - report incomplete rollback
            cleanup_error = cause
        if failure.phase != "transport":
            channel.vote("cleanup", cleanup_error)
        elif cleanup_error is not None:
            raise CaptureStartupError("cleanup") from cleanup_error
        raise
    coordinator.activation.set()


def coordinate_target_startup(
    *,
    group,
    build_local: Callable[[], RankTargetContract],
    tp_size: int,
    pp_size: int,
    dp_rank: int = 0,
    aux_tp_rank: int = 0,
    timeout_seconds: float = 120.0,
):
    """Use the existing Gloo group in PP-major, TP-minor rank order.

    Every member must enter in the same collective order. Local binding,
    allocation and validation failures are voted before the next phase. A
    transport failure poisons startup; callers must tear down the workers,
    never retry on that process group. Local callbacks also need the serving
    supervisor's watchdog if they can block before entering a collective.
    """
    channel = _StartupCollectives(group, timeout_seconds)
    world_size = channel.world_size

    error, payload = None, b""
    try:
        payload = canonical_bytes(
            _StartupRecord(
                tp_size=tp_size,
                pp_size=pp_size,
                dp_rank=dp_rank,
                aux_tp_rank=aux_tp_rank,
                contract=build_local(),
            )
        )
        if not 0 < len(payload) <= MAX_RECORD_BYTES:
            raise ContractError("rank startup metadata exceeds its byte limit")
    except Exception as cause:  # noqa: BLE001 - peers must vote before raising
        error = cause
    lengths = [item[2] for item in channel.vote("binding", error, length=len(payload))]

    error = None
    send, received = None, None
    try:
        padded = max(lengths)
        if (
            any(not 0 < size <= MAX_RECORD_BYTES for size in lengths)
            or padded * world_size > MAX_EXCHANGE_BYTES
        ):
            raise ContractError("startup metadata exceeds the aggregate byte limit")
        send = torch.zeros(padded, dtype=torch.uint8, device="cpu")
        send[: len(payload)].copy_(
            torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        )
        received = [torch.empty_like(send) for _ in range(world_size)]
    except Exception as cause:  # noqa: BLE001 - allocation failure must reach the vote
        error = cause
    channel.vote("allocation", error)
    channel.gather(received, send)

    error, result, fingerprint = None, None, ()
    try:
        records = [
            msgspec.json.decode(memoryview(value.numpy())[:size], type=_StartupRecord)
            for value, size in zip(received, lengths, strict=True)
        ]
        expected = (tp_size, pp_size, dp_rank, aux_tp_rank)
        if world_size != tp_size * pp_size or any(
            (record.tp_size, record.pp_size, record.dp_rank, record.aux_tp_rank)
            != expected
            or record.contract.pp_rank * tp_size + record.contract.tp_rank != origin
            for origin, record in enumerate(records)
        ):
            raise ContractError("startup topology or rank origin disagrees")
        result = assemble_target_contract(
            [record.contract for record in records],
            tp_size=tp_size,
            pp_size=pp_size,
            dp_rank=dp_rank,
            aux_tp_rank=aux_tp_rank,
        )
        fingerprint = _fingerprint(result, max_bytes=MAX_EXCHANGE_BYTES)
    except Exception as cause:  # noqa: BLE001 - no rank may return before the final vote
        error = cause
    votes = channel.vote("validation", error, fingerprint=fingerprint)
    if len({tuple(item[3:7]) for item in votes}) != 1:
        raise CaptureStartupError("agreement", range(world_size))
    channel.vote("identity_ready", None)
    return result
