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

PROTOCOL_VERSION = 1
MAX_RECORD_BYTES = 1 << 20
MAX_EXCHANGE_BYTES = 64 << 20


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
    if (
        not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
        or str(dist.get_backend(group)) != "gloo"
    ):
        raise ContractError("capture startup requires Gloo and a finite timeout")
    world_size = dist.get_world_size(group)
    if world_size < 1:
        raise ContractError("capture startup caller is outside its process group")
    timeout = timedelta(seconds=timeout_seconds)
    # Reuse small control buffers even if a later payload allocation fails.
    header = torch.zeros(7, dtype=torch.int64, device="cpu")
    headers = [torch.empty_like(header) for _ in range(world_size)]

    def gather(outputs, value):
        try:
            work = dist.all_gather(outputs, value, group=group, async_op=True)
            if not work.wait(timeout):
                raise TimeoutError("startup collective did not complete")
        except Exception as error:
            raise CaptureStartupError("transport") from error

    def vote(phase, error, *, length=0, fingerprint=()):
        header.zero_()
        header[0], header[1], header[2] = (
            PROTOCOL_VERSION,
            int(error is not None),
            length,
        )
        for index, word in enumerate(fingerprint, start=3):
            header[index] = word
        gather(headers, header)
        values = [item.tolist() for item in headers]
        if {item[0] for item in values} != {PROTOCOL_VERSION}:
            raise CaptureStartupError("protocol", range(world_size)) from error
        failures = [rank for rank, item in enumerate(values) if item[1] != 0]
        if failures:
            raise CaptureStartupError(phase, failures) from error
        return values

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
    lengths = [item[2] for item in vote("binding", error, length=len(payload))]

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
    vote("allocation", error)
    gather(received, send)

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
        raw_digest = bytes.fromhex(digest_bytes(canonical_bytes(result)))
        if len(raw_digest) != 32:
            raise ContractError("startup agreement requires a SHA-256 digest")
        fingerprint = tuple(
            int.from_bytes(raw_digest[offset : offset + 8], "little", signed=True)
            for offset in range(0, 32, 8)
        )
    except Exception as cause:  # noqa: BLE001 - no rank may return before the final vote
        error = cause
    votes = vote("validation", error, fingerprint=fingerprint)
    if len({tuple(item[3:]) for item in votes}) != 1:
        raise CaptureStartupError("agreement", range(world_size))
    return result
