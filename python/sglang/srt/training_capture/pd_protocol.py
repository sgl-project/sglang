"""Bounded P/D handoff of one raw teacher row; decode owns publication."""

from __future__ import annotations

import math
from typing import Literal

import msgspec
import torch

from sglang.srt.training_capture.protocol import (
    ContractError,
    Digest,
    Identifier,
    Nonnegative,
    Positive,
    StrictStruct,
    canonical_bytes,
    digest_bytes,
)
from sglang.srt.training_capture.teacher import TeacherRows

MAX_HANDOFF_BYTES = 16384


class CaptureTransferContext(StrictStruct, kw_only=True):
    version: Literal[1] = 1
    capture_id: Identifier
    fencing_token: Positive
    dataset_id: Identifier
    sample_id: Identifier
    generation_id: Identifier
    bootstrap_room: Positive
    contract_sha256: Digest
    prompt_sha256: Digest
    prompt_length: Positive
    sampling_sha256: Digest


class PrefillTeacherHandoff(StrictStruct):
    context: CaptureTransferContext
    output_token_id: Nonnegative
    topk_ids: list[Nonnegative]
    topk_logits: list[float]
    logsumexp: float

    def teacher_rows(self, vocab_size: int) -> TeacherRows:
        if (
            not 0 <= self.output_token_id < vocab_size
            or len(self.topk_ids) != 128
            or len(set(self.topk_ids)) != 128
            or len(self.topk_logits) != 128
            or any(not 0 <= token < vocab_size for token in self.topk_ids)
            or not all(math.isfinite(value) for value in self.topk_logits)
            or not math.isfinite(self.logsumexp)
            or any(a < b for a, b in zip(self.topk_logits, self.topk_logits[1:]))
            or self.logsumexp < self.topk_logits[0]
        ):
            raise ContractError("invalid prefill raw teacher row")
        return TeacherRows(
            token_ids=torch.tensor([self.topk_ids], dtype=torch.int32),
            logits=torch.tensor([self.topk_logits], dtype=torch.float32),
            logsumexp=torch.tensor([self.logsumexp], dtype=torch.float32),
        )


def contract_digest(teacher, kv) -> str:
    return digest_bytes(canonical_bytes({"teacher": teacher, "kv": kv}))


def prompt_digest(tokens) -> str:
    return digest_bytes(canonical_bytes(list(tokens)))


def sampling_digest(sampling_config) -> str:
    # Routers reduce the prefill request to one generated token.
    return digest_bytes(
        canonical_bytes(
            {
                key: value
                for key, value in sampling_config.items()
                if key != "max_new_tokens"
            }
        )
    )


def encode_handoff(value) -> bytes:
    payload = msgspec.msgpack.encode(value)
    if len(payload) > MAX_HANDOFF_BYTES:
        raise ContractError("PD capture handoff exceeds its byte budget")
    return payload


def decode_handoff(payload: bytes, target_type):
    if not payload or len(payload) > MAX_HANDOFF_BYTES:
        raise ContractError("missing or oversized PD capture handoff")
    try:
        return msgspec.msgpack.decode(payload, type=target_type)
    except (msgspec.DecodeError, ValueError) as error:
        raise ContractError("invalid PD capture handoff") from error
