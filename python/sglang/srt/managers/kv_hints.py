# SPDX-License-Identifier: Apache-2.0
"""Versioned KV-hint envelope carried on a request.

A KV hint is advisory routing metadata attached by a trusted orchestrator (a
router or a scheduler) *after* it has picked a worker -- never by an application
client. SGLang transports it unchanged from the entrypoint down to the HiCache
storage backends; nothing in the core reads a payload.

The envelope is a list of independent actions. Each action names its own type
and version, so one consumer implementing ``foo.bar@1.0`` must ignore -- not
reject -- an action it does not implement sitting next to it in the same
envelope. ``payload`` is therefore deliberately untyped here: the component
that implements an action owns its schema and validates it.

Wire-compatible with vLLM's ``vllm.v1.kv_hints`` (vllm-project/vllm#53423) so
one orchestrator can address either engine with the same envelope.
"""

from __future__ import annotations

from typing import Any, Dict, List

import msgspec

from sglang.srt.utils.msgspec_utils import msgspec_struct_pydantic_core_schema


class KvHintAction(msgspec.Struct, frozen=True, kw_only=True):
    """One versioned action inside a KV-hint envelope."""

    action_id: str
    action_type: str
    action_version: str
    # JSON-compatible values only; validated by whichever component implements
    # `action_type`, not here.
    payload: Dict[str, Any] = {}

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return msgspec_struct_pydantic_core_schema(cls, handler)


class KvHintsEnvelope(msgspec.Struct, frozen=True, kw_only=True):
    """Orchestrator-provided KV hints attached to one request."""

    protocol_version: str
    message_id: str
    actions: List[KvHintAction] = []

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return msgspec_struct_pydantic_core_schema(cls, handler)


# Envelope version this transport was specified against. Not enforced: an
# action carries its own version, and a consumer that finds no action it
# implements already falls back to normal behavior.
KV_HINTS_PROTOCOL_VERSION = "0.1"


def decode_kv_hints_envelope(value: Any) -> KvHintsEnvelope:
    """Convert a JSON/dict envelope into :class:`KvHintsEnvelope`.

    Raises ``ValueError`` on a malformed envelope so an entrypoint rejects the
    request outright. This is the one place the shape is enforced; every layer
    below holds the typed struct.
    """
    if isinstance(value, KvHintsEnvelope):
        return value
    try:
        return msgspec.convert(value, KvHintsEnvelope)
    except (msgspec.ValidationError, TypeError) as e:
        raise ValueError(f"Malformed kv_hints envelope: {e}") from e
