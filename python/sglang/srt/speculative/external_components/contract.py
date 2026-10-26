"""Wire contract for external speculative decoding.

This module is the complete specification a drafter implementer needs.
It depends only on msgspec (no SGLang imports) so the contract and its
reference serialization cannot drift.

Encoding
--------
All messages are ``msgspec.Struct`` subtypes with ``tag=True`` and
``array_like=True``, encoded with ``msgspec.msgpack``.

- ``array_like=True`` means each struct is a positional msgpack array
  (no field names on the wire).
- ``tag=True`` means the **first element** of every array is the struct's
  class name as a string (e.g. ``"DraftTokensRequest"``).  This tag is the
  sole discriminator; a single ``Union[...]`` decoder on either side routes
  to the correct type.

Wire format example::

    DraftTokensReply(request_id=7, token_ids=[12, 55, 3])
    -> msgpack(["DraftTokensReply", 7, [12, 55, 3]])

A drafter in any language builds the same array with a stock msgpack
library: put the class name string at index 0, then the field values in
declaration order.

Message types
-------------
SGLang -> drafter:
    "ConfigureDraftEngineRequest"  [tag, num_draft_tokens, protocol_version]
    "PrefillCacheRequest"          [tag, prefill_contexts]
    "DraftTokensRequest"           [tag, draft_contexts]
    "CancelInferenceRequest"       [tag, request_ids]

Drafter -> SGLang:
    "ConfigureDraftEngineResponse" [tag]
    "DraftTokensReply"             [tag, request_id, token_ids]
    "CancelInferenceResponse"      [tag]
"""

from typing import Union

import msgspec


class _ExternalMessage(msgspec.Struct, tag=True, array_like=True):
    """Base for all external speculative decoding wire messages."""

    pass


# -- ConfigureDraftEngine (once at startup) ----------------------------------


class ConfigureDraftEngineRequest(_ExternalMessage):
    num_draft_tokens: int
    protocol_version: int = 1


class ConfigureDraftEngineResponse(_ExternalMessage):
    pass


# -- PrefillCache (once per request, fire and forget) ------------------------


class PrefillContext(_ExternalMessage):
    request_id: int
    prompt_token_ids: list[int]


class PrefillCacheRequest(_ExternalMessage):
    prefill_contexts: list[PrefillContext]


# -- DraftTokens (every decode step) ----------------------------------------


class DraftContext(_ExternalMessage):
    request_id: int
    seq_len: int
    bonus_token_id: int


class DraftTokensRequest(_ExternalMessage):
    draft_contexts: list[DraftContext]


class DraftTokensReply(_ExternalMessage):
    request_id: int
    # Exactly num_draft_tokens token IDs, or empty on error (e.g. prefill
    # failed or timed out for this request).
    token_ids: list[int]


# -- CancelInference (once per finished request) ----------------------------


class CancelInferenceRequest(_ExternalMessage):
    request_ids: list[int]


class CancelInferenceResponse(_ExternalMessage):
    pass


# -- Decoder for all message types -------------------------------------------

AllMessages = Union[
    ConfigureDraftEngineRequest,
    ConfigureDraftEngineResponse,
    PrefillCacheRequest,
    DraftTokensRequest,
    DraftTokensReply,
    CancelInferenceRequest,
    CancelInferenceResponse,
]
