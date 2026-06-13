"""Compatibility policy and audit records for tool-call parsing."""

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, List, Optional, Tuple, Type, Union

logger = logging.getLogger(__name__)


class CompatibilityEvent(str, Enum):
    """One named tolerance.

    Members carry their compatibility scope and whether the deviation is the model's
    (raises in strict mode) or the caller's / already-failed (never raises).
    """

    def __new__(cls, value: str, scope: int, strict_raises: bool):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.scope = scope
        obj.strict_raises = strict_raises
        return obj

    # -- Scope 1: wire structure -----------------------------------------
    #: Unparseable text dropped up to the next recognizable marker.
    SKIPPED_GARBAGE = ("skipped-garbage", 1, True)
    #: A block whose JSON (or literal) body failed to parse was dropped;
    #: surrounding output is preserved.
    MALFORMED_JSON_DROPPED = ("malformed-json-dropped", 1, True)
    #: The model called a tool that is not in the request's tools; the call
    #: was dropped (set SGLANG_FORWARD_UNKNOWN_TOOLS to forward instead).
    UNKNOWN_TOOL_DROPPED = ("unknown-tool-dropped", 1, True)
    #: The rest of an invoke was dropped and its streamed arguments closed.
    DROPPED_INVOKE_TAIL = ("dropped-invoke-tail", 1, True)
    #: A Step3 tool-call entry whose type part is not ``function``.
    SKIPPED_NON_FUNCTION_ENTRY = ("skipped-non-function-entry", 1, True)
    #: A parameter value terminated by the next tag instead of its close tag.
    MISSING_CLOSE_TAG = ("missing-close-tag", 1, True)
    #: Generation ended mid-call (e.g. max_tokens); the half-specified call
    #: was dropped and its raw text re-surfaced as content.
    TRUNCATED_CALL_DROPPED = ("truncated-call-dropped", 1, True)

    # -- Scope 2: nested parameter structure -----------------------------
    #: A closing tag closed tags the model never closed itself.
    MISMATCHED_CLOSING_TAG = ("mismatched-closing-tag", 2, True)
    #: Tags still open at the end of a parameter body were flushed.
    UNCLOSED_TAGS_AT_END = ("unclosed-tags-at-end", 2, True)
    #: A duplicate tag inside an object was collapsed into a list.
    DUPLICATE_TAG_AS_LIST = ("duplicate-tag-as-list", 2, True)
    #: Stray text inside an object was captured under ``$text``.
    MIXED_TEXT_CAPTURED = ("mixed-text-captured", 2, True)
    #: An array element used a tag other than ``<item>``.
    NON_ITEM_ARRAY_CHILD = ("non-item-array-child", 2, True)
    #: Nested tags were parsed as structure although the schema says scalar.
    STRUCTURE_OVERRODE_SCHEMA = ("structure-overrode-schema", 2, True)

    # -- Scope 3: value conversion ---------------------------------------
    #: No candidate type could convert the value; the raw string was kept.
    UNCONVERTIBLE_VALUE_KEPT_RAW = ("unconvertible-value-kept-raw", 3, True)
    #: The tool's JSON schema is itself invalid and was ignored (caller
    #: fault -- never raises in strict mode).
    INVALID_SCHEMA_IGNORED = ("invalid-schema-ignored", 3, False)

    # -- Scope 4: fail-open ----------------------------------------------
    #: The detector raised; the parse fell open to normal text (recorded at
    #: the FunctionCallParser boundary after the fact -- never raises).
    FAIL_OPEN = ("fail-open", 4, False)


#: Cap on stored records; pathological inputs degrade to counting only.
_MAX_RECORDS = 256


@dataclass(slots=True)
class CompatibilityRecord:
    event: CompatibilityEvent
    detail: str = ""
    #: Character offset into the full received text, when known.
    offset: Optional[int] = None


class CompatibilityViolation(Exception):
    """A model-output deviation hit a strict-mode policy."""

    def __init__(self, record: CompatibilityRecord):
        self.record = record
        where = f" at offset {record.offset}" if record.offset is not None else ""
        detail = f": {record.detail}" if record.detail else ""
        super().__init__(f"strict mode: {record.event.value}{where}{detail}")


@dataclass(slots=True)
class _Absorbed:
    """Outcome of one :meth:`CompatibilityMode.absorb` block, for flow control."""

    fired: bool = False
    record: Optional[CompatibilityRecord] = None


@dataclass
class CompatibilityMode:
    """Per-request tolerance policy plus the audit trail of applied tolerances.

    ``BaseFormatDetector`` creates one per detector instance (detectors are
    per-request locals); a detector's internal parser shares the detector's
    policy so all tolerances for a request land on one audit trail.
    """

    strict: bool = False
    records: List[CompatibilityRecord] = field(default_factory=list)
    #: Records not stored because ``_MAX_RECORDS`` was reached.
    dropped: int = 0

    def note(
        self,
        event: CompatibilityEvent,
        detail: str = "",
        *,
        offset: Optional[int] = None,
    ) -> Optional[CompatibilityRecord]:
        """Record (and log) one applied tolerance; raise instead in strict mode."""
        record = CompatibilityRecord(event=event, detail=detail, offset=offset)
        if self.strict and event.strict_raises:
            raise CompatibilityViolation(record)
        logger.warning("compatibility: %s (%s)", event.value, detail)
        if len(self.records) >= _MAX_RECORDS:
            self.dropped += 1
            return None
        self.records.append(record)
        return record

    @contextmanager
    def absorb(
        self,
        event: CompatibilityEvent,
        exceptions: Union[Type[BaseException], Tuple[Type[BaseException], ...]],
        detail: str = "",
    ) -> Iterator[_Absorbed]:
        """Tolerance as a context: suppress ``exceptions``, record ``event``.

        In strict mode the recording raises ``CompatibilityViolation`` instead
        of suppressing. Exceptions not listed propagate untouched.
        """
        outcome = _Absorbed()
        try:
            yield outcome
        except exceptions as e:
            outcome.fired = True
            outcome.record = self.note(event, detail=detail or str(e)[:120])

    def summary(self) -> str:
        """One-line ``event x count`` summary for logs."""
        if not self.records and not self.dropped:
            return "none"
        counts: dict = {}
        for record in self.records:
            counts[record.event.value] = counts.get(record.event.value, 0) + 1
        parts = [f"{name} x{count}" for name, count in counts.items()]
        if self.dropped:
            parts.append(f"(+{self.dropped} dropped)")
        return ", ".join(parts)
