"""Dedicated, flag-gated artifact sink for the DS selection-recall oracle.

Records are keyed by ``(request_id, trial_id, layer_id, decode_step)`` — a
schema the production metrics path does not have. This sink is intentionally
SEPARATE from :mod:`metrics` (``record_selection`` is Prometheus production
counter plumbing for selected/valid token counts): the recall oracle must NOT
route per-trial rank / percentile payloads through ``record_selection`` (it
would pollute production counters and, worse, trap a host sync during
CUDA-graph capture).

Activation is opt-in via the ``SGLANG_DS_RECALL_ORACLE`` environment flag and
off by default. When disabled, :func:`oracle_enabled` is ``False`` and
:func:`record_oracle_sample` is a cheap early return — no sink object, no
allocation, no host sync — so the production hot path is byte-for-byte
unaffected (the "zero hot-path cost" property the equivalence test pins).

Callers on a CUDA-graph-captured path MUST additionally gate the record call
behind ``not torch.cuda.is_current_stream_capturing()`` (host writes / ``.item()``
are illegal during capture), mirroring the existing radix-capture guard.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_ORACLE_ENV_FLAG = "SGLANG_DS_RECALL_ORACLE"
_ORACLE_SINK_PATH_ENV = "SGLANG_DS_RECALL_ORACLE_PATH"


def oracle_enabled() -> bool:
    """``True`` iff the recall oracle is opt-in enabled via the env flag.

    Read fresh each call so tests can toggle it; the cost is one ``os.environ``
    lookup, only paid when a caller explicitly probes the oracle.
    """
    return os.environ.get(_ORACLE_ENV_FLAG, "0") not in ("0", "", "false", "False")


@dataclass
class OracleArtifactSink:
    """In-memory + optional JSONL sink for per-(request, trial, layer, step) records.

    Kept deliberately dumb: it appends fully-formed dict records. All numeric
    extraction (``.item()`` / ``.tolist()``) happens in the caller BEFORE
    ``record`` so this module never touches a live device tensor.
    """

    path: Optional[str] = None
    records: List[Dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, payload: Dict[str, Any]) -> None:
        required = ("request_id", "trial_id", "layer_id", "decode_step")
        missing = [k for k in required if k not in payload]
        if missing:
            raise ValueError(
                f"oracle record missing required key(s) {missing}; the sink is "
                "keyed by (request_id, trial_id, layer_id, decode_step)."
            )
        with self._lock:
            self.records.append(payload)
            if self.path is not None:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(payload) + "\n")

    def clear(self) -> None:
        with self._lock:
            self.records.clear()


_ACTIVE_SINK: Optional[OracleArtifactSink] = None
_ACTIVE_SINK_LOCK = threading.Lock()


def get_sink() -> Optional[OracleArtifactSink]:
    """Return the active sink, lazily creating it when the oracle is enabled.

    Returns ``None`` when the oracle is disabled so callers can short-circuit
    without constructing anything.
    """
    if not oracle_enabled():
        return None
    global _ACTIVE_SINK
    if _ACTIVE_SINK is None:
        with _ACTIVE_SINK_LOCK:
            if _ACTIVE_SINK is None:
                _ACTIVE_SINK = OracleArtifactSink(
                    path=os.environ.get(_ORACLE_SINK_PATH_ENV) or None
                )
    return _ACTIVE_SINK


def reset_sink_for_testing(sink: Optional[OracleArtifactSink] = None) -> None:
    """Install (or clear) the active sink. Test-only."""
    global _ACTIVE_SINK
    with _ACTIVE_SINK_LOCK:
        _ACTIVE_SINK = sink


@dataclass
class OracleTrialContext:
    """The active NIAH oracle trial. NIAH diagnostic trials are single-request,
    so a module-level active context (set by the harness before a trial) is the
    correct, simplest way to carry the harness-provided needle span down to the
    selector score path. ``sample_counter`` advances once per recorded selector
    call so each (layer, decode-step) sample is distinguishable.
    """

    request_id: Any
    trial_id: Any
    needle_positions: List[int]
    sample_counter: int = 0


_ACTIVE_TRIAL: Optional[OracleTrialContext] = None
_ACTIVE_TRIAL_LOCK = threading.Lock()


def set_active_trial(request_id: Any, trial_id: Any, needle_positions) -> None:
    """Register the active NIAH oracle trial's needle span (harness-side)."""
    span = [int(p) for p in needle_positions]
    if not span:
        raise ValueError("needle_positions must be non-empty (the oracle never guesses).")
    global _ACTIVE_TRIAL
    with _ACTIVE_TRIAL_LOCK:
        _ACTIVE_TRIAL = OracleTrialContext(
            request_id=request_id, trial_id=trial_id, needle_positions=span
        )


def clear_active_trial() -> None:
    global _ACTIVE_TRIAL
    with _ACTIVE_TRIAL_LOCK:
        _ACTIVE_TRIAL = None


def get_active_trial() -> Optional[OracleTrialContext]:
    return _ACTIVE_TRIAL


def next_sample_index() -> int:
    """Advance and return the active trial's per-call sample counter."""
    with _ACTIVE_TRIAL_LOCK:
        if _ACTIVE_TRIAL is None:
            return 0
        idx = _ACTIVE_TRIAL.sample_counter
        _ACTIVE_TRIAL.sample_counter = idx + 1
        return idx


def record_oracle_sample(
    *,
    request_id: Any,
    trial_id: Any,
    layer_id: int,
    decode_step: int,
    payload: Dict[str, Any],
) -> bool:
    """Write one oracle sample to the active sink, keyed and host-resident.

    No-op (returns ``False``) when the oracle is disabled, so this is safe to
    call unconditionally from instrumentation sites. ``payload`` must already be
    host-resident (plain Python numbers / lists) — never pass live device
    tensors. Returns ``True`` iff a record was written.
    """
    sink = get_sink()
    if sink is None:
        return False
    record = {
        "request_id": request_id,
        "trial_id": trial_id,
        "layer_id": int(layer_id),
        "decode_step": int(decode_step),
        **payload,
    }
    sink.record(record)
    return True
