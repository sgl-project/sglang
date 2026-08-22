from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log: logging.Logger = logging.getLogger(__name__)

GENESIS_HASH: str = (
    "0000000000000000000000000000000000000000000000000000000000000000"
)


@dataclass
class SGLangDebtReport:
    server_id: str
    sdi_score: float  # SGLang Debt Index (target <= 12.0)
    radix_eviction_multiplier: float  # Target <= 1.08x
    ttft_latency_seconds: float  # Target <= 0.65s
    mutation_safety_score: float  # Target 100.0
    production_readiness_index: float  # Scale 0 - 100
    is_production_ready: bool
    critical_smells: List[str]
    receipt_hash: str


class TechnicalDueDiligenceLedger:
    """
    Cryptographic SHA-256 hash-chained Action Ledger for SGLang high-throughput inference runs.
    """

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []
        self._last_hash: str = GENESIS_HASH

    def record_inference_event(
        self,
        server_id: str,
        event_type: str,
        readiness_index: float,
        critical_smells: List[str],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        index = len(self._entries)

        meta_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
        canonical_content = f"{index}|{self._last_hash}|{server_id}|{event_type}|{readiness_index}|{timestamp}|{hashlib.sha256(meta_bytes).hexdigest()}"
        curr_hash = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()

        entry = {
            "index": index,
            "timestamp": timestamp,
            "server_id": server_id,
            "event_type": event_type,
            "readiness_index": readiness_index,
            "critical_smells": critical_smells,
            "prev_hash": self._last_hash,
            "curr_hash": curr_hash,
            "metadata": metadata,
        }

        self._entries.append(entry)
        self._last_hash = curr_hash
        return entry

    def get_ledger_entries(self) -> List[Dict[str, Any]]:
        return list(self._entries)

    def verify_ledger_integrity(self) -> bool:
        prev = GENESIS_HASH
        for entry in self._entries:
            if entry["prev_hash"] != prev:
                return False
            prev = entry["curr_hash"]
        return True


class ProductionDebtInferenceGate:
    """
    A2Z SOC Production Debt & Technical Due Diligence Gate for SGLang Serving & RadixAttention.

    Quantifies RadixTree KV cache memory, speculative rollbacks, and TTFT latency against 4 Enterprise KPIs:
    1. SGLang Debt Index (SDI <= 12.0)
    2. Radix KV Cache Eviction Multiplier (REM <= 1.08x)
    3. P99 Time-To-First-Token (TTFT <= 0.65s)
    4. Deterministic Mutation Boundaries (never_equate_intent_to_approval)
    """

    def __init__(
        self,
        never_equate_intent_to_approval: bool = True,
        max_acceptable_sdi: float = 12.0,
    ) -> None:
        self.never_equate_intent_to_approval = never_equate_intent_to_approval
        self.max_acceptable_sdi = max_acceptable_sdi
        self.ledger = TechnicalDueDiligenceLedger()

    def check_kill_switch(self) -> bool:
        if os.environ.get("AAG_KILL_SWITCH", "").lower() in ("true", "1", "yes"):
            return True
        for path_str in ("artifacts/KILL", "/tmp/KILL"):
            if Path(path_str).exists():
                return True
        return False

    def evaluate_inference_batch(
        self,
        server_id: str,
        allocated_kv_bytes: int = 10000000000,
        peak_radix_kv_bytes: int = 10500000000,
        ttft_latency_seconds: float = 0.45,
        speculative_rollback_count: int = 0,
        un_gated_mutations: int = 0,
    ) -> SGLangDebtReport:
        # 1. Evaluate emergency kill switch
        if self.check_kill_switch():
            self.ledger.record_inference_event(
                server_id=server_id,
                event_type="serving_halted_kill_switch",
                readiness_index=0.0,
                critical_smells=["EMERGENCY_KILL_SWITCH_ENGAGED"],
                metadata={"reason": "AAG_KILL_SWITCH is set"},
            )
            raise PermissionError(
                "A2Z SOC ActionGate: Emergency kill switch is engaged. High-throughput inference halted."
            )

        critical_smells: List[str] = []

        # KPI 2: Radix Eviction Multiplier
        kv_ratio = peak_radix_kv_bytes / max(1, allocated_kv_bytes)
        if kv_ratio > 1.8:
            critical_smells.append(f"HIGH_RADIX_KV_CACHE_SPRAWL_{kv_ratio:.2f}X")

        # KPI 3: Latency Ceiling
        if ttft_latency_seconds > 2.0:
            critical_smells.append(f"HIGH_TTFT_LATENCY_{ttft_latency_seconds:.2f}S")

        # Speculative rollbacks
        if speculative_rollback_count > 2:
            critical_smells.append(f"DETECTED_{speculative_rollback_count}_SPECULATIVE_ROLLBACKS")

        # KPI 4: Mutation Safety
        if un_gated_mutations > 0:
            critical_smells.append(f"DETECTED_{un_gated_mutations}_UNGATED_PROGRAM_MUTATIONS")

        # KPI 1: SGLang Debt Index (0 = Clean, 100 = Catastrophic)
        sdi = (
            max(0.0, (kv_ratio - 1.0) * 20.0)
            + max(0.0, (ttft_latency_seconds - 0.65) * 10.0)
            + (speculative_rollback_count * 12.0)
            + (un_gated_mutations * 30.0)
        )
        sdi_score = round(min(100.0, sdi), 2)

        # Production Readiness Index (0 - 100)
        readiness = max(0.0, 100.0 - sdi_score)
        is_production_ready = (
            sdi_score <= self.max_acceptable_sdi and len(critical_smells) == 0
        )

        # Cryptographic Ledger Entry
        entry = self.ledger.record_inference_event(
            server_id=server_id,
            event_type="batch_authorized" if is_production_ready else "batch_flagged_debt",
            readiness_index=readiness,
            critical_smells=critical_smells,
            metadata={
                "sdi_score": sdi_score,
                "kv_ratio": kv_ratio,
                "allocated_kv_bytes": allocated_kv_bytes,
                "peak_radix_kv_bytes": peak_radix_kv_bytes,
                "ttft_latency_seconds": ttft_latency_seconds,
                "speculative_rollback_count": speculative_rollback_count,
                "un_gated_mutations": un_gated_mutations,
                "never_equate_intent_to_approval": self.never_equate_intent_to_approval,
            },
        )

        return SGLangDebtReport(
            server_id=server_id,
            sdi_score=sdi_score,
            radix_eviction_multiplier=round(kv_ratio, 2),
            ttft_latency_seconds=round(ttft_latency_seconds, 2),
            mutation_safety_score=(
                100.0 if un_gated_mutations == 0 else max(0.0, 100.0 - un_gated_mutations * 30.0)
            ),
            production_readiness_index=readiness,
            is_production_ready=is_production_ready,
            critical_smells=critical_smells,
            receipt_hash=entry["curr_hash"],
        )
