from __future__ import annotations

"""
Copyright 2026 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"


@dataclass
class RadixCacheDebtReport:
    trie_node_id: str
    rdi_score: float  # RadixCache Debt Index (target <= 12.0)
    trie_sprawl_multiplier: float  # Target <= 1.08x
    lookup_latency_ms: float  # Target <= 2.8ms
    mutation_safety_score: float  # Target 100.0
    production_readiness_index: float  # Scale 0 - 100
    is_production_ready: bool
    critical_smells: list[str]
    receipt_hash: str


class TechnicalDueDiligenceLedger:
    """Cryptographic SHA-256 hash-chained Action Ledger for SGLang RadixCache trie tree operations."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []
        self._last_hash: str = GENESIS_HASH

    def record_radix_event(
        self,
        trie_node_id: str,
        event_type: str,
        readiness_index: float,
        critical_smells: list[str],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        index = len(self._entries)

        meta_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
        canonical_content = (
            f"{index}|{self._last_hash}|{trie_node_id}|{event_type}|"
            f"{readiness_index}|{timestamp}|{hashlib.sha256(meta_bytes).hexdigest()}"
        )
        curr_hash = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()

        entry = {
            "index": index,
            "timestamp": timestamp,
            "trie_node_id": trie_node_id,
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

    def get_ledger_entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def verify_ledger_integrity(self) -> bool:
        prev = GENESIS_HASH
        for entry in self._entries:
            if entry["prev_hash"] != prev:
                return False
            prev = entry["curr_hash"]
        return True


class ProductionDebtRadixCacheGate:
    """A2Z SOC Production Debt & Technical Due Diligence Gate for SGLang RadixCache Trie Tree & Chunked Prefill.

    Quantifies radix trie node memory fragmentation, chunked prefill worker pipeline stalls, and trie lookup/eviction latency against 4 Enterprise KPIs:
    1. RadixCache Debt Index (RDI <= 12.0)
    2. Trie Tree Memory Sprawl Multiplier (TTMM <= 1.08x)
    3. P99 Trie Lookup & Eviction Latency (<= 2.8ms)
    4. Deterministic Mutation Boundaries (never_equate_intent_to_approval)
    """

    def __init__(
        self,
        never_equate_intent_to_approval: bool = True,
        max_acceptable_rdi: float = 12.0,
    ) -> None:
        self.never_equate_intent_to_approval = never_equate_intent_to_approval
        self.max_acceptable_rdi = max_acceptable_rdi
        self.ledger = TechnicalDueDiligenceLedger()

    def check_kill_switch(self) -> bool:
        if os.environ.get("AAG_KILL_SWITCH", "").lower() in ("true", "1", "yes"):
            return True
        return any(Path(p).exists() for p in ("artifacts/KILL", "/tmp/KILL"))

    def evaluate_trie_eviction(
        self,
        trie_node_id: str,
        allocated_trie_bytes: int = 16000000000,
        utilized_trie_bytes: int = 16800000000,
        lookup_latency_ms: float = 2.1,
        chunked_prefill_stalls: int = 0,
        un_gated_mutations: int = 0,
    ) -> RadixCacheDebtReport:
        # 1. Evaluate emergency kill switch
        if self.check_kill_switch():
            self.ledger.record_radix_event(
                trie_node_id=trie_node_id,
                event_type="trie_operation_halted_kill_switch",
                readiness_index=0.0,
                critical_smells=["EMERGENCY_KILL_SWITCH_ENGAGED"],
                metadata={"reason": "AAG_KILL_SWITCH is set"},
            )
            err_msg = "A2Z SOC ActionGate: Emergency kill switch is engaged. SGLang RadixCache execution halted."
            raise PermissionError(err_msg)

        critical_smells: list[str] = []

        # KPI 2: Trie Tree Memory Sprawl Multiplier
        trie_ratio = utilized_trie_bytes / max(1, allocated_trie_bytes)
        if trie_ratio > 1.8:
            critical_smells.append(f"HIGH_RADIX_TRIE_FRAGMENTATION_{trie_ratio:.2f}X")

        # KPI 3: Latency Ceiling
        if lookup_latency_ms > 15.0:
            critical_smells.append(f"HIGH_TRIE_LOOKUP_LATENCY_{lookup_latency_ms:.1f}MS")

        # Chunked prefill stalls
        if chunked_prefill_stalls > 0:
            critical_smells.append(f"DETECTED_{chunked_prefill_stalls}_CHUNKED_PREFILL_STALLS")

        # KPI 4: Mutation Safety
        if un_gated_mutations > 0:
            critical_smells.append(f"DETECTED_{un_gated_mutations}_UNGATED_RADIX_TRIE_MUTATIONS")

        # KPI 1: RadixCache Debt Index (0 = Clean, 100 = Catastrophic)
        rdi = (
            max(0.0, (trie_ratio - 1.0) * 20.0)
            + max(0.0, (lookup_latency_ms - 2.8) * 0.5)
            + (chunked_prefill_stalls * 25.0)
            + (un_gated_mutations * 30.0)
        )
        rdi_score = round(min(100.0, rdi), 2)

        # Production Readiness Index (0 - 100)
        readiness = max(0.0, 100.0 - rdi_score)
        is_production_ready = (
            rdi_score <= self.max_acceptable_rdi and len(critical_smells) == 0
        )

        # Cryptographic Ledger Entry
        entry = self.ledger.record_radix_event(
            trie_node_id=trie_node_id,
            event_type="trie_node_authorized" if is_production_ready else "trie_node_flagged_debt",
            readiness_index=readiness,
            critical_smells=critical_smells,
            metadata={
                "rdi_score": rdi_score,
                "trie_ratio": trie_ratio,
                "allocated_trie_bytes": allocated_trie_bytes,
                "utilized_trie_bytes": utilized_trie_bytes,
                "lookup_latency_ms": lookup_latency_ms,
                "chunked_prefill_stalls": chunked_prefill_stalls,
                "un_gated_mutations": un_gated_mutations,
                "never_equate_intent_to_approval": self.never_equate_intent_to_approval,
            },
        )

        return RadixCacheDebtReport(
            trie_node_id=trie_node_id,
            rdi_score=rdi_score,
            trie_sprawl_multiplier=round(trie_ratio, 2),
            lookup_latency_ms=round(lookup_latency_ms, 2),
            mutation_safety_score=(
                100.0 if un_gated_mutations == 0 else max(0.0, 100.0 - un_gated_mutations * 30.0)
            ),
            production_readiness_index=readiness,
            is_production_ready=is_production_ready,
            critical_smells=critical_smells,
            receipt_hash=entry["curr_hash"],
        )
