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

import importlib.util
import os
import sys
import unittest

# Load module directly
file_path = os.path.join(
    os.path.dirname(__file__),
    "../../python/sglang/srt/mem_cache/production_debt.py",
)
spec = importlib.util.spec_from_file_location("sglang_radix_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["sglang_radix_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtRadixCacheGate = production_debt_mod.ProductionDebtRadixCacheGate
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtRadixCacheGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ProductionDebtRadixCacheGate(
            never_equate_intent_to_approval=True,
            max_acceptable_rdi=12.0,
        )

    def test_clean_trie_eviction_passes_readiness(self) -> None:
        report = self.gate.evaluate_trie_eviction(
            trie_node_id="sglang_radix_trie_prefix_node",
            allocated_trie_bytes=16000000000,
            utilized_trie_bytes=16800000000,
            lookup_latency_ms=2.1,
            chunked_prefill_stalls=0,
            un_gated_mutations=0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.rdi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_trie_eviction_fails_debt(self) -> None:
        report = self.gate.evaluate_trie_eviction(
            trie_node_id="uncalibrated_radix_trie_node",
            allocated_trie_bytes=16000000000,
            utilized_trie_bytes=45000000000,  # 2.81x trie tree fragmentation sprawl
            lookup_latency_ms=28.0,  # High trie lookup latency
            chunked_prefill_stalls=3,  # 3 chunked prefill stalls
            un_gated_mutations=2,  # 2 un-gated mutations
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.rdi_score, 50.0)
        self.assertIn("HIGH_RADIX_TRIE_FRAGMENTATION_2.81X", report.critical_smells)
        self.assertIn("HIGH_TRIE_LOOKUP_LATENCY_28.0MS", report.critical_smells)
        self.assertIn("DETECTED_3_CHUNKED_PREFILL_STALLS", report.critical_smells)
        self.assertIn("DETECTED_2_UNGATED_RADIX_TRIE_MUTATIONS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.gate.evaluate_trie_eviction("node-1")
        self.gate.evaluate_trie_eviction("node-2")
        self.gate.evaluate_trie_eviction("node-3")

        entries = self.gate.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.gate.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
