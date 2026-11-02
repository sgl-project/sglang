import importlib.util
import os
import sys
import unittest

# Load module directly
file_path = os.path.join(
    os.path.dirname(__file__),
    "../../python/sglang/srt/production_debt.py",
)
spec = importlib.util.spec_from_file_location("sglang_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["sglang_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtInferenceGate = production_debt_mod.ProductionDebtInferenceGate
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtInferenceGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ProductionDebtInferenceGate(
            never_equate_intent_to_approval=True,
            max_acceptable_sdi=12.0,
        )

    def test_clean_batch_passes_readiness(self) -> None:
        report = self.gate.evaluate_inference_batch(
            server_id="sglang_deepseek_cluster_node_01",
            allocated_kv_bytes=10000000000,
            peak_radix_kv_bytes=10400000000,
            ttft_latency_seconds=0.45,
            speculative_rollback_count=0,
            un_gated_mutations=0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.sdi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_batch_fails_debt(self) -> None:
        report = self.gate.evaluate_inference_batch(
            server_id="unoptimized_radix_sprawl_node",
            allocated_kv_bytes=10000000000,
            peak_radix_kv_bytes=24000000000,  # High KV cache sprawl (2.4x)
            ttft_latency_seconds=3.5,  # High TTFT latency
            speculative_rollback_count=4,  # 4 speculative rollbacks
            un_gated_mutations=2,  # 2 un-gated mutations
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.sdi_score, 50.0)
        self.assertIn("HIGH_RADIX_KV_CACHE_SPRAWL_2.40X", report.critical_smells)
        self.assertIn("HIGH_TTFT_LATENCY_3.50S", report.critical_smells)
        self.assertIn("DETECTED_4_SPECULATIVE_ROLLBACKS", report.critical_smells)
        self.assertIn("DETECTED_2_UNGATED_PROGRAM_MUTATIONS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.gate.evaluate_inference_batch("server-1")
        self.gate.evaluate_inference_batch("server-2")
        self.gate.evaluate_inference_batch("server-3")

        entries = self.gate.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.gate.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
