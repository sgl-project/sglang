"""Unit tests for the AC-4 SLO gate in development/benchmark_compare.py.

The comparator must gate on the resolved per-request decode-throughput metric
`median_decode_throughput_tps` (= output_tokens / (e2e - ttft)) that
bench_serving emits, NOT silently fall back to a differently-derived number
when that field is present. These tests pin: (a) a current task9-style artifact
carrying only the new decode field gates correctly (not missing-data); (b) a
contradictory legacy field does not override the new decode field; (c) a
new-style artifact missing the decode field AND the derivation arrays fails
closed; (d) the strict `P99 TTFT < 22 s` boundary; (e) legacy fixtures still parse.

The comparator lives under development/ (not the package) and defines
@dataclass classes, so it is loaded by file path with sys.modules registered
before exec_module (else the dataclass decorator AttributeErrors).

    python -m pytest test/registered/unit/test_benchmark_compare_slo.py -v
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest


def _load_comparator():
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..", "development", "benchmark_compare.py",
        )
    )
    spec = importlib.util.spec_from_file_location("_bench_compare_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_bench_compare_under_test"] = mod  # before exec (dataclass contract)
    spec.loader.exec_module(mod)
    return mod


BC = _load_comparator()


def _verdict_from_row(row: dict) -> tuple:
    """Write a one-row bench JSONL, read it back, return (metrics, verdict)."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "native_nsa_gsp_isl4096_osl512_c64.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        _ctx, metrics = BC._read_bench_jsonl(p)
    return metrics, BC._slo_verdict(metrics)


class TestDecodeTpsSloGate(unittest.TestCase):
    def test_new_decode_field_gates_pass(self):
        # task9-style row: only the new decode field + p99 ttft, no legacy/arrays.
        m, verdict = _verdict_from_row(
            {
                "max_concurrency": 64,
                "median_decode_throughput_tps": 73.0,
                "p99_ttft_ms": 21000.0,
            }
        )
        self.assertEqual(m.output_tps_p50, 73.0)
        self.assertAlmostEqual(m.ttft_p99_s, 21.0)
        self.assertEqual(verdict, "pass")  # NOT missing-data

    def test_new_field_below_floor_fails(self):
        m, verdict = _verdict_from_row(
            {"max_concurrency": 64, "median_decode_throughput_tps": 12.0, "p99_ttft_ms": 5000.0}
        )
        self.assertEqual(m.output_tps_p50, 12.0)
        self.assertEqual(verdict, "fail")

    def test_legacy_field_does_not_override_new_decode_field(self):
        # Contradictory low legacy scalar must NOT win over the new decode field.
        m, _ = _verdict_from_row(
            {
                "max_concurrency": 64,
                "median_decode_throughput_tps": 73.0,
                "per_req_output_tps_p50": 5.0,
                "output_throughput_p50": 6.0,
                "p99_ttft_ms": 10000.0,
            }
        )
        self.assertEqual(m.output_tps_p50, 73.0)

    def test_missing_decode_field_and_arrays_fails_closed(self):
        # New-style artifact missing the decode field, the --output-details
        # arrays, AND legacy scalars → unmeasurable → missing-data (fail closed).
        m, verdict = _verdict_from_row({"max_concurrency": 64, "p99_ttft_ms": 21000.0})
        self.assertIsNone(m.output_tps_p50)
        self.assertEqual(verdict, "missing-data")

    def test_strict_22s_ttft_boundary_fails(self):
        # Plan bar is strict "< 22 s"; exactly 22.0 s must FAIL.
        _m, verdict = _verdict_from_row(
            {"max_concurrency": 64, "median_decode_throughput_tps": 73.0, "p99_ttft_ms": 22000.0}
        )
        self.assertEqual(verdict, "fail")
        # Just under the bar passes.
        _m2, verdict2 = _verdict_from_row(
            {"max_concurrency": 64, "median_decode_throughput_tps": 73.0, "p99_ttft_ms": 21999.0}
        )
        self.assertEqual(verdict2, "pass")

    def test_legacy_fixture_still_parses(self):
        # No decode field, no arrays → legacy scalar fallback preserved.
        m, verdict = _verdict_from_row(
            {"max_concurrency": 64, "per_req_output_tps_p50": 40.0, "p99_ttft_ms": 10000.0}
        )
        self.assertEqual(m.output_tps_p50, 40.0)
        self.assertEqual(verdict, "pass")

    def test_output_details_arrays_still_derive(self):
        # When the new field is absent but --output-details arrays exist, the
        # derivation path (output_lens / sum(itls)) still produces a value.
        m, verdict = _verdict_from_row(
            {
                "max_concurrency": 64,
                "output_lens": [100, 100],
                "itls": [[0.01] * 100, [0.01] * 100],  # 100 tok / 1.0 s = 100 tok/s
                "p99_ttft_ms": 5000.0,
            }
        )
        self.assertIsNotNone(m.output_tps_p50)
        self.assertAlmostEqual(m.output_tps_p50, 100.0, places=1)
        self.assertEqual(verdict, "pass")


if __name__ == "__main__":
    unittest.main()
