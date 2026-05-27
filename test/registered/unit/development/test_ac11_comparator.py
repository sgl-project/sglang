"""Registered regressions for the AC-11 directional comparator semantics
added to ``development/benchmark_compare.py`` in Round 30.

Per plan §AC-11 / DEC-2:
* DS TPS must be >= 95% of DSA TPS at conc=64.
* DS P99 TTFT must be <= DSA P99 TTFT * 1.10.
* Fixed seed, 120s warmup, 600s measurement, 3 trials per concurrency,
  median across trials.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import pathlib
import sys
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
BC_PATH = REPO_ROOT / "development" / "benchmark_compare.py"


def _load_bc():
    spec = importlib.util.spec_from_file_location("_bc", BC_PATH)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so any @dataclass inside can resolve __module__
    # (per BL-20260527-importlib-dataclass-sys-modules).
    sys.modules["_bc"] = mod
    spec.loader.exec_module(mod)
    return mod


bc = _load_bc()


def _write_bench_jsonl(
    path: str, *,
    concurrency: int,
    tps_p50: float,
    ttft_p99_s: float,
    num_prompts: int = 320,
    input_len: int = 4096,
    output_len: int = 512,
    extra: dict = None,
    sidecar: bool = True,
    sidecar_overrides: dict = None,
    mode: str = "double_sparsity",
    tp_size: int = 8,
    gpu_id: str = "0",
    page_size: int = 64,
    disable_radix_cache: bool = True,
) -> None:
    """Build a minimal bench_serving JSONL with the metrics under test.

    When ``sidecar=True`` (default), also writes a valid AC-11 ``.meta.json``
    sidecar with the timing floors satisfied. Tests that exercise the
    validation gauntlet either suppress the sidecar (``sidecar=False``) or
    pass ``sidecar_overrides`` to inject specific bad fields.
    """
    summary = {
        "max_concurrency": concurrency,
        "num_prompts": num_prompts,
        "input_len": input_len,
        "output_len": output_len,
        "output_throughput_p50": tps_p50,
        "output_throughput_p99": tps_p50 * 1.1,
        "median_ttft_ms": ttft_p99_s * 1000.0 * 0.5,
        "p99_ttft_ms": ttft_p99_s * 1000.0,
        "median_tpot_ms": 5.0,
        "p99_tpot_ms": 12.0,
        "goodput_under_slo": 0.95,
        "selected_tokens_mean": 1024.0,
        "total_tokens_mean": 4096.0,
        "dense_fallback_total": 0,
        "server_info": {
            "gpu_id": gpu_id,
            "tp_size": tp_size,
            "page_size": page_size,
            "disable_radix_cache": disable_radix_cache,
        },
    }
    if extra:
        summary.update(extra)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(summary) + "\n")

    if sidecar:
        sa = {
            "tp_size": tp_size,
            "page_size": page_size,
            "chunked_prefill_size": 8192,
        }
        if mode == "double_sparsity":
            sa["enable_double_sparsity"] = True
            sa["double_sparsity_config"] = "/tmp/x.safetensors"
            sa["disable_radix_cache"] = True
        meta = {
            "commit_sha": "abc123",
            "mode": mode,
            "concurrency": concurrency,
            "seed": 213,
            "num_prompts": num_prompts,
            "isl_total_tokens": input_len,
            "osl_tokens": output_len,
            "timestamp_utc": "2026-05-27T12:00:00Z",
            "chunked_prefill_size": 8192,
            "warmup_seconds": 120.0,
            "measurement_window_seconds": 600.0,
            "trial_id": "1",
            "server_args": sa,
            "server_args_error": None,
        }
        if sidecar_overrides:
            meta.update(sidecar_overrides)
        with open(path + ".meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh)


class TestMedianHelper(unittest.TestCase):

    def test_median_odd(self):
        self.assertEqual(bc._median([1.0, 2.0, 3.0, 4.0, 5.0]), 3.0)

    def test_median_even(self):
        self.assertEqual(bc._median([1.0, 2.0, 3.0, 4.0]), 2.5)

    def test_median_single(self):
        self.assertEqual(bc._median([42.0]), 42.0)

    def test_median_drops_none(self):
        # None entries are skipped, not converted to 0.
        self.assertEqual(bc._median([None, 4.0, 6.0, None]), 5.0)

    def test_median_empty_returns_none(self):
        self.assertIsNone(bc._median([]))

    def test_median_all_none_returns_none(self):
        self.assertIsNone(bc._median([None, None, None]))


class TestMedianMetrics(unittest.TestCase):

    def _row(self, *, concurrency=32, tps=100.0, ttft_p99=10.0, df=0):
        return bc.RunMetrics(
            concurrency=concurrency,
            num_prompts=320, isl=4096, osl=512,
            output_tps_p50=tps, output_tps_p99=tps * 1.1,
            ttft_p50_s=ttft_p99 * 0.5, ttft_p99_s=ttft_p99,
            tpot_p50_ms=5.0, tpot_p99_ms=12.0,
            goodput_under_slo=0.95,
            selected_tokens_mean=1024.0, dense_fallback_total=df,
            total_tokens_mean=4096.0,
        )

    def test_median_metrics_takes_field_medians(self):
        trials = [
            self._row(tps=90.0, ttft_p99=9.0),
            self._row(tps=100.0, ttft_p99=10.0),
            self._row(tps=110.0, ttft_p99=11.0),
        ]
        m = bc._median_metrics(trials)
        self.assertEqual(m.output_tps_p50, 100.0)
        self.assertEqual(m.ttft_p99_s, 10.0)

    def test_median_metrics_sums_dense_fallback(self):
        """dense_fallback_total is a counter — sum, not median."""
        trials = [
            self._row(df=3), self._row(df=5), self._row(df=7),
        ]
        m = bc._median_metrics(trials)
        self.assertEqual(m.dense_fallback_total, 15)

    def test_median_metrics_refuses_concurrency_mismatch(self):
        trials = [
            self._row(concurrency=32),
            self._row(concurrency=64),
            self._row(concurrency=32),
        ]
        with self.assertRaises(ValueError) as ctx:
            bc._median_metrics(trials)
        self.assertIn("concurrency", str(ctx.exception))


class TestGroupByConcurrency(unittest.TestCase):

    def test_group_by_concurrency_from_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, conc in enumerate([16, 32, 32, 64]):
                p = os.path.join(tmp, f"t{i}.jsonl")
                _write_bench_jsonl(p, concurrency=conc, tps_p50=100.0, ttft_p99_s=10.0)
                paths.append(p)
            grouped = bc._group_by_concurrency(paths)
            self.assertEqual(sorted(grouped.keys()), [16, 32, 64])
            self.assertEqual(len(grouped[32]), 2)

    def test_group_by_concurrency_falls_back_to_filename(self):
        # JSONL with no per-row concurrency, but filename has _c64.jsonl.
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "any_c64.jsonl")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"num_prompts": 320}) + "\n")
            grouped = bc._group_by_concurrency([p])
            self.assertEqual(list(grouped.keys()), [64])

    def test_group_by_concurrency_refuses_unresolvable(self):
        # No row concurrency + filename without _c<N>.jsonl suffix.
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "mystery.jsonl")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"num_prompts": 1}) + "\n")
            with self.assertRaises(ValueError) as ctx:
                bc._group_by_concurrency([p])
            self.assertIn("cannot resolve concurrency", str(ctx.exception))


class TestEvaluateAC11Gates(unittest.TestCase):

    def _med(self, *, tps, ttft_p99):
        return bc.RunMetrics(
            concurrency=64, num_prompts=320, isl=4096, osl=512,
            output_tps_p50=tps, output_tps_p99=tps * 1.1,
            ttft_p50_s=ttft_p99 * 0.5, ttft_p99_s=ttft_p99,
            tpot_p50_ms=5.0, tpot_p99_ms=12.0,
            goodput_under_slo=0.95,
            selected_tokens_mean=1024.0, dense_fallback_total=0,
            total_tokens_mean=4096.0,
        )

    def test_tps_pass_at_equality(self):
        dsa = self._med(tps=100.0, ttft_p99=10.0)
        ds = self._med(tps=100.0, ttft_p99=10.0)
        g = bc._evaluate_ac11_gates(dsa, ds)
        self.assertAlmostEqual(g["tps_ratio"], 1.0)
        self.assertTrue(g["tps_pass"])
        self.assertTrue(g["ttft_pass"])
        self.assertEqual(g["reason"], "")

    def test_tps_pass_at_floor(self):
        # 0.95 ratio exactly satisfies the >= 0.95 gate.
        dsa = self._med(tps=100.0, ttft_p99=10.0)
        ds = self._med(tps=95.0, ttft_p99=10.0)
        g = bc._evaluate_ac11_gates(dsa, ds)
        self.assertTrue(g["tps_pass"])

    def test_tps_fail_below_floor(self):
        dsa = self._med(tps=100.0, ttft_p99=10.0)
        ds = self._med(tps=90.0, ttft_p99=10.0)
        g = bc._evaluate_ac11_gates(dsa, ds)
        self.assertFalse(g["tps_pass"])
        self.assertIn("AC-11 TPS gate failed", g["reason"])

    def test_ttft_pass_at_ceiling(self):
        # 1.10 ratio exactly satisfies the <= 1.10 gate.
        dsa = self._med(tps=100.0, ttft_p99=10.0)
        ds = self._med(tps=100.0, ttft_p99=11.0)
        g = bc._evaluate_ac11_gates(dsa, ds)
        self.assertTrue(g["ttft_pass"])

    def test_ttft_fail_above_ceiling(self):
        dsa = self._med(tps=100.0, ttft_p99=10.0)
        ds = self._med(tps=100.0, ttft_p99=12.0)
        g = bc._evaluate_ac11_gates(dsa, ds)
        self.assertFalse(g["ttft_pass"])
        self.assertIn("AC-11 TTFT gate failed", g["reason"])

    def test_missing_data_marks_both_failed(self):
        dsa = self._med(tps=100.0, ttft_p99=10.0)
        # Construct DS with missing TPS directly (helper requires non-None).
        ds = bc.RunMetrics(
            concurrency=64, num_prompts=320, isl=4096, osl=512,
            output_tps_p50=None, output_tps_p99=None,
            ttft_p50_s=5.0, ttft_p99_s=10.0,
            tpot_p50_ms=5.0, tpot_p99_ms=12.0,
            goodput_under_slo=0.95,
            selected_tokens_mean=1024.0, dense_fallback_total=0,
            total_tokens_mean=4096.0,
        )
        g = bc._evaluate_ac11_gates(dsa, ds)
        self.assertFalse(g["tps_pass"])
        self.assertFalse(g["ttft_pass"])
        self.assertIn("missing-data", g["reason"])


class TestAC11EndToEnd(unittest.TestCase):

    def _make_trials(self, tmp, label, conc, tps_values, ttft_values, *,
                     sidecar=True, sidecar_overrides=None, disable_radix=None):
        paths = []
        is_dsa = label.startswith("dsa") or label.startswith("native")
        mode = "native_nsa" if is_dsa else "double_sparsity"
        # DSA baseline runs with radix ON; DS runs with radix OFF (AC-10
        # gap). The comparator allows this via _DS_ONLY_SERVER_ARG_KEYS.
        if disable_radix is None:
            disable_radix = (mode == "double_sparsity")
        for i, (tps, ttft) in enumerate(zip(tps_values, ttft_values)):
            p = os.path.join(tmp, f"{label}_c{conc}_t{i}.jsonl")
            _write_bench_jsonl(
                p, concurrency=conc, tps_p50=tps, ttft_p99_s=ttft,
                mode=mode,
                disable_radix_cache=disable_radix,
                sidecar=sidecar,
                sidecar_overrides=sidecar_overrides,
            )
            paths.append(p)
        return paths

    def _capture_stdout(self, args):
        # Suppress stdout (the AC-11 mode writes Markdown to it). We don't
        # need to inspect the report's exact text here; tests below that
        # do use --output.
        buf = io.StringIO()
        orig = sys.stdout
        sys.stdout = buf
        try:
            code = bc.main(args)
        finally:
            sys.stdout = orig
        return code, buf.getvalue()

    def test_full_pass_exit_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            dsa = self._make_trials(tmp, "dsa", 64, [100, 102, 98], [10.0, 9.8, 10.2])
            ds = self._make_trials(tmp, "ds", 64, [99, 100, 101], [10.0, 9.5, 10.1])
            out = os.path.join(tmp, "report.md")
            jout = os.path.join(tmp, "report.json")
            code = bc.main([
                "--ac11",
                "--ac11-baseline-results", *dsa,
                "--ac11-ds-results", *ds,
                "--output", out,
                "--json-output", jout,
            ])
            self.assertEqual(code, 0)
            md = open(out).read()
            self.assertIn("AC-11 verdict: PASS", md)
            payload = json.loads(open(jout).read())
            self.assertEqual(payload["verdict"], "PASS")
            self.assertIn("64", payload["per_concurrency"])

    def test_tps_gate_fail_exit_3(self):
        with tempfile.TemporaryDirectory() as tmp:
            dsa = self._make_trials(tmp, "dsa", 64, [100, 102, 98], [10.0, 9.8, 10.2])
            # DS TPS = 50% of DSA → fails the 0.95 floor.
            ds = self._make_trials(tmp, "ds", 64, [50, 50, 50], [10.0, 10.0, 10.0])
            out = os.path.join(tmp, "report.md")
            code = bc.main([
                "--ac11",
                "--ac11-baseline-results", *dsa,
                "--ac11-ds-results", *ds,
                "--output", out,
            ])
            self.assertEqual(code, 3)
            md = open(out).read()
            self.assertIn("AC-11 verdict: FAIL", md)
            self.assertIn("AC-11 TPS gate failed", md)
            self.assertIn("Profiling obligation", md)

    def test_ttft_gate_fail_exit_3(self):
        with tempfile.TemporaryDirectory() as tmp:
            dsa = self._make_trials(tmp, "dsa", 64, [100, 100, 100], [10.0, 10.0, 10.0])
            # DS TTFT = 1.5× DSA → fails the 1.10 ceiling.
            ds = self._make_trials(tmp, "ds", 64, [100, 100, 100], [15.0, 15.0, 15.0])
            out = os.path.join(tmp, "report.md")
            code = bc.main([
                "--ac11",
                "--ac11-baseline-results", *dsa,
                "--ac11-ds-results", *ds,
                "--output", out,
            ])
            self.assertEqual(code, 3)
            md = open(out).read()
            self.assertIn("AC-11 TTFT gate failed", md)

    def test_too_few_trials_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            dsa = self._make_trials(tmp, "dsa", 64, [100, 100], [10.0, 10.0])  # 2 trials
            ds = self._make_trials(tmp, "ds", 64, [100, 100, 100], [10.0, 10.0, 10.0])
            code, _ = self._capture_stdout([
                "--ac11",
                "--ac11-baseline-results", *dsa,
                "--ac11-ds-results", *ds,
            ])
            self.assertEqual(code, 2)

    def test_concurrency_set_mismatch_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            # DSA at 32; DS at 64 → concurrency sets disagree.
            dsa = self._make_trials(tmp, "dsa", 32, [100, 100, 100], [10.0, 10.0, 10.0])
            ds = self._make_trials(tmp, "ds", 64, [100, 100, 100], [10.0, 10.0, 10.0])
            code, _ = self._capture_stdout([
                "--ac11",
                "--ac11-baseline-results", *dsa,
                "--ac11-ds-results", *ds,
            ])
            self.assertEqual(code, 2)

    # ---- Round 31: validation gauntlet refusals -----------------------

    def _ac11_main(self, dsa: List[str], ds: List[str], *, extra=None) -> int:
        """Run main() with stdout suppressed. Returns the exit code."""
        args = [
            "--ac11",
            "--ac11-baseline-results", *dsa,
            "--ac11-ds-results", *ds,
        ]
        if extra:
            args.extend(extra)
        buf = io.StringIO()
        orig = sys.stdout
        sys.stdout = buf
        try:
            return bc.main(args)
        finally:
            sys.stdout = orig

    def test_missing_sidecar_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            dsa = self._make_trials(tmp, "dsa", 64, [100, 100, 100], [10, 10, 10])
            # DS side without sidecars.
            ds = self._make_trials(tmp, "ds", 64, [100, 100, 100], [10, 10, 10],
                                    sidecar=False)
            self.assertEqual(self._ac11_main(dsa, ds), 2)

    def test_malformed_sidecar_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            dsa = self._make_trials(tmp, "dsa", 64, [100, 100, 100], [10, 10, 10])
            ds = self._make_trials(tmp, "ds", 64, [100, 100, 100], [10, 10, 10])
            # Corrupt the first DS sidecar.
            with open(ds[0] + ".meta.json", "w") as fh:
                fh.write("{not json")
            self.assertEqual(self._ac11_main(dsa, ds), 2)

    def test_sidecar_server_args_error_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            dsa = self._make_trials(tmp, "dsa", 64, [100, 100, 100], [10, 10, 10])
            ds = self._make_trials(
                tmp, "ds", 64, [100, 100, 100], [10, 10, 10],
                sidecar_overrides={"server_args_error": "simulated"},
            )
            self.assertEqual(self._ac11_main(dsa, ds), 2)

    def test_short_warmup_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            dsa = self._make_trials(tmp, "dsa", 64, [100, 100, 100], [10, 10, 10])
            ds = self._make_trials(
                tmp, "ds", 64, [100, 100, 100], [10, 10, 10],
                sidecar_overrides={"warmup_seconds": 30.0},  # < 120
            )
            self.assertEqual(self._ac11_main(dsa, ds), 2)

    def test_short_measurement_window_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            dsa = self._make_trials(tmp, "dsa", 64, [100, 100, 100], [10, 10, 10])
            ds = self._make_trials(
                tmp, "ds", 64, [100, 100, 100], [10, 10, 10],
                sidecar_overrides={"measurement_window_seconds": 60.0},
            )
            self.assertEqual(self._ac11_main(dsa, ds), 2)

    def test_seed_mismatch_within_side_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Manually build a side where one trial's seed differs.
            dsa = self._make_trials(tmp, "dsa", 64, [100, 100, 100], [10, 10, 10])
            ds_paths = []
            for i, seed in enumerate([213, 999, 213]):
                p = os.path.join(tmp, f"ds_c64_t{i}.jsonl")
                _write_bench_jsonl(
                    p, concurrency=64, tps_p50=100, ttft_p99_s=10,
                    sidecar_overrides={"seed": seed}, mode="double_sparsity",
                )
                ds_paths.append(p)
            self.assertEqual(self._ac11_main(dsa, ds_paths), 2)

    def test_seed_mismatch_across_sides_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            dsa = self._make_trials(tmp, "dsa", 64, [100, 100, 100], [10, 10, 10],
                                    sidecar_overrides={"seed": 1000})
            ds = self._make_trials(tmp, "ds", 64, [100, 100, 100], [10, 10, 10],
                                   sidecar_overrides={"seed": 2000})
            self.assertEqual(self._ac11_main(dsa, ds), 2)

    def test_commit_sha_mismatch_across_sides_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            dsa = self._make_trials(tmp, "dsa", 64, [100, 100, 100], [10, 10, 10],
                                    sidecar_overrides={"commit_sha": "v1"})
            ds = self._make_trials(tmp, "ds", 64, [100, 100, 100], [10, 10, 10],
                                   sidecar_overrides={"commit_sha": "v2"})
            self.assertEqual(self._ac11_main(dsa, ds), 2)

    def test_chunked_prefill_size_mismatch_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            dsa = self._make_trials(tmp, "dsa", 64, [100, 100, 100], [10, 10, 10],
                                    sidecar_overrides={"chunked_prefill_size": 4096})
            ds = self._make_trials(tmp, "ds", 64, [100, 100, 100], [10, 10, 10],
                                   sidecar_overrides={"chunked_prefill_size": 8192})
            self.assertEqual(self._ac11_main(dsa, ds), 2)

    def test_workload_num_prompts_mismatch_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            dsa_paths = []
            ds_paths = []
            for i in range(3):
                p = os.path.join(tmp, f"dsa_c64_t{i}.jsonl")
                _write_bench_jsonl(p, concurrency=64, tps_p50=100, ttft_p99_s=10,
                                   num_prompts=320, mode="native_nsa",
                                   disable_radix_cache=False)
                dsa_paths.append(p)
                p2 = os.path.join(tmp, f"ds_c64_t{i}.jsonl")
                _write_bench_jsonl(p2, concurrency=64, tps_p50=100, ttft_p99_s=10,
                                   num_prompts=160,  # DS differs
                                   mode="double_sparsity")
                ds_paths.append(p2)
            self.assertEqual(self._ac11_main(dsa_paths, ds_paths), 2)

    def test_server_args_mismatch_tp_size_exit_2(self):
        """TP size differs between DSA and DS — not in the DS-only allowlist."""
        with tempfile.TemporaryDirectory() as tmp:
            dsa = self._make_trials(tmp, "dsa", 64, [100, 100, 100], [10, 10, 10])
            # DS with tp_size=1 (DSA is 8) — should refuse.
            ds_paths = []
            for i in range(3):
                p = os.path.join(tmp, f"ds_c64_t{i}.jsonl")
                _write_bench_jsonl(
                    p, concurrency=64, tps_p50=100, ttft_p99_s=10,
                    tp_size=1, mode="double_sparsity",
                )
                ds_paths.append(p)
            self.assertEqual(self._ac11_main(dsa, ds_paths), 2)

    def test_allowed_ds_only_differences_still_pass(self):
        """DS has enable_double_sparsity + disable_radix_cache; DSA doesn't.
        Both are in _DS_ONLY_SERVER_ARG_KEYS so the comparator must PASS."""
        with tempfile.TemporaryDirectory() as tmp:
            # _make_trials with default mode logic: DSA → native_nsa (no DS
            # flags, radix on), DS → double_sparsity (DS flags, radix off).
            dsa = self._make_trials(tmp, "dsa", 64, [100, 100, 100], [10, 10, 10])
            ds = self._make_trials(tmp, "ds", 64, [100, 100, 100], [10, 10, 10])
            self.assertEqual(self._ac11_main(dsa, ds), 0)

    def test_missing_tps_metric_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            dsa = self._make_trials(tmp, "dsa", 64, [100, 100, 100], [10, 10, 10])
            # Build DS trials with no `output_throughput_p50` field.
            ds_paths = []
            for i in range(3):
                p = os.path.join(tmp, f"ds_c64_t{i}.jsonl")
                with open(p, "w") as fh:
                    fh.write(json.dumps({
                        "max_concurrency": 64, "num_prompts": 320,
                        "input_len": 4096, "output_len": 512,
                        # Missing output_throughput_p50 + p99_ttft_ms.
                        "median_ttft_ms": 5000.0,
                        "server_info": {"gpu_id": "0", "tp_size": 8,
                                         "page_size": 64, "disable_radix_cache": True},
                    }) + "\n")
                # Valid sidecar so the failure is the metric absence.
                meta = {
                    "commit_sha": "abc123", "mode": "double_sparsity",
                    "concurrency": 64, "seed": 213, "num_prompts": 320,
                    "isl_total_tokens": 4096, "osl_tokens": 512,
                    "timestamp_utc": "2026-05-27T12:00:00Z",
                    "chunked_prefill_size": 8192,
                    "warmup_seconds": 120.0, "measurement_window_seconds": 600.0,
                    "trial_id": str(i), "server_args": {"tp_size": 8, "page_size": 64,
                                                          "chunked_prefill_size": 8192,
                                                          "enable_double_sparsity": True,
                                                          "double_sparsity_config": "/tmp/x.safetensors",
                                                          "disable_radix_cache": True},
                    "server_args_error": None,
                }
                with open(p + ".meta.json", "w") as fh:
                    json.dump(meta, fh)
                ds_paths.append(p)
            self.assertEqual(self._ac11_main(dsa, ds_paths), 2)

    def test_malformed_jsonl_exits_2_not_traceback(self):
        """Round 30 review counter-evidence: malformed *_c64.jsonl raised
        an uncaught JSONDecodeError. Round 31 must report clean exit 2."""
        with tempfile.TemporaryDirectory() as tmp:
            dsa = self._make_trials(tmp, "dsa", 64, [100, 100, 100], [10, 10, 10])
            # Build three "DS" trial files but with malformed JSON.
            ds_paths = []
            for i in range(3):
                p = os.path.join(tmp, f"ds_c64_t{i}.jsonl")
                with open(p, "w") as fh:
                    fh.write("not-json-at-all\n")
                ds_paths.append(p)
            self.assertEqual(self._ac11_main(dsa, ds_paths), 2)

    def test_legacy_mode_still_works(self):
        """The single-trial AC-7/AC-8 report path must still work after Round 30."""
        with tempfile.TemporaryDirectory() as tmp:
            b = os.path.join(tmp, "baseline.jsonl")
            d = os.path.join(tmp, "ds.jsonl")
            _write_bench_jsonl(b, concurrency=64, tps_p50=42.0, ttft_p99_s=2.0)
            _write_bench_jsonl(d, concurrency=64, tps_p50=42.0, ttft_p99_s=2.0)
            out = os.path.join(tmp, "report.md")
            code = bc.main(["--baseline", b, "--ds", d, "--output", out])
            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
