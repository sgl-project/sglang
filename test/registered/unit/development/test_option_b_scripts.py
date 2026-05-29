"""Registered regression for the locked Option B contract.

Per plan §13 / DEC-1 the AC-8 / AC-9 / AC-11 comparison must run both DS
and DSA at the SAME locked operating point (FP8 KV cache, ``flashmla_kv``
backends, overlap off, piecewise CUDA graph off, page size 64), and the
benchmark sweep must cover concurrency 16 / 32 / 64. The reference
launchers live at ``development/serve_double_sparsity.sh`` (DS) and
``development/serve_native_nsa.sh`` (DSA), and the benchmark sweeps at
``development/benchmark.sh`` (DS) and ``development/benchmark_baseline.sh``
(DSA).

This test reads the four shell scripts as text and locks in the Option B
contract so a future "small cleanup" doesn't quietly invalidate
AC-8/AC-9/AC-11 evidence.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
DS_SERVER = REPO_ROOT / "development" / "serve_double_sparsity.sh"
DSA_SERVER = REPO_ROOT / "development" / "serve_native_nsa.sh"
DS_BENCH = REPO_ROOT / "development" / "benchmark.sh"
DSA_BENCH = REPO_ROOT / "development" / "benchmark_baseline.sh"

LOCKED_OPTION_B_FLAGS = (
    "--dsa-prefill-backend flashmla_kv",
    "--dsa-decode-backend flashmla_kv",
    "--disable-overlap-schedule",
    "--disable-piecewise-cuda-graph",
)


def _non_comment_lines(path: Path) -> str:
    """Return the script body with bash # comment lines stripped.

    Inline trailing ``# ...`` comments are intentionally kept (a flag on
    the same line as a comment still counts as present).
    """
    out_lines = []
    for line in path.read_text().splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#") and not stripped.startswith("#!"):
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


class TestOptionBLockedFlagsServerScripts(unittest.TestCase):
    """Both DS and DSA launchers must carry the same Option B locked flags."""

    def test_all_four_scripts_exist(self):
        for p in (DS_SERVER, DSA_SERVER, DS_BENCH, DSA_BENCH):
            self.assertTrue(p.exists(), f"missing script: {p}")

    def test_ds_server_has_locked_flags(self):
        text = _non_comment_lines(DS_SERVER)
        for flag in LOCKED_OPTION_B_FLAGS:
            self.assertIn(
                flag, text,
                f"serve_double_sparsity.sh missing locked Option B flag {flag!r}",
            )

    def test_dsa_server_has_locked_flags(self):
        text = _non_comment_lines(DSA_SERVER)
        for flag in LOCKED_OPTION_B_FLAGS:
            self.assertIn(
                flag, text,
                f"serve_native_nsa.sh missing locked Option B flag {flag!r}",
            )

    def test_dsa_server_radix_on_by_default_with_smoke_knob(self):
        """The DSA baseline runs with radix cache ON by default so the
        DS-vs-DSA TPS gap reflects DS configuration alone. A
        ``DISABLE_RADIX_CACHE=1`` knob may add ``--disable-radix-cache``
        for the radix-off smoke parity run, but only inside that guard —
        never unconditionally."""
        text = _non_comment_lines(DSA_SERVER)
        # Default off => radix cache stays ON unless the operator opts in.
        self.assertIn(
            'DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-0}"', text,
            "serve_native_nsa.sh must default the radix-off knob to 0 "
            "(radix cache ON by default).",
        )
        # --disable-radix-cache appears only inside the knob guard.
        self.assertIn('if [[ "${DISABLE_RADIX_CACHE}" == "1" ]]; then', text)
        self.assertIn('RADIX_CACHE_ARG="--disable-radix-cache"', text)

    def test_ds_server_radix_off_by_default(self):
        """The DS launcher serves radix-off by default (the radix-cache
        validator gate is satisfied only via the fixture-artifact path
        below)."""
        text = _non_comment_lines(DS_SERVER)
        self.assertIn(
            "RADIX_ARGS=(--disable-radix-cache)", text,
            "serve_double_sparsity.sh default branch must keep "
            "--disable-radix-cache (radix-off) when no fixture artifact "
            "is provided.",
        )

    def test_both_servers_have_host_knob_defaulting_localhost(self):
        """Both launchers expose a HOST env knob (default 127.0.0.1) and
        pass it through as ``--host`` so the AC-12 two-node quality gate can
        bind the baseline to 0.0.0.0 for cross-node reach, without changing
        the default localhost-only behavior or the locked Option B flags."""
        for path, name in ((DS_SERVER, "serve_double_sparsity.sh"),
                           (DSA_SERVER, "serve_native_nsa.sh")):
            text = _non_comment_lines(path)
            self.assertIn(
                'HOST="${HOST:-127.0.0.1}"', text,
                f"{name} must default HOST to 127.0.0.1 (localhost-only).",
            )
            self.assertIn(
                '--host "${HOST}"', text,
                f"{name} must pass --host \"${{HOST}}\" to launch_server.",
            )

    def test_ds_server_artifact_driven_radix_on(self):
        """Radix cache ON is enabled via a config-bound fixture-passed
        state file (``RADIX_FIXTURE_ARTIFACT`` ->
        ``--double-sparsity-radix-fixture-artifact``), which the validator
        re-verifies before serving. No environment override and no fixed
        edit-point marker is required."""
        text = _non_comment_lines(DS_SERVER)
        self.assertIn('RADIX_FIXTURE_ARTIFACT="${RADIX_FIXTURE_ARTIFACT:-}"', text)
        self.assertIn(
            "--double-sparsity-radix-fixture-artifact", text,
            "serve_double_sparsity.sh must pass the fixture artifact when "
            "RADIX_FIXTURE_ARTIFACT is set (the radix-on path).",
        )
        self.assertIn('"${RADIX_ARGS[@]}"', text)


class TestOptionBBenchmarkSweeps(unittest.TestCase):
    """Both benchmark scripts must default to conc 16 / 32 / 64 and emit a
    metadata sidecar."""

    _CONC_DEFAULT_RE = re.compile(
        r'CONCURRENCIES\s*=\s*"\$\{CONCURRENCIES:-16\s+32\s+64\}"'
    )

    def test_ds_bench_defaults_to_three_concurrencies(self):
        text = DS_BENCH.read_text()
        self.assertRegex(
            text, self._CONC_DEFAULT_RE,
            "benchmark.sh must default CONCURRENCIES to '16 32 64'.",
        )

    def test_dsa_bench_defaults_to_three_concurrencies(self):
        text = DSA_BENCH.read_text()
        self.assertRegex(
            text, self._CONC_DEFAULT_RE,
            "benchmark_baseline.sh must default CONCURRENCIES to '16 32 64'.",
        )

    def test_ds_bench_emits_meta_sidecar(self):
        text = _non_comment_lines(DS_BENCH)
        self.assertIn(".meta.json", text,
                      "benchmark.sh must emit a .meta.json sidecar.")

    def test_dsa_bench_emits_meta_sidecar(self):
        text = _non_comment_lines(DSA_BENCH)
        self.assertIn(".meta.json", text,
                      "benchmark_baseline.sh must emit a .meta.json sidecar.")

    # ---- Round 31: AC-11 3-trial loop + timing env ------------------

    def test_ds_bench_has_three_trial_loop(self):
        text = _non_comment_lines(DS_BENCH)
        self.assertIn('TRIALS="${TRIALS:-3}"', text,
                      "benchmark.sh must default TRIALS=3 for AC-11 sweep.")
        self.assertIn('for TRIAL_ID in $(seq 1 "${TRIALS}")', text,
                      "benchmark.sh must iterate trial_id from 1 to TRIALS.")
        # Output filename must include _t${TRIAL_ID}.
        self.assertIn('_c${CONCURRENCY}_t${TRIAL_ID}.jsonl', text,
                      "benchmark.sh trial outputs must include _t${TRIAL_ID} suffix.")

    def test_dsa_bench_has_three_trial_loop(self):
        text = _non_comment_lines(DSA_BENCH)
        self.assertIn('TRIALS="${TRIALS:-3}"', text)
        self.assertIn('for TRIAL_ID in $(seq 1 "${TRIALS}")', text)
        self.assertIn('_c${CONCURRENCY}_t${TRIAL_ID}.jsonl', text)

    def test_ds_bench_defaults_warmup_and_window(self):
        text = _non_comment_lines(DS_BENCH)
        self.assertIn('WARMUP_SECONDS="${WARMUP_SECONDS:-120}"', text,
                      "benchmark.sh must default WARMUP_SECONDS=120 per AC-11.")
        self.assertIn('MEASUREMENT_WINDOW_S="${MEASUREMENT_WINDOW_S:-600}"', text,
                      "benchmark.sh must default MEASUREMENT_WINDOW_S=600 per AC-11.")

    def test_dsa_bench_defaults_warmup_and_window(self):
        text = _non_comment_lines(DSA_BENCH)
        self.assertIn('WARMUP_SECONDS="${WARMUP_SECONDS:-120}"', text)
        self.assertIn('MEASUREMENT_WINDOW_S="${MEASUREMENT_WINDOW_S:-600}"', text)

    # ---- Round 33: AC-11 producer-side timing enforcement -----------

    def test_ds_bench_passes_warmup_seconds_flag(self):
        text = _non_comment_lines(DS_BENCH)
        self.assertIn('--warmup-seconds "${WARMUP_SECONDS}"', text,
                      "benchmark.sh must pass --warmup-seconds to bench_serving "
                      "so the AC-11 120s warmup is actually run, not just "
                      "metadata.")

    def test_dsa_bench_passes_warmup_seconds_flag(self):
        text = _non_comment_lines(DSA_BENCH)
        self.assertIn('--warmup-seconds "${WARMUP_SECONDS}"', text)

    def test_ds_bench_passes_measurement_window_seconds_flag(self):
        text = _non_comment_lines(DS_BENCH)
        self.assertIn('--measurement-window-seconds "${MEASUREMENT_WINDOW_S}"',
                      text,
                      "benchmark.sh must pass --measurement-window-seconds "
                      "to bench_serving so the AC-11 600s measured window "
                      "is enforced at the producer, not just at the "
                      "comparator.")

    def test_dsa_bench_passes_measurement_window_seconds_flag(self):
        text = _non_comment_lines(DSA_BENCH)
        self.assertIn('--measurement-window-seconds "${MEASUREMENT_WINDOW_S}"',
                      text)

    def test_ds_bench_fails_on_short_observed_duration(self):
        """benchmark.sh must refuse to publish an AC-11 artifact whose
        observed JSONL `duration` falls short of MEASUREMENT_WINDOW_S —
        guards against bench_serving bailing out before the time loop
        met its threshold."""
        text = _non_comment_lines(DS_BENCH)
        self.assertIn("OBSERVED_DURATION", text,
                      "benchmark.sh must inspect the JSONL duration.")
        self.assertIn("MEASUREMENT_WINDOW_S", text)
        self.assertIn("refusing to publish AC-11 artifact", text,
                      "benchmark.sh must fail loudly when duration < window.")

    def test_dsa_bench_fails_on_short_observed_duration(self):
        text = _non_comment_lines(DSA_BENCH)
        self.assertIn("OBSERVED_DURATION", text)
        self.assertIn("MEASUREMENT_WINDOW_S", text)
        self.assertIn("refusing to publish AC-11 artifact", text)

    def test_ds_bench_uses_meta_writer_helper(self):
        """The Round 23 inline heredoc spliced JSON as Python source and
        crashed on real `/get_server_info` `true`/`false`/`null`. Round 24
        extracted the writer into `_bench_meta_writer.py`. Lock that here
        so a future refactor can't reintroduce the heredoc bug."""
        text = _non_comment_lines(DS_BENCH)
        self.assertIn("_bench_meta_writer.py", text,
                      "benchmark.sh must invoke development/_bench_meta_writer.py.")
        # Forbid the unsafe inline heredoc pattern.
        self.assertNotIn("PYEOF", text,
                         "benchmark.sh must not splice JSON via inline heredoc.")
        self.assertNotIn('server_args": ${SERVER_ARGS_JSON', text,
                         "benchmark.sh must not interpolate JSON as Python source.")
        # SERVER_ARGS_JSON must be passed as ENV var to the helper, not
        # spliced into source.
        self.assertIn('SERVER_ARGS_JSON="${SERVER_ARGS_JSON}"', text,
                      "benchmark.sh must pass SERVER_ARGS_JSON via env var.")

    def test_dsa_bench_uses_meta_writer_helper(self):
        text = _non_comment_lines(DSA_BENCH)
        self.assertIn("_bench_meta_writer.py", text,
                      "benchmark_baseline.sh must invoke development/_bench_meta_writer.py.")
        self.assertNotIn("PYEOF", text)
        self.assertNotIn('server_args": ${SERVER_ARGS_JSON', text)
        self.assertIn('SERVER_ARGS_JSON="${SERVER_ARGS_JSON}"', text)


class TestOptionBScriptsSyntax(unittest.TestCase):
    """All four scripts must be syntactically valid bash."""

    @unittest.skipUnless(shutil.which("bash"), "bash unavailable")
    def test_bash_parses_all_four_scripts(self):
        for p in (DS_SERVER, DSA_SERVER, DS_BENCH, DSA_BENCH):
            proc = subprocess.run(
                ["bash", "-n", str(p)],
                capture_output=True, text=True,
            )
            self.assertEqual(
                proc.returncode, 0,
                f"bash -n failed on {p.name}:\n{proc.stderr}",
            )


if __name__ == "__main__":
    unittest.main()
