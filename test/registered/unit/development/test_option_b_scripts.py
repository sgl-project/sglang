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

    def test_dsa_server_does_not_disable_radix_cache(self):
        """Per plan §13 the DSA baseline runs with radix cache ON, so the
        DS-vs-DSA TPS gap reflects DS configuration alone, not the AC-10
        radix-cache gate that DS still has to clear."""
        text = _non_comment_lines(DSA_SERVER)
        self.assertNotIn(
            "--disable-radix-cache", text,
            "serve_native_nsa.sh must NOT pass --disable-radix-cache "
            "(DSA baseline runs with radix on per plan §13).",
        )

    def test_ds_server_does_disable_radix_cache_until_ac10(self):
        """The DS launcher keeps --disable-radix-cache until AC-10 passes."""
        text = _non_comment_lines(DS_SERVER)
        self.assertIn(
            "--disable-radix-cache", text,
            "serve_double_sparsity.sh must keep --disable-radix-cache "
            "until AC-10 (radix-cache fixture) passes.",
        )


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
