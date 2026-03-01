"""E2E test: source patcher + dumper + comparator on SGLang server.

Patches Qwen3DecoderLayer.forward to insert dumper.dump() calls,
launches 1-GPU baseline and 2-GPU TP=2 target servers, runs inference,
verifies patched dump fields exist, then runs comparator to verify
numerical consistency.

The dumper.apply_source_patches() auto-injects ``from ... import dumper``
so the YAML only needs ``dumper.dump(...)`` calls.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest
import requests

from sglang.srt.debug_utils.comparator.output_types import (
    AnyRecord,
    SummaryRecord,
    parse_record_json,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

register_cuda_ci(est_time=120, suite="nightly-2-gpu", nightly=True)

MODEL = "Qwen/Qwen3-0.6B"
EXP_NAME = "e2e_source_patcher"
DUMPER_FILTER = r"layer_id=[012]"

PATCH_CONFIG_YAML: str = """\
patches:
  - target: sglang.srt.models.qwen3.Qwen3DecoderLayer.forward
    edits:
      - match: "hidden_states = self.mlp(hidden_states)"
        prepend: "dumper.dump('patched_attn_output', hidden_states, dims='t h')"
      - match: "return hidden_states, residual"
        prepend: "dumper.dump('patched_mlp_output', hidden_states, dims='t h')"
"""


class TestSourcePatcherE2ESGLang:
    """E2E: patch Qwen3 forward -> dump -> compare 1gpu vs 2gpu-tp2."""

    @pytest.mark.timeout(300)
    def test_patch_dump_and_compare(self, tmp_path: Path) -> None:
        patched_fields: list[str] = ["patched_attn_output", "patched_mlp_output"]
        base_url: str = DEFAULT_URL_FOR_TEST

        config_path: Path = tmp_path / "patch_config.yaml"
        config_path.write_text(PATCH_CONFIG_YAML)

        # Run 1: baseline (1 GPU)
        baseline_dir: Path = tmp_path / "baseline"
        _run_server_and_generate(
            dump_dir=baseline_dir,
            config_path=config_path,
            tp=1,
            base_url=base_url,
        )
        _verify_patched_fields(dump_dir=baseline_dir, field_names=patched_fields)

        # Run 2: target (2 GPU TP=2)
        target_dir: Path = tmp_path / "target"
        _run_server_and_generate(
            dump_dir=target_dir,
            config_path=config_path,
            tp=2,
            base_url=base_url,
        )
        _verify_patched_fields(dump_dir=target_dir, field_names=patched_fields)

        # Compare baseline vs target
        baseline_exp: Path = baseline_dir / EXP_NAME
        target_exp: Path = target_dir / EXP_NAME

        result: subprocess.CompletedProcess[str] = subprocess.run(
            [
                "python",
                "-m",
                "sglang.srt.debug_utils.comparator",
                "--baseline-path",
                str(baseline_exp),
                "--target-path",
                str(target_exp),
                "--output-format",
                "json",
                "--grouping",
                "logical",
            ],
            capture_output=True,
            text=True,
        )

        debug_file: Path = _save_comparator_output(
            stdout=result.stdout, stderr=result.stderr
        )
        print(f"Comparator debug output: {debug_file}")

        assert result.returncode == 0, (
            f"Comparator failed (rc={result.returncode}). "
            f"Debug output: {debug_file}"
        )

        records: list[AnyRecord] = [
            parse_record_json(line)
            for line in result.stdout.strip().splitlines()
            if line.strip()
        ]
        assert (
            len(records) > 0
        ), f"Comparator produced no output records. Debug: {debug_file}"

        summary: SummaryRecord = _find_summary(records=records, debug_file=debug_file)
        assert (
            summary.passed > 0
        ), f"No comparisons passed (total={summary.total}). Debug: {debug_file}"
        assert summary.failed == 0, (
            f"{summary.failed} comparisons failed "
            f"(passed={summary.passed}, skipped={summary.skipped}). "
            f"Debug: {debug_file}"
        )


# --------------------------------- helpers ---------------------------------


def _run_server_and_generate(
    *,
    dump_dir: Path,
    config_path: Path,
    tp: int,
    base_url: str,
) -> None:
    """Launch SGLang server with source patcher + dumper, send a generate request."""
    env: dict[str, str] = {
        **os.environ,
        "DUMPER_SOURCE_PATCHER_CONFIG": str(config_path),
        "DUMPER_DIR": str(dump_dir),
        "DUMPER_EXP_NAME": EXP_NAME,
        "DUMPER_SERVER_PORT": "reuse",
    }

    proc = popen_launch_server(
        MODEL,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=["--tp", str(tp), "--max-total-tokens", "128"],
        env=env,
    )
    try:
        requests.post(
            f"{base_url}/dumper/configure",
            json={
                "enable": True,
                "filter": DUMPER_FILTER,
                "cleanup_previous": True,
            },
        ).raise_for_status()

        resp = requests.post(
            f"{base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"max_new_tokens": 8},
            },
        )
        assert resp.status_code == 200, f"Generate failed: {resp.text}"
    finally:
        kill_process_tree(proc.pid)


def _verify_patched_fields(*, dump_dir: Path, field_names: list[str]) -> None:
    """Verify that patched dump fields exist as .pt files."""
    for field in field_names:
        matches: list[Path] = list(dump_dir.rglob(f"*name={field}*.pt"))
        assert len(matches) > 0, (
            f"Expected patched field '{field}' not found under {dump_dir}. "
            f"Available files: {sorted(f.name for f in dump_dir.rglob('*.pt'))[:20]}"
        )


def _find_summary(*, records: list[AnyRecord], debug_file: Path) -> SummaryRecord:
    """Extract the SummaryRecord from comparator output."""
    summaries: list[SummaryRecord] = [
        r for r in records if isinstance(r, SummaryRecord)
    ]
    assert len(summaries) == 1, (
        f"Expected 1 summary record, got {len(summaries)}. "
        f"Record types: {[type(r).__name__ for r in records]}. "
        f"Debug: {debug_file}"
    )
    return summaries[0]


def _save_comparator_output(*, stdout: str, stderr: str) -> Path:
    """Save comparator stdout+stderr to a temp file that persists for debugging."""
    fd, path_str = tempfile.mkstemp(prefix="comparator_e2e_", suffix=".log", dir="/tmp")
    with os.fdopen(fd, "w") as f:
        f.write("=== STDOUT ===\n")
        f.write(stdout)
        f.write("\n=== STDERR ===\n")
        f.write(stderr)
    return Path(path_str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
