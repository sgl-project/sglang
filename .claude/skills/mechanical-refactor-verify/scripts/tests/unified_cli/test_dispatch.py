import subprocess
import sys
from pathlib import Path

import pytest

from conftest import _git
from mechanical_refactor import main

_SCRIPT = Path(__file__).resolve().parents[2] / "mechanical_refactor.py"

_PASSING_PROOF = (
    "import sys\n"
    'print("PASS: reproduces the commit byte-for-byte.")\n'
    "sys.exit(0)\n"
)
_FAILING_PROOF = (
    "import sys\n" 'print("RESIDUAL (2 lines):\\n+x\\n-y")\n' "sys.exit(1)\n"
)


def _write(repo: Path, **files: "str | None") -> None:
    for name, content in files.items():
        path = repo / name
        if content is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _move_chain(repo: Path) -> "tuple[str, str]":
    """A base commit plus one pure function-move commit on a `chain` branch."""
    _write(
        repo,
        **{
            "model.py": "def keep():\n    return 0\n\n\ndef resolve(m):\n    return m\n",
            "util.py": "import os\n",
        },
    )
    base = _commit(repo, "base")
    _git(repo, "switch", "-q", "-c", "chain")
    _write(
        repo,
        **{
            "model.py": "def keep():\n    return 0\n\n\n",
            "util.py": "import os\n\ndef resolve(m):\n    return m\n",
        },
    )
    move_sha = _commit(repo, "mechanical_provable: move resolve to util")
    return base, move_sha


def _run_cli(repo: Path, *args: str) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        cwd=repo,
        capture_output=True,
        text=True,
    )


def test_split_reports_pass_for_a_pure_move_commit(repo: Path) -> None:
    """split on a pure relocation commit prints the inferred script and exits 0."""
    _, move_sha = _move_chain(repo)

    result = _run_cli(repo, "split", move_sha)

    assert result.returncode == 0, result.stderr
    assert "move_symbol" in result.stdout
    assert "PASS" in result.stdout


def test_split_reports_unsupported_for_a_non_relocation_commit(repo: Path) -> None:
    """split on a value-edit commit exits non-zero and names the UNSUPPORTED verdict."""
    _write(repo, **{"model.py": "VALUE = 1\n"})
    _commit(repo, "base")
    _write(repo, **{"model.py": "VALUE = 2\n"})
    edit_sha = _commit(repo, "edit the value")

    result = _run_cli(repo, "split", edit_sha)

    assert result.returncode != 0
    assert "UNSUPPORTED" in result.stderr


def test_construct_writes_a_proof_folder_for_matching_commits(
    repo: Path, tmp_path: Path
) -> None:
    """construct over a range emits one repro script per matching commit plus the logs."""
    _, move_sha = _move_chain(repo)
    out = tmp_path / "proof"

    result = _run_cli(
        repo,
        "construct",
        "main..chain",
        "--match",
        r"(?<!_)mechanical_provable",
        "--out",
        str(out),
    )

    assert result.returncode == 0, result.stderr
    assert (out / "repro_scripts" / f"{move_sha[:9]}.py").exists()
    assert (out / "output.log").exists()
    assert (out / "mechanical_refactor_reproduction_utils.py").exists()


def test_verify_passes_a_chain_whose_every_proof_passes(
    repo: Path, tmp_path: Path
) -> None:
    """verify exits 0 for a fully classified chain whose proofs all PASS."""
    base, move_sha = _move_chain(repo)
    proof = tmp_path / "proof"
    (proof / "repro_scripts").mkdir(parents=True)
    (proof / "repro_scripts" / f"{move_sha[:9]}.py").write_text(_PASSING_PROOF)

    code = main(
        [
            "verify",
            "--base",
            base,
            "--branch",
            "chain",
            "--proof",
            str(proof),
            "--repo-root",
            str(repo),
            "--jobs",
            "1",
        ]
    )

    assert code == 0
    assert (proof / "chain_report.md").exists()


def test_verify_fails_the_chain_when_a_proof_fails(
    repo: Path, tmp_path: Path
) -> None:
    """verify exits non-zero when a provable commit's proof reports a residual."""
    base, move_sha = _move_chain(repo)
    proof = tmp_path / "proof"
    (proof / "repro_scripts").mkdir(parents=True)
    (proof / "repro_scripts" / f"{move_sha[:9]}.py").write_text(_FAILING_PROOF)

    code = main(
        [
            "verify",
            "--base",
            base,
            "--branch",
            "chain",
            "--proof",
            str(proof),
            "--repo-root",
            str(repo),
        ]
    )

    assert code != 0


def test_missing_subcommand_is_a_usage_error() -> None:
    """invoking with no subcommand exits with argparse's usage error code."""
    with pytest.raises(SystemExit) as excinfo:
        main([])

    assert excinfo.value.code == 2
