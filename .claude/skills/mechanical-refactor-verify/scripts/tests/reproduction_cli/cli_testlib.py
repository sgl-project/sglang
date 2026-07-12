import subprocess
from pathlib import Path

_PASSING_PROOF = (
    "import sys\n"
    'print("PASS: reproduces the commit byte-for-byte.")\n'
    "sys.exit(0)\n"
)
_FAILING_PROOF = (
    "import sys\n"
    'print("RESIDUAL (2 lines):\\n+x\\n-y")\n'
    "sys.exit(1)\n"
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write(repo: Path, **files: "str | None") -> None:
    for name, content in files.items():
        path = repo / name.replace("__", "/")
        if content is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _chain(repo: Path, messages: "list[str]") -> "tuple[str, list[str]]":
    """A base commit plus one single-file commit per message, on a `chain` branch.
    Returns (base_sha, commit_shas)."""
    _write(repo, **{"seed.py": "SEED = 0\n"})
    base = _commit(repo, "base")
    _git(repo, "switch", "-q", "-c", "chain")
    shas: "list[str]" = []
    for i, message in enumerate(messages):
        _write(repo, **{f"file_{i}.py": f"VALUE = {i}\n"})
        shas.append(_commit(repo, message))
    return base, shas


def _write_stub_proof(
    proof_dir: Path,
    sha: str,
    *,
    passing: bool = True,
    flat: bool = False,
    stem_len: int = 9,
) -> Path:
    """A stand-in proof script printing the arbiter's verdict line and exiting to match."""
    directory = proof_dir if flat else proof_dir / "repro_scripts"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{sha[:stem_len]}.py"
    path.write_text(_PASSING_PROOF if passing else _FAILING_PROOF)
    return path
