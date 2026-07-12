import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mechanical_refactor_proof_generator as generator
from cli_testlib import _commit, _git, _write
from mechanical_refactor_reproduction_cli import (
    VERDICT_HUMAN_REVIEW,
    VERDICT_MISLABELED_PROVABLE,
    render_report,
    verify_chain,
)

_BASE_FILES = {
    "model.py": "def keep():\n    return 0\n\n\ndef resolve(m):\n    return m\n",
    "util.py": "import os\n",
}
# The after-state is the primitives' exact output (this bare repo has no formatter
# to absorb the cut's leftover blank lines, unlike a pre-commit-clean real repo).
_MOVED_FILES = {
    "model.py": "def keep():\n    return 0\n\n\n",
    "util.py": "import os\n\ndef resolve(m):\n    return m\n",
}


def _base_then_branch(repo: Path, extra_base: "dict[str, str] | None" = None) -> str:
    _write(repo, **{**_BASE_FILES, **(extra_base or {})})
    base = _commit(repo, "base")
    _git(repo, "switch", "-q", "-c", "chain")
    return base


def test_fully_provable_commit_declared_non_mechanical_is_mislabeled(
    repo: Path, tmp_path: Path
) -> None:
    """A pure-relocation commit labeled non_mechanical_provable fails the chain."""
    proof = tmp_path / "proof"
    proof.mkdir()
    base = _base_then_branch(repo)
    _write(repo, **_MOVED_FILES)
    _commit(repo, "non_mechanical_provable: rework the util module")

    result = verify_chain(base=base, branch="chain", proof=proof, repo_root=str(repo))

    assert result.verdicts[0].verdict == VERDICT_MISLABELED_PROVABLE
    assert not result.passed
    assert "MISLABELED_PROVABLE" in render_report(result)


def test_no_provable_sniff_flag_accepts_the_same_commit(
    repo: Path, tmp_path: Path
) -> None:
    """With provable_sniff disabled the mislabeled commit degrades to HUMAN_REVIEW."""
    proof = tmp_path / "proof"
    proof.mkdir()
    base = _base_then_branch(repo)
    _write(repo, **_MOVED_FILES)
    _commit(repo, "non_mechanical_provable: rework the util module")

    result = verify_chain(
        base=base,
        branch="chain",
        proof=proof,
        repo_root=str(repo),
        provable_sniff=False,
    )

    assert result.verdicts[0].verdict == VERDICT_HUMAN_REVIEW
    assert result.passed


def test_genuinely_semantic_commit_stays_human_review_without_warning(
    repo: Path, tmp_path: Path
) -> None:
    """A commit relocating nothing keeps HUMAN_REVIEW with no sniff warning."""
    proof = tmp_path / "proof"
    proof.mkdir()
    base = _base_then_branch(repo)
    _write(repo, **{"model.py": "def keep():\n    return 42\n"})
    _commit(repo, "non_mechanical_provable: change keep's return value")

    result = verify_chain(base=base, branch="chain", proof=proof, repo_root=str(repo))

    assert result.verdicts[0].verdict == VERDICT_HUMAN_REVIEW
    assert result.verdicts[0].warning == ""
    assert result.passed
    assert "## Warnings" not in render_report(result)


def test_relocation_bundled_with_a_semantic_change_warns_but_passes(
    repo: Path, tmp_path: Path
) -> None:
    """A non-mech commit bundling a real move gets a sniff warning in the report."""
    proof = tmp_path / "proof"
    proof.mkdir()
    base = _base_then_branch(repo, extra_base={"const.py": "LIMIT = 1\n"})
    _write(repo, **{**_MOVED_FILES, "const.py": "LIMIT = 2\n"})
    _commit(repo, "non_mechanical_provable: rework util and bump the limit")

    result = verify_chain(base=base, branch="chain", proof=proof, repo_root=str(repo))

    assert result.verdicts[0].verdict == VERDICT_HUMAN_REVIEW
    assert "resolve" in result.verdicts[0].warning
    assert "mechanical_provable split" in result.verdicts[0].warning
    assert result.passed
    report = render_report(result)
    assert "## Warnings" in report
    assert "provable-sniff" in report


def test_sniff_error_degrades_to_human_review_with_a_note(
    repo: Path, tmp_path: Path, monkeypatch
) -> None:
    """A sniff crash never fails the chain walk; it becomes a warning note."""
    proof = tmp_path / "proof"
    proof.mkdir()
    base = _base_then_branch(repo)
    _write(repo, **_MOVED_FILES)
    _commit(repo, "non_mechanical_provable: rework the util module")

    def boom(commit: str, root: str):
        raise RuntimeError("inference exploded")

    monkeypatch.setattr(generator, "infer_recipe", boom)
    result = verify_chain(base=base, branch="chain", proof=proof, repo_root=str(repo))

    assert result.verdicts[0].verdict == VERDICT_HUMAN_REVIEW
    assert "provable-sniff errored" in result.verdicts[0].warning
    assert result.passed
