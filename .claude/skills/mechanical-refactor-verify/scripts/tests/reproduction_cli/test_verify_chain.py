import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cli_testlib import _chain, _commit, _git, _write, _write_stub_proof
from mechanical_refactor_proof_generator import generate_range
from mechanical_refactor_reproduction_cli import (
    VERDICT_FAIL,
    VERDICT_HUMAN_REVIEW,
    VERDICT_PASS,
    ChainVerificationError,
    main,
    verify_chain,
)


def test_chain_of_proved_and_declared_commits_passes(
    repo: Path, tmp_path: Path
) -> None:
    """A proved mechanical commit plus a declared non-mechanical one verifies as PASS."""
    proof = tmp_path / "proof"
    base, shas = _chain(
        repo,
        ["mechanical_provable: move foo", "non_mechanical_provable: rework bar"],
    )
    _write_stub_proof(proof, shas[0])

    result = verify_chain(base=base, branch="chain", proof=proof, repo_root=str(repo))

    assert [v.verdict for v in result.verdicts] == [VERDICT_PASS, VERDICT_HUMAN_REVIEW]
    assert result.passed


def test_failing_proof_fails_the_commit_and_the_chain(
    repo: Path, tmp_path: Path
) -> None:
    """A proof that exits non-zero yields FAIL with the output tail in the detail."""
    proof = tmp_path / "proof"
    base, shas = _chain(repo, ["mechanical_provable: move foo"])
    _write_stub_proof(proof, shas[0], passing=False)

    result = verify_chain(base=base, branch="chain", proof=proof, repo_root=str(repo))

    assert result.verdicts[0].verdict == VERDICT_FAIL
    assert "RESIDUAL" in result.verdicts[0].detail
    assert not result.passed


def test_proof_exiting_zero_without_a_pass_line_is_a_fail(
    repo: Path, tmp_path: Path
) -> None:
    """PASS needs exit 0 AND the PASS: verdict line, so a residual under exit 0 fails."""
    proof = tmp_path / "proof"
    base, shas = _chain(repo, ["mechanical_provable: move foo"])
    script = _write_stub_proof(proof, shas[0])
    script.write_text('print("RESIDUAL (1 lines):\\n+x")\n')

    result = verify_chain(base=base, branch="chain", proof=proof, repo_root=str(repo))

    assert result.verdicts[0].verdict == VERDICT_FAIL


def test_main_exit_codes_reflect_the_chain_verdict(repo: Path, tmp_path: Path) -> None:
    """main returns 0 for a verified chain and 1 once an unverifiable commit appears."""
    proof = tmp_path / "proof"
    base, shas = _chain(repo, ["mechanical_provable: move"])
    _write_stub_proof(proof, shas[0])
    args = [
        "--base",
        base,
        "--branch",
        "chain",
        "--proof",
        str(proof),
        "--repo-root",
        str(repo),
    ]

    assert main(args) == 0

    _git(repo, "commit", "-q", "--allow-empty", "-m", "plain subject with no kind word")
    assert main(args) == 1


def test_unresolvable_refs_and_missing_proof_folder_are_setup_errors(
    repo: Path, tmp_path: Path
) -> None:
    """Bad --base/--branch/--proof inputs raise ChainVerificationError (exit code 2)."""
    proof = tmp_path / "proof"
    proof.mkdir()
    base, _ = _chain(repo, ["mechanical_provable: move"])

    with pytest.raises(ChainVerificationError):
        verify_chain(
            base=base, branch="no-such-branch", proof=proof, repo_root=str(repo)
        )
    with pytest.raises(ChainVerificationError):
        verify_chain(
            base=base,
            branch="chain",
            proof=tmp_path / "missing",
            repo_root=str(repo),
        )
    assert (
        main(
            [
                "--base",
                base,
                "--branch",
                "no-such-branch",
                "--proof",
                str(proof),
                "--repo-root",
                str(repo),
            ]
        )
        == 2
    )


def test_non_ancestor_base_and_empty_range_are_setup_errors(
    repo: Path, tmp_path: Path
) -> None:
    """A base off the branch or an empty base..branch range refuses to verify."""
    proof = tmp_path / "proof"
    proof.mkdir()
    base, shas = _chain(repo, ["mechanical_provable: move"])
    _git(repo, "switch", "-q", "main")
    _write(repo, **{"other.py": "OTHER = 1\n"})
    off_branch = _commit(repo, "unrelated main-side commit")

    with pytest.raises(ChainVerificationError):
        verify_chain(base=off_branch, branch="chain", proof=proof, repo_root=str(repo))
    with pytest.raises(ChainVerificationError):
        verify_chain(base=shas[0], branch=shas[0], proof=proof, repo_root=str(repo))


def test_merge_commit_in_the_chain_is_a_setup_error(repo: Path, tmp_path: Path) -> None:
    """A non-linear chain (contains a merge commit) refuses to verify."""
    proof = tmp_path / "proof"
    proof.mkdir()
    base, _ = _chain(repo, ["mechanical_provable: move"])
    _git(repo, "switch", "-q", "main")
    _write(repo, **{"other.py": "OTHER = 1\n"})
    _commit(repo, "mechanical_provable: main-side")
    _git(repo, "switch", "-q", "chain")
    _git(repo, "merge", "-q", "--no-ff", "-m", "non_mechanical_provable: merge", "main")

    with pytest.raises(ChainVerificationError):
        verify_chain(base=base, branch="chain", proof=proof, repo_root=str(repo))


def test_end_to_end_with_a_generated_proof_folder(repo: Path, tmp_path: Path) -> None:
    """A real move commit proved by generate_range verifies through the CLI end-to-end."""
    _write(
        repo,
        **{
            "model.py": "def keep():\n    return 0\n\n\ndef resolve(m):\n    return m\n",
            "util.py": "import os\n",
        },
    )
    base = _commit(repo, "base")
    _git(repo, "switch", "-q", "-c", "chain")
    # The after-state is the primitives' exact output (this bare repo has no formatter
    # to absorb the cut's leftover blank lines, unlike a pre-commit-clean real repo).
    _write(
        repo,
        **{
            "model.py": "def keep():\n    return 0\n\n\n",
            "util.py": "import os\n\ndef resolve(m):\n    return m\n",
        },
    )
    move_sha = _commit(repo, "mechanical_provable: move resolve to util")

    proof = tmp_path / "proof"
    generate_range(f"{base}..chain", out_dir=str(proof), repo_root=str(repo))
    result = verify_chain(base=base, branch="chain", proof=proof, repo_root=str(repo))

    assert result.verdicts[0].sha == move_sha
    assert result.verdicts[0].verdict == VERDICT_PASS
    assert result.passed
