import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cli_testlib import _chain, _write_stub_proof
from mechanical_refactor_reproduction_cli import (
    KIND_MECHANICAL,
    KIND_NON_MECHANICAL,
    VERDICT_AMBIGUOUS_KIND,
    VERDICT_HUMAN_REVIEW,
    VERDICT_PASS,
    VERDICT_UNCLASSIFIED,
    verify_chain,
)


def _single_verdict(repo: Path, tmp_path: Path, message: str, *, with_proof: bool):
    proof = tmp_path / "proof"
    proof.mkdir(exist_ok=True)
    base, shas = _chain(repo, [message])
    if with_proof:
        _write_stub_proof(proof, shas[0])
    result = verify_chain(base=base, branch="chain", proof=proof, repo_root=str(repo))
    return result.verdicts[0]


def test_mechanical_provable_word_classifies_the_commit_as_mechanical(
    repo: Path, tmp_path: Path
) -> None:
    """A message carrying mechanical_provable is classified mechanical and needs a proof."""
    verdict = _single_verdict(
        repo, tmp_path, "grp(step,mechanical_provable): move foo", with_proof=True
    )
    assert verdict.kind == KIND_MECHANICAL
    assert verdict.verdict == VERDICT_PASS


def test_non_mechanical_provable_word_is_not_double_counted_as_the_bare_word(
    repo: Path, tmp_path: Path
) -> None:
    """non_mechanical_provable classifies as non-mechanical, not as both words at once."""
    verdict = _single_verdict(
        repo,
        tmp_path,
        "grp(step,non_mechanical_provable): rework foo",
        with_proof=False,
    )
    assert verdict.kind == KIND_NON_MECHANICAL
    assert verdict.verdict == VERDICT_HUMAN_REVIEW


def test_message_without_either_word_is_unclassified_and_fails_the_chain(
    repo: Path, tmp_path: Path
) -> None:
    """A commit missing both words gets UNCLASSIFIED and the chain does not pass."""
    proof = tmp_path / "proof"
    proof.mkdir()
    base, _ = _chain(repo, ["plain subject with no kind word"])
    result = verify_chain(base=base, branch="chain", proof=proof, repo_root=str(repo))
    assert result.verdicts[0].verdict == VERDICT_UNCLASSIFIED
    assert result.verdicts[0].kind is None
    assert not result.passed


def test_message_with_both_words_is_ambiguous(repo: Path, tmp_path: Path) -> None:
    """A commit declaring both kinds gets AMBIGUOUS_KIND and fails the chain."""
    verdict = _single_verdict(
        repo,
        tmp_path,
        "subject mechanical_provable\n\nbody also says non_mechanical_provable",
        with_proof=True,
    )
    assert verdict.verdict == VERDICT_AMBIGUOUS_KIND
    assert verdict.kind is None


def test_kind_word_must_stand_alone_not_as_a_substring(
    repo: Path, tmp_path: Path
) -> None:
    """xmechanical_provable / mechanical_provable_x do not count as the standalone word."""
    verdict = _single_verdict(
        repo,
        tmp_path,
        "xmechanical_provable and mechanical_provable_x only",
        with_proof=False,
    )
    assert verdict.verdict == VERDICT_UNCLASSIFIED


def test_kind_word_delimited_by_punctuation_counts(repo: Path, tmp_path: Path) -> None:
    """The word inside punctuation, e.g. (step,mechanical_provable), is a valid match."""
    verdict = _single_verdict(
        repo, tmp_path, "grp(step,mechanical_provable): move", with_proof=True
    )
    assert verdict.kind == KIND_MECHANICAL


def test_repeating_the_same_kind_word_is_accepted(repo: Path, tmp_path: Path) -> None:
    """Multiple occurrences of one kind word still classify unambiguously."""
    verdict = _single_verdict(
        repo,
        tmp_path,
        "mechanical_provable move\n\nthis commit is mechanical_provable",
        with_proof=True,
    )
    assert verdict.kind == KIND_MECHANICAL
    assert verdict.verdict == VERDICT_PASS


def test_kind_word_in_the_body_counts_when_subject_is_free_form(
    repo: Path, tmp_path: Path
) -> None:
    """Classification scans the whole message, so a body-only word is enough."""
    verdict = _single_verdict(
        repo,
        tmp_path,
        "Move resolve to util\n\nKind: non_mechanical_provable",
        with_proof=False,
    )
    assert verdict.kind == KIND_NON_MECHANICAL
