"""Verify a whole mechanical-refactor chain: classification, proofs, and a full report.

Every commit in ``base..branch`` must classify itself by carrying exactly one of the two
words ``mechanical_provable`` or ``non_mechanical_provable`` anywhere in its message (the
rest of the message format is free). Every ``mechanical_provable`` commit must ship a
proof script in the proof folder (``<proof>/repro_scripts/<sha-prefix>.py`` or a flat
``<proof>/<sha-prefix>.py``), and running the proof must PASS -- reproduce the commit
byte-for-byte. A ``non_mechanical_provable`` commit carries no machine proof and is left
to human review.

The run prints a markdown report, writes it into the proof folder (``chain_report.md``),
and exits 0 iff the whole chain verifies. Normative contract: spec-reproduction-cli.md.

    python3 mechanical_refactor_reproduction_cli.py \
        --base <base-commit> --branch <pr-branch-name> --proof path/to/proof/folder
"""

import argparse
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_JOBS = 3

KIND_MECHANICAL = "mechanical_provable"
KIND_NON_MECHANICAL = "non_mechanical_provable"

VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"
VERDICT_MISSING_PROOF = "MISSING_PROOF"
VERDICT_AMBIGUOUS_PROOF = "AMBIGUOUS_PROOF"
VERDICT_HUMAN_REVIEW = "HUMAN_REVIEW"
VERDICT_UNCLASSIFIED = "UNCLASSIFIED"
VERDICT_AMBIGUOUS_KIND = "AMBIGUOUS_KIND"

_OK_VERDICTS = (VERDICT_PASS, VERDICT_HUMAN_REVIEW)

# The words are matched standalone: delimited by any non-[0-9A-Za-z_] character or the
# string boundary, so `non_mechanical_provable` never also counts as the bare word.
_KIND_WORD_RE = re.compile(
    r"(?<![0-9A-Za-z_])(non_)?mechanical_provable(?![0-9A-Za-z_])"
)

# The arbiter's verdict line (Repro.run / verify_mechanical_refactor both print `PASS:`).
_PASS_LINE_RE = re.compile(r"^PASS:", re.MULTILINE)

_MIN_PROOF_STEM_LEN = 7
_REPORT_FILENAME = "chain_report.md"
_FAIL_OUTPUT_TAIL_LINES = 60


class ChainVerificationError(Exception):
    """A setup problem (bad refs, non-linear range, missing proof folder): exit code 2."""


@dataclass(frozen=True)
class CommitVerdict:
    sha: str
    subject: str
    kind: "str | None"
    verdict: str
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict in _OK_VERDICTS


@dataclass(frozen=True)
class _PendingProof:
    sha: str
    subject: str
    kind: str
    script: Path


@dataclass(frozen=True)
class ChainResult:
    base: str
    branch: str
    base_sha: str
    branch_sha: str
    proof_dir: Path
    verdicts: "list[CommitVerdict]" = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.verdicts) and all(v.ok for v in self.verdicts)


def main(argv: "list[str]") -> int:
    parser = argparse.ArgumentParser(
        description="Verify a whole mechanical-refactor chain against its proof folder."
    )
    parser.add_argument("--base", required=True, help="base commit of the chain")
    parser.add_argument("--branch", required=True, help="PR branch name (chain tip)")
    parser.add_argument("--proof", required=True, help="proof folder path")
    parser.add_argument("--repo-root", default=None, help="repo root (default: cwd's)")
    parser.add_argument(
        "--report",
        default=None,
        help=f"report file path (default: <proof>/{_REPORT_FILENAME})",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=_DEFAULT_JOBS,
        help=f"max concurrent proof runs (default {_DEFAULT_JOBS})",
    )
    args = parser.parse_args(argv)

    try:
        result = verify_chain(
            base=args.base,
            branch=args.branch,
            proof=Path(args.proof),
            repo_root=args.repo_root,
            jobs=args.jobs,
        )
    except ChainVerificationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = render_report(result)
    report_path = (
        Path(args.report) if args.report else result.proof_dir / _REPORT_FILENAME
    )
    report_path.write_text(report)
    print(report)
    print(f"report written to: {report_path}")
    return 0 if result.passed else 1


def verify_chain(
    *,
    base: str,
    branch: str,
    proof: Path,
    repo_root: "str | None" = None,
    jobs: int = _DEFAULT_JOBS,
) -> ChainResult:
    """Classify every commit in ``base..branch`` and run every provable commit's proof.

    Classification and proof resolution are sequential (cheap); the proof runs execute
    concurrently, up to ``jobs`` at a time — safe because each proof works in its own
    throwaway worktree. The verdict list keeps chain order."""
    root = repo_root or _repo_root()
    if not proof.is_dir():
        raise ChainVerificationError(f"proof folder does not exist: {proof}")
    base_sha = _rev_parse(base, root)
    branch_sha = _rev_parse(branch, root)
    commits = _linear_commits(base_sha=base_sha, branch_sha=branch_sha, root=root)

    resolved: "list[CommitVerdict | _PendingProof]" = []
    for sha in commits:
        subject = _git_output(["log", "-1", "--format=%s", sha], root).strip()
        message = _git_output(["log", "-1", "--format=%B", sha], root)
        resolved.append(
            _resolve_commit(sha=sha, subject=subject, message=message, proof=proof)
        )

    verdicts: "list[CommitVerdict]" = _run_pending_proofs(
        resolved=resolved, root=root, jobs=jobs
    )
    return ChainResult(
        base=base,
        branch=branch,
        base_sha=base_sha,
        branch_sha=branch_sha,
        proof_dir=proof,
        verdicts=verdicts,
    )


def render_report(result: ChainResult) -> str:
    """The full chain report as markdown: header, per-commit table, failure details."""
    n_mech = sum(1 for v in result.verdicts if v.kind == KIND_MECHANICAL)
    n_non_mech = sum(1 for v in result.verdicts if v.kind == KIND_NON_MECHANICAL)
    n_unclassified = sum(1 for v in result.verdicts if v.kind is None)
    n_pass = sum(1 for v in result.verdicts if v.verdict == VERDICT_PASS)

    lines = [
        "# Mechanical refactor chain report",
        "",
        f"- base: `{result.base}` (`{result.base_sha[:12]}`)",
        f"- branch: `{result.branch}` (`{result.branch_sha[:12]}`)",
        f"- proof folder: `{result.proof_dir}`",
        f"- chain verdict: **{'PASS' if result.passed else 'FAIL'}**",
        f"- commits: {len(result.verdicts)} total — {n_mech} {KIND_MECHANICAL}, "
        f"{n_non_mech} {KIND_NON_MECHANICAL}, {n_unclassified} classification error(s)",
        f"- proofs: {n_pass}/{n_mech} PASS",
        "",
        "| # | commit | kind | verdict | subject |",
        "|---|--------|------|---------|---------|",
    ]
    for i, v in enumerate(result.verdicts, start=1):
        kind = v.kind or "?"
        subject = v.subject.replace("|", "\\|")
        lines.append(f"| {i} | `{v.sha[:9]}` | {kind} | {v.verdict} | {subject} |")

    failures = [v for v in result.verdicts if not v.ok]
    if failures:
        lines += ["", "## Failure details"]
        for v in failures:
            lines += [
                "",
                f"### `{v.sha[:9]}` — {v.verdict}",
                "",
                v.detail or "(no detail)",
            ]
    return "\n".join(lines) + "\n"


def _run_pending_proofs(
    *, resolved: "list[CommitVerdict | _PendingProof]", root: str, jobs: int
) -> "list[CommitVerdict]":
    """Execute the pending proofs on a bounded thread pool; keep chain order."""
    pending = [
        (i, item) for i, item in enumerate(resolved) if isinstance(item, _PendingProof)
    ]
    finished: "dict[int, CommitVerdict]" = {}
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            futures = {
                i: pool.submit(_proof_verdict, item, root=root) for i, item in pending
            }
            for i, future in futures.items():
                finished[i] = future.result()
    return [
        finished[i] if isinstance(item, _PendingProof) else item
        for i, item in enumerate(resolved)
    ]


def _proof_verdict(pending: _PendingProof, *, root: str) -> CommitVerdict:
    passed, output = _run_proof(script=pending.script, root=root)
    if passed:
        verdict = CommitVerdict(
            sha=pending.sha,
            subject=pending.subject,
            kind=pending.kind,
            verdict=VERDICT_PASS,
            detail="",
        )
    else:
        tail = "\n".join(output.splitlines()[-_FAIL_OUTPUT_TAIL_LINES:])
        verdict = CommitVerdict(
            sha=pending.sha,
            subject=pending.subject,
            kind=pending.kind,
            verdict=VERDICT_FAIL,
            detail=(
                f"proof `{pending.script}` did not PASS; output tail:\n\n"
                f"```\n{tail}\n```"
            ),
        )
    print(f"proof {pending.sha[:9]}  {verdict.verdict}", flush=True)
    return verdict


def _resolve_commit(
    *, sha: str, subject: str, message: str, proof: Path
) -> "CommitVerdict | _PendingProof":
    kind, classification_error = _classify(message)
    if kind is None:
        return CommitVerdict(
            sha=sha,
            subject=subject,
            kind=None,
            verdict=classification_error,
            detail=(
                f"the commit message must contain exactly one of the words "
                f"`{KIND_MECHANICAL}` or `{KIND_NON_MECHANICAL}`"
            ),
        )
    if kind == KIND_NON_MECHANICAL:
        return CommitVerdict(
            sha=sha,
            subject=subject,
            kind=kind,
            verdict=VERDICT_HUMAN_REVIEW,
            detail="declared non_mechanical_provable: no machine proof, review by hand",
        )

    scripts = _find_proof_scripts(proof=proof, sha=sha)
    if not scripts:
        return CommitVerdict(
            sha=sha,
            subject=subject,
            kind=kind,
            verdict=VERDICT_MISSING_PROOF,
            detail=(
                f"no proof script found; searched `{proof / 'repro_scripts'}` and "
                f"`{proof}` for `<sha-prefix>.py` (>= {_MIN_PROOF_STEM_LEN} hex chars)"
            ),
        )
    if len(scripts) > 1:
        listing = ", ".join(f"`{p}`" for p in scripts)
        return CommitVerdict(
            sha=sha,
            subject=subject,
            kind=kind,
            verdict=VERDICT_AMBIGUOUS_PROOF,
            detail=f"multiple proof scripts match this commit: {listing}",
        )

    return _PendingProof(sha=sha, subject=subject, kind=kind, script=scripts[0])


def _classify(message: str) -> "tuple[str | None, str]":
    """The commit's declared kind, or (None, error-verdict) when the word rule is broken.

    Exactly one of the two words must appear (any number of times, but only one of the
    two): zero occurrences is UNCLASSIFIED, both words present is AMBIGUOUS_KIND."""
    kinds = {
        KIND_NON_MECHANICAL if match.group(1) else KIND_MECHANICAL
        for match in _KIND_WORD_RE.finditer(message)
    }
    if not kinds:
        return None, VERDICT_UNCLASSIFIED
    if len(kinds) > 1:
        return None, VERDICT_AMBIGUOUS_KIND
    return kinds.pop(), ""


def _find_proof_scripts(*, proof: Path, sha: str) -> "list[Path]":
    """Proof scripts naming this commit: a ``<sha-prefix>.py`` (lowercase hex, >= 7 chars)
    under ``<proof>/repro_scripts/`` or flat in ``<proof>/``."""
    found: "list[Path]" = []
    for directory in (proof / "repro_scripts", proof):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.py")):
            stem = path.stem
            is_sha_prefix = (
                len(stem) >= _MIN_PROOF_STEM_LEN
                and all(c in "0123456789abcdef" for c in stem)
                and sha.startswith(stem)
            )
            if is_sha_prefix:
                found.append(path)
    return found


def _run_proof(*, script: Path, root: str) -> "tuple[bool, str]":
    """Run one proof script from the repo root. A PASS is exit code 0 AND the arbiter's
    ``PASS:`` verdict line on stdout (an old-style script that exits 0 with a residual is
    therefore still a FAIL)."""
    result = subprocess.run(
        [sys.executable, str(script.resolve())],
        cwd=root,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    passed = result.returncode == 0 and bool(_PASS_LINE_RE.search(result.stdout))
    return passed, output


def _linear_commits(*, base_sha: str, branch_sha: str, root: str) -> "list[str]":
    if not _is_ancestor(base_sha=base_sha, branch_sha=branch_sha, root=root):
        raise ChainVerificationError(
            f"base {base_sha[:12]} is not an ancestor of branch {branch_sha[:12]}"
        )
    commits = _git_output(
        ["rev-list", "--reverse", f"{base_sha}..{branch_sha}"], root
    ).split()
    if not commits:
        raise ChainVerificationError(
            f"no commits in {base_sha[:12]}..{branch_sha[:12]}"
        )
    merges = [
        sha
        for sha in commits
        if len(_git_output(["rev-list", "--parents", "-n", "1", sha], root).split()) > 2
    ]
    if merges:
        listing = ", ".join(sha[:9] for sha in merges)
        raise ChainVerificationError(
            f"the chain must be linear, but it contains merge commit(s): {listing}"
        )
    return commits


def _is_ancestor(*, base_sha: str, branch_sha: str, root: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", base_sha, branch_sha],
        cwd=root,
        capture_output=True,
    )
    return result.returncode == 0


def _rev_parse(ref: str, root: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ChainVerificationError(f"cannot resolve {ref!r}: {result.stderr.strip()}")
    return result.stdout.strip()


def _git_output(args: "list[str]", root: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    )
    return result.stdout


def _repo_root() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
