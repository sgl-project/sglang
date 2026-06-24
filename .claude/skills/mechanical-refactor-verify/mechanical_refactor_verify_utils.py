"""Utilities for mechanical refactor verification.

Two complementary checks, both self-contained in this file (no external scripts):

- Reproduce mode (``verify_mechanical_refactor``): a transform script regenerates a
  whole mechanical change's diff byte-for-byte (pre-commit reformatting included).
  Use for a single mechanical PR, and for renames / inlines where a formatter
  re-wraps lines. See SKILL.md for the script template.

- Verify mode (``verify_move_commit``): inspect a single commit and certify it is a
  pure relocation (function move, file split, module extraction) without any
  reproduce script -- it confirms every changed line is either relocated byte-for-byte
  or an import, and prints whatever is left over for review. Use for a stack of commits
  (each its own PR) where only some commits are mechanical. Runnable directly:

      python3 mechanical_refactor_verify_utils.py move <commit>

The exact rule the verify mode enforces is specified in verifier-spec.md (the source
of truth); this module implements that file and nothing more.
"""

import ast
import shlex
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path


def exec_command(cmd: str, cwd: str | None = None, check: bool = True) -> str:
    print(f"  $ {cmd}", flush=True)
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        print(f"FAILED: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def git_add_and_commit(message: str, cwd: str) -> None:
    exec_command(f"git add -A && git commit -m {shlex.quote(message)}", cwd=cwd)


def dedent(text: str, n: int) -> str:
    """Remove exactly n leading spaces from each line."""
    lines = text.splitlines(keepends=True)
    return "".join(line[n:] if line[:n] == " " * n else line for line in lines)


# ---------------------------------------------------------------------------
# Reproduce mode: regenerate a whole mechanical change and diff it byte-for-byte.
# ---------------------------------------------------------------------------


def verify_mechanical_refactor(
    base_commit: str,
    target_commit: str,
    transform: "Callable[[Path], None]",
) -> None:
    repo_root = exec_command("git rev-parse --show-toplevel")
    worktree_dir = tempfile.mkdtemp(prefix="verify-mechanical-")
    branch_name = f"verify-mechanical-{base_commit[:8]}"

    try:
        print(f"[1/4] Creating worktree at {base_commit[:8]}...")
        exec_command(
            f"git worktree add -b {branch_name} {worktree_dir} {base_commit}",
            cwd=repo_root,
        )

        print("[2/4] Running transformation...")
        transform(Path(worktree_dir))

        print("[3/4] Running pre-commit...")
        exec_command("pre-commit run --all-files", cwd=worktree_dir, check=False)
        if exec_command("git status --porcelain", cwd=worktree_dir):
            git_add_and_commit("pre-commit fixes", cwd=worktree_dir)

        print(f"[4/4] Diffing against {target_commit[:8]}...")
        diff = exec_command(
            f"git diff {target_commit} -- .",
            cwd=worktree_dir,
            check=False,
        )

        if diff:
            print(f"\nFAIL: diff is non-empty:\n{diff}")
            sys.exit(1)
        else:
            print("\nPASS: transform reproduces the commit exactly.")

    finally:
        print(f"\nWorktree left at: {worktree_dir}")
        print(f"Branch: {branch_name}")
        print("To clean up manually:")
        print(f"  git worktree remove {worktree_dir} && git branch -D {branch_name}")


# ---------------------------------------------------------------------------
# Verify mode: certify one commit is a pure relocation by inspecting its diff.
# The rule below is specified in verifier-spec.md; keep the two in lockstep.
# ---------------------------------------------------------------------------


def _git_show_file(commit_ish: str, path: str, repo_root: str) -> str:
    """Exact content of ``path`` at ``commit_ish`` ("" if it does not exist there).

    Unlike ``exec_command`` this does not strip, so ``ast`` line numbers stay aligned
    with the file's real lines.
    """
    print(f"  $ git show {commit_ish}:{path}", flush=True)
    result = subprocess.run(
        ["git", "show", f"{commit_ish}:{path}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    return result.stdout if result.returncode == 0 else ""


def _import_line_texts(file_text: str) -> set[str]:
    """Stripped text of every line that is part of an import statement.

    Found structurally via ``ast`` (not regex), so single-line imports and
    parenthesised multi-line imports (each member line) are both covered. A file
    that does not parse as Python contributes nothing. See verifier-spec.md step 2.
    """
    try:
        tree = ast.parse(file_text)
    except SyntaxError:
        return set()
    lines = file_text.splitlines()
    texts: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            end = node.end_lineno or node.lineno
            for lineno in range(node.lineno, end + 1):
                if 1 <= lineno <= len(lines):
                    texts.add(lines[lineno - 1].strip())
    return texts


def _commit_import_texts(commit: str, repo_root: str) -> tuple[set[str], set[str]]:
    """Import-line texts across every file the commit touches, before and after.

    Renames are not followed (``-M`` off), so a rename shows as delete + add and both
    its sides are parsed. See verifier-spec.md step 2.
    """
    status = exec_command(
        f"git show {shlex.quote(commit)} --name-status --format= --no-color --no-ext-diff",
        cwd=repo_root,
    )
    before: set[str] = set()
    after: set[str] = set()
    for line in status.splitlines():
        if "\t" not in line:
            continue
        code, path = line.split("\t", 1)
        status_code = code[:1]
        if status_code in ("A", "M", "C"):
            after |= _import_line_texts(_git_show_file(commit, path, repo_root))
        if status_code in ("D", "M"):
            before |= _import_line_texts(_git_show_file(f"{commit}^", path, repo_root))
    return before, after


def _commit_changed_lines(commit: str, repo_root: str) -> tuple[list[str], list[str]]:
    """Return (removed, added) content lines of a commit's diff, headers excluded."""
    out = exec_command(
        f"git show {shlex.quote(commit)} --format= --no-color --no-ext-diff",
        cwd=repo_root,
    )
    removed: list[str] = []
    added: list[str] = []
    for line in out.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith("+"):
            added.append(line[1:])
        elif line.startswith("-"):
            removed.append(line[1:])
    return removed, added


def _normalize_block(lines: list[str]) -> Counter:
    """Strip indentation/trailing space and drop blank lines. See verifier-spec.md step 3."""
    counts: Counter = Counter()
    for raw in lines:
        stripped = raw.strip()
        if stripped:
            counts[stripped] += 1
    return counts


def verify_move_commit(commit: str, *, repo_root: str | None = None) -> bool:
    """Certify a commit is a pure relocation (function move / file split / rename).

    Implements verifier-spec.md exactly: every changed line must either be relocated
    byte-for-byte (indentation ignored) or be an import statement. Imports are the only
    tolerated non-moved change; anything else -- a call-site rewrite, a dropped
    decorator, a re-derived constant, a line that changed by even one byte -- is listed
    for review and the commit is not certified.
    """
    root = repo_root or exec_command("git rev-parse --show-toplevel")
    removed, added = _commit_changed_lines(commit, root)
    imports_before, imports_after = _commit_import_texts(commit, root)
    rem, add = _normalize_block(removed), _normalize_block(added)

    relocated = sum((rem & add).values())
    imports: list[str] = []
    to_review: list[str] = []
    for line in sorted((rem - add).elements()):
        (imports if line in imports_before else to_review).append(line)
    for line in sorted((add - rem).elements()):
        (imports if line in imports_after else to_review).append(line)

    print(f"commit {commit}: move check (rule: verifier-spec.md)")
    print(f"  {relocated} line(s) relocated byte-for-byte (indentation ignored)")
    print(f"  {len(imports)} import line(s) (the only allowed non-moved change):")
    for line in imports:
        print(f"    [import] {line}")
    print(f"  {len(to_review)} line(s) to review:")
    for line in to_review:
        print(f"    [review] {line}")

    nothing_changed = not removed and not added
    clean = not to_review and (relocated > 0 or nothing_changed)
    if clean and relocated > 0:
        print(
            "  => CLEAN MOVE (every changed line is a byte-identical move or an import)"
        )
    elif clean:
        print("  => CLEAN (pure rename; no content changed)")
    else:
        print("  => NEEDS REVIEW (a non-import line changed, or nothing was relocated)")
    return clean


def _main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "move":
        return 0 if verify_move_commit(argv[1]) else 1
    print(
        "usage: python3 mechanical_refactor_verify_utils.py move <commit>",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
