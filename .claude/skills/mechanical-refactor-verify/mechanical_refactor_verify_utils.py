"""Utilities for mechanical refactor verification.

Two complementary checks, both self-contained in this file (no external scripts):

- Reproduce mode (``verify_mechanical_refactor``): a transform script regenerates a
  whole mechanical change's diff byte-for-byte (pre-commit reformatting included).
  Use for a single mechanical PR, and for renames / inlines where a formatter
  re-wraps lines. See SKILL.md for the script template.

- Verify mode (``verify_move_commit``): inspect a single commit and certify it is a
  faithful relocation (function move, file split, module extraction) without any
  reproduce script -- it confirms the moved block is byte-identical on both sides
  and prints whatever is left over for review. Use for a stack of commits (each its
  own PR) where only some commits are mechanical. Runnable directly:

      python3 mechanical_refactor_verify_utils.py move <commit>
"""

import re
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
# Verify mode: certify one commit is a faithful relocation by inspecting its diff.
# ---------------------------------------------------------------------------

# Header lines that legitimately appear on only one side of a relocation: a
# decorator dropped/added when a method becomes a free function, and the
# ``__future__`` import a freshly created module file carries.
_MOVE_ONE_SIDED = {
    "@staticmethod",
    "@classmethod",
    "from __future__ import annotations",
}

_IMPORT_ITEM = re.compile(r"[A-Za-z_][A-Za-z0-9_]*( as [A-Za-z_][A-Za-z0-9_]*)?,?$")
# A line that is entirely a call expression -- the rewritten call site of the moved
# symbol (optionally assigned or returned). Args are unchanged by a move, so the
# only difference vs the old call site is the qualifier (``self.``/``ClassName.``).
_CALL_LINE = re.compile(
    r"(self\.[A-Za-z_]\w* = |return )?[A-Za-z_][A-Za-z0-9_.]*\(.*\)"
)


def _is_plumbing(line: str) -> bool:
    """A residual line that is expected wiring of a move rather than logic:
    imports, the moved symbol's new import-list items, call-site openings, a
    module logger, a TYPE_CHECKING guard, and lone bracket/punctuation lines that
    a formatter leaves behind when wrapping changes.
    """
    s = line.strip()
    if not s:
        return True
    if s.startswith(("import ", "from ", "@")):
        return True
    if s == "if TYPE_CHECKING:":
        return True
    if s.startswith("logger = logging.getLogger"):
        return True
    if all(c in "()[]{}:,." for c in s):
        return True
    if _IMPORT_ITEM.fullmatch(s):
        return True
    if s.endswith("("):  # a call-site opening (the new free-function call)
        return True
    if _CALL_LINE.fullmatch(s):  # a one-line call site of the moved symbol
        return True
    return False


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
    """Strip indentation/trailing space, drop blanks and one-sided header lines."""
    counts: Counter = Counter()
    for raw in lines:
        stripped = raw.strip()
        if stripped and stripped not in _MOVE_ONE_SIDED:
            counts[stripped] += 1
    return counts


def verify_move_commit(commit: str, *, repo_root: str | None = None) -> bool:
    """Certify a commit is a faithful relocation (function move / file split).

    A clean move deletes a block from one place and adds the byte-identical block
    (ignoring indentation and decorator / ``__future__`` header lines) somewhere
    else. Whatever is left over is split into expected wiring (imports, call sites)
    and "to review" lines. A move is certified only when nothing needs review;
    otherwise the to-review lines -- the only ones a human must read -- are listed.
    """
    root = repo_root or exec_command("git rev-parse --show-toplevel")
    removed, added = _commit_changed_lines(commit, root)
    rem, add = _normalize_block(removed), _normalize_block(added)

    relocated = sum((rem & add).values())
    residual = sorted((rem - add).elements()) + sorted((add - rem).elements())
    wiring = [r for r in residual if _is_plumbing(r)]
    to_review = [r for r in residual if not _is_plumbing(r)]

    print(f"commit {commit}: move check")
    print(f"  {relocated} line(s) relocated byte-for-byte (indentation ignored)")
    print(f"  {len(wiring)} wiring line(s) (imports / call sites):")
    for line in wiring:
        print(f"    [wiring] {line}")
    print(f"  {len(to_review)} line(s) to review:")
    for line in to_review:
        print(f"    [review] {line}")

    clean = relocated > 0 and not to_review
    if clean:
        print(
            "  => CLEAN MOVE (relocation is byte-faithful; only imports/call sites changed)"
        )
    else:
        print("  => NEEDS REVIEW (no relocation detected, or non-wiring lines changed)")
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
