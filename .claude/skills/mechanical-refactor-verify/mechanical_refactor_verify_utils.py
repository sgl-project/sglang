"""Certify that a commit is a pure relocation (function move / file split / rename).

Inspect a single commit's diff and confirm every changed line is either part of the
moved block -- relocated in the same order, allowing one uniform indentation shift for
the whole block -- or a whitelisted move artifact: an import, a dropped
@staticmethod/@classmethod (and the self type annotation dropped with it), or a call site
requalified for a moved symbol. Anything else is listed for review.

This module is self-contained (only git + the standard library) and independent of the
reproduce-mode helper. The exact rule it enforces is specified in verifier-spec.md (the
source of truth); this module implements that file and nothing more. Runnable directly:

    python3 mechanical_refactor_verify_utils.py <commit>
    python3 mechanical_refactor_verify_utils.py <base>..<tip> [--match REGEX]

The range form verifies every commit in the range oldest first; ``--match`` restricts it
to commits whose subject matches the regex (e.g. ``--match -move``) and skips the rest.
"""

import ast
import difflib
import re
import subprocess
import sys
from collections import Counter


def _git_output(args: list[str], cwd: str) -> str:
    """Raw stdout of a git command ("" if it fails). Not stripped, so ``ast`` line
    numbers stay aligned with a file's real lines."""
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else ""


def _repo_root() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _import_line_texts(file_text: str) -> set[str]:
    """Stripped text of every line that is part of an import statement.

    Found structurally via ``ast`` (not regex), so single-line imports and parenthesised
    multi-line imports (each member line) are both covered. A file that does not parse as
    Python contributes nothing. See verifier-spec.md step 2.
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

    Renames are not followed (``-M`` off), so a rename shows as delete + add and both its
    sides are parsed. See verifier-spec.md step 2.
    """
    status = _git_output(
        ["show", commit, "--name-status", "--format=", "--no-color", "--no-ext-diff"],
        repo_root,
    )
    before: set[str] = set()
    after: set[str] = set()
    for line in status.splitlines():
        if "\t" not in line:
            continue
        code, path = line.split("\t", 1)
        status_code = code[:1]
        if status_code in ("A", "M", "C"):
            after |= _import_line_texts(
                _git_output(["show", f"{commit}:{path}"], repo_root)
            )
        if status_code in ("D", "M"):
            before |= _import_line_texts(
                _git_output(["show", f"{commit}^:{path}"], repo_root)
            )
    return before, after


def _drop_counts(lines: list[str], to_drop: Counter) -> list[str]:
    """Return ``lines`` with up to ``to_drop[line]`` occurrences of each line removed,
    preserving order."""
    budget = Counter(to_drop)
    kept: list[str] = []
    for line in lines:
        if budget.get(line, 0) > 0:
            budget[line] -= 1
        else:
            kept.append(line)
    return kept


def _commit_changed_lines(commit: str, repo_root: str) -> tuple[list[str], list[str]]:
    """Return (removed, added) content lines of a commit's diff, in patch order, with
    their whitespace intact; file/hunk headers excluded.

    Diffs are accumulated per file and lines that appear byte-for-byte as BOTH removed and
    added within the same file are cancelled -- those are git's own diff artifacts
    (an unchanged line re-represented as remove + add when nearby lines change), not a
    relocation. A real relocation keeps its lines because they cross files. See
    verifier-spec.md step 1.
    """
    out = _git_output(
        ["show", commit, "--format=", "--no-color", "--no-ext-diff"], repo_root
    )
    per_file: list[tuple[list[str], list[str]]] = []
    file_removed: list[str] = []
    file_added: list[str] = []
    for line in out.splitlines():
        if line.startswith("diff --git"):
            per_file.append((file_removed, file_added))
            file_removed, file_added = [], []
        elif line.startswith(("+++", "---")):
            continue
        elif line.startswith("+"):
            file_added.append(line[1:])
        elif line.startswith("-"):
            file_removed.append(line[1:])
    per_file.append((file_removed, file_added))

    removed: list[str] = []
    added: list[str] = []
    for file_rem, file_add in per_file:
        artifacts = Counter(file_rem) & Counter(file_add)
        removed.extend(_drop_counts(file_rem, artifacts))
        added.extend(_drop_counts(file_add, artifacts))
    return removed, added


# Decorators a de-self drops when a method becomes a free function; the move carries them
# on only one side. See verifier-spec.md (whitelist).
_MOVE_DECORATORS = {"@staticmethod", "@classmethod"}


def _moved_symbol_names(relocated_lines: list[str]) -> set[str]:
    """Names defined (def / class) within lines present on both sides -- the symbols that
    moved. A call site may be requalified for these names only, so a requalification of a
    symbol that did NOT move in this commit is never silently allowed. See verifier-spec.md.
    """
    names: set[str] = set()
    for line in relocated_lines:
        match = re.match(r"(?:async\s+)?def\s+(\w+)", line) or re.match(
            r"class\s+(\w+)", line
        )
        if match:
            names.add(match.group(1))
    return names


def _strip_moved_qualifiers(line: str, names: set[str]) -> str:
    """Drop a ``Qualifier.`` prefix before any moved symbol name (``self.foo`` -> ``foo``).

    Only the qualifier is removed; the rest of the line -- including leading whitespace and
    arguments -- is untouched, so a call whose arguments also changed will not match its
    old form. See verifier-spec.md.
    """
    for name in names:
        line = re.sub(rf"\b[\w.]+\.{re.escape(name)}\b", name, line)
    return line


_SELF_ANNOTATION = re.compile(r"(\bdef\s+\w+\(\s*self)\s*:[^,)]+")


def _strip_self_annotation(line: str) -> str:
    """Drop a type annotation on a definition's ``self`` parameter
    (``def foo(self: Target)`` -> ``def foo(self)``).

    Relocating a ``@staticmethod def foo(self: Target)`` into ``Target`` as a normal
    instance method ``def foo(self)`` drops the decorator (a whitelisted artifact) and the
    now-redundant annotation on ``self``; both are mechanical side-effects of the move, so
    the annotation is normalised away before the block comparison. See verifier-spec.md.
    """
    return _SELF_ANNOTATION.sub(r"\1", line)


def _block_signature(lines: list[str]) -> list[str]:
    """The block's non-blank lines with its common leading indent removed, in order.

    Blank lines are dropped: separator blank lines legitimately collapse when code is split
    across files or relocated, and a blank line never changes Python behavior. A *uniform*
    indentation shift is absorbed (the common prefix is removed), while relative
    indentation, trailing whitespace, and order are preserved -- so a non-uniform indent
    change, a trailing-whitespace change, a line merge, or a reorder makes the signature
    differ. See verifier-spec.md.
    """
    non_blank = [line for line in lines if line.strip()]
    indents = [len(line) - len(line.lstrip()) for line in non_blank]
    common = min(indents) if indents else 0
    return [line[common:] for line in non_blank]


def _peel_artifacts(
    lines: list[str], import_set: set[str]
) -> tuple[list[str], list[str], list[str]]:
    """Split lines into (imports, decorators, block), preserving block order and
    whitespace. Imports and decorators are matched on their stripped text."""
    imports, decorators, block = [], [], []
    for line in lines:
        stripped = line.strip()
        if stripped in import_set:
            imports.append(stripped)
        elif stripped in _MOVE_DECORATORS:
            decorators.append(stripped)
        else:
            block.append(line)
    return imports, decorators, block


def _peel_requalifications(
    rem_block: list[str], add_block: list[str], names: set[str]
) -> tuple[list[str], list[str], int]:
    """Remove call-site requalification pairs: a removed line and an added line that match
    after dropping a moved symbol's qualifier (a qualifier must actually be present).
    These are caller lines that stay in place; what remains is the relocated block."""
    add_pool = list(add_block)
    rem_remaining: list[str] = []
    count = 0
    for removed_line in rem_block:
        requalified = _strip_moved_qualifiers(removed_line, names)
        if requalified != removed_line:
            match = next(
                (
                    a
                    for a in add_pool
                    if _strip_moved_qualifiers(a, names) == requalified
                ),
                None,
            )
            if match is not None:
                add_pool.remove(match)
                count += 1
                continue
        rem_remaining.append(removed_line)
    return rem_remaining, add_pool, count


def verify_move_commit(commit: str, *, repo_root: str | None = None) -> bool:
    """Certify a commit is a pure relocation. Implements verifier-spec.md exactly."""
    root = repo_root or _repo_root()
    removed, added = _commit_changed_lines(commit, root)
    imports_before, imports_after = _commit_import_texts(commit, root)

    rem_keys = Counter(line.strip() for line in removed if line.strip())
    add_keys = Counter(line.strip() for line in added if line.strip())
    moved_names = _moved_symbol_names(list((rem_keys & add_keys).elements()))

    rem_imports, rem_decos, rem_block = _peel_artifacts(removed, imports_before)
    add_imports, add_decos, add_block = _peel_artifacts(added, imports_after)
    rem_block, add_block, requalified = _peel_requalifications(
        rem_block, add_block, moved_names
    )

    rem_signature = [
        _strip_self_annotation(line) for line in _block_signature(rem_block)
    ]
    add_signature = [
        _strip_self_annotation(line) for line in _block_signature(add_block)
    ]
    block_matches = rem_signature == add_signature
    relocated = len(rem_signature)
    nothing_changed = not removed and not added
    clean = block_matches and (relocated > 0 or nothing_changed)

    print(f"commit {commit}: move check (rule: verifier-spec.md)")
    for text in sorted(set(rem_imports + add_imports)):
        print(f"    [import] {text}")
    for text in sorted(set(rem_decos + add_decos)):
        print(f"    [decorator] {text}")
    if requalified:
        print(f"  {requalified} call-site requalification(s) of moved symbol(s)")
    if clean and relocated > 0:
        print(
            f"  {relocated} line(s) relocated in order (uniform indentation shift allowed)"
        )
        print("  => CLEAN MOVE")
    elif clean:
        print("  => CLEAN (pure rename; no content changed)")
    else:
        print("  the moved block is not an in-order, uniform-shift match:")
        for line in difflib.unified_diff(
            rem_signature,
            add_signature,
            fromfile="removed",
            tofile="added",
            lineterm="",
        ):
            print(f"    [review] {line}")
        print("  => NEEDS REVIEW")
    return clean


def verify_move_range(
    rev_range: str, *, match: str | None = None, repo_root: str | None = None
) -> bool:
    """Verify every commit in a git range (e.g. ``main..branch``), oldest first.

    With ``match`` (a regex) only commits whose subject matches are verified; the rest
    are skipped, so a long chain's log stays focused on, say, the ``-move`` commits.
    Prints each verified commit's report and a final summary; returns True iff every
    verified commit is clean.
    """
    root = repo_root or _repo_root()
    commits = _git_output(["rev-list", "--reverse", rev_range], root).split()
    pattern = re.compile(match) if match else None

    results: list[tuple[str, str, bool]] = []
    skipped = 0
    for commit in commits:
        subject = _git_output(["log", "-1", "--format=%s", commit], root).strip()
        if pattern is not None and not pattern.search(subject):
            skipped += 1
            continue
        clean = verify_move_commit(commit, repo_root=root)
        print()
        results.append((commit[:9], subject, clean))

    n_clean = sum(1 for _, _, clean in results if clean)
    print("=" * 72)
    print(
        f"verified {len(results)} commit(s)"
        + (f", skipped {skipped} (no subject match)" if pattern is not None else "")
        + f": {n_clean} clean, {len(results) - n_clean} need review"
    )
    for short, subject, clean in results:
        print(f"  {'CLEAN ' if clean else 'REVIEW'}  {short}  {subject}")
    return all(clean for _, _, clean in results)


def _main(argv: list[str]) -> int:
    match: str | None = None
    if "--match" in argv:
        i = argv.index("--match")
        if i + 1 < len(argv):
            match = argv[i + 1]
            argv = argv[:i] + argv[i + 2 :]
    if len(argv) == 1:
        target = argv[0]
        if ".." in target:
            return 0 if verify_move_range(target, match=match) else 1
        return 0 if verify_move_commit(target) else 1
    print(
        "usage: python3 mechanical_refactor_verify_utils.py <commit>\n"
        "       python3 mechanical_refactor_verify_utils.py <base>..<tip> [--match REGEX]",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
