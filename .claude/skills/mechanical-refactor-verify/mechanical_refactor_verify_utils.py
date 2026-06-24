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
    python3 mechanical_refactor_verify_utils.py <base>..<tip> [--match REGEX] [--html OUT.html]

The range form verifies every commit in the range oldest first; ``--match`` restricts it
to commits whose subject matches the regex (e.g. ``--match -move:``) and skips the rest;
``--html`` also writes a standalone, self-contained HTML report for eyeballing.
"""

import ast
import difflib
import html
import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


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


def _module_level_lines(file_text: str) -> set[str]:
    """Stripped text of every non-blank top-level (indent 0) line in a file."""
    return {
        line.strip()
        for line in file_text.splitlines()
        if line.strip() and not line[:1].isspace()
    }


def _commit_before_module_lines(commit: str, repo_root: str) -> set[str]:
    """Top-level lines in the before-version of every file the commit touches -- the pool
    of module scaffolding (a logger, module-level constants, a ``TYPE_CHECKING`` guard) a
    new destination module may legitimately copy verbatim. See verifier-spec.md (whitelist).
    """
    status = _git_output(
        ["show", commit, "--name-status", "--format=", "--no-color", "--no-ext-diff"],
        repo_root,
    )
    lines: set[str] = set()
    for line in status.splitlines():
        if "\t" not in line:
            continue
        code, path = line.split("\t", 1)
        if code[:1] in ("D", "M"):
            lines |= _module_level_lines(
                _git_output(["show", f"{commit}^:{path}"], repo_root)
            )
    return lines


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


_LOGGER_RE = re.compile(r"^logger = logging\.getLogger\(")


def _is_universal_scaffold(stripped: str) -> bool:
    """A top-level line that is benign module boilerplate regardless of the source: a
    ``TYPE_CHECKING`` guard or a logger definition (these never carry logic)."""
    return stripped == "if TYPE_CHECKING:" or bool(_LOGGER_RE.match(stripped))


def _peel_scaffold(
    lines: list[str], source_module_lines: set[str], removed_stripped: set[str]
) -> tuple[list[str], list[str]]:
    """Remove top-level module scaffolding the destination module needs: a line copied
    byte-for-byte from a source file that was not itself relocated, or universal boilerplate
    (a logger / ``TYPE_CHECKING`` guard). The "not relocated" guard keeps a moved
    module-level function in the block. See verifier-spec.md (whitelist)."""
    scaffold, block = [], []
    for line in lines:
        stripped = line.strip()
        top_level = bool(stripped) and not line[:1].isspace()
        carried = top_level and stripped in source_module_lines
        universal = top_level and _is_universal_scaffold(stripped)
        if (carried or universal) and stripped not in removed_stripped:
            scaffold.append(stripped)
        else:
            block.append(line)
    return block, scaffold


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


@dataclass
class MoveCheck:
    """The structured verdict for one commit (so the CLI text and the HTML report render
    from the same data)."""

    commit: str
    clean: bool
    kind: str
    relocated: int
    imports: list[str]
    decorators: list[str]
    scaffold: list[str]
    requalified: int
    review_diff: list[str]
    subject: str = ""


def _check_move(commit: str, repo_root: str) -> MoveCheck:
    """Compute the verdict for a commit (no printing). Implements verifier-spec.md."""
    removed, added = _commit_changed_lines(commit, repo_root)
    imports_before, imports_after = _commit_import_texts(commit, repo_root)
    before_module_lines = _commit_before_module_lines(commit, repo_root)
    removed_stripped = {line.strip() for line in removed if line.strip()}

    rem_keys = Counter(line.strip() for line in removed if line.strip())
    add_keys = Counter(line.strip() for line in added if line.strip())
    moved_names = _moved_symbol_names(list((rem_keys & add_keys).elements()))

    rem_imports, rem_decos, rem_block = _peel_artifacts(removed, imports_before)
    add_imports, add_decos, add_block = _peel_artifacts(added, imports_after)
    add_block, scaffold = _peel_scaffold(
        add_block, before_module_lines, removed_stripped
    )
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

    if clean and relocated > 0:
        kind, review_diff = "CLEAN MOVE", []
    elif clean:
        kind, review_diff = "CLEAN (pure rename)", []
    else:
        kind = "NEEDS REVIEW"
        review_diff = list(
            difflib.unified_diff(
                rem_signature, add_signature, "removed", "added", lineterm=""
            )
        )
    return MoveCheck(
        commit=commit,
        clean=clean,
        kind=kind,
        relocated=relocated,
        imports=sorted(set(rem_imports + add_imports)),
        decorators=sorted(set(rem_decos + add_decos)),
        scaffold=sorted(set(scaffold)),
        requalified=requalified,
        review_diff=review_diff,
    )


def _print_check(check: MoveCheck) -> None:
    print(f"commit {check.commit}: move check (rule: verifier-spec.md)")
    for text in check.imports:
        print(f"    [import] {text}")
    for text in check.decorators:
        print(f"    [decorator] {text}")
    for text in check.scaffold:
        print(f"    [scaffold] {text}")
    if check.requalified:
        print(f"  {check.requalified} call-site requalification(s) of moved symbol(s)")
    if check.clean and check.relocated > 0:
        print(
            f"  {check.relocated} line(s) relocated in order (uniform indentation shift allowed)"
        )
        print("  => CLEAN MOVE")
    elif check.clean:
        print("  => CLEAN (pure rename; no content changed)")
    else:
        print("  the moved block is not an in-order, uniform-shift match:")
        for line in check.review_diff:
            print(f"    [review] {line}")
        print("  => NEEDS REVIEW")


def verify_move_commit(commit: str, *, repo_root: str | None = None) -> bool:
    """Certify a commit is a pure relocation. Implements verifier-spec.md exactly."""
    check = _check_move(commit, repo_root or _repo_root())
    _print_check(check)
    return check.clean


def verify_move_range(
    rev_range: str,
    *,
    match: str | None = None,
    html_path: str | None = None,
    repo_root: str | None = None,
) -> bool:
    """Verify every commit in a git range (e.g. ``main..branch``), oldest first.

    With ``match`` (a regex) only commits whose subject matches are verified; the rest are
    skipped, so a long chain's log stays focused on, say, the ``-move`` commits. With
    ``html_path`` a standalone HTML report is also written. Prints each verified commit's
    report and a final summary; returns True iff every verified commit is clean.
    """
    root = repo_root or _repo_root()
    commits = _git_output(["rev-list", "--reverse", rev_range], root).split()
    pattern = re.compile(match) if match else None

    checks: list[MoveCheck] = []
    skipped = 0
    for commit in commits:
        subject = _git_output(["log", "-1", "--format=%s", commit], root).strip()
        if pattern is not None and not pattern.search(subject):
            skipped += 1
            continue
        check = _check_move(commit, root)
        check.subject = subject
        _print_check(check)
        print()
        checks.append(check)

    n_clean = sum(1 for c in checks if c.clean)
    print("=" * 72)
    print(
        f"verified {len(checks)} commit(s)"
        + (f", skipped {skipped} (no subject match)" if pattern is not None else "")
        + f": {n_clean} clean, {len(checks) - n_clean} need review"
    )
    for check in checks:
        mark = "CLEAN " if check.clean else "REVIEW"
        print(f"  {mark}  {check.commit[:9]}  {check.subject}")
    if html_path is not None:
        _write_verify_html(html_path, rev_range, checks)
        print(f"html: {html_path}")
    return all(c.clean for c in checks)


def _write_verify_html(path: str, title: str, checks: list[MoveCheck]) -> None:
    """Write a standalone HTML report; the data is embedded as a JSON blob and rendered by
    the template's own JS, so the page is self-contained and offline."""
    payload = {
        "title": title,
        "clean": sum(1 for c in checks if c.clean),
        "review": sum(1 for c in checks if not c.clean),
        "checks": [asdict(c) for c in checks],
    }
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    content = _VERIFY_HTML_TEMPLATE.replace("__TITLE__", html.escape(title)).replace(
        "__DATA_JSON__", data_json
    )
    Path(path).write_text(content)


_VERIFY_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>move verify __TITLE__</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#fff;color:#1f2328;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
header{position:sticky;top:0;z-index:2;background:#fff;border-bottom:1px solid #d1d9e0;padding:10px 16px}
header h1{margin:0 0 4px;font-size:15px}
#meta{font-size:12px;color:#59636e}
#meta code{background:#f6f8fa;padding:1px 4px;border-radius:4px}
.chip{display:inline-block;padding:1px 8px;border-radius:10px;margin-right:6px;font-size:12px}
.chip.clean{background:#e6ffec;border:1px solid #4ac26b}
.chip.review{background:#ffebe9;border:1px solid #ff8182}
label.toggle{font-size:12px;color:#59636e;margin-left:8px;cursor:pointer;user-select:none}
main{padding:8px 16px 40px}
.card{border:1px solid #d1d9e0;border-left-width:4px;border-radius:6px;margin:8px 0}
.card.clean{border-left-color:#4ac26b}
.card.review{border-left-color:#ff8182}
.hd{padding:6px 10px;cursor:pointer;font-size:13px;display:flex;align-items:center;gap:8px}
.hd:hover{background:#f6f8fa}
.hd .sub{overflow-wrap:anywhere}
.badge{font-size:11px;font-weight:600;padding:1px 7px;border-radius:10px;flex:none}
.badge.clean{background:#e6ffec;border:1px solid #4ac26b;color:#1a7f37}
.badge.review{background:#ffebe9;border:1px solid #ff8182;color:#cf222e}
.hd code{color:#59636e;flex:none}
.bd{display:none;padding:6px 10px;border-top:1px solid #f0f1f3;font:11.5px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.card.open .bd{display:block}
.sec{margin:3px 0}
.sec.ok{color:#1a7f37}
.sec .k{color:#59636e}
.diff{border-collapse:collapse;width:100%;margin-top:4px;table-layout:fixed}
.diff td{white-space:pre-wrap;word-break:break-all;padding:0 6px}
.diff tr.add td{background:#e6ffec}
.diff tr.del td{background:#ffebe9}
.diff tr.hdr td{background:#eff5ff;color:#0550ae}
.diff tr.meta td{color:#59636e}
</style>
</head>
<body>
<header>
<h1>move verify: <code>__TITLE__</code></h1>
<div id="meta"></div>
<div style="margin-top:6px"><span id="counts"></span><label class="toggle"><input type="checkbox" id="onlyReview"> show only NEEDS REVIEW</label></div>
</header>
<main id="list"></main>
<script id="data" type="application/json">__DATA_JSON__</script>
<script>
'use strict';
const DATA = JSON.parse(document.getElementById('data').textContent);
const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
function lineClass(l){
  if(l.startsWith('@@')) return 'hdr';
  if(l.startsWith('+++')||l.startsWith('---')) return 'meta';
  if(l[0]==='+') return 'add';
  if(l[0]==='-') return 'del';
  return 'ctx';
}
function render(onlyReview){
  let out = '';
  DATA.checks.forEach((c,i) => {
    if(onlyReview && c.clean) return;
    const cls = c.clean ? 'clean' : 'review';
    out += '<div class="card '+cls+(c.clean?'':' open')+'">';
    out += '<div class="hd"><span class="badge '+cls+'">'+(c.clean?'CLEAN':'REVIEW')+'</span>';
    out += '<code>'+esc(c.commit.slice(0,9))+'</code><span class="sub">'+esc(c.subject||c.commit)+'</span></div>';
    out += '<div class="bd">';
    if(c.imports.length) out += '<div class="sec"><span class="k">imports:</span> '+c.imports.map(esc).join('<br>&nbsp;&nbsp;')+'</div>';
    if(c.decorators.length) out += '<div class="sec"><span class="k">decorators:</span> '+c.decorators.map(esc).join(', ')+'</div>';
    if(c.scaffold.length) out += '<div class="sec"><span class="k">scaffold:</span> '+c.scaffold.map(esc).join('<br>&nbsp;&nbsp;')+'</div>';
    if(c.requalified) out += '<div class="sec"><span class="k">requalified call sites:</span> '+c.requalified+'</div>';
    if(c.review_diff.length){
      out += '<table class="diff"><tbody>';
      c.review_diff.forEach(l => { out += '<tr class="'+lineClass(l)+'"><td>'+esc(l)+'</td></tr>'; });
      out += '</tbody></table>';
    } else {
      out += '<div class="sec ok">'+c.relocated+' line(s) relocated in order &mdash; '+esc(c.kind)+'</div>';
    }
    out += '</div></div>';
  });
  document.getElementById('list').innerHTML = out || '<p>(no commits matched)</p>';
}
document.getElementById('meta').innerHTML = 'range <code>'+esc(DATA.title)+'</code>';
document.getElementById('counts').innerHTML = '<span class="chip clean">'+DATA.clean+' CLEAN</span><span class="chip review">'+DATA.review+' NEEDS REVIEW</span>';
document.getElementById('list').addEventListener('click', e => {
  const hd = e.target.closest('.hd');
  if(hd) hd.parentNode.classList.toggle('open');
});
const onlyReview = document.getElementById('onlyReview');
onlyReview.addEventListener('change', () => render(onlyReview.checked));
render(false);
</script>
</body>
</html>"""


def _take_option(argv: list[str], name: str) -> tuple[str | None, list[str]]:
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return argv[i + 1], argv[:i] + argv[i + 2 :]
    return None, argv


def _main(argv: list[str]) -> int:
    match, argv = _take_option(argv, "--match")
    html_path, argv = _take_option(argv, "--html")
    if len(argv) == 1:
        target = argv[0]
        if ".." in target:
            return (
                0 if verify_move_range(target, match=match, html_path=html_path) else 1
            )
        return 0 if verify_move_commit(target) else 1
    print(
        "usage: python3 mechanical_refactor_verify_utils.py <commit>\n"
        "       python3 mechanical_refactor_verify_utils.py <base>..<tip> "
        "[--match REGEX] [--html OUT.html]",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
