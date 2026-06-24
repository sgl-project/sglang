"""Infer a faithful reproduce recipe for a move commit, then emit and run a self-contained
reproduce script. Lets the verifier turn a commit a formatter re-wrapped into an auditable,
runnable reproduce script -- no one hand-writes it.

A recipe is inferred from the commit's diff and its before-state AST: which symbol moved
(src -> dst, into which class), which call sites were lowered, which local imports the
lowering orphaned, and which module-level imports the destination gained. ``recipe_to_script``
emits a standalone ``repro_scripts/<sha>.py`` (importing only the reproduce util); running it
reproduces the commit and diffs it byte-for-byte. ``generate_range`` writes a whole folder
(scripts + output.log + output.html) for a commit range.

Currently handles the "method moved onto an existing class, call sites lowered, local
imports removed" pattern. New-file extracts, free-function moves, and renames are not yet
inferred (they are reported as unsupported). Runnable directly:

    python3 mechanical_refactor_reproduce_gen_utils.py <commit>
    python3 mechanical_refactor_reproduce_gen_utils.py <base>..<tip> --match -move: --out DIR
"""

import ast
import html
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mechanical_refactor_reproduce_utils as rr
from mechanical_refactor_verify_utils import _git_output, _repo_root


def _def_indent(lines: list[str], name: str) -> int | None:
    for line in lines:
        match = re.match(r"(\s*)(?:async\s+)?def\s+" + re.escape(name) + r"\b", line)
        if match:
            return len(match.group(1))
    return None


def _per_file_diff(commit: str, root: str) -> dict[str, dict]:
    """Per-file removed/added content lines (whitespace intact) and a new-file flag."""
    out = _git_output(
        ["show", commit, "--format=", "--no-color", "--no-ext-diff"], root
    )
    files: dict[str, dict] = {}
    path: str | None = None
    for line in out.splitlines():
        if line.startswith("diff --git"):
            path = line.split(" b/")[-1]
            files[path] = {"removed": [], "added": [], "new": False}
        elif line.startswith("new file"):
            files[path]["new"] = True
        elif line.startswith(("+++", "---", "@@", "index ")):
            continue
        elif line.startswith("+"):
            files[path]["added"].append(line[1:])
        elif line.startswith("-"):
            files[path]["removed"].append(line[1:])
    return files


def _enclosing_function(tree: ast.AST, lineno: int) -> str | None:
    best: tuple[int, str] | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno <= lineno <= node.end_lineno:
                if best is None or node.lineno > best[0]:
                    best = (node.lineno, node.name)
    return best[1] if best else None


def _enclosing_class_of_def(tree: ast.AST, name: str) -> str | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if (
                    isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name == name
                ):
                    return node.name
    return None


def _nested_in_function(tree: ast.AST, name: str) -> bool:
    target = rr._find_def(tree, name)
    if target is None:
        return False
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node is not target
            and node.lineno <= target.lineno <= node.end_lineno
        ):
            return True
    return False


def _module_of_path(path: str) -> str:
    return path.removeprefix("python/").removesuffix(".py").replace("/", ".")


def _module_imports(text: str) -> dict[str, ast.AST]:
    out: dict[str, ast.AST] = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out[ast.unparse(node)] = node
    return out


def _local_import_of(
    tree: ast.AST, fn_name: str, module: str, symbol: str
) -> str | None:
    fn = rr._find_def(tree, fn_name)
    if fn is None:
        return None
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == module
            and any(alias.name == symbol for alias in node.names)
        ):
            return f"from {module} import {symbol}"
    return None


@dataclass
class Recipe:
    base: str
    target: str
    supported: bool = True
    moves: list = field(default_factory=list)
    lowerings: list = field(default_factory=list)
    import_removals: list = field(default_factory=list)
    import_additions: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def infer_recipe(commit: str, root: str) -> Recipe:
    """Infer a faithful relocation recipe for a move commit from its diff + before-state."""
    files = _per_file_diff(commit, root)
    recipe = Recipe(base=f"{commit}~1", target=commit)

    def def_names(lines: list[str]) -> set[str]:
        return {
            m.group(1)
            for ln in lines
            if (m := re.match(r"\s*(?:async\s+)?def\s+(\w+)", ln))
        }

    all_removed = [ln for f in files.values() for ln in f["removed"]]
    all_added = [ln for f in files.values() for ln in f["added"]]
    moved_names = def_names(all_removed) & def_names(all_added)
    if any(f["new"] for f in files.values()):
        recipe.supported = False
        recipe.notes.append("new-file extract: not yet inferred")

    for name in sorted(moved_names):
        src = next(
            (p for p, f in files.items() if name in def_names(f["removed"])), None
        )
        dst = next((p for p, f in files.items() if name in def_names(f["added"])), None)
        if src is None or dst is None or src == dst:
            continue
        src_before = _git_output(["show", f"{commit}^:{src}"], root)
        if _nested_in_function(ast.parse(src_before), name):
            recipe.notes.append(f"skip {name}: nested function (moves with parent)")
            continue

        dst_after = _git_output(["show", f"{commit}:{dst}"], root)
        dst_tree = ast.parse(dst_after)
        into_class = _enclosing_class_of_def(dst_tree, name)
        dst_def = rr._find_def(dst_tree, name)
        src_indent = _def_indent(files[src]["removed"], name) or 0
        dst_indent = _def_indent(files[dst]["added"], name) or 0
        recipe.moves.append(
            {
                "name": name,
                "src": src,
                "dst": dst,
                "into_class": into_class,
                "dedent": src_indent - dst_indent,
                "dst_order": dst_def.lineno if dst_def else 0,
            }
        )

        src_class = _enclosing_class_of_def(ast.parse(src_before), name)
        if src_class is None:
            recipe.supported = False
            recipe.notes.append(
                f"{name}: source is already a free function; callers not inferred"
            )
            continue
        # A move onto a class lowers the receiver out of the args; a move to a module-level
        # free function only drops the qualifier (requalify).
        kind = "lower" if into_class is not None else "requalify"
        src_module = _module_of_path(src)

        # A caller is a before-state call `<src_class>.name(...)`; its receiver moves out of
        # the argument list. The moved body's own calls have a different receiver, so they
        # are excluded -- which is why we match on src_class, not a loose text search.
        for path, f in files.items():
            before = _git_output(["show", f"{commit}^:{path}"], root)
            try:
                tree = ast.parse(before)
            except SyntaxError:
                continue
            caller_fns: set[str] = set()
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == name
                    and (node.args or kind == "requalify")
                    and ast.unparse(node.func.value) == src_class
                ):
                    fn = _enclosing_function(tree, node.lineno)
                    if fn is not None:
                        caller_fns.add(fn)
            if not caller_fns:
                continue
            recipe.lowerings.append(
                {"name": name, "owner": src_class, "path": path, "kind": kind}
            )
            for fn in sorted(caller_fns):
                imp = _local_import_of(tree, fn, src_module, src_class)
                if imp is not None and any(imp in r for r in f["removed"]):
                    recipe.import_removals.append(
                        {"path": path, "text": imp, "in_function": fn}
                    )

    # Module-level imports any touched file gained: the destination needs the moved code's
    # imports, and a caller of a moved free function gains an import of it. (In a pure move
    # commit the only import changes are move-related.)
    for path in sorted(files):
        before = _git_output(["show", f"{commit}^:{path}"], root)
        after = _git_output(["show", f"{commit}:{path}"], root)
        before_imports = _module_imports(before) if before.strip() else {}
        for stmt in _module_imports(after):
            if stmt not in before_imports:
                recipe.import_additions.append({"path": path, "text": stmt})
    return recipe


def build_repro(recipe: Recipe) -> rr.Repro:
    """A Repro that lowers/fixes imports BEFORE moving, so a call to a moved method from
    inside another moved method is lowered while still in the source and travels with the
    body; moves run last in destination order so multiple methods land correctly."""
    repro = rr.Repro(base=recipe.base, target=recipe.target)
    for lo in recipe.lowerings:
        if lo["kind"] == "requalify":
            repro.requalify_call_sites(lo["name"], lo["owner"], paths=[lo["path"]])
        else:
            repro.lower_call_sites(lo["name"], lo["owner"], paths=[lo["path"]])
    for im in recipe.import_removals:
        repro.remove_import(im["path"], im["text"], in_function=im["in_function"])
    for im in recipe.import_additions:
        repro.add_import(im["path"], im["text"])
    for mv in sorted(recipe.moves, key=lambda m: (m["dst"], m["dst_order"])):
        repro.move_symbol(
            mv["name"],
            src=mv["src"],
            dst=mv["dst"],
            into_class=mv["into_class"],
            dedent=mv["dedent"],
        )
    return repro


def recipe_to_script(recipe: Recipe, subject: str) -> str:
    """A standalone, auditable reproduce script (imports only the reproduce util)."""
    lines = [
        '"""Auto-generated reproduce script. Audit each call, then run.',
        "",
        f"commit:  {recipe.target}",
        f"subject: {subject}",
        "",
        "Each call is a faithful relocation primitive. Running this reproduces the commit",
        "in a throwaway worktree and diffs it byte-for-byte; PASS means the commit is",
        "exactly these relocations.",
        '"""',
        "import sys",
        "from pathlib import Path",
        "",
        "sys.path.insert(0, str(Path(__file__).resolve().parent.parent))",
        "from mechanical_refactor_reproduce_utils import Repro",
        "",
        f"r = Repro(base={recipe.base!r}, target={recipe.target!r})",
    ]
    for lo in recipe.lowerings:
        method = (
            "requalify_call_sites" if lo["kind"] == "requalify" else "lower_call_sites"
        )
        lines.append(
            f"r.{method}({lo['name']!r}, {lo['owner']!r}, paths=[{lo['path']!r}])"
        )
    for im in recipe.import_removals:
        lines.append(
            f"r.remove_import({im['path']!r}, {im['text']!r}, "
            f"in_function={im['in_function']!r})"
        )
    for im in recipe.import_additions:
        lines.append(f"r.add_import({im['path']!r}, {im['text']!r})")
    for mv in sorted(recipe.moves, key=lambda m: (m["dst"], m["dst_order"])):
        lines.append(
            f"r.move_symbol({mv['name']!r}, src={mv['src']!r}, dst={mv['dst']!r}, "
            f"into_class={mv['into_class']!r}, dedent={mv['dedent']})"
        )
    lines += ["r.run()", ""]
    return "\n".join(lines)


@dataclass
class GenResult:
    commit: str
    subject: str
    supported: bool
    passed: bool
    residual: str
    script: str
    notes: list


def generate_range(
    rev_range: str,
    *,
    match: str | None = None,
    out_dir: str,
    repo_root: str | None = None,
) -> list[GenResult]:
    """For each matched commit: infer a recipe, emit repro_scripts/<sha>.py, run it, and
    record PASS / residual. Writes output.log + output.html and copies the reproduce util
    so the folder is self-contained."""
    root = repo_root or _repo_root()
    commits = _git_output(["rev-list", "--reverse", rev_range], root).split()
    pattern = re.compile(match) if match else None

    out = Path(out_dir)
    scripts_dir = out / "repro_scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (out / "mechanical_refactor_reproduce_utils.py").write_text(
        Path(rr.__file__).read_text()
    )

    results: list[GenResult] = []
    for commit in commits:
        subject = _git_output(["log", "-1", "--format=%s", commit], root).strip()
        if pattern is not None and not pattern.search(subject):
            continue
        recipe = infer_recipe(commit, root)
        script = recipe_to_script(recipe, subject)
        (scripts_dir / f"{commit[:9]}.py").write_text(script)
        passed, residual = False, ""
        if recipe.supported and recipe.moves:
            residual = build_repro(recipe).run()
            passed = residual == ""
        results.append(
            GenResult(
                commit=commit,
                subject=subject,
                supported=recipe.supported and bool(recipe.moves),
                passed=passed,
                residual=residual,
                script=script,
                notes=recipe.notes,
            )
        )

    _write_log(out / "output.log", rev_range, results)
    _write_html(out / "output.html", rev_range, results)
    return results


def _write_log(path: Path, rev_range: str, results: list[GenResult]) -> None:
    n_pass = sum(1 for r in results if r.passed)
    lines = [
        f"reproduce-gen: {rev_range}",
        f"{len(results)} commit(s): {n_pass} reproduced, {len(results) - n_pass} not",
        "",
    ]
    for r in results:
        if r.passed:
            verdict = "PASS"
        elif not r.supported:
            verdict = "UNSUPPORTED (" + "; ".join(r.notes) + ")"
        else:
            verdict = f"RESIDUAL ({len(r.residual.splitlines())} lines)"
        lines.append(f"{r.commit[:9]}  {verdict}  {r.subject}")
    path.write_text("\n".join(lines) + "\n")


def _write_html(path: Path, rev_range: str, results: list[GenResult]) -> None:
    payload = {
        "title": rev_range,
        "passed": sum(1 for r in results if r.passed),
        "total": len(results),
        "results": [asdict(r) for r in results],
    }
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    path.write_text(
        _HTML_TEMPLATE.replace("__TITLE__", html.escape(rev_range)).replace(
            "__DATA_JSON__", data
        )
    )


_HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>reproduce-gen __TITLE__</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#fff;color:#1f2328;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #d1d9e0;padding:10px 16px}
header h1{margin:0 0 4px;font-size:15px}
.chip{display:inline-block;padding:1px 8px;border-radius:10px;margin-right:6px;font-size:12px}
.chip.ok{background:#e6ffec;border:1px solid #4ac26b}
.chip.no{background:#ffebe9;border:1px solid #ff8182}
main{padding:8px 16px 40px}
.card{border:1px solid #d1d9e0;border-left-width:4px;border-radius:6px;margin:8px 0}
.card.ok{border-left-color:#4ac26b}.card.no{border-left-color:#ff8182}.card.un{border-left-color:#d4a72c}
.hd{padding:6px 10px;cursor:pointer;font-size:13px;display:flex;align-items:center;gap:8px}
.hd:hover{background:#f6f8fa}
.badge{font-size:11px;font-weight:600;padding:1px 7px;border-radius:10px;flex:none}
.badge.ok{background:#e6ffec;border:1px solid #4ac26b;color:#1a7f37}
.badge.no{background:#ffebe9;border:1px solid #ff8182;color:#cf222e}
.badge.un{background:#fff8c5;border:1px solid #d4a72c;color:#7d4e00}
.hd code{color:#59636e;flex:none}
.bd{display:none;padding:6px 10px;border-top:1px solid #f0f1f3;font:11.5px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.card.open .bd{display:block}
.k{color:#59636e;margin-top:6px}
pre{white-space:pre-wrap;word-break:break-word;background:#f6f8fa;padding:8px;border-radius:6px;overflow-x:auto}
.res .d{white-space:pre-wrap;word-break:break-word}
.res .add{background:#e6ffec}.res .del{background:#ffebe9}.res .h{background:#eff5ff;color:#0550ae}
</style></head>
<body>
<header><h1>reproduce-gen: <code>__TITLE__</code></h1>
<div><span id="counts"></span></div></header>
<main id="list"></main>
<script id="data" type="application/json">__DATA_JSON__</script>
<script>
'use strict';
const DATA = JSON.parse(document.getElementById('data').textContent);
const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
function cls(r){ return r.passed?'ok':(r.supported?'no':'un'); }
function label(r){ return r.passed?'PASS':(r.supported?'RESIDUAL':'UNSUPPORTED'); }
let out='';
DATA.results.forEach(r => {
  const c = cls(r);
  out += '<div class="card '+c+'"><div class="hd"><span class="badge '+c+'">'+label(r)+'</span>';
  out += '<code>'+esc(r.commit.slice(0,9))+'</code><span>'+esc(r.subject)+'</span></div><div class="bd">';
  if(r.notes.length) out += '<div class="k">notes:</div><pre>'+esc(r.notes.join('\\n'))+'</pre>';
  out += '<div class="k">repro script:</div><pre>'+esc(r.script)+'</pre>';
  if(r.residual){
    out += '<div class="k">residual diff:</div><div class="res">';
    r.residual.split('\\n').forEach(l => {
      const k = l.startsWith('@@')?'h':(l[0]==='+'?'add':(l[0]==='-'?'del':''));
      out += '<div class="d '+k+'">'+esc(l)+'</div>';
    });
    out += '</div>';
  }
  out += '</div></div>';
});
document.getElementById('counts').innerHTML =
  '<span class="chip ok">'+DATA.passed+' reproduced</span><span class="chip no">'+(DATA.total-DATA.passed)+' not</span>';
document.getElementById('list').innerHTML = out;
document.getElementById('list').addEventListener('click', e => {
  const hd = e.target.closest('.hd'); if(hd) hd.parentNode.classList.toggle('open');
});
</script></body></html>"""


def _main(argv: list[str]) -> int:
    out_dir = None
    if "--out" in argv:
        i = argv.index("--out")
        out_dir = argv[i + 1]
        argv = argv[:i] + argv[i + 2 :]
    match = None
    if "--match" in argv:
        i = argv.index("--match")
        match = argv[i + 1]
        argv = argv[:i] + argv[i + 2 :]
    if len(argv) != 1:
        print(
            "usage: python3 mechanical_refactor_reproduce_gen_utils.py <commit>\n"
            "       python3 mechanical_refactor_reproduce_gen_utils.py <base>..<tip> "
            "[--match REGEX] --out DIR",
            file=sys.stderr,
        )
        return 2
    target = argv[0]
    if ".." in target:
        assert out_dir, "--out DIR is required for a range"
        results = generate_range(target, match=match, out_dir=out_dir)
        n = sum(1 for r in results if r.passed)
        print(f"{n}/{len(results)} reproduced; folder: {out_dir}")
        return 0
    root = _repo_root()
    recipe = infer_recipe(target, root)
    print(
        recipe_to_script(
            recipe, _git_output(["log", "-1", "--format=%s", target], root)
        )
    )
    if recipe.supported and recipe.moves:
        build_repro(recipe).run()
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
