"""类成员归属对账脚本（cherry-pick 迁移盲区形态 5 检查，见 SKILL.md 5.4）。

用法：
    python audit_class_placement.py <base_ref> [<source_tip_ref>]

- base_ref：迁移基线（如 backup-pre-pick-v0519），脚本审计 base_ref..HEAD 全部提交
- 对每个提交，按 subject 在全部分支中寻找源拷贝（同 subject 的原始 PPU 提交），
  用 AST 比较新增 def 的所属类：源所属类 != 目标所属类即 MISPLACED 告警
- 「兄弟方法集重叠度」投票给出目标终态中最匹配的类，辅助人工确认正确归属
- v0.5.19 实战：65 提交全量审计，唯一命中 = is_acext 落进 DeepEPv2Fp8ScaleFormat
"""

import ast
import subprocess
import sys
from collections import defaultdict


def git_out(*args):
    r = subprocess.run(
        ["git"] + list(args),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return r.stdout if r.returncode == 0 else None


_blob_cache = {}


def blob(rev, path):
    key = (rev, path)
    if key not in _blob_cache:
        _blob_cache[key] = git_out("show", f"{rev}:{path}")
    return _blob_cache[key]


_ana_cache = {}


class Collector(ast.NodeVisitor):
    def __init__(self):
        self.stack = []
        self.class_methods = {}
        self.defs = []

    def visit_ClassDef(self, node):
        self.stack.append(node.name)
        self.class_methods.setdefault(".".join(self.stack), set())
        self.generic_visit(node)
        self.stack.pop()

    def _func(self, node):
        owner = ".".join(self.stack) if self.stack else "<module>"
        self.defs.append((owner, node.name))
        if self.stack:
            self.class_methods[".".join(self.stack)].add(node.name)
        self.generic_visit(node)

    visit_FunctionDef = _func
    visit_AsyncFunctionDef = _func


def analyze(rev, path):
    key = (rev, path)
    if key in _ana_cache:
        return _ana_cache[key]
    src = blob(rev, path)
    res = (None, set(), None)
    if src is not None:
        try:
            tree = ast.parse(src)
            c = Collector()
            c.visit(tree)
            res = (c, {n for _, n in c.defs}, None)
        except SyntaxError:
            pass
    _ana_cache[key] = res
    return res


import sys

BASE = sys.argv[1] if len(sys.argv) > 1 else "backup-pre-pick-v0519"
revs = git_out("rev-list", "--reverse", f"{BASE}..HEAD").split()
print(f"base ref: {BASE}", flush=True)
in_range = set(revs)

subject_map = defaultdict(list)
for line in git_out("log", "--all", "--format=%H%x09%aI%x09%s").splitlines():
    if line.count("\t") >= 2:
        h, ad, s = line.split("\t", 2)
        subject_map[s].append((ad, h))

print(f"range commits: {len(revs)}; subjects indexed: {len(subject_map)}", flush=True)
findings = []
for C in revs:
    subject = git_out("log", "-1", "--format=%s", C).strip()
    files = [
        f
        for f in (
            git_out("diff-tree", "--no-commit-id", "--name-only", "-r", C) or ""
        ).splitlines()
        if f.endswith(".py")
    ]
    srcs = [
        h
        for _, h in sorted(subject_map.get(subject, []), reverse=True)
        if h not in in_range
    ][:2]
    if not srcs or not files:
        continue
    for path in files:
        c_new, names_new, _ = analyze(C, path)
        _, names_old, _ = analyze(C + "^", path)
        if c_new is None:
            continue
        added = names_new - names_old
        if not added:
            continue
        for S in srcs:
            c_s, names_s, _ = analyze(S, path)
            _, names_s_old, _ = analyze(S + "^", path)
            if c_s is None:
                continue
            added_s = names_s - names_s_old
            for name in sorted(added):
                if name not in added_s:
                    continue
                src_owner = next((o for o, n in c_s.defs if n == name), None)
                tgt_owner = next((o for o, n in c_new.defs if n == name), None)
                if src_owner is None or tgt_owner is None or src_owner == tgt_owner:
                    continue
                src_methods = c_s.class_methods.get(src_owner, set())
                tgt_methods = c_new.class_methods.get(tgt_owner, set())
                overlap = len(src_methods & tgt_methods)
                best, best_n = None, 0
                for q, ms in c_new.class_methods.items():
                    n = len(ms & src_methods)
                    if n > best_n:
                        best, best_n = q, n
                findings.append(
                    (
                        C[:12],
                        S[:12],
                        path,
                        name,
                        src_owner,
                        tgt_owner,
                        len(src_methods),
                        overlap,
                        best,
                        best_n,
                    )
                )
    print(f"done {C[:12]} {subject[:60]}", flush=True)

print(f"\ncross-class def placements found: {len(findings)}", flush=True)
for C, S, path, name, so, to, ns, no, best, bn in findings:
    # landed == best matching class (sibling-overlap vote) => the move mirrors
    # the community refactor (e.g. enum-body methods -> mixin): legitimate.
    # landed != best => method landed in an unrelated class: MISPLACED.
    status = "ADAPTED" if (best is not None and to == best) else "MISPLACED"
    print(f"[{status}] {C} <- src {S}")
    print(f"    {path} :: def {name}")
    print(
        f"    source class = {so} ({ns} methods) -> landed in target class = {to} (sibling overlap {no})"
    )
    print(f"    best matching target class = {best} (overlap {bn})")
sys.stdout.flush()
