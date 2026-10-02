"""配置读取对账（形态 6：访问惯用法漂移 / server_args 原始字段直读审计）

背景：v0.5.19 的声明式解析管线不再把声明「物化」回 ServerArgs 字段
（materialize_declarations 已删除），声明只进 stash、投影进 config bag
（get_schedule()/get_disagg()/... 分层访问器）。源分支若在解析结束有
物化步骤，则其 `self.server_args.<field>` 原始直读在源上是正确的读法，
cherry-pick 逐字搬运后在目标基线上变成读到 None/旧值——文本零信号。

判定（两级，均可机械执行）：
- DIVERGENT（高危，必须人工甄别为 0）：本提交**新增** `server_args.<field>`
  原始直读，且 parent 提交同文件中该字段已有 `get_<bag>().<field>` 访问器
  读法（同文件读法分歧——正是 page_size 炸点的特征）
- RESOLVABLE-RAW（中危，人工复核）：字段在 server_args.py 中标注
  `resolvable=True`（值可能来自模型覆盖/后处理 pass 的声明），而新增代码
  原始直读。CLI 直填字段（resolvable=False）直读合法 → BENIGN。

用法: python audit_serverargs_reads.py [BASE] [TIP]
默认 BASE=backup-pre-pick-v0519, TIP=HEAD
"""

import re
import subprocess
import sys
from collections import defaultdict

BASE = sys.argv[1] if len(sys.argv) > 1 else "backup-pre-pick-v0519"
TIP = sys.argv[2] if len(sys.argv) > 2 else "HEAD"


def git_out(*args):
    r = subprocess.run(
        ["git"] + list(args),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return r.stdout if r.returncode == 0 else None


def resolvable_field_names():
    """从目标基线 server_args.py 静态提取 resolvable=True 字段名集合。"""
    src = git_out("show", f"{TIP}:python/sglang/srt/server_args.py") or ""
    names = set()
    for m in re.finditer(r"\n    (\w+): A\[(.*?)\n    \]", src, re.S):
        if "resolvable=True" in m.group(2):
            names.add(m.group(1))
    for m in re.finditer(r"\n    (\w+): A\[([^\]\n]*)\]", src):
        if "resolvable=True" in m.group(2):
            names.add(m.group(1))
    return names


RESOLVABLE = resolvable_field_names()
print(f"resolvable fields: {len(RESOLVABLE)}", flush=True)

revs = git_out("rev-list", "--reverse", f"{BASE}..{TIP}").split()
print(f"base={BASE} tip={TIP} commits={len(revs)}", flush=True)

SRV_RE = re.compile(r"server_args\.(\w+)")
ACC_RE = r"get_\w+\(\)\.{}\b"
RAW_RE = r"server_args\.{}\b"

findings = []
for C in revs:
    subject = (git_out("log", "-1", "--format=%s", C) or "").strip()
    files = [
        f
        for f in (
            git_out("diff-tree", "--no-commit-id", "--name-only", "-r", C) or ""
        ).splitlines()
        if f.endswith(".py")
    ]
    if not files:
        continue
    diff = git_out("diff", f"{C}^", C, "--", *files)
    if not diff:
        continue
    cur_file = None
    added_by_file = defaultdict(list)
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            cur_file = line[6:]
        elif line.startswith("+") and not line.startswith("+++"):
            body = line[1:].strip()
            if not body.startswith("#"):  # 跳过纯注释行
                added_by_file[cur_file].append(body)
    for path, added_lines in added_by_file.items():
        parent = git_out("show", f"{C}^:{path}")
        if parent is None:  # 新文件: 无基线惯例可对比
            continue
        seen = set()
        for ln in added_lines:
            for m in SRV_RE.finditer(ln):
                field = m.group(1)
                if field in seen:
                    continue
                seen.add(field)
                acc_hits = re.findall(ACC_RE.format(field), parent)
                if not acc_hits:
                    continue  # parent 无访问器惯例: 非 DIVERGENT
                accs = sorted({h.split("(")[0] for h in acc_hits})
                raw_n = len(re.findall(RAW_RE.format(field), parent))
                level = (
                    "DIVERGENT"
                    if field in RESOLVABLE
                    else f"DIVERGENT-NO-RESOLVE(raw field)"
                )
                findings.append(
                    (level, C[:12], path, field, len(acc_hits), raw_n, accs, ln[:110])
                )
    print(f"done {C[:12]} {subject[:60]}", flush=True)

print(
    f"\n=== server_args raw-read divergence findings: {len(findings)} ===", flush=True
)
for level, C, path, field, acc, raw, accs, line in findings:
    print(f"[{level}] {C} {path}")
    print(f"    field={field}  parent: accessor={acc}({','.join(accs)}) raw={raw}")
    print(f"    added: {line}")
    print(
        f"    -> field resolvable={field in RESOLVABLE}"
        + (
            " (declared by model override/post-process pass: raw read is a None/stale-value hazard)"
            if field in RESOLVABLE
            else " (CLI-filled: raw read legitimate, but house style is the accessor)"
        )
    )
sys.stdout.flush()
