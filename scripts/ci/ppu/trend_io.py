"""趋势文件读取边界：严格JSON、无符号链接、原始身份可核验。"""

import json
import os
import re
import sys
from pathlib import Path

# 只导入同一受信任checkout中的stdlib协议模块，不初始化sglang。
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python/sglang/test/kits"))
import trend_contract as contract

ARCHIVE = re.compile(
    r"(\d{4}-\d{2}-\d{2})-([1-9][0-9]{0,19})-([1-9][0-9]{0,19})\.jsonl\Z"
)


def no_links(path):
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        contract.require(not part.is_symlink(), "拒绝符号链接")


def files(root, *, suffix):
    root = Path(root)
    no_links(root)
    if not root.exists():
        return []
    result = []
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs + names:
            path = Path(directory) / name
            contract.require(not path.is_symlink(), "拒绝符号链接")
        result.extend(Path(directory) / name for name in names if name.endswith(suffix))
    return sorted(result)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        contract.require(key not in result, "重复JSON键")
        result[key] = value
    return result


def _constant(_value):
    raise ValueError("非有限JSON数值")


def read_rows(path, *, allow_legacy_perf=False):
    no_links(path)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line, object_pairs_hook=_pairs, parse_constant=_constant)
        contract.validate_row(value, allow_legacy_perf=allow_legacy_perf)
        rows.append((value, line))
    contract.require(bool(rows), "空趋势文件")
    return rows


def archived_rows(root):
    records = []
    for path in files(root, suffix=".jsonl"):
        relative = path.relative_to(root)
        match = ARCHIVE.fullmatch(path.name)
        contract.require(len(relative.parts) == 2 and match is not None, "无效归档路径")
        day, run, attempt = match.groups()
        # 已发布历史Perf行仍可读取；派生器只投影白名单数值，不重新发布原始字段。
        for row, line in read_rows(path, allow_legacy_perf=True):
            contract.require(relative.parts[0] == row["test_id"], "目录与行身份不符")
            contract.require(
                str(contract.utc(row["generated_at"]).date()) == day, "归档日期不符"
            )
            provenance = row["provenance"]
            claimed_run = provenance.get("github_run_id")
            if claimed_run is not None:
                contract.require(claimed_run == run, "归档run身份不符")
            if (
                row["schema_version"] != contract.PERF
                or provenance.get("github_run_attempt") is not None
            ):
                contract.require(
                    provenance.get("github_run_attempt") == attempt,
                    "归档attempt身份不符",
                )
            records.append(
                {
                    "row": row,
                    "run": run,
                    "attempt": attempt,
                    "time": contract.utc(row["generated_at"]),
                    "content": contract.canonical(row),
                }
            )
    return records
