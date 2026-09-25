"""固定数据分支的隔离发布事务；竞争时重新读取远端并重算。"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path

from derive_trend import derive
from render_trend import escape, render
from stage_trend_rows import stage
from trend_io import contract, no_links

BRANCH = "nightly-test-data"


def _git(repo, *args, check=True):
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True
    )
    if check and result.returncode:
        raise RuntimeError("Git操作失败；未确认发布")
    return result


def _push(repo):
    return (
        _git(
            repo, "push", "--quiet", "origin", "HEAD:refs/heads/" + BRANCH, check=False
        ).returncode
        == 0
    )


def _status(output, status, **fields):
    result = {"status": status, **fields}
    if output:
        no_links(output / "publication.json")
        (output / "publication.json").write_text(
            contract.canonical(result) + "\n", encoding="utf-8"
        )
    return result


_RESULTS = ("success", "failure", "cancelled", "skipped")


def run_context(needs, *, source_sha, workflow_sha, tool_sha, executed_at):
    """global运行上下文，仅进入运行summary；确定性trend.json不含这些字段。"""
    lines = [
        "",
        "## 本次运行上下文",
        "",
        "作业结束不代表全部报告已归档；缺报告以各原始发布job的对账为准。",
        "",
        "| 上游作业 | 结果 |",
        "| --- | --- |",
    ]
    for name in sorted((needs or {}).keys()):
        outcome = (needs[name] or {}).get("result")
        # 仅投影已知枚举，任意状态与outputs一律不回显。
        outcome = outcome if outcome in _RESULTS else "unknown"
        lines.append("| " + escape(name) + " | " + escape(outcome) + " |")
    lines += [
        "",
        "源码 SHA：" + escape(source_sha),
        "workflow SHA：" + escape(workflow_sha),
        "工具 SHA：" + escape(tool_sha),
        "执行 UTC 时间：" + escape(executed_at),
    ]
    return "\n".join(lines) + "\n"


def _derive_candidate(checkout, *, output, as_of_date, revision, input_commit):
    target = checkout / "trend.json"
    no_links(target)
    if target.exists():
        previous = json.loads(target.read_text(encoding="utf-8"))
        previous_date = date.fromisoformat(previous["as_of_date"])
        if previous_date > date.fromisoformat(as_of_date):
            return "superseded"
    tree = _git(checkout, "rev-parse", "HEAD:data").stdout.strip()
    summary = derive(
        checkout / "data",
        as_of_date=as_of_date,
        input_data_tree=tree,
        derivation_revision=revision,
    )
    content = contract.canonical(summary) + "\n"
    (output / "trend.json").write_text(content, encoding="utf-8")
    (output / "summary.md").write_text(render(summary), encoding="utf-8")
    (output / "input.json").write_text(
        contract.canonical(
            {
                "input_commit": input_commit,
                "input_data_tree": tree,
                "derivation_revision": revision,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    if target.exists() and target.read_text(encoding="utf-8") == content:
        return "unchanged"
    target.write_text(content, encoding="utf-8")
    _git(checkout, "add", "--", "trend.json")
    return "candidate"


def _commit(checkout, *, mode):
    staged = _git(checkout, "diff", "--cached", "--name-only").stdout.splitlines()
    contract.require(bool(staged), "没有候选变更")
    if mode == "derived":
        contract.require(staged == ["trend.json"], "派生发布只能写trend.json")
    else:
        contract.require(
            all(
                name.startswith("data/") and name.endswith(".jsonl") for name in staged
            ),
            "原始发布只能写data文件",
        )
        statuses = _git(
            checkout, "diff", "--cached", "--name-status"
        ).stdout.splitlines()
        contract.require(
            all(line.startswith("A\t") for line in statuses),
            "原始发布禁止修改或删除历史",
        )
    # 临时命令级身份不修改仓库Git配置。
    _git(
        checkout,
        "-c",
        "user.name=github-actions[bot]",
        "-c",
        "user.email=41898282+github-actions[bot]@users.noreply.github.com",
        "commit",
        "--quiet",
        "-m",
        "data(ppu): " + mode,
    )


@contextmanager
def _checkout(repo, tip):
    temporary = tempfile.TemporaryDirectory(prefix="ppu-trend-publish-")
    checkout = Path(temporary.name).resolve() / "checkout"
    try:
        _git(repo, "worktree", "add", "--detach", str(checkout), tip)
        yield checkout
    finally:
        # 校验失败可留下脏暂存区；先清理独占临时目录，再移除其注册。
        _git(repo, "worktree", "remove", str(checkout), check=False)
        temporary.cleanup()
        _git(repo, "worktree", "remove", str(checkout), check=False)


def publish(
    repo,
    *,
    mode,
    incoming=None,
    output=None,
    as_of_date=None,
    revision=None,
    run_id=None,
    attempt=None
):
    repo = Path(repo).resolve()
    contract.require(mode in {"derived", "rows"}, "未知发布模式")
    if output:
        no_links(output)
        output = Path(output).resolve()
        no_links(output / "publication.json")
        output.mkdir(parents=True, exist_ok=True)
        for name in ("trend.json", "summary.md", "input.json"):
            path = output / name
            no_links(path)
            if path.exists():
                path.unlink()
    _status(output, "failed")
    contract.require(
        not _git(repo, "status", "--porcelain").stdout, "发布checkout必须干净"
    )
    if mode == "derived":
        contract.require(
            output is not None and contract.SHA1.fullmatch(revision or ""),
            "派生输出或工具SHA缺失",
        )
        contract.require(
            date.fromisoformat(as_of_date).isoformat() == as_of_date, "无效发布日期"
        )
    if incoming:
        no_links(incoming)
        incoming = Path(incoming).resolve()
    for retry in range(6):
        _git(repo, "fetch", "--quiet", "origin", "refs/heads/" + BRANCH)
        tip = _git(repo, "rev-parse", "FETCH_HEAD").stdout.strip()
        with _checkout(repo, tip) as checkout:
            if mode == "derived":
                status = _derive_candidate(
                    checkout,
                    output=output,
                    as_of_date=as_of_date,
                    revision=revision,
                    input_commit=tip,
                )
                if status != "candidate":
                    return _status(
                        output, status, input_commit=tip, published_commit=tip
                    )
            else:
                stage(incoming, checkout / "data", run_id=run_id, attempt=attempt)
                if not _git(checkout, "status", "--porcelain").stdout:
                    return _status(output, "unchanged", published_commit=tip)
                _git(checkout, "add", "--", "data")
            _commit(checkout, mode=mode)
            if _push(checkout):
                commit = _git(checkout, "rev-parse", "HEAD").stdout.strip()
                return _status(
                    output, "published", input_commit=tip, published_commit=commit
                )
            _status(output, "failed", input_commit=tip, candidate_only=True)
        if retry < 5:
            time.sleep(min(5 * (retry + 1), 25))
    raise RuntimeError("六次发布未成功，候选结果未归档")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["rows", "derived"])
    args = parser.parse_args()
    output = (
        Path(os.environ["TREND_OUTPUT_DIR"])
        if os.environ.get("TREND_OUTPUT_DIR")
        else None
    )
    try:
        result = publish(
            Path.cwd(),
            mode=args.mode,
            incoming=Path(os.environ.get("INCOMING_DIR", "incoming")),
            output=output,
            as_of_date=os.environ.get("TREND_AS_OF_DATE"),
            revision=os.environ.get("TREND_TOOL_SHA"),
            run_id=os.environ.get("GITHUB_RUN_ID"),
            attempt=os.environ.get("GITHUB_RUN_ATTEMPT"),
        )
        print("趋势发布状态：" + result["status"])
    except (ValueError, RuntimeError, OSError, KeyError, TypeError):
        print("::error::趋势校验或发布失败；保留原始行与已发布产物，artifact仅为候选")
        raise SystemExit(1)
    finally:
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_path:
            with open(summary_path, "a", encoding="utf-8") as stream:
                if output:
                    status_file = output / "publication.json"
                    status = (
                        json.loads(status_file.read_text())
                        if status_file.exists()
                        else {"status": "failed"}
                    )
                    stream.write(
                        "\n发布状态："
                        + escape(status["status"])
                        + "（失败时附件仅为候选）\n"
                    )
                    for key in ("input_commit", "published_commit"):
                        stream.write(key + ": " + escape(status.get(key)) + "\n")
                    candidate = output / "summary.md"
                    if candidate.is_file():
                        stream.write(candidate.read_text(encoding="utf-8"))
                if args.mode == "derived":
                    raw = os.environ.get("NEEDS_JSON")
                    try:
                        needs = json.loads(raw) if raw else {}
                    except ValueError:
                        needs = {}
                    stream.write(
                        run_context(
                            needs,
                            source_sha=os.environ.get("RUN_SOURCE_SHA"),
                            workflow_sha=os.environ.get("RUN_WORKFLOW_SHA"),
                            tool_sha=os.environ.get("TREND_TOOL_SHA"),
                            executed_at=datetime.now(timezone.utc)
                            .replace(microsecond=0)
                            .isoformat()
                            .replace("+00:00", "Z"),
                        )
                    )


if __name__ == "__main__":
    main()
