"""Summarize the case outputs without repeating their pass/fail checks."""

import json
import os
import subprocess
import sys
from pathlib import Path


def main(output):
    root = Path(os.environ["SGLANG_ROOT"])
    kernel_root = Path(os.environ["KERNEL_ROOT"])
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip() or "未知"
    kernel_commit = subprocess.run(
        ["git", "-C", str(kernel_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip() or "未知"
    lines = [
        "# Qwen3-8B MHA HiCache KV split 端到端测试报告",
        "",
        f"- 代码：`{root}`，提交 `{commit}`",
        f"- 算子：`{kernel_root}`，提交 `{kernel_commit}`",
        f"- 模型：`{os.environ['MODEL_PATH']}`",
        f"- NPU 卡号：`{os.environ['NPU_DEVICES']}`；TP：`{os.environ['TP_SIZE']}`",
        f"- 端口：`{os.environ['PORT']}`；Host 内存后端：`{os.environ['SGLANG_HICACHE_HOST_MEM_BACKEND']}`",
        f"- 每布局请求数：`{int(os.environ['FILLER_REQUESTS']) + 2}`；生成质量/精度：不适用",
        "",
        "| 布局 | 结论 | D2H KV token 增量 | H2D KV token 增量 | Host 命中 token | 输出 token 一致数 | 原因 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    passed = True
    for layout in ("layer_first", "page_first"):
        result_file = output / f"{layout}.json"
        result = json.loads(result_file.read_text(encoding="utf-8")) if result_file.exists() else {}
        code_file = output / f"{layout}.exit_code"
        code = code_file.read_text(encoding="utf-8").strip() if code_file.exists() else "未运行"
        okay = result.get("passed") is True and code == "0"
        passed &= okay
        host_tokens = result.get("cached_tokens_details", {}).get("host", "—")
        first_ids = result.get("first_token_ids")
        replay_ids = result.get("replay_token_ids")
        tokens_match = (
            sum(a == b for a, b in zip(first_ids, replay_ids))
            if first_ids is not None and replay_ids is not None
            else "—"
        )
        reason = result.get("error", "" if okay else f"exit_code={code}")
        lines.append(
            f"| {layout} | {'通过' if okay else '失败'} | "
            f"{result.get('backup_tokens_delta', '—')} | "
            f"{result.get('load_back_tokens_delta', '—')} | {host_tokens} | "
            f"{tokens_match} | {str(reason).replace('|', '/')} |"
        )
    lines.insert(2, f"**结论：{'通过' if passed else '失败'}。**")
    lines += [
        "",
        "## 逐用例结果",
        "",
        "| 用例 | 首次输出 token ID（真值） | 重放输出 token ID（实际） | 是否通过 |",
        "| --- | --- | --- | --- |",
    ]
    for layout in ("layer_first", "page_first"):
        path = output / f"{layout}.json"
        result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        lines.append(
            f"| {layout} | `{result.get('first_token_ids', '—')}` | "
            f"`{result.get('replay_token_ids', '—')}` | "
            f"{'是' if result.get('passed') else '否'} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
