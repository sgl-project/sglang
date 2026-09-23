#!/usr/bin/env python3
"""
扫描 HLE 评测日志目录，统计 predictions/reviews 文件完整性并汇总可评分样本。

用法:
    python analyze_hle_results.py <日志目录路径>

示例:
    python analyze_hle_results.py C:/heyao/work/20260922_125841
"""

import json
import os
import re
import sys
from pathlib import Path

# 文件小于此值视为截断/空文件（仅含文件头）
TRUNCATED_THRESHOLD = 500  # bytes


def walk_jsonl_files(directory: str, folder: str) -> dict:
    """扫描 {directory}/{folder}/ 下所有 JSONL 文件，返回 {学科名: 文件路径}"""
    base = Path(directory) / folder
    result = {}
    if not base.exists():
        return result
    for jsonl in base.rglob("*.jsonl"):
        # 学科名 = 相对路径去掉 .jsonl
        subject = str(jsonl.relative_to(base)).replace(".jsonl", "").replace("\\", "/")
        result[subject] = str(jsonl)
    return result


def analyze_file(filepath: str) -> dict:
    """分析单个 JSONL 文件：大小、行数、是否完整"""
    path = Path(filepath)
    size = path.stat().st_size
    is_complete = size > TRUNCATED_THRESHOLD

    lines = 0
    if is_complete:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = sum(1 for line in f if line.strip())
        except Exception:
            pass

    return {
        "path": filepath,
        "size": size,
        "lines": lines,
        "is_complete": is_complete,
    }


def extract_target_from_review(filepath: str) -> dict:
    """从 reviews JSONL 提取每题正确答案，按 index 索引返回 "{index: target}" """
    targets = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                t = d.get("target", [])
                idx = d.get("index")
                if idx is not None:
                    targets[idx] = t
            except json.JSONDecodeError:
                continue
    return targets


def extract_answer_from_prediction(filepath: str) -> dict:
    """从 predictions JSONL 提取模型回答，按 index 返回 {index: "answer"}"""
    answers = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                idx = d.get("index")
                if idx is None:
                    continue
                content = (
                    d.get("model_output", {})
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", [])
                )
                # content 可能是 list[dict] 或纯字符串
                if isinstance(content, str):
                    full_text = content
                elif isinstance(content, list):
                    text_blocks = [
                        b.get("text", b.get("reasoning", ""))
                        for b in content
                        if isinstance(b, dict) and b.get("type") in ("text", "reasoning")
                    ]
                    full_text = "\n".join(text_blocks)
                else:
                    full_text = ""

                # 方式1：提取 Exact Answer: xxx 或 Answer: xxx
                m = re.search(r"(?:Exact\s+)?Answer:\s*(.+)", full_text, re.IGNORECASE)
                answer = m.group(1).strip() if m else ""

                # 方式2 fallback：若正则失败，取 full_text 最后一行非空内容
                if not answer:
                    lines = [l.strip() for l in full_text.split("\n") if l.strip()]
                    if lines:
                        answer = lines[-1]
                        # 清理 markdown 包裹
                        answer = re.sub(r"^\*\*|\*\*$|^`|`$|^\$|\$$", "", answer).strip()

                answers[idx] = answer
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
    return answers


def normalize_answer(a: str) -> str:
    """归一化答案用于比对"""
    return a.strip().lower().rstrip(".")


def score_answers(targets: dict, answers: dict) -> list:
    """比对 targets 和 answers（均为 {index: ...}），返回评分结果列表"""
    results = []

    for idx in sorted(targets.keys()):
        t = targets[idx]
        target_vals = [normalize_answer(v) for v in t]
        model_ans = normalize_answer(answers.get(idx, ""))

        # 匹配：模型答案是否在 target 列表中
        is_correct = model_ans in target_vals
        results.append({
            "index": idx,
            "target": t,
            "model_answer": model_ans,
            "correct": is_correct,
        })

    return results


def main():
    if len(sys.argv) < 2:
        print(f"用法: python {os.path.basename(__file__)} <日志目录路径>")
        print(f"示例: python {os.path.basename(__file__)} C:/heyao/work/20260922_125841")
        sys.exit(1)

    log_dir = sys.argv[1]
    if not os.path.isdir(log_dir):
        print(f"错误: 目录不存在: {log_dir}")
        sys.exit(1)

    # 扫描 predictions 和 reviews
    pred_files = walk_jsonl_files(log_dir, "predictions")
    review_files = walk_jsonl_files(log_dir, "reviews")

    all_subjects = sorted(set(pred_files.keys()) | set(review_files.keys()))

    print("=" * 80)
    print(f"日志目录: {log_dir}")
    print(f"发现 {len(all_subjects)} 个学科")
    print("=" * 80)

    # 统计文件状态
    complete_preds = 0
    complete_reviews = 0
    scorable = 0
    scorable_subjects = []
    total_correct = 0
    total_samples = 0

    for subject in all_subjects:
        print(f"\n--- {subject} ---")

        # Predictions
        if subject in pred_files:
            info = analyze_file(pred_files[subject])
            status = "完整" if info["is_complete"] else "截断/空"
            print(f"  predictions: {info['size']:>8} B, {info['lines']} 行 [{status}]")
            if info["is_complete"]:
                complete_preds += 1
        else:
            print(f"  predictions: 缺失")

        # Reviews
        if subject in review_files:
            info = analyze_file(review_files[subject])
            status = "完整" if info["is_complete"] else "截断/空"
            print(f"  reviews:     {info['size']:>8} B, {info['lines']} 行 [{status}]")
            if info["is_complete"]:
                complete_reviews += 1
        else:
            print(f"  reviews:     缺失")

        # 可评分判断
        pred_ok = subject in pred_files and analyze_file(pred_files[subject])["is_complete"]
        rev_ok = subject in review_files and analyze_file(review_files[subject])["is_complete"]

        if pred_ok and rev_ok:
            scorable += 1
            scorable_subjects.append(subject)

            targets = extract_target_from_review(review_files[subject])
            answers = extract_answer_from_prediction(pred_files[subject])
            results = score_answers(targets, answers)

            for r in results:
                total_samples += 1
                if r["correct"]:
                    total_correct += 1
                symbol = "✅" if r["correct"] else "❌"
                answer_display = r["model_answer"][:80] if r["model_answer"] else "(空)"
                print(f"    #{r['index']}: target={r['target']} | answer={answer_display} {symbol}")
        else:
            print(f"    ⚠️ 不可评分（文件不完整）")

    # 汇总
    print("\n" + "=" * 80)
    print("汇总")
    print("=" * 80)
    print(f"  完整 predictions: {complete_preds}/{len(all_subjects)}")
    print(f"  完整 reviews:     {complete_reviews}/{len(all_subjects)}")
    print(f"  可评分学科:       {scorable}/{len(all_subjects)}")
    print(f"  可评分样本:       {total_samples}")
    if total_samples > 0:
        acc = total_correct / total_samples
        print(f"  正确:             {total_correct}/{total_samples} = {acc:.2%}")
        # 判断是否通过 accuracy 阈值
        threshold = 0.33
        passed = "PASS" if acc >= threshold else "FAIL"
        print(f"  阈值 {threshold}:  {passed}")
    else:
        print(f"  正确:             0（无可评分样本）")

    print(f"\n可评分的学科: {', '.join(scorable_subjects) if scorable_subjects else '无'}")


if __name__ == "__main__":
    main()
