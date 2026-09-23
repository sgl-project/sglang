#!/usr/bin/env python3
"""
扫描 HLE 评测日志目录，使用 LLM Judge (对齐 evalscope 官方评分) 或正则 fallback 评分。

用法:
    # LLM Judge 模式 (对齐官方评分，需 judge 模型 API 可用)
    python analyze_hle_results.py <日志目录路径> --judge-api http://127.0.0.1:8010/v1 --judge-model MiMo-V2.5-Pro-FP4-DFlash

    # 正则 fallback 模式 (无需 API)
    python analyze_hle_results.py <日志目录路径>

示例:
    python analyze_hle_results.py C:/heyao/work/20260922_125841 --judge-api http://127.0.0.1:8010/v1 --judge-model MiMo-V2.5-Pro-FP4-DFlash
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# 文件小于此值视为截断/空文件（仅含文件头）
TRUNCATED_THRESHOLD = 500  # bytes

# LLM Judge 提示词（完全对齐 evalscope HLE judge 格式）
JUDGE_TEMPLATE = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgment must focus only on if there are meaningful differences between [correct_answer] and the [response]. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match. Explain why the [response] is correct or incorrect based on [correct_answer] in one or two sentences. Finally, write your answer in the format 'GRADE: C' for correct answer or 'GRADE: I' for incorrect answer.


Reply with a single JSON object and no other text, containing these required keys (additional keys are allowed):
- "reasoning": a string
- "verdict": exactly one of "C" or "I\""""


def walk_jsonl_files(directory: str, folder: str) -> dict:
    """扫描 {directory}/{folder}/ 下所有 JSONL 文件，返回 {学科名: 文件路径}"""
    base = Path(directory) / folder
    result = {}
    if not base.exists():
        return result
    for jsonl in base.rglob("*.jsonl"):
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


def extract_sample_info(review_file: str) -> list:
    """从 reviews JSONL 提取每个样本的 question、model response、target"""
    samples = []
    with open(review_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                idx = d.get("index")
                target = d.get("target", [])

                # 提取 question (含 system prompt + user message)
                question = ""
                response_text = ""
                messages = d.get("messages", [])
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "system":
                        question += f"**System**: {content}\n\n"
                    elif role == "user":
                        user_text = ""
                        if isinstance(content, list):
                            user_text = "\n".join(
                                b.get("text", "") for b in content
                                if isinstance(b, dict) and b.get("type") == "text"
                            )
                        elif isinstance(content, str):
                            user_text = content
                        question += f"**User**: {user_text}"
                    elif role == "assistant":
                        if isinstance(content, list):
                            text_blocks = [
                                b.get("text", "") for b in content
                                if isinstance(b, dict) and b.get("type") == "text"
                            ]
                            response_text = "\n".join(text_blocks)
                        elif isinstance(content, str):
                            response_text = content

                samples.append({
                    "index": idx,
                    "question": question.strip(),
                    "response": response_text.strip() if response_text else "",
                    "target": target,
                })
            except json.JSONDecodeError:
                continue
    return samples


def call_judge_api(api_url: str, model: str, api_key: str,
                   question: str, response: str, correct_answer: str,
                   max_retries: int = 3) -> dict:
    """调用 OpenAI 兼容 API 执行 LLM Judge 评分"""
    import urllib.request
    import urllib.error

    prompt = JUDGE_TEMPLATE.format(
        question=question,
        response=response,
        correct_answer=correct_answer,
    )

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 512,
    }).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                f"{api_url.rstrip('/')}/chat/completions",
                data=payload,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"]
                return parse_judge_verdict(content, attempt)
        except urllib.error.URLError as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return {"verdict": None, "reasoning": str(e), "error": str(e)}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return {"verdict": None, "reasoning": str(e), "error": str(e)}

    return {"verdict": None, "reasoning": "max retries exceeded", "error": "max retries exceeded"}


def parse_judge_verdict(content: str, attempt: int) -> dict:
    """解析 Judge 返回的 verdict (C 或 I)"""
    # 尝试 JSON 解析
    try:
        # 清理可能的 markdown 包裹
        clean = content.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        data = json.loads(clean)
        verdict = data.get("verdict", "").strip().upper()
        reasoning = data.get("reasoning", content[:200])
        if verdict in ("C", "I"):
            return {"verdict": verdict, "reasoning": reasoning}
    except json.JSONDecodeError:
        pass

    # 正则 fallback
    m = re.search(r'GRADE:\s*([CI])', content, re.IGNORECASE)
    if m:
        return {"verdict": m.group(1).upper(), "reasoning": content[:200]}

    m = re.search(r'"verdict":\s*"([CI])"', content, re.IGNORECASE)
    if m:
        return {"verdict": m.group(1).upper(), "reasoning": content[:200]}

    return {"verdict": None, "reasoning": content[:200], "error": "failed to parse verdict"}


def score_with_judge(samples: list, api_url: str, model: str, api_key: str = "EMPTY",
                     verbose: bool = False) -> tuple:
    """使用 LLM Judge 评分，返回 (结果列表, 正确数, 总样本数)"""
    results = []
    correct = 0
    total = 0

    for sample in samples:
        idx = sample["index"]
        target = sample["target"]
        correct_answer = "; ".join(target)  # 拼接多个 target

        verdict = call_judge_api(
            api_url, model, api_key,
            sample["question"], sample["response"], correct_answer,
        )

        total += 1
        is_correct = verdict["verdict"] == "C"
        if is_correct:
            correct += 1
        symbol = "✅" if is_correct else ("⚠️" if verdict["verdict"] is None else "❌")

        result = {
            "index": idx,
            "target": target,
            "verdict": verdict["verdict"],  # C / I / None
            "correct": is_correct,
            "reasoning": verdict.get("reasoning", "")[:120],
        }
        results.append(result)

        if verbose:
            print(f"    #{idx}: target={target} | verdict={verdict['verdict']} "
                  f"| {result['reasoning']} {symbol}")

    return results, correct, total


def score_with_regex(review_file: str, verbose: bool = False) -> tuple:
    """正则 fallback 评分（简单字符串匹配）"""
    samples = extract_sample_info(review_file)
    results = []
    correct = 0
    total = 0

    for sample in samples:
        idx = sample["index"]
        target = sample["target"]
        response = sample["response"]

        # 正则提取 answer
        m = re.search(r"(?:Exact\s+)?Answer:\s*(.+)", response, re.IGNORECASE)
        answer = m.group(1).strip() if m else ""

        # 归一化比对
        target_vals = [t.strip().lower().rstrip(".") for t in target]
        model_ans = answer.strip().lower().rstrip(".")

        is_correct = model_ans in target_vals

        total += 1
        if is_correct:
            correct += 1
        symbol = "✅" if is_correct else "❌"

        answer_display = model_ans[:80] if model_ans else "(空)"
        results.append({
            "index": idx,
            "target": target,
            "model_answer": model_ans,
            "correct": is_correct,
        })

        if verbose:
            print(f"    #{idx}: target={target} | answer={answer_display} {symbol}")

    return results, correct, total


def main():
    parser = argparse.ArgumentParser(
        description="HLE 评测结果分析 (支持 LLM Judge / 正则 fallback)"
    )
    parser.add_argument("log_dir", help="日志目录路径")
    parser.add_argument("--judge-api", default=None,
                        help="Judge 模型 API URL (e.g. http://127.0.0.1:8010/v1)")
    parser.add_argument("--judge-model", default="MiMo-V2.5-Pro-FP4-DFlash",
                        help="Judge 模型名称")
    parser.add_argument("--judge-api-key", default="EMPTY",
                        help="API Key (默认 EMPTY)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="打印每个样本的评分详情")
    args = parser.parse_args()

    log_dir = args.log_dir
    if not os.path.isdir(log_dir):
        print(f"错误: 目录不存在: {log_dir}")
        sys.exit(1)

    use_judge = args.judge_api is not None

    # 扫描 predictions 和 reviews
    pred_files = walk_jsonl_files(log_dir, "predictions")
    review_files = walk_jsonl_files(log_dir, "reviews")

    all_subjects = sorted(set(pred_files.keys()) | set(review_files.keys()))

    mode_label = "LLM Judge (对齐 evalscope 官方评分)" if use_judge else "正则 fallback"
    print("=" * 80)
    print(f"日志目录: {log_dir}")
    print(f"评分模式: {mode_label}")
    if use_judge:
        print(f"Judge API: {args.judge_api} | Model: {args.judge_model}")
    print(f"发现 {len(all_subjects)} 个学科")
    print("=" * 80)

    # 统计
    complete_preds = 0
    complete_reviews = 0
    total_correct = 0
    total_samples = 0
    scorable_subjects = []

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

        # 评分
        pred_ok = subject in pred_files and analyze_file(pred_files[subject])["is_complete"]
        rev_ok = subject in review_files and analyze_file(review_files[subject])["is_complete"]

        if pred_ok and rev_ok:
            try:
                if use_judge:
                    samples = extract_sample_info(review_files[subject])
                    results, corr, tot = score_with_judge(
                        samples, args.judge_api, args.judge_model,
                        args.judge_api_key, verbose=args.verbose,
                    )
                else:
                    results, corr, tot = score_with_regex(
                        review_files[subject], verbose=args.verbose,
                    )

                total_correct += corr
                total_samples += tot
                scorable_subjects.append(subject)

                for r in results:
                    if use_judge:
                        symbol = "✅" if r["correct"] else ("⚠️" if r.get("verdict") is None else "❌")
                        print(f"    #{r['index']}: target={r['target']} | verdict={r.get('verdict','?')} {symbol}")
                    else:
                        symbol = "✅" if r["correct"] else "❌"
                        print(f"    #{r['index']}: target={r['target']} | answer={r['model_answer'][:80]} {symbol}")
            except Exception as e:
                print(f"    ⚠️ 评分出错: {e}")
        else:
            print(f"    ⚠️ 不可评分（文件不完整）")

    # 汇总
    print("\n" + "=" * 80)
    print("汇总")
    print("=" * 80)
    print(f"  评分模式:        {mode_label}")
    print(f"  完整 predictions: {complete_preds}/{len(all_subjects)}")
    print(f"  完整 reviews:     {complete_reviews}/{len(all_subjects)}")
    print(f"  可评分学科:       {len(scorable_subjects)}/{len(all_subjects)}")
    print(f"  可评分样本:       {total_samples}")
    if total_samples > 0:
        acc = total_correct / total_samples
        print(f"  正确:             {total_correct}/{total_samples} = {acc:.2%}")
        threshold = 0.33
        passed = "PASS" if acc >= threshold else "FAIL"
        print(f"  阈值 {threshold}:  {passed}")
    else:
        print(f"  正确:             0（无可评分样本）")

    if scorable_subjects:
        print(f"\n  可评分学科: {', '.join(scorable_subjects)}")


if __name__ == "__main__":
    main()