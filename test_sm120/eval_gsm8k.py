"""Full GSM8K test set evaluation via OpenAI-compatible API.

Downloads the GSM8K test set (1319 questions) from HuggingFace datasets
and evaluates using 0-shot prompting.

Usage:
    python eval_gsm8k_full.py [--max-questions N] [--max-tokens 512] [--workers 4]
"""
import argparse
import json
import os
import re
import time
import urllib.request
import concurrent.futures
import sys

SERVER = "http://localhost:30000"
MODEL = "deepseek-ai/DeepSeek-V4-Flash"

GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
GSM8K_CACHE = os.path.join(os.path.dirname(__file__), "gsm8k_test.jsonl")


def download_gsm8k():
    """Download GSM8K test set if not cached."""
    if os.path.exists(GSM8K_CACHE):
        print(f"Using cached GSM8K: {GSM8K_CACHE}")
    else:
        print(f"Downloading GSM8K test set...")
        urllib.request.urlretrieve(GSM8K_URL, GSM8K_CACHE)
        print(f"Saved to {GSM8K_CACHE}")

    questions = []
    with open(GSM8K_CACHE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # Extract ground truth answer from "#### <number>" pattern in answer field
            answer_text = item["answer"]
            match = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_text)
            if match:
                gt = match.group(1).replace(",", "")
            else:
                # fallback: last number
                numbers = re.findall(r"-?\d[\d,]*\.?\d*", answer_text)
                gt = numbers[-1].replace(",", "") if numbers else "N/A"
            questions.append({
                "question": item["question"],
                "answer": gt,
            })
    return questions


def ask(question, max_tokens=512):
    """Send a single question to the server."""
    prompt = (
        "Solve this math problem step by step. "
        "At the end, give your final numerical answer after '#### '.\n\n"
        f"Question: {question}\nAnswer:"
    )
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"{SERVER}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=600)
        data = json.loads(resp.read())
        content = data["choices"][0]["message"]["content"]
        elapsed = time.time() - t0
        return content, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        return f"ERROR: {e}", elapsed


def extract_answer(text):
    """Extract numerical answer from model output."""
    # Try #### pattern first
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")
    # Try boxed pattern (common in reasoning models)
    match = re.search(r"\\boxed\{(-?[\d,]+\.?\d*)\}", text)
    if match:
        return match.group(1).replace(",", "")
    # Fallback: last number in text
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def eval_question(idx, q, max_tokens):
    """Evaluate a single question, return result dict."""
    response, elapsed = ask(q["question"], max_tokens)
    predicted = extract_answer(response)
    expected = q["answer"]

    # Normalize comparison (handle floats like 70000.0 vs 70000)
    try:
        is_correct = predicted is not None and float(predicted) == float(expected)
    except (ValueError, TypeError):
        is_correct = predicted is not None and str(predicted).strip() == str(expected).strip()

    return {
        "idx": idx,
        "expected": expected,
        "predicted": predicted,
        "correct": is_correct,
        "elapsed": elapsed,
        "response_snippet": response[:200] if not is_correct else "",
    }


def main():
    parser = argparse.ArgumentParser(description="Full GSM8K evaluation")
    parser.add_argument("--max-questions", type=int, default=0,
                        help="Max questions to eval (0 = all)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max output tokens per question")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent workers for parallel eval")
    parser.add_argument("--tp", type=int, default=8,
                        help="TP value (for display/logging)")
    args = parser.parse_args()

    # Health check
    try:
        req = urllib.request.Request(f"{SERVER}/health")
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"Server not reachable: {e}")
        sys.exit(1)

    questions = download_gsm8k()
    if args.max_questions > 0:
        questions = questions[:args.max_questions]

    total = len(questions)
    print(f"GSM8K Full Eval: {total} questions, {args.workers} workers, "
          f"max_tokens={args.max_tokens}, TP={args.tp}")
    print("=" * 70)

    start_time = time.time()
    results = []
    correct = 0
    errors = 0

    # Run with concurrent workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx = {}
        for i, q in enumerate(questions):
            f = executor.submit(eval_question, i, q, args.max_tokens)
            future_to_idx[f] = i

        for f in concurrent.futures.as_completed(future_to_idx):
            r = f.result()
            results.append(r)

            if r["correct"]:
                correct += 1
            if r["predicted"] is None:
                errors += 1

            done = len(results)
            acc_so_far = correct / done * 100
            status = "✓" if r["correct"] else "✗"
            print(f"  [{done:4d}/{total}] {status} "
                  f"Q{r['idx']+1:4d} exp={r['expected']:>10s} "
                  f"pred={str(r['predicted']):>10s} "
                  f"{r['elapsed']:.1f}s  "
                  f"(running acc: {acc_so_far:.1f}%)")

    total_time = time.time() - start_time
    accuracy = correct / total * 100 if total > 0 else 0.0

    # Sort results by index for final report
    results.sort(key=lambda x: x["idx"])

    print()
    print("=" * 70)
    print(f"GSM8K Results: {correct}/{total} = {accuracy:.1f}% accuracy")
    print(f"Total time: {total_time:.0f}s ({total_time/total:.1f}s per question)")
    print(f"Errors (no answer extracted): {errors}")
    print("=" * 70)

    # Show wrong answers
    wrong = [r for r in results if not r["correct"]]
    if wrong:
        print(f"\nWrong answers ({len(wrong)}):")
        for r in wrong[:20]:  # Show first 20
            print(f"  Q{r['idx']+1}: expected={r['expected']}, "
                  f"predicted={r['predicted']}")
            if r["response_snippet"]:
                snippet = r["response_snippet"].replace("\n", " ")[:120]
                print(f"    → {snippet}...")

    # Save detailed results
    outfile = f"gsm8k_full_tp{args.tp}.json"
    out = {
        "config": {
            "total_questions": total,
            "max_tokens": args.max_tokens,
            "workers": args.workers,
            "tp": args.tp,
            "gpu": "RTX PRO 6000 (SM120, 96GB GDDR7)",
        },
        "summary": {
            "correct": correct,
            "total": total,
            "accuracy_pct": accuracy,
            "total_time_s": total_time,
            "avg_time_per_q_s": total_time / total if total > 0 else 0,
            "errors": errors,
        },
        "results": results,
    }
    with open(outfile, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDetailed results saved to {outfile}")


if __name__ == "__main__":
    main()
