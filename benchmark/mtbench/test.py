#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import requests
from collections import defaultdict

# ========== 配置区 ==========
# 1) SGLang HTTP 服务地址（如果跑在本机 30000 端口，直接填下面这个即可）
SG_LANG_URL = "http://0.0.0.0:30000/v1/chat/completions"

# 2) 如果需要在请求里指定 model（与启动 SGLang 时使用相同的 model_id），就填在这里；否则留空
MODEL_ID = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"

# 3) system 提示，和 bench_sglang_eagle.py 中的 answer_mt_bench 一致
SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.  
Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.  
If you don't know the answer to a question, please don't share false information.
""".strip()

# 4) question.jsonl 文件路径
QUESTION_FILE = "/workspace/sglang/benchmark/mtbench/question.jsonl"

# 5) 并发线程数（可根据需要调整，如果不想并发，留 1 即可）
PARALLELISM = 1

# 6) 每次请求的解码参数
TEMPERATURE = 0
MAX_NEW_TOKENS = 2048
# ============================


def load_questions(path):
    """
    从本地 JSONL 文件中加载所有问题。
    每一行格式示例：
      {
        "question_id": "...",
        "category": "writing",
        "turns": ["第一轮问句", "第二轮问句"]
      }
    """
    questions = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            questions.append(obj)
    return questions


def call_sglang_api(user_question):
    """
    向 SGLang HTTP 服务发送一次 /v1/chat/completions 请求，
    参数：只包含 system + user 两轮对话，不带上下文。
    返回：从 response JSON 中提取出的 (completion_tokens, spec_verify_ct)。
    如果响应里没有 spec_verify_ct，就把它当成 1。
    """
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_question},
        ],
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    # 如果需要指定 model，就把它加上
    if MODEL_ID:
        payload["model"] = MODEL_ID

    resp = requests.post(SG_LANG_URL, json=payload)
    resp.raise_for_status()
    data = resp.json()

    # 假定返回里有 usage 字段，结构类似：
    # "usage": {
    #   "prompt_tokens": 37,
    #   "completion_tokens": 256,
    #   "spec_verify_ct": 101,
    #   ...
    # }
    usage = data.get("usage", {})
    comp_tokens = usage.get("completion_tokens", 0)
    verify_ct = usage.get("spec_verify_ct", None)

    # 如果没有 spec_verify_ct，直接让它等于 1（避免除零）
    if verify_ct is None or verify_ct <= 0:
        verify_ct = 1

    return comp_tokens, verify_ct


def main():
    # 1. 读取所有题目，按 category 分组
    questions = load_questions(QUESTION_FILE)
    by_category = defaultdict(list)
    for q in questions:
        cat = q.get("category", "unknown")
        turns = q.get("turns", [])
        if len(turns) < 2:
            continue
        by_category[cat].append((turns[0], turns[1]))

    # 2. 对每个 category，累加输出 token 和 验收 token
    results = {}
    for cat, qa_list in by_category.items():
        total_output_tokens = 0
        total_verify_tokens = 0

        print(f"正在处理类别：{cat}，共 {len(qa_list)} 个问题 ...")
        for idx, (q1, q2) in enumerate(qa_list, start=1):
            # 2.1 第一轮
            c1, v1 = call_sglang_api(q1)
            total_output_tokens += c1
            total_verify_tokens += v1

            # 2.2 第二轮
            c2, v2 = call_sglang_api(q2)
            total_output_tokens += c2
            total_verify_tokens += v2

            if idx % 10 == 0:
                print(f"  [{cat}] 已处理 {idx}/{len(qa_list)} ...")

        # 2.3 计算 Acceptance Rate
        acc_rate = total_output_tokens / total_verify_tokens if total_verify_tokens > 0 else 0.0
        results[cat] = acc_rate

    # 3. 打印最终结果
    print("\nCategory\tMT Bench Acceptance Rate")
    for cat, rate in sorted(results.items()):
        print(f"{cat}\t{rate:.2f}")


if __name__ == "__main__":
    main()
