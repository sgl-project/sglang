"""分析JSONL文件中的token统计信息

JSONL格式每行应为:
[{"role":"system","content":"xxx"},{"role":"user","content":"xxx"},...]
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from tqdm import tqdm


def count_tokens_for_conversation(tokenizer, messages):
    formatted_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    return len(tokenizer(formatted_text).input_ids)


def analyze_jsonl_tokens(jsonl_path, tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    dataset_name = jsonl_path.split("/")[-1].replace(".jsonl", "")
    token_counts = []
    line_tokens = []

    print("正在统计token数...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(tqdm(f), 1):
            messages = json.loads(line.strip())

            if not isinstance(messages, list):
                raise ValueError(f"第{line_idx}行不是有效的消息列表格式")

            token_count = count_tokens_for_conversation(tokenizer, messages)
            token_counts.append(token_count)
            line_tokens.append((line_idx, token_count))

    token_stats_file = f"token_stats_{dataset_name}.txt"
    with open(token_stats_file, "w", encoding="utf-8") as f:
        f.write("line,tokens\n")
        for line_idx, token_count in line_tokens:
            f.write(f"{line_idx},{token_count}\n")
    print(f"\nToken统计已保存至 {token_stats_file}")

    token_counts = np.array(token_counts)
    stats = {
        "Mean": np.mean(token_counts),
        "Median": np.median(token_counts),
        "Max": np.max(token_counts),
        "Min": np.min(token_counts),
        "Std": np.std(token_counts),
        "Total Lines": len(token_counts),
    }

    print("\nToken Statistics:")
    for metric, value in stats.items():
        if metric == "Total Lines":
            print(f"{metric}: {int(value)}")
        else:
            print(f"{metric}: {value:.2f}")

    plt.figure(figsize=(10, 6))
    plt.hist(token_counts, bins=50, edgecolor="black")
    plt.title(f"Token Length Distribution ({dataset_name})")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")

    stats_text = "\n".join([f'{k}: {int(v) if k == "Total Lines" else v:.2f}' for k, v in stats.items()])
    plt.text(
        0.95,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plot_filename = f"distribution_{dataset_name}.png"
    plt.savefig(plot_filename)
    print(f"\n分布图已保存为 {plot_filename}")


def main():
    parser = argparse.ArgumentParser(description="分析JSONL数据集的token统计信息")
    parser.add_argument("--jsonl-path", type=str, required=True, help="JSONL文件路径")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Tokenizer路径")

    args = parser.parse_args()
    analyze_jsonl_tokens(args.jsonl_path, args.tokenizer_path)


if __name__ == "__main__":
    main()
