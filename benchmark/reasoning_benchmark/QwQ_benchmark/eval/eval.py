import json
import argparse
from tqdm import tqdm
import os

from math_opensource import compute_scores as compute_scores_math_opensource
from livecodebench_v5 import compute_scores as compute_scores_livecodebench_v5
from ifeval import compute_scores as compute_scores_ifeval

def get_after_think(text):
    parts = text.split("\n</think>\n\n", 1)
    if len(parts) > 1:
        return parts[1]
    else:
        return text
def main():
    parser = argparse.ArgumentParser(description="Evaluate model outputs")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input jsonl file")
    parser.add_argument("--cache_path", type=str, required=True, help="Path to save cache results")
    parser.add_argument("--task_name", type=str, required=True, help="Task should be in ['math_opensource/aime24', 'math_opensource/aime25' ,'livecodebench', 'ifeval']")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.cache_path), exist_ok=True)

    with open(args.input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    for item in data:
        item["task"] = args.task_name
        temp = get_after_think(item['gen'][0])
        item['gen'][0] = temp
    if "math_opensource" in args.task_name:
        acc = compute_scores_math_opensource(data, args.cache_path)
        print(f"Task: {args.task_name}, Accuracy: {acc}")
    elif "livecodebench" in args.task_name:
        acc = compute_scores_livecodebench_v5(data, args.cache_path)
        print(f"Task: {args.task_name}, Pass@1: {acc}")
    elif "ifeval" in args.task_name:
        acc = compute_scores_ifeval(data, args.cache_path)
        print(f"Task: {args.task_name}, Strict_prompt_acc: {acc}")
    else:
        print(f"No evaluation function found for task name: {args.task_name}")
    
    print("Evaluation complete!")

if __name__ == "__main__":
    main()