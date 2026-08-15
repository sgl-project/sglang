"""
Usage:
python3 bench_hf.py --model-path meta-llama/Llama-2-7b-hf --data-dir data --ntrain 5
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about{}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def main(args):
    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    ).eval()

    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    all_cors = []
    num_requests = 0
    total_latency = 0

    for subject in tqdm(subjects[: args.nsub]):
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        k = args.ntrain
        few_shot_examples = gen_prompt(dev_df, subject, k)
        while len(tokenizer.encode(few_shot_examples)) > 1536:
            k -= 1
            if k < 0:
                break
            few_shot_examples = gen_prompt(dev_df, subject, k)

        preds = []
        labels = []
        tic = time.perf_counter()

        for i in range(test_df.shape[0]):
            prompt_end = format_example(test_df, i, include_answer=False)
            prompt = few_shot_examples + prompt_end

            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            output_ids = model.generate(
                input_ids,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            output_str = tokenizer.decode(
                output_ids[0][input_ids.shape[-1] :], skip_special_tokens=True
            )
            preds.append(output_str.strip()[0] if len(output_str.strip()) > 0 else "")
            labels.append(test_df.iloc[i, test_df.shape[1] - 1])

        latency = time.perf_counter() - tic
        total_latency += latency

        cors = [pred == label for pred, label in zip(preds, labels)]
        all_cors.append(cors)
        num_requests += len(test_df)

        print(
            f"Subject: {subject}, Accuracy: {np.mean(cors):.3f}, Latency: {latency:.3f}s"
        )

    weighted_acc = np.mean(np.concatenate(all_cors))
    print(f"Total Latency: {total_latency:.3f}s")
    print(f"Average Accuracy: {weighted_acc:.3f}")

    if args.output:
        with open(args.output, "a") as fout:
            value = {
                "task": "mmlu",
                "backend": "hf",
                "model": args.model_path,
                "latency": round(total_latency, 3),
                "accuracy": round(weighted_acc, 3),
                "num_requests": num_requests,
                "other": {
                    "nsub": args.nsub,
                    "ntrain": args.ntrain,
                },
            }
            fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--nsub", type=int, default=60)
    parser.add_argument("--output", type=str, help="Output file path")
    args = parser.parse_args()
    main(args)
