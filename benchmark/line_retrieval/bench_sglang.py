import argparse
import json
import re
import time

import numpy as np

import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import dump_state_text


@sgl.function
def line_retrieval(s, prefix, suffix, body_0, body_1, body_2, body_3):
    s += prefix + "\n"

    contexts = [body_0, body_1, body_2, body_3]
    position_ids_offset = [i * 1000 for i in range(len(contexts))]
    forks = s.fork(len(contexts), position_ids_offset)
    forks += lambda i: contexts[i] + "\n"
    forks.join(mode="concate_and_append")

    s += "\n" + suffix
    s += sgl.gen("answer", max_tokens=16)


def eval_model(args, line_obj, num_hoops, src_indices, dst_percents):
    arguments = []
    labels = []
    sum_src_indices = []
    sum_dst_indices = []

    for i in range(len(src_indices)):
        for j in range(len(dst_percents)):
            src_index = src_indices[i]
            dst_percent = dst_percents[j]

            query_indices = line_obj["group_by_num_hoops"][str(num_hoops)]
            query_indices = [
                q
                for q in query_indices
                if all(l <= src_index for l in line_obj["links"][q]) and q < src_index
            ]
            dst_index = query_indices[
                min(int(len(query_indices) * dst_percent), len(query_indices) - 1)
            ]
            label = line_obj["values"][dst_index]

            body = line_obj["lines"][: src_index + 1]
            suffix = line_obj["suffix"].replace("???", line_obj["indices"][dst_index])
            body_part_len = len(body) // 4

            arguments.append(
                {
                    "prefix": line_obj["prefix"],
                    "body_0": "\n".join(body[:body_part_len]),
                    "body_1": "\n".join(body[body_part_len : 2 * body_part_len]),
                    "body_2": "\n".join(body[2 * body_part_len : 3 * body_part_len]),
                    "body_3": "\n".join(body[3 * body_part_len :]),
                    "suffix": suffix,
                }
            )
            labels.append(label)
            sum_src_indices.append(src_index)
            sum_dst_indices.append(dst_index)

    # Select backend
    backend = select_sglang_backend(args)

    tic = time.perf_counter()
    states = line_retrieval.run_batch(
        arguments,
        temperature=0,
        backend=backend,
        num_threads=args.parallel,
        progress_bar=True,
    )
    latency = time.perf_counter() - tic

    corrects = []
    for i in range(len(arguments)):
        output = states[i]["answer"]
        prompt_len = states[i].get_meta_info("answer").get("prompt_length", -1)
        label = labels[i]

        # Try all numbers
        findall = re.findall("\d+", output)
        if not findall:
            response_number = output
        else:
            for response_number in findall:
                if response_number == label:
                    break

        correct = response_number == label
        corrects.append(correct)

        # Log results
        summary = (
            f"Line index: {sum_src_indices[i]} -> {sum_dst_indices[i]}, "
            f"Prompt len: {prompt_len}, "
            f"Correct: {correct}, "
            f"Label: {label}, Predicted: {response_number}, "
        )
        print(summary)

    accuracy = np.mean(corrects)
    print(f"Accuracy: {accuracy:.3f}, latency: {latency:.2f} s")

    # Write results
    dump_state_text(f"tmp_output_{args.backend}.txt", states)

    with open(args.result_file, "a") as fout:
        value = {
            "task": "line_retrieval",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(latency, 3),
            "num_requests": len(arguments),
            "other": {
                "num_questions": len(arguments),
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")


def main(args):
    line_obj = json.load(open(args.data_path, "r"))

    num_hoops = args.num_hoops
    for src_index in args.src_index:
        src_indices = [src_index]
        num_queries = args.num_queries_per_src
        dst_percents = [i * (1 / (num_queries)) for i in range(num_queries)]
        eval_model(args, line_obj, num_hoops, src_indices, dst_percents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="lines_1000_0.0.json")
    parser.add_argument("--src-index", type=int, nargs="+", default=[100])
    parser.add_argument("--num-queries-per-src", type=int, default=10)
    parser.add_argument("--num-hoops", type=int, default=1)
    args = add_common_sglang_args_and_parse(parser)
    main(args)
