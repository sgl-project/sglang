"""
    Bench the huggingface vLM with benchmark MMMU

    Usage:
        python benchmark/mmmu/bench_hf.py --model-path Qwen/Qwen2-VL-7B-Instruct --dataset-path

    The eval output will be logged
"""

import argparse
import random
import re

import torch
from data_utils import save_json
from eval_utils import (
    EvalArgs,
    eval_result,
    get_sampling_params,
    load_model,
    prepare_samples,
    process_result,
)
from Qwen2VLchat import Qwen2VLchat
from tqdm import tqdm


@torch.no_grad()
def eval_mmmu(args):
    eval_args = EvalArgs.from_cli_args(args)
    model = load_model(args.model_path)
    model.build_model()
    samples = prepare_samples(eval_args)
    out_samples = dict()
    answer_dict = {}
    for sample in tqdm(samples):
        image = sample["image_1"]
        if image is not None:
            response = model.chat(sample)
        else:  # multiple images actually
            if sample["question_type"] == "multiple-choice":
                all_choices = sample["all_choices"]
                response = random.choice(all_choices)
            else:
                response = "INVALID GENERATION FOR MULTIPLE IMAGE INPUTS"
        process_result(response, sample, answer_dict, out_samples)

    args.output_path = f"{args.model_path}_val_hf.json"
    save_json(args.output_path, out_samples)
    eval_result(model_answer_path=args.output_path, answer_dict=answer_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
        required=True,
    )
    EvalArgs.add_cli_args(parser)
    args = parser.parse_args()

    eval_mmmu(args)
