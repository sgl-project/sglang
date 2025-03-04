"""
    Bench the sglang-hosted vLM with benchmark MMMU

    Usage:
        python benchmark/mmmu/bench_sglang.py --model-path Qwen/Qwen2-VL-7B-Instruct --chat-template qwen2-vl

    The eval output will be logged
"""

import argparse
import dataclasses
import random
import re
from io import BytesIO

from data_utils import save_json
from eval_utils import (
    EvalArgs,
    eval_result,
    get_sampling_params,
    parse_multi_choice_response,
    prepare_samples,
)
from tqdm import tqdm

from sglang import Engine
from sglang.srt.conversation import chat_templates
from sglang.srt.server_args import ServerArgs


def eval_mmmu(args):
    server_args = ServerArgs.from_cli_args(args)
    eval_args = EvalArgs.from_cli_args(args)

    if server_args.chat_template is None:
        raise ValueError("Chat template must be provided for this benchmark")

    samples = prepare_samples(eval_args)

    backend = Engine(**dataclasses.asdict(server_args))

    out_samples = dict()

    sampling_params = get_sampling_params(eval_args)

    conv = chat_templates[server_args.chat_template].copy()
    image_token = conv.image_token
    answer_dict = {}
    for sample in tqdm(samples):
        prompt = sample["final_input_prompt"]
        image = sample["image"]
        bytes_io = BytesIO()
        image.save(bytes_io, format="PNG")
        png_bytes = bytes_io.getvalue()

        prompt = re.sub(r"<[^>]*>", image_token, prompt)

        if image is not None:
            gen_out = backend.generate(
                prompt=prompt, image_data=[png_bytes], sampling_params=sampling_params
            )["text"]

            response = gen_out
        else:  # multiple images actually
            if sample["question_type"] == "multiple-choice":
                all_choices = sample["all_choices"]
                response = random.choice(all_choices)

            else:
                response = "INVALID GENERATION FOR MULTIPLE IMAGE INPUTS"

        if sample["question_type"] == "multiple-choice":
            pred_ans = parse_multi_choice_response(
                response, sample["all_choices"], sample["index2ans"]
            )
        else:  # open question
            pred_ans = response
        out_samples[sample["id"]] = pred_ans

        # set ground truth answer
        answer_dict[sample["id"]] = {
            "question_type": sample["question_type"],
            "ground_truth": (
                sample["correct_choice"]
                if "correct_choice" in samples
                else sample["answer"]
            ),
        }

    args.output_path = f"{args.model_path}_val_sglang.json"
    save_json(args.output_path, out_samples)
    eval_result(output_path=args.output_path, answer_dict=answer_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    EvalArgs.add_cli_args(parser)
    args = parser.parse_args()

    eval_mmmu(args)
