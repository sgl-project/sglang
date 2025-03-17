"""
    Bench the sglang-hosted vLM with benchmark MMMU

    Usage:
        python benchmark/mmmu/bench_sglang.py --model-path Qwen/Qwen2-VL-7B-Instruct --chat-template qwen2-vl

    The eval output will be logged
"""

import argparse
import base64
import dataclasses
import random
from io import BytesIO

from data_utils import save_json
from eval_utils import (
    EvalArgs,
    eval_result,
    get_sampling_params,
    load_model,
    prepare_samples,
    process_result,
)
from tqdm import tqdm

from sglang import Engine
from sglang.srt.conversation import generate_chat_conv
from sglang.srt.openai_api.protocol import ChatCompletionRequest
from sglang.srt.server_args import ServerArgs


def eval_mmmu(args):
    server_args = ServerArgs.from_cli_args(args)
    eval_args = EvalArgs.from_cli_args(args)

    if server_args.chat_template is None:
        raise ValueError("Chat template must be provided for this benchmark")
    model = load_model(args.model_path)
    backend = Engine(**dataclasses.asdict(server_args))
    out_samples = dict()
    samples = prepare_samples(eval_args)

    answer_dict = {}

    for sample in tqdm(samples):
        image = sample["image_1"]
        if image is not None:
            request_dict = model.build_prompt_sglang(sample)
            conv = generate_chat_conv(
                ChatCompletionRequest(**request_dict),
                template_name=server_args.chat_template,
            )
            prompt = conv.get_prompt()
            print(f"\033[31m{prompt}\033[0m")
            gen_out = backend.generate(
                prompt=prompt,
                image_data=conv.image_data,
                sampling_params=model.sampling_params,
            )["text"]
            response = gen_out
            print(f"\033[32m{response}\033[0m")
        else:  # multiple images actually
            if sample["question_type"] == "multiple-choice":
                all_choices = sample["all_choices"]
                response = random.choice(all_choices)
            else:
                response = "INVALID GENERATION FOR MULTIPLE IMAGE INPUTS"

        process_result(response, sample, answer_dict, out_samples)
    args.output_path = f"{args.model_path}_val_sglang.json"
    save_json(args.output_path, out_samples)
    eval_result(model_answer_path=args.output_path, answer_dict=answer_dict)

    backend.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    EvalArgs.add_cli_args(parser)
    args = parser.parse_args()

    eval_mmmu(args)
