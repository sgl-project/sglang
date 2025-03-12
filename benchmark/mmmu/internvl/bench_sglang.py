"""
    Bench the sglang-hosted vLM with benchmark MMMU

    Usage:
        python benchmark/mmmu/bench_sglang.py --model-path your_model_path --chat-template internvl2_5 --trust-remote-code

    The eval output will be logged
"""

import argparse
import dataclasses
import os
import random
import re
import sys
from io import BytesIO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import save_json
from eval_utils import (
    EvalArgs,
    eval_result,
    get_sampling_params,
    parse_multi_choice_response,
    prepare_samples,
    process_result
)
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from sglang import Engine
from sglang.srt.conversation import chat_templates, generate_chat_conv
from sglang.srt.server_args import ServerArgs

chat = "<|im_start|>system\n你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。<|im_end|>\n<|im_start|>user\nquestion<|im_end|>\n<|im_start|>assistant\n"

def eval_mmmu(args):
    server_args = ServerArgs.from_cli_args(args)
    eval_args = EvalArgs.from_cli_args(args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=False
    )
    if server_args.chat_template is None:
        raise ValueError("Chat template must be provided for this benchmark")
    samples = prepare_samples(eval_args)
    backend = Engine(**dataclasses.asdict(server_args))
    out_samples = dict()
    sampling_params = get_sampling_params(eval_args)
    sampling_params.update(
        {"stop_token_ids": [tokenizer.added_tokens_encoder["<|im_end|>"]]}
    )
    conv = chat_templates[server_args.chat_template].copy()
    image_token = conv.image_token
    answer_dict = {}
    for sample in tqdm(samples):
        if sample["image_1"] is not None:
            image_data = []
            prompt = sample["final_input_prompt"]
            count = len(
                sorted(set(int(m) for m in re.findall(r"<image ([1-7])>", prompt)))
            )
            if count == 1:
                prompt = "<image>\n" + prompt
                image = sample["image_1"]
                bytes_io = BytesIO()
                image.save(bytes_io, format="PNG")
                png_bytes = bytes_io.getvalue()
                image_data.append(png_bytes)
            else:
                for idx in range(1, count + 1):
                    prompt = f"Image-{count+1-idx}: <image>\n" + prompt
                    image = sample[f"image_{idx}"]
                    bytes_io = BytesIO()
                    image.save(bytes_io, format="PNG")
                    png_bytes = bytes_io.getvalue()
                    image_data.append(png_bytes)
            prompt = chat.replace("question", prompt)
            gen_out = backend.generate(
                prompt=prompt,
                input_ids=None,
                image_data=[png_bytes],
                sampling_params=sampling_params,
            )["text"]
            response = gen_out
            print(prompt)
            print(response)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    EvalArgs.add_cli_args(parser)
    args = parser.parse_args()

    eval_mmmu(args)
