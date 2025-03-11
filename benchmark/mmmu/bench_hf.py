"""
    Bench the huggingface vLM with benchmark MMMU

    Usage:
        python benchmark/mmmu/bench_hf.py --model-path Qwen/Qwen2-VL-7B-Instruct

    The eval output will be logged
"""

import argparse
import random

import torch
from data_utils import save_json
from eval_utils import (
    EvalArgs,
    eval_result,
    get_sampling_params,
    prepare_samples,
    process_result,
)
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig


@torch.no_grad()
def eval_mmmu(args):
    eval_args = EvalArgs.from_cli_args(args)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model = model.eval().cuda()

    processor = AutoProcessor.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="auto"
    )

    samples = prepare_samples(eval_args)
    out_samples = dict()

    sampling_params = get_sampling_params(eval_args)
    generation_config = GenerationConfig(
        max_new_tokens=sampling_params["max_new_tokens"],
        do_sample=False,
    )

    answer_dict = {}
    for sample in tqdm(samples):
        prompt = sample["final_input_prompt"]
        image = sample["image"]
        prefix = prompt.split("<")[0]
        suffix = prompt.split(">")[1]
        if image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prefix},
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": suffix},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            generated_ids = model.generate(
                **inputs, generation_config=generation_config
            )

            response = processor.decode(
                generated_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[len(text) :]
            print(f"response: {response}")
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
