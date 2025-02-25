"""
    Bench the huggingface vLM with benchmark MMMU

    Usage:
        python benchmark/mmmu/bench_hf.py --model-path Qwen/Qwen2-VL-7B-Instruct

    The eval output will be logged
"""

import argparse
import random

import torch
from bench_sglang import EvalArgs, prepare_samples
from data_utils import save_json
from eval_utils import eval_result, get_sampling_params, parse_multi_choice_response
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


@torch.no_grad()
def eval_mmmu(args):
    eval_args = EvalArgs.from_cli_args(args)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model = model.eval().cuda()
    model = torch.compile(model)

    processor = AutoProcessor.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="auto"
    )

    samples = prepare_samples(eval_args)
    out_samples = dict()

    sampling_params = get_sampling_params(eval_args)

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

            generated_ids = model.generate(**inputs, **sampling_params)

            response = processor.decode(
                generated_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[len(text) :]
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

        torch.cuda.empty_cache()
        # set ground truth answer
        answer_dict[sample["id"]] = {
            "question_type": sample["question_type"],
            "ground_truth": sample["answer"],
        }

    args.output_path = f"{args.model_path}_val_hf.json"
    save_json(args.output_path, out_samples)
    eval_result(output_path=args.output_path, answer_dict=answer_dict)


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
