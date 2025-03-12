"""
    Bench the huggingface vLM with benchmark MMMU

    Usage:
        python benchmark/mmmu/bench_hf.py --model-path your_hfmodel_path --dataset_path your_dataset_path
    The eval output will be logged
"""

import argparse
import random
import re

import torch
from bench_sglang import EvalArgs, prepare_samples
from data_utils import save_json
from eval_utils import eval_result, get_sampling_params, parse_multi_choice_response,process_result
from PIL import Image
from preprocess import load_image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer


@torch.no_grad()
def eval_mmmu(args):
    eval_args = EvalArgs.from_cli_args(args)

    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        use_flash_attn=True,
        trust_remote_code=True,
    )
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=False
    )
    samples = prepare_samples(eval_args)
    out_samples = dict()
    sampling_params = get_sampling_params(eval_args)
    generation_config = dict(max_new_tokens=4096, do_sample=False, eos_token_id=151645)
    answer_dict = {}
    for sample in tqdm(samples):
        if sample["image_1"] is not None:
            prompt = sample["final_input_prompt"]
            count = len(
                sorted(set(int(m) for m in re.findall(r"<image ([1-7])>", prompt)))
            )
            num_patches_list = []
            if count == 1:
                prompt = "<image>\n" + prompt
                pixel_values = (
                    load_image(sample["image_1"], upscale=True)
                    .to(torch.bfloat16)
                    .cuda()
                )
                num_patches_list = [pixel_values.size(0)]
            else:
                pixel_values_list = []
                for idx in range(1, count + 1):
                    prompt = f"Image-{count+1-idx}: <image>\n" + prompt
                    pixel_values = load_image(sample[f"image_{idx}"], upscale=True).to(
                        torch.bfloat16
                    )
                    num_patches_list.append(pixel_values.size(0))
                    pixel_values_list.append(pixel_values)
                pixel_values = torch.cat(pixel_values_list, dim=0).cuda()
            response = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                num_patches_list=num_patches_list,
                question=prompt,
                generation_config=generation_config,
                verbose=True,
            )
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
        default="/high_perf_store3/ad-compute-data/models/OpenGVLab/InternVL2_5-38B",
    )
    EvalArgs.add_cli_args(parser)
    args = parser.parse_args()

    eval_mmmu(args)
