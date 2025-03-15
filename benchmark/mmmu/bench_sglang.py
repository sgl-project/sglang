"""
    Bench the sglang-hosted vLM with benchmark MMMU

    Usage:
        python benchmark/mmmu/bench_sglang.py --model-path Qwen/Qwen2-VL-7B-Instruct --chat-template qwen2-vl

    The eval output will be logged
"""

import sglang as sgl


@sgl.function
def mmmu_qa(s, prefix, image_file, suffix, max_tokens):
    s += sgl.user(prefix + sgl.image(image_file) + suffix)
    s += sgl.assistant(sgl.gen("answer", max_tokens=max_tokens))


import argparse
import base64
import random
from io import BytesIO

from data_utils import save_json
from eval_utils import (
    EvalArgs,
    eval_result,
    get_sampling_params,
    prepare_samples,
    process_result,
)
from tqdm import tqdm

from sglang.test.test_utils import add_common_sglang_args_and_parse


def eval_mmmu(args):
    eval_args = EvalArgs.from_cli_args(args)

    out_samples = dict()

    sampling_params = get_sampling_params(eval_args)

    samples = prepare_samples(eval_args)

    answer_dict = {}

    import openai

    # had to use an openai server, since SglImage doesn't support image data
    client = openai.Client(api_key="sk", base_url=f"http://127.0.0.1:{args.port}/v1")

    for sample in tqdm(samples):
        prompt = sample["final_input_prompt"]
        image = sample["image"]
        buff = BytesIO()
        image.save(buff, format="PNG")
        base64_str = base64.b64encode(buff.getvalue()).decode("utf-8")
        prefix = prompt.split("<")[0]
        suffix = prompt.split(">")[1]
        if image is not None:
            # TODO: batch
            response = client.chat.completions.create(
                model="default",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prefix,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_str}"
                                },
                            },
                            {
                                "type": "text",
                                "text": suffix,
                            },
                        ],
                    }
                ],
                temperature=0,
                max_completion_tokens=sampling_params["max_new_tokens"],
                max_tokens=sampling_params["max_new_tokens"],
            )
            response = response.choices[0].message.content
        else:  # multiple images actually
            if sample["question_type"] == "multiple-choice":
                all_choices = sample["all_choices"]
                response = random.choice(all_choices)
            else:
                response = "INVALID GENERATION FOR MULTIPLE IMAGE INPUTS"

        process_result(response, sample, answer_dict, out_samples)
    args.output_path = f"./val_sglang.json"
    save_json(args.output_path, out_samples)
    eval_result(model_answer_path=args.output_path, answer_dict=answer_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_common_sglang_args_and_parse(parser)
    EvalArgs.add_cli_args(parser)
    args = parser.parse_args()

    eval_mmmu(args)
