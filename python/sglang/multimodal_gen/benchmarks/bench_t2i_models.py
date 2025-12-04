# Copied and adapted from: https://github.com/A113N-W3I/TIIF-Bench/blob/main/eval/inference_t2i_models.py
# Copied and adapted from: https://github.com/A113N-W3I/TIIF-Bench/blob/main/eval/eval_with_vlm.py

import argparse
import base64
import glob
import io
import json
import os
import random
import re

import numpy as np
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.sync_scheduler_client import sync_scheduler_client

raw_prompt = """
You are tasked with conducting a careful examination of the provided image. Based on the content of the image, please answer the following yes or no questions:

Questions:
##YNQuestions##

Note that:
1. Each answer should be on a separate line, starting with "yes" or "no", followed by the reason.
2. The order of answers must correspond exactly to the order of the questions.
3. Each question must have only one answer.
4. Directly return the answers to each question, without any additional content.
5. Each answer must be on its own line!
6. Make sure the number of output answers equal to the number of questions!
"""

raw_prompt_1 = """
You are tasked with conducting a careful examination of the image. Based on the content of the image, please answer the following yes or no questions:

Questions:
##YNQuestions##

Note that:
Each answer should be on a separate line, starting with "yes" or "no", followed by the reason.
The order of answers must correspond exactly to the order of the questions.
Each question must have only one answer. Output one answer if there is only one question.
Directly return the answers to each question, without any additional content.
Each answer must be on its own line!
Make sure the number of output answers equal to the number of questions!
"""

raw_prompt_2 = """
You are tasked with carefully examining the provided image and answering the following yes or no questions:

Questions:
##YNQuestions##

Instructions:

1. Answer each question on a separate line, starting with "yes" or "no", followed by a brief reason.
2. Maintain the exact order of the questions in your answers.
3. Provide only one answer per question.
4. Return only the answers—no additional commentary.
5. Each answer must be on its own line.
6. Ensure the number of answers matches the number of questions.
"""


def get_jsonl_files(input_folder, eval_folder, file_prefix):
    if file_prefix is not None:
        prompt_files = [os.path.join(input_folder, f"{file_prefix}_prompts.jsonl")]
        eval_files = [os.path.join(eval_folder, f"{file_prefix}_eval_prompts.jsonl")]
    else:
        prompt_files = glob.glob(os.path.join(input_folder, "*.jsonl"))
        eval_files = glob.glob(os.path.join(eval_folder, "*.jsonl"))
    return prompt_files, eval_files


def format_questions_prompt(questions):
    question_texts = [item.strip() for item in questions]
    formatted_questions = "\n".join(question_texts)
    prompt_template = random.choice([raw_prompt, raw_prompt_1, raw_prompt_2])
    formatted_prompt = prompt_template.replace("##YNQuestions##", formatted_questions)
    return formatted_prompt


def get_prompt(prompt, image_data):
    messages = [
        {"role": "system", "content": "You are a professional image critic."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"},
                },
            ],
        },
    ]
    return messages


class OutputFormatError(Exception):
    pass


def extract_yes_no(model_output, questions):
    lines = [line.strip() for line in model_output.strip().split("\n") if line.strip()]
    preds = []
    for idx, line in enumerate(lines):
        m = re.match(r"^(yes|no)\b", line.strip(), flags=re.IGNORECASE)
        if m:
            preds.append(m.group(1).lower())
        else:
            continue
    if len(preds) != len(questions):
        raise OutputFormatError(
            f"Preds count {len(preds)} != questions count {len(questions)}"
        )
    return preds


def tensor_to_base64(tensor):
    tensor = tensor.squeeze(0).squeeze(1)  # Now shape [3, 1024, 1024]
    array = tensor.numpy()
    array = np.transpose(array, (1, 2, 0))  # [1024, 1024, 3]
    array = (array * 255).astype(np.uint8)
    img = Image.fromarray(array)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def add_timings(stages, steps, total_duration_ms, timings):
    for k, v in timings["stages"].items():
        if k not in stages:
            stages[k] = [v]
        else:
            stages[k].append(v)
    steps.extend(timings["steps"])
    total_duration_ms.append(timings["total_duration_ms"])


def get_model_pred(vlm_client, eval_question, img_data, yn_question_list):
    while True:
        try:
            messages = get_prompt(eval_question, img_data)
            completion = vlm_client.chat.completions.create(
                model="VL",
                messages=messages,
                temperature=1.0,
            )
            return extract_yes_no(
                completion.choices[0].message.content, yn_question_list
            )
        except Exception as e:
            print(f"{e=}")


def get_correct(total_questions, correct_answers, model_pred, yn_answer_list):
    for gt, pred in zip(yn_answer_list, model_pred):
        total_questions += 1
        if gt.strip().lower() == pred.strip().lower():
            correct_answers += 1
    return total_questions, correct_answers


def print_perf(perf_name, perf_list, max_len, is_head=False):
    if is_head:
        perf_name = ""
        mean_perf = "mean"
        p99_perf = "p99"
    else:
        perf_name = f"{perf_name}:"
        mean_perf = f"{(sum(perf_list) / len(perf_list)):6.6f}"
        p99_perf = f"{(sorted(perf_list)[int(0.99* len(perf_list))]):6.6f}"
    print(f"{perf_name:>{max_len}}" f"{mean_perf:^15}" f"{p99_perf:^15}")


def process_jsonl_files(
    input_folder, eval_folder, model_name, file_prefix, scheduler_port
):
    server_args = ServerArgs.from_kwargs(model_path=model_name)
    server_args.scheduler_port = scheduler_port
    print(server_args)
    sync_scheduler_client.initialize(server_args)
    assert sync_scheduler_client.ping()
    diffusion_client = OpenAI(
        api_key="sk-proj-1234567890", base_url="http://localhost:30010/v1"
    )
    vlm_client = OpenAI(
        api_key="sk-proj-1234567890", base_url="http://localhost:30000/v1"
    )

    prompt_files, eval_files = get_jsonl_files(input_folder, eval_folder, file_prefix)
    total_questions, correct_answers = 0, 0
    stages = {}
    steps = []
    total_duration_ms = []
    for prompt_file, eval_file in zip(prompt_files, eval_files):
        with open(prompt_file, "r") as pfile, open(eval_file, "r") as efile:
            for idx, (pline, eline) in tqdm(enumerate(zip(pfile, efile))):
                # prompt
                pdata = json.loads(pline)
                pdata_type = pdata["type"]
                short_description = pdata["short_description"]
                long_description = pdata["long_description"]

                # eval
                edata = json.loads(eline)
                edata_type = edata["type"]
                yn_question_list = edata["yn_question_list"]
                yn_answer_list = edata["yn_answer_list"]
                eval_question = format_questions_prompt(yn_question_list)

                # t2i & eval
                sampling_params = SamplingParams.from_user_sampling_params_args(
                    model_name,
                    server_args=server_args,
                    prompt=short_description,
                    height=1024,
                    width=1024,
                )
                short_output = sync_scheduler_client.forward(
                    [
                        prepare_request(
                            server_args=server_args,
                            sampling_params=sampling_params,
                        )
                    ]
                )
                short_img = short_output.output
                short_img_b64 = tensor_to_base64(short_img)
                timings = short_output.timings.to_dict()
                add_timings(stages, steps, total_duration_ms, timings)

                long_img = diffusion_client.images.generate(
                    prompt=long_description,
                    size="1024x1024",
                    n=1,
                    response_format="b64_json",
                )
                sampling_params = SamplingParams.from_user_sampling_params_args(
                    model_name,
                    server_args=server_args,
                    prompt=long_description,
                    height=1024,
                    width=1024,
                )
                long_output = sync_scheduler_client.forward(
                    [
                        prepare_request(
                            server_args=server_args,
                            sampling_params=sampling_params,
                        )
                    ]
                )
                long_img = long_output.output
                long_img_b64 = tensor_to_base64(long_img)
                timings = long_output.timings.to_dict()
                add_timings(stages, steps, total_duration_ms, timings)

                short_model_pred, long_model_pred = None, None

                short_model_pred = get_model_pred(
                    vlm_client, eval_question, short_img_b64, yn_question_list
                )
                long_model_pred = get_model_pred(
                    vlm_client, eval_question, long_img_b64, yn_question_list
                )

                total_questions, correct_answers = get_correct(
                    total_questions, correct_answers, short_model_pred, yn_answer_list
                )
                total_questions, correct_answers = get_correct(
                    total_questions, correct_answers, long_model_pred, yn_answer_list
                )

    if total_questions > 0:
        accuracy = correct_answers / total_questions
    else:
        accuracy = None
    print(
        f"Total questions: {total_questions}, Correct answers: {correct_answers}, Accuracy: {accuracy if accuracy is not None else 'N/A'}"
    )
    stages["steps"] = steps
    stages["total_duration_ms"] = total_duration_ms
    perf_max_len = max([len(k) + 5 for k in stages.keys()])
    print_perf("", [], perf_max_len, is_head=True)
    for k, v in stages.items():
        print_perf(k, v, perf_max_len, is_head=False)


def main():
    parser = argparse.ArgumentParser(
        description="Generate images from JSONL files using a specified T2I model."
    )
    parser.add_argument("--model", required=True, help="Name of the T2I model to use.")
    parser.add_argument(
        "--file_prefix",
        type=str,
        default="2d_spatial_relation",
        help="Prefix of the JSONL files.",
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default="data/testmini_prompts",
        help="Path to the folder containing JSONL files.",
    )
    parser.add_argument(
        "--eval_folder",
        type=str,
        default="data/testmini_eval_prompts",
        help="Path to the folder containing JSONL files.",
    )
    parser.add_argument(
        "--scheduler-port", type=int, default=5555, help="Port for the scheduler server"
    )

    args = parser.parse_args()

    process_jsonl_files(
        args.input_folder,
        args.eval_folder,
        args.model,
        args.file_prefix,
        args.scheduler_port,
    )


if __name__ == "__main__":
    main()
