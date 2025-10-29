"""Response Parsing and Evaluation for various models"""

import argparse
import dataclasses
import json
import os
import pprint
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

import numpy as np
import torch
from data_utils import (
    CAT_SHORT2LONG,
    DOMAIN_CAT2SUB_CAT,
    construct_prompt,
    load_yaml,
    process_single_sample,
    save_json,
)
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm


@dataclasses.dataclass
class EvalArgs:
    seed: int = 42
    split: str = "validation"
    image_pixels_limit: int = -1
    result_filename: str = f"./val_sglang.json"
    prompt_format_file: str = "prompt_format.yaml"
    dataset_path: str = "MMMU/MMMU"
    extra_request_body: Optional[str] = None
    profile: bool = False
    profile_number: int = 5
    concurrency: int = 1
    max_new_tokens: int = 30
    response_answer_regex: str = "(.*)"
    lora_path: Optional[str] = None

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--result-filename",
            type=str,
            default=EvalArgs.result_filename,
            help="The filename to save the evaluation results.",
        )
        parser.add_argument(
            "--image-pixels-limit",
            type=int,
            default=EvalArgs.image_pixels_limit,
            help="The maximum number of pixels allowed for an image. If an image exceeds this limit, it will be skipped during evaluation.",
        )
        parser.add_argument(
            "--dataset-path",
            type=str,
            default=EvalArgs.dataset_path,
            help="path to the dataset",
        )
        parser.add_argument("--seed", type=int, default=1, help="The random seed.")
        parser.add_argument(
            "--prompt-format-file",
            type=str,
            help="The path to the prompt format of mmmu. If not, a default format llava_config.yaml will be used",
        )
        parser.add_argument(
            "--split",
            type=str,
            default=EvalArgs.split,
            help='Split of the dataset to use for evaluation. Default is "validation".',
        )
        parser.add_argument(
            "--extra-request-body",
            metavar='{"key1": "value1", "key2": "value2"}',
            type=str,
            default=EvalArgs.extra_request_body,
            help="Append given JSON object to the request payload. You can use this to specify"
            "additional generate params like sampling params.",
        )
        parser.add_argument(
            "--profile", action="store_true", help="enable mmmu profile"
        )
        parser.add_argument(
            "--profile-number",
            type=int,
            default=EvalArgs.profile_number,
            help="Number of samples to profile. If not set, will profile all samples.",
        )
        parser.add_argument(
            "--concurrency",
            type=int,
            default=EvalArgs.concurrency,
            help="Number of concurrent requests to make during evaluation. Default is 1, which means no concurrency.",
        )
        parser.add_argument(
            "--max-new-tokens",
            type=int,
            default=EvalArgs.max_new_tokens,
            help="Maximum number of new tokens to generate per sample.",
        )
        parser.add_argument(
            "--response-answer-regex",
            type=str,
            default=EvalArgs.response_answer_regex,
            help="Specific regex to capture the answer from the response, string",
        )
        parser.add_argument(
            "--lora-path",
            type=str,
            default=EvalArgs.lora_path,
            help="Specify the LoRA path to use for evaluation. If specified, the value will be specified in the body of every request as `lora-path`.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_samples(eval_args: EvalArgs):
    print("Preparing samples...")
    # Build prompts
    set_seed(eval_args.seed)

    prompt_format_file = (
        eval_args.prompt_format_file
        if eval_args.prompt_format_file is not None
        else os.path.join(os.path.dirname(__file__), "prompt_format.yaml")
    )
    # load config and process to one value
    eval_args.config = load_yaml(prompt_format_file)
    for key, value in eval_args.config.items():
        if key != "eval_params" and type(value) == list:
            assert len(value) == 1, "key {} has more than one value".format(key)
            eval_args.config[key] = value[0]

    # run for each subject in parallel
    sub_dataset_list = []
    subjects = list(CAT_SHORT2LONG.values())  # Get a fixed list of subjects

    print(f"Loading datasets for {len(subjects)} subjects...")
    with ThreadPoolExecutor() as executor:
        # Submit all load_dataset tasks
        future_to_subject = {
            executor.submit(
                load_dataset, eval_args.dataset_path, subject, split=eval_args.split
            ): subject
            for subject in subjects
        }

        # Collect results as they complete
        results = {}
        for future in tqdm(
            as_completed(future_to_subject),
            total=len(subjects),
            desc="Loading datasets",
        ):
            subject = future_to_subject[future]
            try:
                results[subject] = future.result()
            except Exception as exc:
                print(f"{subject} generated an exception: {exc}")

    # Ensure datasets are added in the original order for consistency
    for subject in subjects:
        if subject in results:
            sub_dataset_list.append(results[subject])
        else:
            # Handle cases where a dataset failed to load (optional, depends on desired behavior)
            print(f"Warning: Dataset for subject '{subject}' could not be loaded.")

    # merge all dataset
    dataset = concatenate_datasets(sub_dataset_list)

    # Prepare images in parallel
    images_path = os.path.expanduser("~/.cache/mmmu/images")
    os.makedirs(images_path, exist_ok=True)
    print(f"Saving images to: {images_path}")

    samples = []
    skip_count = 0

    def process_sample(i, sample):
        sample = process_single_sample(sample)
        sample = construct_prompt(sample, eval_args.config)
        image = sample["image"]
        width, height = image.size
        if 0 < eval_args.image_pixels_limit <= width * height:
            return None, True
        # Use a unique identifier for the image path to avoid potential collisions if indices reset
        image_path = f"{images_path}/image_{sample['id']}.png"
        if not os.path.exists(image_path):
            image.save(image_path)
        sample["image_path"] = image_path
        return sample, False

    print("Processing samples...")
    with ThreadPoolExecutor() as executor:
        # Pass the sample itself to process_sample, index is less reliable now
        futures = [
            executor.submit(
                process_sample, i, sample
            )  # Keep index i for tqdm maybe? Or remove it. Let's keep it for now.
            for i, sample in enumerate(dataset)
        ]
        for future in tqdm(
            as_completed(futures), total=len(dataset), desc="Processing samples"
        ):
            sample, skipped = future.result()
            if skipped:
                skip_count += 1
            elif sample:
                samples.append(sample)

    samples.sort(key=lambda x: x["final_input_prompt"])

    print(
        f"Skipping {skip_count} samples with large images, {round((float(skip_count) / len(dataset)) * 100, 2)}% of dataset"
    )
    print("Samples have been prepared")
    return samples


def get_sampling_params(eval_args):
    max_new_tokens = eval_args.max_new_tokens
    temperature = 0.001

    extra_request_body = {}
    if eval_args.extra_request_body:
        extra_request_body = json.loads(eval_args.extra_request_body)

    return {
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        **extra_request_body,
    }


# ----------- Process Multi-choice -------------
def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f" {choice} " in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


# ----------- Process Open -------------
def check_is_number(string):
    """
    Check if the given string a number.
    """
    try:
        float(string.replace(",", ""))
        return True
    except ValueError:
        # check if there's comma inside
        return False


def normalize_str(string):
    """
    Normalize the str to lower case and make them float numbers if possible.
    """
    # check if characters in the string

    # if number, numerize it.
    string = string.strip()

    is_number = check_is_number(string)

    if is_number:
        string = string.replace(",", "")
        string = float(string)
        # leave 2 decimal
        string = round(string, 2)
        return [string]
    else:  # it's likely to be a string
        # lower it
        string = string.lower()
        if len(string) == 1:
            return [" " + string, string + " "]  # avoid trivial matches
        return [string]


def extract_numbers(string):
    """
    Exact all forms of numbers from a string with regex.
    """
    # Pattern for numbers with commas
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    # Pattern for scientific notation
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    # Pattern for simple numbers without commas
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"

    # Extract numbers with commas
    numbers_with_commas = re.findall(pattern_commas, string)
    # Extract numbers in scientific notation
    numbers_scientific = re.findall(pattern_scientific, string)
    # Extract simple numbers without commas
    numbers_simple = re.findall(pattern_simple, string)

    # Combine all extracted numbers
    all_numbers = numbers_with_commas + numbers_scientific + numbers_simple
    return all_numbers


def parse_open_response(response):
    """
    Parse the prediction from the generated response.
    Return a list of predicted strings or numbers.
    """

    # content = content.strip("\n").strip(".").strip(" ")
    def get_key_subresponses(response):
        key_responses = []
        response = response.strip().strip(".").lower()
        sub_responses = re.split(r"\.\s(?=[A-Z])|\n", response)
        indicators_of_keys = [
            "could be ",
            "so ",
            "is ",
            "thus ",
            "therefore ",
            "final ",
            "answer ",
            "result ",
        ]
        key_responses = []
        for index, resp in enumerate(sub_responses):
            # if last one, accept it's an equation (the entire response can be just one sentence with equation)
            if index == len(sub_responses) - 1:
                indicators_of_keys.extend(["="])
            shortest_key_response = None  # the shortest response that may contain the answer (tail part of the response)
            for indicator in indicators_of_keys:
                if indicator in resp:
                    if not shortest_key_response:
                        shortest_key_response = resp.split(indicator)[-1].strip()
                    else:
                        if len(resp.split(indicator)[-1].strip()) < len(
                            shortest_key_response
                        ):
                            shortest_key_response = resp.split(indicator)[-1].strip()
                    # key_responses.append(resp.split(indicator)[1].strip())

            if shortest_key_response:
                # and it's not trivial
                if shortest_key_response.strip() not in [
                    ":",
                    ",",
                    ".",
                    "!",
                    "?",
                    ";",
                    ":",
                    "'",
                ]:
                    key_responses.append(shortest_key_response)
        if len(key_responses) == 0:  # did not found any
            return [response]
        return key_responses

    # pdb.set_trace()
    key_responses = get_key_subresponses(response)

    pred_list = key_responses.copy()  # keep the original string response
    for resp in key_responses:
        pred_list.extend(extract_numbers(resp))

    tmp_pred_list = []
    for i in range(len(pred_list)):
        tmp_pred_list.extend(normalize_str(pred_list[i]))
    pred_list = tmp_pred_list

    # remove duplicates
    pred_list = list(set(pred_list))

    return pred_list


# ----------- Evaluation -------------


def eval_multi_choice(gold_i, pred_i):
    """
    Evaluate a multiple choice instance.
    """
    correct = False
    # for case like Answer: A, Answer is A, answer is A, answer: A
    for _exp in ["Answer:", "Answer is ", "answer is ", "answer: "]:
        if _exp in pred_i:
            pred_i = pred_i.split(_exp)[1].strip()
            break
    # for case like (A), (B), (C), (D) ......
    if "(" in pred_i and ")" in pred_i:
        try:
            pred_i = re.search(r"\(([A-Z])\)", pred_i).group(1)
        except:
            print(f"Error to extract answer from: {pred_i}")
            pass
    # only they are exactly the same, we consider it as correct
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:  # gold_i is a string
        if gold_i == pred_i:
            correct = True
    return correct


def eval_open(gold_i, pred_i):
    """
    Evaluate an open question instance
    """
    correct = False
    if isinstance(gold_i, list):
        # use float to avoid trivial matches
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(normalize_str(answer))
    else:
        norm_answers = normalize_str(gold_i)
    for pred in pred_i:  # pred is already normalized in parse response phase
        if isinstance(pred, str):  # if it's a string, then find if ans in the pred_i
            for norm_ans in norm_answers:
                # only see if the string answer in the string pred
                if isinstance(norm_ans, str) and norm_ans in pred:
                    if not correct:
                        correct = True
                    break
        else:  # it's a float number
            if pred in norm_answers:
                if not correct:
                    correct = True
                break
    return correct


# ----------- Batch Evaluation -------------
def evaluate(samples):
    """
    Batch evaluation for multiple choice and open questions.
    """
    pred_correct = 0
    judge_dict = dict()
    for sample in samples:
        gold_i = sample["answer"]
        pred_i = sample["parsed_pred"]
        if sample["question_type"] == "multiple-choice":
            correct = eval_multi_choice(gold_i, pred_i)
        else:  # open question
            correct = eval_open(gold_i, pred_i)

        if correct:
            judge_dict[sample["id"]] = "Correct"
            pred_correct += 1
        else:
            # print(f"Wrong! expected {pred_i}, answered with {gold_i}")
            judge_dict[sample["id"]] = "Wrong"

    if len(samples) == 0:
        return {"acc": 0}
    return judge_dict, {"acc": pred_correct / len(samples)}


# ----------- Calculate Accuracy -------------
def calculate_ins_level_acc(results: Dict):
    """Calculate the instruction level accuracy for given Subject results"""
    acc = 0
    ins_num = 0
    for cat_results in results.values():
        acc += cat_results["acc"] * cat_results["num_example"]
        ins_num += cat_results["num_example"]
    if ins_num == 0:
        return 0
    return acc / ins_num


def process_result(response, sample, answer_dict, out_samples):
    if response is None:
        return
    if sample["question_type"] == "multiple-choice":
        pred_ans = parse_multi_choice_response(
            response, sample["all_choices"], sample["index2ans"]
        )
    else:  # open question
        pred_ans = response

    out_samples[sample["id"]] = {
        "pred_ans": pred_ans,
        "original_response": sample["original_response"],
        "ground_truth": sample["answer"],
        "question_type": sample["question_type"],
    }

    # set ground truth answer
    answer_dict[sample["id"]] = {
        "question_type": sample["question_type"],
        "ground_truth": sample["answer"],
    }


def eval_result(model_answer_path, answer_dict, eval_output_path=None):
    if eval_output_path is None:
        eval_output_path = model_answer_path
    print("Evaluating...")
    output_dict = json.load(open(model_answer_path))
    # answer_dict = json.load(open(answer_path))

    # group by category
    output_dict_w_cat = {}
    for data_id, parsed_pred in output_dict.items():
        if isinstance(parsed_pred, str):
            parsed_pred = parsed_pred
        elif isinstance(parsed_pred, dict):
            parsed_pred = parsed_pred["pred_ans"]
        else:
            raise ValueError(f"Unknown type of parsed_pred: {type(parsed_pred)}")
        category = "_".join(data_id.split("_")[1:-1])
        if category not in output_dict_w_cat:
            output_dict_w_cat.update({category: {}})
        output_dict_w_cat[category].update({data_id: parsed_pred})

    # group by category
    answer_dict_w_cat = {}
    for data_id, parsed_pred in answer_dict.items():
        category = "_".join(data_id.split("_")[1:-1])
        if category not in answer_dict_w_cat:
            answer_dict_w_cat.update({category: {}})
        answer_dict_w_cat[category].update({data_id: parsed_pred})

    evaluation_result = {}

    for category in CAT_SHORT2LONG.values():
        # print("Evaluating: {}".format(category))
        # get cat_outputs and cat_answers
        try:
            cat_outputs = output_dict_w_cat[category]
            cat_answers = answer_dict_w_cat[category]
        except KeyError:
            # print("Skipping {} for not found".format(category))
            continue

        exampels_to_eval = []
        for data_id, parsed_pred in cat_outputs.items():
            question_type = cat_answers[data_id]["question_type"]
            if question_type != "multiple-choice":
                parsed_pred = parse_open_response(
                    parsed_pred
                )  # mainly for type consistency (make it number, etc.)
            else:
                parsed_pred = parsed_pred

            exampels_to_eval.append(
                {
                    "id": data_id,
                    "question_type": question_type,
                    "answer": cat_answers[data_id]["ground_truth"],
                    "parsed_pred": parsed_pred,
                }
            )

        judge_dict, metric_dict = evaluate(exampels_to_eval)
        metric_dict.update({"num_example": len(exampels_to_eval)})
        for key, value in judge_dict.items():
            output_dict[key]["judge"] = value

        evaluation_result[category] = metric_dict

    save_json(model_answer_path, output_dict)
    printable_results = {}
    # pdb.set_trace()
    # add domain Subject
    for domain, in_domain_cats in DOMAIN_CAT2SUB_CAT.items():
        in_domain_cat_results = {}
        for cat_name in in_domain_cats:  # use the order in DOMAIN_CAT2SUB_CAT
            if cat_name in evaluation_result.keys():
                in_domain_cat_results[cat_name] = evaluation_result[cat_name]
            else:
                pass
        in_domain_ins_acc = calculate_ins_level_acc(in_domain_cat_results)
        in_domain_data_num = sum(
            [
                cat_results["num_example"]
                for cat_results in in_domain_cat_results.values()
            ]
        )
        printable_results["Overall-" + domain] = {
            "num": int(in_domain_data_num),
            "acc": round(in_domain_ins_acc, 3),
        }
        # add sub category
        for cat_name, cat_results in in_domain_cat_results.items():
            printable_results[cat_name] = {
                "num": int(cat_results["num_example"]),
                "acc": round(cat_results["acc"], 3),
            }

    # table.append(["-----------------------------", "-----", "----"])
    all_ins_acc = calculate_ins_level_acc(evaluation_result)
    overall_acc = round(all_ins_acc, 3)
    printable_results["Overall"] = {
        "num": sum(
            [cat_results["num_example"] for cat_results in evaluation_result.values()]
        ),
        "acc": overall_acc,
    }
    pprint.pprint(printable_results)
    out = eval_output_path
    with open(out, "w", encoding="utf-8") as outfile:
        json.dump(printable_results, outfile)
        print(f"eval out saved to {out}")

    print(f"Overall accuracy: {overall_acc}")
