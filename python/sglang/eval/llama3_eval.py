# Adapt from https://github.com/fw-ai/llm_eval_meta

import argparse
import asyncio
import os
import pickle
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass

import httpx
import numpy as np
import openai
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm import tqdm

# Mapping providers to their clients and models
provider_to_models = {
    "b10": {
        "8b": "meta-llama/Llama-3.1-8B-Instruct",
        "70b": "meta-llama/Llama-3.1-70B-Instruct",
        "405b": "meta-llama/Llama-3.1-405B-Instruct",
    },
    "oai": {
        "8b": "meta-llama/Llama-3.1-8B-Instruct",
        "70b": "meta-llama/Llama-3.1-70B-Instruct",
        "405b": "meta-llama/Llama-3.1-405B-Instruct",
    },
    "sgl": {
        "8b": "meta-llama/Llama-3.1-8B-Instruct",
        "70b": "meta-llama/Llama-3.1-70B-Instruct",
        "405b": "meta-llama/Llama-3.1-405B-Instruct",
    },
}


async def fetch_responses(
    client, prompt, semaphore, index, provider, model_size, output_dir, max_tokens
):
    output_file = os.path.join(output_dir, f"response_{index}.pkl")
    if os.path.exists(output_file):
        print(f"File {output_file} already exists, skipping.")
        return

    async with semaphore:
        response = await client.completions.create(
            model=provider_to_models[provider][model_size],
            prompt=prompt,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        if isinstance(response, openai.BadRequestError):
            with open(output_file, "wb") as f:
                pickle.dump("bad_response", f)
        assert isinstance(response, openai.types.completion.Completion)
        # Save response to a file
        with open(output_file, "wb") as f:
            pickle.dump(response, f)


TASK_TO_MAX_TOKENS = {
    "evals__mmlu__details": 1,
    "evals__mmlu__0_shot__cot__details": 1024,
    # Official meta uses 1024, but a small % (.05) of questions are answered correctly after relaxing
    "evals__mmlu_pro__details": 2048,
    "evals__gsm8k__details": 1024,
}

TASK_TO_EVAL_SET = {
    "mmlu": "evals__mmlu__details",
    "mmlu_cot": "evals__mmlu__0_shot__cot__details",
    "mmlu_pro": "evals__mmlu_pro__details",
    "gsm8k": "evals__gsm8k__details",
}


class CustomAsyncHTTPXClient(httpx.AsyncClient):
    async def send(self, request: httpx.Request, *args, **kwargs) -> httpx.Response:
        request.url = httpx.URL(
            f"https://model-{os.getenv('MODEL_ID')}.api.baseten.co/development/predict"
        )
        return await super().send(request, *args, **kwargs)


def get_client(provider):
    if provider not in "b10":
        if os.getenv("OPENAI_API_KEY") == None:
            os.environ["OPENAI_API_KEY"] = "EMPTY"
    return {
        "oai": AsyncOpenAI(base_url="http://127.0.0.1:8000/v1/"),
        "b10": AsyncOpenAI(
            api_key=f"Api-Key {os.getenv('OPENAI_API_KEY')}",
            base_url=f"https://model-{os.getenv('MODEL_ID')}.api.baseten.co/development/predict",
            http_client=CustomAsyncHTTPXClient(),
        ),
        "sgl": AsyncOpenAI(base_url="http://127.0.0.1:30000/v1/"),
    }[provider]


# Define the benchmark function
async def benchmark(args):
    ds = load_dataset(
        "meta-llama/Llama-3.1-405B-Instruct-evals",
        f"Llama-3.1-405B-Instruct-{TASK_TO_EVAL_SET[args.task]}",
    )
    semaphore = asyncio.Semaphore(args.concurrency)  # Limit to 16 concurrent tasks

    if args.num_examples is None:
        args.num_examples = len(ds["latest"]["input_final_prompts"])
    prompts = ds["latest"]["input_final_prompts"][: args.num_examples]

    # Create the output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)

    tasks = []
    # Create the tasks with tqdm progress bar
    max_tokens = TASK_TO_MAX_TOKENS[TASK_TO_EVAL_SET[args.task]]
    client = get_client(args.provider)
    for idx, prompt in enumerate(tqdm(prompts, desc="Creating tasks")):
        tasks.append(
            asyncio.create_task(
                fetch_responses(
                    client,
                    f"<|begin_of_text|>{prompt[0]}",
                    semaphore,
                    idx,
                    args.provider,
                    args.model_size,
                    args.output_dir,
                    max_tokens=max_tokens,
                )
            )
        )

    # Run the tasks with tqdm progress bar
    for future in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Processing tasks"
    ):
        await future


def get_mmlu_answer(response):
    if response is not None:
        return response.choices[0].text.lstrip().rstrip().upper().replace(".", "")
    return None


def get_mmlu_cot_answer(response):
    pattern = r"The best answer is (.+)\.?"
    match = re.search(pattern, response.choices[0].text)
    if match:
        return match.group(1).replace(".", "").replace("*", "")

    pattern = r"the best answer is (.+)\.?"
    match = re.search(pattern, response.choices[0].text)
    if match:
        return match.group(1).replace(".", "")

    pattern = r"The correct answer is (.+)\.?"
    match = re.search(pattern, response.choices[0].text)
    if match:
        return match.group(1).replace(".", "")

    pattern = r"the correct answer is (.+)\.?"
    match = re.search(pattern, response.choices[0].text)
    if match:
        return match.group(1).replace(".", "")


def get_answer_gsm8k(response):
    pattern = r"The final answer is (.+)\.?"
    match = re.search(pattern, response.choices[0].text)
    if match:
        s = match.group(1)
        for ok_symbol in ["%", "$"]:
            s = s.replace(ok_symbol, "")
        return s


TASK_TO_ANSWER_EXTRACTOR = {
    "evals__mmlu__details": get_mmlu_answer,
    "evals__mmlu__0_shot__cot__details": get_mmlu_cot_answer,
    "evals__gsm8k__details": get_answer_gsm8k,
    "evals__mmlu_pro__details": get_mmlu_cot_answer,
}


def get_dataset_from_task(task, response_path, model_size):
    ds_405b = load_dataset(
        f"meta-llama/Llama-3.1-405B-Instruct-evals",
        f"Llama-3.1-405B-Instruct-{task}",
    )
    ds_405b_hash_order = [x[0] for x in ds_405b["latest"]["input_final_prompts_hash"]]

    if "70b" in model_size or "8b" in model_size:
        if "70" in model_size:
            ref_model_ds = load_dataset(
                f"meta-llama/Llama-3.1-70B-Instruct-evals",
                f"Llama-3.1-70B-Instruct-{task}",
            )
        else:
            ref_model_ds = load_dataset(
                f"meta-llama/Llama-3.1-8B-Instruct-evals",
                f"Llama-3.1-8B-Instruct-{task}",
            )

        hash_to_row = {}
        for row in ref_model_ds["latest"]:
            hash_to_row[row["input_final_prompts_hash"][0]] = row
        reordered_rows = []
        for prompt_hash in ds_405b_hash_order:
            reordered_rows.append(hash_to_row[prompt_hash])
        ref_model_ds["latest"] = reordered_rows
        return ref_model_ds

    return ds_405b


def analyze(task, response_path, model_size):
    ds = get_dataset_from_task(task, response_path, model_size)

    responses = []
    total = len(ds["latest"])

    for i in range(0, total):
        response = pickle.load(
            open(os.path.join(response_path, f"response_{i}.pkl"), "rb")
        )
        responses.append(response)

    @dataclass
    class Stats:
        correct: int = 0
        total: int = 0
        meta_correct: int = 0

        average: float = None

    subtask_name_to_stats = defaultdict(lambda: Stats())

    for response, ds_row in zip(responses, ds["latest"]):
        model_answer = TASK_TO_ANSWER_EXTRACTOR[task](response)

        subtask = ds_row["subtask_name"]

        is_eval_correct = model_answer in ds_row["input_correct_responses"]
        if is_eval_correct:
            subtask_name_to_stats[subtask].correct += 1

        if ds_row["is_correct"]:
            subtask_name_to_stats[subtask].meta_correct += 1

        subtask_name_to_stats[subtask].total += 1

    micro_stats = Stats()
    for subtask, stats in subtask_name_to_stats.items():
        stats.average = stats.correct / stats.total
        stats.meta_average = stats.meta_correct / stats.total

        micro_stats.correct += stats.correct
        micro_stats.total += stats.total
        micro_stats.meta_correct += stats.meta_correct

    micro_stats.average = micro_stats.correct / micro_stats.total
    micro_stats.meta_average = micro_stats.meta_correct / micro_stats.total

    print("Macro average", np.mean([x.average for x in subtask_name_to_stats.values()]))
    print(
        "Meta Macro average",
        np.mean([x.meta_average for x in subtask_name_to_stats.values()]),
    )
    print("Micro average", micro_stats.average)
    print("Meta Micro average", micro_stats.meta_average)


# Entry point for the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to run model with specified parameters."
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="8b",
        help="Size of the model (e.g., 8b or 70b)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="sgl",
        help="Provider name (e.g., sgl, oai, b10)",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task (e.g., mmlu, mmlu_cot, mmlu_pro, gsm8k)",
    )
    parser.add_argument(
        "--num-examples", type=int, default=None, help="Number of examples to process"
    )
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tmp-output-dir",
        help="Directory to save responses",
    )

    args = parser.parse_args()
    asyncio.run(benchmark(args))
    analyze(TASK_TO_EVAL_SET[args.task], args.output_dir, args.model_size)
    shutil.rmtree("tmp-output-dir", ignore_errors=True)
