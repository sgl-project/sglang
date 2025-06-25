import argparse
import asyncio
import os
import pickle
from pathlib import Path
from typing import List

import openai
import torch
from bert_score import BERTScorer
from datasets import load_dataset
from tqdm import tqdm


def get_client(api_url: str) -> openai.AsyncOpenAI:
    if os.getenv("OPENAI_API_KEY") is None:
        os.environ["OPENAI_API_KEY"] = "EMPTY"
    return openai.AsyncOpenAI(base_url=api_url)


def get_dataset():
    return load_dataset("bigai-nlco/LooGLE", "longdep_qa", split="test")


async def fetch_response(
    client: openai.AsyncOpenAI,
    context: str,
    question: str,
    semaphore: asyncio.Semaphore,
    index: int,
    model: str,
    output_dir: Path,
):
    output_file = output_dir / f"response_{index}.pkl"
    if output_file.exists():
        return

    prompt = (
        "Please answer the question based on the long texts below.\n"
        f"{context}\n"
        f"Question: {question}\n"
        "Answer:"
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
            )
        except openai.BadRequestError as e:
            with open(output_file, "wb") as f:
                pickle.dump({"error": str(e)}, f)
            return

    with open(output_file, "wb") as f:
        pickle.dump(response, f)


async def benchmark(args):
    dataset = get_dataset()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = get_client(args.api_url)
    semaphore = asyncio.Semaphore(args.max_concurrency)

    tasks: List[asyncio.Task] = []
    for idx, ex in enumerate(dataset):
        tasks.append(
            asyncio.create_task(
                fetch_response(
                    client,
                    ex["context"],
                    ex["question"],
                    semaphore,
                    idx,
                    args.model,
                    output_dir,
                )
            )
        )

    for _ in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Running benchmark"
    ):
        await _


def analyse(args):
    dataset = get_dataset()
    output_dir = Path(args.output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = BERTScorer(lang="en", device=device)

    hyps: List[str] = []
    refs: List[str] = []
    for idx, ex in enumerate(tqdm(dataset, desc="Loading responses")):
        pkl_file = output_dir / f"response_{idx}.pkl"
        if not pkl_file.exists():
            raise FileNotFoundError(pkl_file)

        response = pickle.load(open(pkl_file, "rb"))
        if isinstance(response, dict) and "error" in response:
            continue

        hyps.append(response.choices[0].message.content.strip())
        refs.append(ex["answer"])

    if not hyps:
        print("No valid responses to score!")
        return

    batch_size = 64
    all_f1: List[float] = []
    for i in tqdm(range(0, len(hyps), batch_size), desc="Scoring batches"):
        h_batch = hyps[i : i + batch_size]
        r_batch = refs[i : i + batch_size]
        _, _, f1_scores = scorer.score(h_batch, r_batch, verbose=False)
        all_f1.extend([float(x) for x in f1_scores])

    avg = sum(all_f1) / len(all_f1)
    print(f"Average BERTScore (F1): {avg:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run benchmark and evaluation in one go."
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:30000/v1",
        help="OpenAI‑compatible API base URL",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        help="Model name or ID, only used for model name",
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=144, help="Maximum concurrent requests"
    )
    parser.add_argument(
        "--output-dir", default="tmp-output-dir", help="Directory for cached responses"
    )
    args = parser.parse_args()

    asyncio.run(benchmark(args))

    analyse(args)
