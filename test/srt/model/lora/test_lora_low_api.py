import multiprocessing
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import transformers
from sglang.srt.managers.router.infer_batch import Batch, ForwardMode, Req
from sglang.srt.managers.router.model_runner import ModelRunner
from sglang.srt.model_config import ModelConfig
from sglang.srt.sampling_params import SamplingParams


@torch.inference_mode()
def test_generate_worker(model_path, lora_paths, tp_rank, tp_size):
    model_config = ModelConfig(path=model_path)
    model = ModelRunner(model_config, 0.8, tp_rank, tp_size, 28888, lora_paths=lora_paths)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model.load_loras_from_path(lora_paths)

    # Input
    prompts = [
        "The capital of France is",
        "Today is a sunny day and I like",
    ]
    sampling_params = SamplingParams(temperature=0)

    output_ids = [[], []]
    # Prefill
    reqs = []
    for i in range(len(prompts)):
        req = Req(i, None, None)
        req.input_ids = tokenizer.encode(prompts[i])
        req.sampling_params = sampling_params
        reqs.append(req)

    batch = Batch.init_new(reqs, model.req_to_token_pool, model.token_to_kv_pool, None)
    batch.prepare_for_extend(model.model_config.vocab_size, None)
    logits, _ = model.forward(batch, ForwardMode.EXTEND)
    next_token_ids, next_token_probs = batch.sample(logits)
    for k in range(len(output_ids)):
        output_ids[k].append(next_token_ids[k])

    print("extend logits (first)", logits)

    # Decode
    for i in range(6):
        batch.prepare_for_decode(next_token_ids.cpu().numpy())
        logits = model.forward(batch, ForwardMode.DECODE)
        next_token_ids, next_token_probs = batch.sample(logits)
        for k in range(len(output_ids)):
            output_ids[k].append(next_token_ids[k])

        print(
            "next_token_ids",
            next_token_ids,
            [tokenizer.decode(x) for x in next_token_ids],
        )

    print([tokenizer.decode(out_ids) for out_ids in output_ids])


def test_generate(model_path, lora_paths, tp_size):
    workers = []
    for tp_rank in range(tp_size):
        proc = multiprocessing.Process(
            target=test_generate_worker,
            args=(
                model_path,
                lora_paths,
                tp_rank,
                tp_size,
            ),
        )
        proc.start()
        workers.append(proc)

    for proc in workers:
        proc.join()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    test_generate("meta-llama/Llama-2-7b-hf",
                  ["yard1/llama-2-7b-sql-lora-test", "tloen/alpaca-lora-7b"], 1)
