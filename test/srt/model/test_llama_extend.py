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


def test_generate_worker(model_path, tp_rank, tp_size):
    model_config = ModelConfig(path=model_path)
    model = ModelRunner(model_config, 0.8, tp_rank, tp_size, 28888)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

    # Input
    prompts = [
        "The capital of France is",
        "Today is a sunny day and I like",
    ]
    sampling_params = SamplingParams(temperature=0)

    cut_num = 4

    reqs = []
    for i in range(len(prompts)):
        req = Req(i, None, None)
        req.input_ids = tokenizer.encode(prompts[i])[:cut_num]
        req.sampling_params = sampling_params
        reqs.append(req)

    # Prefill
    batch = Batch.init_new(reqs, model.req_to_token_pool, model.token_to_kv_pool, None)
    batch.prepare_for_extend(model.model_config.vocab_size, None)
    logits, _ = model.forward(batch, ForwardMode.EXTEND)
    next_token_ids, next_token_probs = batch.sample(logits)
    print("extend logits (first)", logits)

    # Extend
    for i in range(len(prompts)):
        req = reqs[i]
        req.input_ids += tokenizer.encode(prompts[i])[cut_num:]
        req.prefix_indices = model.req_to_token_pool.req_to_token[
            batch.req_pool_indices[i], :cut_num
        ]
    batch = Batch.init_new(reqs, model.req_to_token_pool, model.token_to_kv_pool, None)
    batch.prepare_for_extend(model.model_config.vocab_size, None)
    logits, _ = model.forward(batch, ForwardMode.EXTEND)
    next_token_ids, next_token_probs = batch.sample(logits)

    print("extend logits", logits)
    print(
        "next_token_ids", next_token_ids, [tokenizer.decode(x) for x in next_token_ids]
    )

    # Decode
    for i in range(6):
        batch.prepare_for_decode(next_token_ids.cpu().numpy())
        logits = model.forward(batch, ForwardMode.DECODE)
        next_token_ids, next_token_probs = batch.sample(logits)

        print(
            "next_token_ids",
            next_token_ids,
            [tokenizer.decode(x) for x in next_token_ids],
        )


def test_generate(model_path, tp_size):
    workers = []
    for tp_rank in range(tp_size):
        proc = multiprocessing.Process(
            target=test_generate_worker,
            args=(
                model_path,
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
    test_generate("TinyLlama/TinyLlama-1.1B-Chat-v0.4", 1)

    # Reference output for TinyLlama-1.1B-Chat-v0.4
    # extend logits (first) tensor([[-10.0312,  -9.5000,   0.8896,  ...,  -4.9375,  -3.2402,  -3.3633],
    #             [ -9.1797, -10.2500,   2.7168,  ...,  -4.3359,  -4.0664,  -4.1289]],
    #                    device='cuda:0', dtype=torch.float16)
    # extend logits tensor([[-8.3125, -7.1172,  3.3359,  ..., -4.9531, -4.1289, -3.4121],
    #             [-9.6406, -9.0547,  4.0195,  ..., -5.3086, -4.7188, -4.4609]],
    #                    device='cuda:0', dtype=torch.float16)
    # next_token_ids tensor([3681,  304], device='cuda:0') ['Paris', 'to']
    # next_token_ids tensor([29889,   748], device='cuda:0') ['.', 'go']
    # next_token_ids tensor([ 13, 363], device='cuda:0') ['\n', 'for']
    # next_token_ids tensor([1576,  263], device='cuda:0') ['The', 'a']
    # next_token_ids tensor([7483, 6686], device='cuda:0') ['capital', 'walk']
    # next_token_ids tensor([310, 297], device='cuda:0') ['of', 'in']
    # next_token_ids tensor([278, 278], device='cuda:0') ['the', 'the']
