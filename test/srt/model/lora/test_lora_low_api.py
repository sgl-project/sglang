import multiprocessing
import time

import numpy as np
import torch
import torch.distributed as dist
from sglang.srt.managers.router.infer_batch import ForwardMode
from sglang.srt.managers.router.model_runner import InputMetadata, ModelRunner
from sglang.srt.model_config import ModelConfig


def init_batch_data(model, batch_size, input_len):
    req_pool_indices = model.req_to_token_pool.alloc(batch_size)
    seq_lens = torch.full((batch_size,), input_len, dtype=torch.int32, device="cuda")
    prefix_lens = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    position_ids_offsets = torch.zeros(batch_size, dtype=torch.int32, device="cuda")

    out_cache_loc = model.token_to_kv_pool.alloc(batch_size * input_len)
    for i in range(batch_size):
        model.req_to_token_pool.req_to_token[i, :input_len] = out_cache_loc[
            i * input_len : (i + 1) * input_len
        ]

    return (
        req_pool_indices,
        seq_lens,
        prefix_lens,
        position_ids_offsets,
        out_cache_loc,
    )


def prefill(model, tp_rank, params, print_logits):
    logits, _ = model.forward_extend_multi_modal(
        *params,
        False,
    )
    prob_out = torch.softmax(logits, dim=-1)
    predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    predict_ids = predict_ids.detach().cpu().numpy()

    if print_logits and tp_rank == 0:
        print("prefill logits", logits, logits.shape)

    return predict_ids


def decode(step, model, tp_rank, batch_size, predict_ids, params, print_logits):
    (
        req_pool_indices,
        seq_lens,
        prefix_lens,
        position_ids_offsets,
        out_cache_loc,
    ) = params

    (
        out_cache_loc,
        out_cache_cont_start,
        out_cache_cont_end,
    ) = model.token_to_kv_pool.alloc_contiguous(batch_size)
    model.req_to_token_pool.req_to_token[req_pool_indices, seq_lens] = out_cache_loc
    seq_lens.add_(1)
    logits = model.forward_decode(
        torch.from_numpy(predict_ids).cuda().reshape(-1),
        req_pool_indices,
        seq_lens,
        None,
        position_ids_offsets,
        None,
        out_cache_cont_start,
        out_cache_cont_end,
    )
    prob_out = torch.softmax(logits, dim=-1)
    predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    predict_ids = predict_ids.detach().cpu().numpy()
    if print_logits and tp_rank == 0:
        print("decode", step, logits)
    return predict_ids


def test_generate_worker(
    model_path,
    lora_paths,
    tp_rank,
    tp_size,
):
    model_config = ModelConfig(path=model_path)
    model = ModelRunner(model_config, 0.8, tp_rank, tp_size, 28888, lora_paths=lora_paths)
    model.load_loras_from_path(lora_paths)
    # print(model.model)

    # Prepare data
    prompt = ""

    # inference

    # detokenization


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
    test_generate("meta-llama/Llama-2-7b-hf",
                  ["yard1/llama-2-7b-sql-lora-test", "tloen/alpaca-lora-7b"], 1)
                  # ["tloen/alpaca-lora-7b"], 1)
