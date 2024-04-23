import multiprocessing
import time

import numpy as np
import torch
import torch.distributed as dist
from sglang.srt.managers.router.model_runner import ModelRunner
from sglang.srt.model_config import ModelConfig


def test_generate_worker(
    model_path, tp_rank, tp_size, batch_size, input_len, output_len
):
    model_config = ModelConfig(path=model_path)
    model = ModelRunner(model_config, 0.8, tp_rank, tp_size, 28888)

    # Prepare data
    input_ids = np.vstack([np.arange(5, input_len + 5) for _ in range(batch_size)])
    input_ids = input_ids.reshape(-1)
    input_ids = torch.tensor(input_ids).cuda()

    def init_batch_data(model, batch_size, input_len):
        req_pool_indices = model.req_to_token_pool.alloc(batch_size)
        seq_lens = torch.full(
            (batch_size,), input_len, dtype=torch.int32, device="cuda"
        )
        prefix_lens = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
        position_ids_offsets = torch.zeros(batch_size, dtype=torch.int32, device="cuda")

        out_cache_loc = model.token_to_kv_pool.alloc(batch_size * input_len)
        for i in range(batch_size):
            req_idx = req_pool_indices[i].item()
            model.req_to_token_pool.req_to_token[req_idx, :input_len] = out_cache_loc[
                i * input_len : (i + 1) * input_len
            ]

        return (
            req_pool_indices,
            seq_lens,
            prefix_lens,
            position_ids_offsets,
            out_cache_loc,
        )

    def prefill(print_logits):
        nonlocal predict_ids

        logits, _ = model.forward_prefill(
            input_ids,
            req_pool_indices,
            seq_lens,
            prefix_lens,
            position_ids_offsets,
            out_cache_loc,
            False,
        )
        prob_out = torch.softmax(logits, dim=-1)
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        predict_ids = predict_ids.detach().cpu().numpy()

        if print_logits and tp_rank == 0:
            print("prefill logits", logits, logits.shape)

    def decode(print_logits):
        nonlocal predict_ids

        (
            out_cache_loc,
            out_cache_cont_start,
            out_cache_cont_end,
        ) = model.token_to_kv_pool.alloc_contiguous(batch_size)
        model.req_to_token_pool.req_to_token[req_pool_indices, seq_lens] = out_cache_loc
        seq_lens.add_(1)
        logits, _ = model.forward_decode(
            torch.from_numpy(predict_ids).cuda().reshape(-1),
            req_pool_indices,
            seq_lens,
            None,
            position_ids_offsets,
            None,
            out_cache_cont_start,
            out_cache_cont_end,
            False,
        )
        prob_out = torch.softmax(logits, dim=-1)
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        predict_ids = predict_ids.detach().cpu().numpy()
        if print_logits and tp_rank == 0:
            print("decode", i, logits)

    # Warm up
    (
        req_pool_indices,
        seq_lens,
        prefix_lens,
        position_ids_offsets,
        out_cache_loc,
    ) = init_batch_data(model, batch_size, input_len)
    predict_ids = None

    prefill(True)
    for i in range(output_len):
        decode(True)

    for i in range(batch_size):
        req_idx = req_pool_indices[i].item()
        model.token_to_kv_pool.free(
            model.req_to_token_pool.req_to_token[req_idx, : seq_lens[i]]
        )
    model.req_to_token_pool.free(req_pool_indices)

    # Benchmark
    if tp_size > 1:
        dist.barrier()
    start_time = prefill_start_time = time.time()

    (
        req_pool_indices,
        seq_lens,
        prefix_lens,
        position_ids_offsets,
        out_cache_loc,
    ) = init_batch_data(model, batch_size, input_len)

    prefill(False)

    if tp_rank == 0:
        print(f"prefill cost: {(time.time() - prefill_start_time) * 1000:.2f} ms")

    for i in range(output_len):
        step_start = time.time()

        decode(False)

        step_end = time.time()

        if i % 100 == 0 or i == output_len - 1:
            if tp_rank == 0:
                print(f"step {i} cost: {(step_end - step_start) * 1000:.2f} ms")

    end_time = time.time()

    if tp_rank == 0:
        print(f"total cost: {(end_time - start_time) * 1000:.2f}")


def test_generate(model_path, tp_size, batch_size, input_len, output_len):
    workers = []
    for tp_rank in range(tp_size):
        proc = multiprocessing.Process(
            target=test_generate_worker,
            args=(
                model_path,
                tp_rank,
                tp_size,
                batch_size,
                input_len,
                output_len,
            ),
        )
        proc.start()
        workers.append(proc)

    for proc in workers:
        proc.join()


if __name__ == "__main__":
    test_generate("TinyLlama/TinyLlama-1.1B-Chat-v0.4", 1, 1, 256, 8)
    # test_generate("meta-llama/Llama-2-7b-chat-hf", 1, 16, 256, 8)

    # Reference output for TinyLlama-1.1B-Chat-v0.4 (1, 32, 8)
    # prefill logits tensor([[-1.3380e-03,  4.4702e-01,  2.9082e+00,  ..., -1.8398e+00,
    #               1.8281e+00,  2.1816e+00]], device='cuda:0')
    # decode 0 tensor([[-0.3904,  0.8784,  3.6934,  ..., -2.4473,  1.5811,  2.0098]],
    #                device='cuda:0')
    # decode 1 tensor([[-0.3552,  0.0635,  2.5781,  ..., -2.5820,  1.3047,  1.7607]],
    #                device='cuda:0')
    # decode 2 tensor([[-1.5645, -1.1963,  3.8145,  ..., -2.9766,  1.0244,  1.0645]],
    #                device='cuda:0')
    # decode 3 tensor([[-1.3682, -0.6548,  4.2734,  ..., -2.8711,  1.1172,  1.1494]],
    #                device='cuda:0')
    # decode 4 tensor([[-1.0205, -0.0060,  4.4844,  ..., -2.7090,  1.6143,  1.8135]],
    #                device='cuda:0')
    # decode 5 tensor([[ 0.4260,  1.6006,  4.3633,  ..., -2.2480,  2.5547,  2.8379]],
    #                device='cuda:0')
    # decode 6 tensor([[ 0.7095,  2.1816,  5.0078,  ..., -2.1309,  3.0293,  3.0840]],
    #                device='cuda:0')
    # decode 7 tensor([[-0.2883,  1.1289,  4.7188,  ..., -2.4023,  2.1055,  2.1836]],
    #                device='cuda:0')

    # Reference output for TinyLlama-1.1B-Chat-v0.4 (1, 256, 8)
    # prefill logits tensor([[-2.5840, -2.7227,  6.8047,  ..., -2.3613,  0.1224,  0.5952]],
    #        device='cuda:0')
    # decode 0 tensor([[-0.6235, -0.7690,  9.2891,  ..., -1.4922,  2.8008,  2.9531]],
    #        device='cuda:0')
    # decode 1 tensor([[-1.3662, -1.4648,  7.1250,  ..., -1.7861,  1.7363,  1.8857]],
    #        device='cuda:0')
    # decode 2 tensor([[-0.8540, -0.5947,  9.1328,  ..., -2.1211,  2.9707,  2.8945]],
    #        device='cuda:0')
    # decode 3 tensor([[ 0.0652,  1.0312,  8.1250,  ..., -2.0586,  3.4727,  3.6172]],
    #        device='cuda:0')
    # decode 4 tensor([[-0.0459,  1.0098,  9.1406,  ..., -2.1797,  3.8320,  3.9355]],
    #        device='cuda:0')
    # decode 5 tensor([[ 0.2964,  1.3564,  9.8828,  ..., -2.1602,  4.1836,  4.2422]],
    #        device='cuda:0')
    # decode 6 tensor([[ 0.6475,  1.8105, 10.1250,  ..., -2.0098,  4.2578,  4.4062]],
    #        device='cuda:0')
    # decode 7 tensor([[ 0.4985,  1.4746,  9.9062,  ..., -1.9141,  3.9863,  4.3047]],
    #        device='cuda:0')
