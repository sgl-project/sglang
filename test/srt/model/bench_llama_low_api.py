import multiprocessing as mp
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
from sglang.srt.managers.router.model_runner import ModelRunner
from sglang.srt.model_config import ModelConfig


@dataclass
class BenchBatch:
    req_to_token_pool: torch.Tensor
    token_to_kv_pool: torch.Tensor

    input_ids: torch.Tensor = None
    position_ids_offsets: torch.Tensor = None
    seq_lens: torch.Tensor = None
    prefix_lens: torch.Tensor = None
    req_pool_indices: torch.Tensor = None
    out_cache_loc: torch.Tensor = None
    out_cache_cont_start: torch.Tensor = None
    out_cache_cont_end: torch.Tensor = None

    def __init__(self, model_runner: ModelRunner):
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool

    def init_prefill_batch(self, input_ids, batch_size, seq_len):
        self.input_ids = input_ids
        self.position_ids_offsets = torch.zeros(
            batch_size, dtype=torch.int32, device="cuda"
        )
        self.seq_lens = torch.full(
            (batch_size,), seq_len, dtype=torch.int32, device="cuda"
        )
        self.prefix_lens = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
        self.req_pool_indices = self.req_to_token_pool.alloc(batch_size)
        self.out_cache_loc = self.token_to_kv_pool.alloc(batch_size * seq_len)

        for i in range(batch_size):
            n_idx = self.req_pool_indices[i].item()
            self.req_to_token_pool.req_to_token[n_idx, :seq_len] = self.out_cache_loc[
                i * seq_len : (i + 1) * seq_len
            ]

    def update_extend(
        self, input_ids, batch_size, prefix_len, extend_len, prefix_req_idx
    ):
        self.input_ids = input_ids
        self.position_ids_offsets = torch.zeros(
            batch_size, dtype=torch.int32, device="cuda"
        )
        self.seq_lens = torch.full(
            (batch_size,), prefix_len + extend_len, dtype=torch.int32, device="cuda"
        )
        self.prefix_lens = torch.full(
            (batch_size,), prefix_len, dtype=torch.int32, device="cuda"
        )
        self.req_pool_indices = self.req_to_token_pool.alloc(batch_size)
        self.out_cache_loc = self.token_to_kv_pool.alloc(batch_size * extend_len)

        req_to_token = self.req_to_token_pool.req_to_token
        fork_num = batch_size // prefix_req_idx.shape[0]
        for i in range(batch_size):
            p_idx = prefix_req_idx[i // fork_num].item()
            n_idx = self.req_pool_indices[i].item()
            req_to_token[n_idx, :prefix_len] = req_to_token[p_idx, :prefix_len]
            req_to_token[
                n_idx, prefix_len : prefix_len + extend_len
            ] = self.out_cache_loc[i * extend_len : (i + 1) * extend_len]

    def update_decode(self, predict_ids, batch_size):
        assert predict_ids.shape[0] == batch_size
        assert batch_size == self.req_pool_indices.shape[0]

        self.input_ids = predict_ids.reshape(-1)
        self.prefix_lens = None
        (
            self.out_cache_loc,
            self.out_cache_cont_start,
            self.out_cache_cont_end,
        ) = self.token_to_kv_pool.alloc_contiguous(batch_size)
        self.req_to_token_pool.req_to_token[
            self.req_pool_indices, self.seq_lens
        ] = self.out_cache_loc
        self.seq_lens.add_(1)


def prefill(model_runner: ModelRunner, batch: BenchBatch):
    logits, _ = model_runner.forward_extend(
        batch.input_ids,
        batch.req_pool_indices,
        batch.seq_lens,
        batch.prefix_lens,
        batch.position_ids_offsets,
        batch.out_cache_loc,
        False,
    )

    prob_out = torch.softmax(logits, dim=-1)
    predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    predict_ids = predict_ids.detach().cpu().numpy()

    return predict_ids


def extend(model_runner: ModelRunner, batch: BenchBatch):
    logits, _ = model_runner.forward_extend(
        batch.input_ids,
        batch.req_pool_indices,
        batch.seq_lens,
        batch.prefix_lens,
        batch.position_ids_offsets,
        batch.out_cache_loc,
        True,
    )

    prob_out = torch.softmax(logits, dim=-1)
    predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    predict_ids = predict_ids.detach().cpu().numpy()

    return predict_ids


def decode(model_runner: ModelRunner, batch: BenchBatch):
    logits = model_runner.forward_decode(
        batch.input_ids,
        batch.req_pool_indices,
        batch.seq_lens,
        None,
        batch.position_ids_offsets,
        None,
        batch.out_cache_cont_start,
        batch.out_cache_cont_end,
    )

    prob_out = torch.softmax(logits, dim=-1)
    predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    predict_ids = predict_ids.detach().cpu().numpy()

    return predict_ids


def bench_generate_worker(
    model_path,
    tp_rank,
    tp_size,
    shared_num,
    unique_num,
    shared_len,
    unique_len,
    decode_len,
    server_args_dict,
):
    assert unique_num % shared_num == 0

    model_config = ModelConfig(path=model_path)
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=0.8,
        tp_rank=tp_rank,
        tp_size=tp_size,
        nccl_port=28888,
        server_args_dict=server_args_dict,
    )

    batch = BenchBatch(model_runner)

    # warm up
    for _ in range(1):
        input_ids = torch.randint(
            low=5, high=100, size=(shared_num * shared_len,)
        ).cuda()
        batch.init_prefill_batch(input_ids, shared_num, shared_len)
        _ = prefill(model_runner, batch)

        input_ids = torch.randint(
            low=5, high=100, size=(unique_num * unique_len,)
        ).cuda()
        batch.update_extend(
            input_ids, unique_num, shared_len, unique_len, batch.req_pool_indices
        )
        predict_ids = extend(model_runner, batch)

        for i in range(decode_len):
            predict_ids = torch.from_numpy(predict_ids).cuda()
            batch.update_decode(predict_ids, unique_num)
            predict_ids = decode(model_runner, batch)

        model_runner.req_to_token_pool.clear()
        model_runner.token_to_kv_pool.clear()

    if tp_size > 1:
        dist.barrier()

    prefill_start = time.time()
    input_ids = torch.randint(low=5, high=100, size=(shared_num * shared_len,)).cuda()
    batch.init_prefill_batch(input_ids, shared_num, shared_len)
    _ = prefill(model_runner, batch)
    if tp_rank == 0:
        print(f"prefill: {(time.time() - prefill_start) * 1000:.2f} ms")

    extend_start = time.time()
    input_ids = torch.randint(low=5, high=100, size=(unique_num * unique_len,)).cuda()
    batch.update_extend(
        input_ids, unique_num, shared_len, unique_len, batch.req_pool_indices
    )
    predict_ids = extend(model_runner, batch)
    if tp_rank == 0:
        print(f"extend: {(time.time() - extend_start) * 1000:.2f} ms")

    for i in range(decode_len):
        decode_start = time.time()
        predict_ids = torch.from_numpy(predict_ids).cuda()
        batch.update_decode(predict_ids, unique_num)
        predict_ids = decode(model_runner, batch)
        if tp_rank == 0:
            print(f"decode {i}: {(time.time() - decode_start) * 1000:.2f} ms")


def bench_generate(
    model_path,
    tp_size,
    shared_num,
    unique_num,
    shared_len,
    unique_len,
    decode_len,
    server_args_dict,
):
    print(
        f"tp_size: {tp_size}, "
        f"shared_num: {shared_num}, "
        f"unique_num: {unique_num}, "
        f"shared_len: {shared_len}, "
        f"unique_len: {unique_len}, "
        f"decode_len: {decode_len}, "
        f"server_args: {server_args_dict}"
    )
    workers = []
    for tp_rank in range(tp_size):
        proc = mp.Process(
            target=bench_generate_worker,
            args=(
                model_path,
                tp_rank,
                tp_size,
                shared_num,
                unique_num,
                shared_len,
                unique_len,
                decode_len,
                server_args_dict,
            ),
        )
        proc.start()
        workers.append(proc)

    for proc in workers:
        proc.join()


if __name__ == "__main__":
    bench_generate(
        model_path="meta-llama/Llama-2-7b-chat-hf",
        tp_size=1,
        shared_num=1,
        unique_num=32,
        shared_len=256,
        unique_len=256,
        decode_len=8,
        server_args_dict={},
    )
