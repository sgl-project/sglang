"""
Usage:
python bench_llama_low_api.py --model-path TinyLlama/TinyLlama-1.1B-Chat-v0.4 --correct



"""

import argparse
import dataclasses
import multiprocessing
import time

import numpy as np
import torch.distributed as dist


from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.controller.infer_batch import Batch, ForwardMode, Req
from sglang.srt.managers.controller.model_runner import ModelRunner
from sglang.srt.model_config import ModelConfig
from sglang.srt.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs


@dataclasses.dataclass
class BenchArgs:
    batch_size: int = 1
    input_len: int = 128
    output_len: int = 16
    cut_len: int = 4
    correctness_test: bool = False

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--batch-size", type=int, default=BenchArgs.batch_size)
        parser.add_argument("--input-len", type=int, default=BenchArgs.input_len)
        parser.add_argument("--output-len", type=int, default=BenchArgs.output_len)
        parser.add_argument("--cut-len", type=int, default=BenchArgs.cut_len)
        parser.add_argument("--correctness-test", action="store_true")

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


def load_model(server_args, tp_rank):
    model_config = ModelConfig(path=server_args.model_path)
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=tp_rank,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        nccl_port=28888,
        server_args=server_args,
    )
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )
    if server_args.tp_size > 1:
        dist.barrier()
    return model_runner, tokenizer


def prepare_inputs(bench_args, tokenizer):
    prompts = [
        "The capital of France is",
        "The capital of the United Kindom is",
        "Today is a sunny day and I like",
    ]
    input_ids = [tokenizer.encode(p) for p in prompts]
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )

    reqs = []
    for i in range(len(prompts)):
        assert len(input_ids[i]) > bench_args.cut_len

        tmp_input_ids = input_ids[i][:bench_args.cut_len]
        req = Req(rid=i, origin_input_text=prompts[i], origin_input_ids=tmp_input_ids)
        req.prefix_indices = []
        req.sampling_params = sampling_params
        req.input_ids = req.origin_input_ids
        reqs.append(req)

    return input_ids, reqs


def prepare_extend_inputs(bench_args, input_ids, reqs, model_runner):
    for i in range(len(reqs)):
        req = reqs[i]
        req.input_ids += input_ids[i][bench_args.cut_len:]
        req.prefix_indices = model_runner.req_to_token_pool.req_to_token[
            i, :bench_args.cut_len
        ]
    return reqs


def extend(reqs, model_runner):
    batch = Batch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool=model_runner.token_to_kv_pool,
        tree_cache=None)
    batch.prepare_for_extend(model_runner.model_config.vocab_size, None)
    output = model_runner.forward(batch, ForwardMode.EXTEND)
    next_token_ids, _ = batch.sample(output.next_token_logits)
    return next_token_ids, output.next_token_logits, batch


def decode(input_token_ids, batch, model_runner):
    batch.prepare_for_decode(input_token_ids.cpu().numpy())
    output = model_runner.forward(batch, ForwardMode.DECODE)
    next_token_ids, _ = batch.sample(output.next_token_logits)
    return next_token_ids, output.next_token_logits


def correctness_test(
    server_args,
    bench_args,
    tp_rank,
):
    # Load the model
    model_runner, tokenizer = load_model(server_args, tp_rank)

    # Prepare inputs
    input_ids, reqs = prepare_inputs(bench_args, tokenizer)

    # Prefill
    next_token_ids, next_token_logits, batch = extend(reqs, model_runner)
    print("prefill logits (first half)", next_token_logits)

    # Prepare extend inputs
    reqs = prepare_extend_inputs(bench_args, input_ids, reqs, model_runner)

    # Extend
    next_token_ids, next_token_logits, batch = extend(reqs, model_runner)
    print("prefill logits (final)", next_token_logits)

    # Decode
    output_ids = [list(req.input_ids) for req in reqs]
    for _ in range(bench_args.output_len):
        next_token_ids, _ = decode(next_token_ids, batch, model_runner)
        for i in range(len(reqs)):
            output_ids[i].append(next_token_ids[i])

    # Print
    for i in range(len(reqs)):
        print(tokenizer.decode(output_ids[i]))


def latency_test(
    server_args,
    bench_args,
    tp_rank,
):
    pass


def main(server_args, bench_args):
    print(bench_args)

    if bench_args.correctness_test:
        work_func = correctness_test
    else:
        work_func = latency_test

    workers = []
    for tp_rank in range(server_args.tp_size):
        proc = multiprocessing.Process(
            target=work_func,
            args=(
                server_args,
                bench_args,
                tp_rank,
            ),
        )
        proc.start()
        workers.append(proc)

    for proc in workers:
        proc.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()

    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    main(server_args, bench_args)