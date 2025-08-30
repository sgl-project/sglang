# Adapted from https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py
import argparse
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple, TypedDict, Optional

import ray
import torch
import triton
import random
from ray.experimental.tqdm_ray import tqdm
from transformers import AutoConfig

from sgl_kernel import ScalarType, scalar_types
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts_machete_impl

from utils import (BenchmarkConfig, TypeConfig, sort_config, 
                    get_configs_compute_bound, get_schedule_name, create_moe_data)

MNK_SHAPES = [
    (1, 7168, 2048),
    (1, 2048, 7168),
]

GROUP_SIZES_TO_TEST: list[Optional[int]] = [128]

ddtype = torch.float16

TUNE_TYPE = TypeConfig(act_type=torch.float8_e4m3fn,
                 weight_type=scalar_types.uint4b8,
                 output_type=ddtype,
                 group_scale_type=ddtype,
                 group_zero_type=None,
                 channel_scale_type=None,
                 token_scale_type=None)


def generate_topk_dist(E, num_tokens, num_topks):
    topk_dist = []
    for i in range(num_tokens*num_topks):
        r = random.random()
        if len(topk_dist) < E and (len(topk_dist) == 0 or r < 0.2):
            topk_dist.append(1)
        else:
            is_added = False
            for j in range(len(topk_dist)):
                r = random.random()
                if r > 0.98 and topk_dist[j] < num_tokens:
                    topk_dist[j] += 1
                    is_added = True
                    break
            if is_added:
                continue
    topk_dist.sort(key=lambda x: -x)
    return topk_dist

def create_hidden_states(num_tokens, dim, dtype, num_experts, num_topk, device):
    from copy import deepcopy
    hidden_states = torch.randn([num_tokens, dim], dtype=ddtype, device=device)
    ll = [i for i in range(num_experts-1)]
    topks = []
    # Simulated fusion of shared_experts
    for i in range(num_tokens):
        ll_cp = deepcopy(ll)
        random.shuffle(ll_cp)
        topks.append(ll_cp[:num_topk-1] + [num_experts-1])

    topk_dist = generate_topk_dist(num_experts, num_tokens, num_topk)
    topks = torch.tensor(topks, device=device)

    topk_weights = torch.randn([num_tokens, num_experts], dtype=ddtype, device=device)
    return hidden_states, topks, topk_weights


def save_configs(
    configs: Dict[int, BenchmarkConfig],
    num_experts: int,
) -> None:
    filename = f"experts_num={num_experts},awq_w4a8_machete_moe.json"
    d = {k: get_schedule_name(configs[k][0]) for k in configs}
    #d["bench_time"] = {k: configs[k][1] for k in configs}

    print(f"Writing best config to {filename}...")
    with open(filename, "w") as f:
        json.dump(d, f, indent=4)
        f.write("\n")

def benchmark_config(
    config: BenchmarkConfig,
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    types: TypeConfig,
    num_iters: int = 100,
) -> float:
    schedule = get_schedule_name(config)

    # create weight, hidden_states
    tensors = create_moe_data(num_experts, [num_tokens, shard_intermediate_size, hidden_size], types, 128, 8)
    hidden_states, topk_ids, topk_weights = create_hidden_states(num_tokens, hidden_size, torch.float8_e4m3fn, num_experts, topk, tensors.w_q.device)

    def run():
        fused_experts_machete_impl(
            hidden_states,
            tensors.w_q,
            tensors.w_q2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            w1_scale=tensors.w_g_s,
            w2_scale=tensors.w_g_s2,
            w1_zp=tensors.w_g_zp,
            w2_zp=tensors.w_g_zp2,
            no_combine=False,
            has_zp=True,
            routed_scaling_factor=None,
            schedule=schedule
        )

    # JIT compilation & warmup
    try:
        run()
    except:
        import traceback
        print("error schedule", schedule)
        traceback.print_exc()
    torch.cuda.synchronize()
    # print("run ok")

    # Capture 10 invocations with CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(5):
            run()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    

    latencies: List[float] = []
    torch.cuda.synchronize()
    start_event.record()
    for i in range(num_iters):
        # prepare(i)
        graph.replay()
    end_event.record()
    end_event.synchronize()
    latencies.append(start_event.elapsed_time(end_event))
    avg = sum(latencies) / num_iters * 100  # us
    graph.reset()
    return avg


@ray.remote(num_gpus=1)
class BenchmarkWorker:
    def __init__(self, seed: int) -> None:
        torch.set_default_device("cuda")
        torch.cuda.manual_seed_all(0)
        self.seed = seed

    def tune(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        search_space: List[Dict[str, int]],
    ) -> Dict[str, int]:
        best_config = None
        best_time = float("inf")
        types = TUNE_TYPE
        all_times = []  
        for config in tqdm(search_space):
            try:
                kernel_time = benchmark_config(
                    config,
                    num_tokens,
                    num_experts,
                    shard_intermediate_size,
                    hidden_size,
                    topk,
                    types,
                    num_iters=20,
                )
            except triton.runtime.autotuner.OutOfResources:
                # Some configurations may be invalid and fail to compile.
                continue

            if kernel_time < best_time:
                best_time = kernel_time
                best_config = config

            all_times.append([get_schedule_name(config), kernel_time])
        now = datetime.now()
        print(f"{now.ctime()}] Completed tuning for batch_size={num_tokens}")
        res = "============== res for bs {}\n".format(num_tokens)
        for line in all_times:
            res += ">>>>> {} {}\n".format(line[0], line[1])
        print(res)

        assert best_config is not None
        return best_config, res



def main(args: argparse.Namespace):
    print(args)
    
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if config.architectures[0] in ["DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"]:
        experts_num = (
            config.n_routed_experts + (0 if args.disable_shared_experts_fusion else 1)
            if config.architectures[0] in ["DeepseekV3ForCausalLM"]
            else config.n_routed_experts
        )
        intermediate_size = config.moe_intermediate_size
    else:
        experts_num = config.num_local_experts
        intermediate_size = config.intermediate_size
    
    shard_intermediate_size = 2 * intermediate_size // args.tp_size
    topk = config.num_experts_per_tok
    hidden_size = getattr(config, "hidden_size", None) or config.text_config.hidden_size
    dtype = config.torch_dtype

    if args.batch_size is None:
        batch_sizes = [
            1, 8, 16, 32, 64, 128
        ]
    else:
        batch_sizes = [args.batch_size]

    ray.init()
    num_gpus = int(ray.available_resources()["GPU"])
    workers = [BenchmarkWorker.remote(args.seed) for _ in range(num_gpus)]

    def _distribute(method: str, inputs: List[Any]) -> List[Any]:
        outputs = []
        worker_idx = 0
        for input_args in inputs:
            worker = workers[worker_idx]
            worker_method = getattr(worker, method)
            output = worker_method.remote(*input_args)
            outputs.append(output)
            worker_idx = (worker_idx + 1) % num_gpus
        return ray.get(outputs)


    search_space = get_configs_compute_bound()
    print(f"Start tuning over {len(search_space)} configurations...")

    start = time.perf_counter()
    configs = _distribute(
        "tune",
        [
            (
                batch_size,
                experts_num,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype,
                search_space,
            )
            for batch_size in batch_sizes
        ],
    )
    best_configs = {
        M: [sort_config(config[0]), config[1]] for M, config in zip(batch_sizes, configs)
    }
    save_configs(best_configs, experts_num)
    end = time.perf_counter()
    print(f"Tuning took {end - start:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/data/models/DeepSeek/DeepSeek-R1-BF16-AWQ-ADD8/")
    parser.add_argument("--tp-size", "--tp", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument("--disable-shared-experts-fusion", action="store_true")
    args = parser.parse_args()

    main(args)
