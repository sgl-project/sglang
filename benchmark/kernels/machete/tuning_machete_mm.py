# Adapted from https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py
import argparse
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple, TypedDict, Optional
from dataclasses import dataclass, fields

import ray
import torch
import triton
from ray.experimental.tqdm_ray import tqdm

from sgl_kernel import scalar_types, machete_mm

from utils import (BenchmarkConfig, TypeConfig, sort_config, 
                    get_configs_compute_bound, get_schedule_name, create_gemm_data)

dd_type = torch.float16
TUNE_TYPE = TypeConfig(act_type=torch.float8_e4m3fn,
                 weight_type=scalar_types.uint4b8,
                 output_type=dd_type,
                 group_scale_type=dd_type,
                 group_zero_type=None,
                 channel_scale_type=None,
                 token_scale_type=None)

def benchmark_config(
    config: BenchmarkConfig,
    shape: List, 
    types: TypeConfig,
    group_size: int = 128,
    num_iters: int = 100,
) -> float:
    schedule = get_schedule_name(config)

    # 创造 weight
    mnk_shape = [shape[0], shape[2], shape[1]]
    tensors = create_gemm_data(mnk_shape, types, group_size)

    def run():
        output = machete_mm(
            a=tensors.a,
            b_q=tensors.w_q,
            b_type=types.weight_type,
            b_group_scales=tensors.w_g_s,
            b_group_zeros=tensors.w_g_zp,
            b_group_size=group_size,
            b_channel_scales=tensors.w_ch_s,
            a_token_scales=tensors.w_tok_s,
            out_type=types.output_type,
            schedule=schedule,
        )

    # JIT compilation & warmup
    run()
    torch.cuda.synchronize()

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
    for i in range(num_iters):
        # prepare(i)
        torch.cuda.synchronize()

        start_event.record()
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
        shape,
        types,
        group_size,
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
                    shape, 
                    types,
                    group_size,
                    num_iters=100,
                )
            except triton.runtime.autotuner.OutOfResources:
                # Some configurations may be invalid and fail to compile.
                continue

            if kernel_time < best_time:
                best_time = kernel_time
                best_config = config

            
            all_times.append([get_schedule_name(config), kernel_time])
        now = datetime.now()
        print(f"{now.ctime()}] Completed tuning for shape={shape}")
        res = "============== res for shape {}\n".format(shape)
        for line in all_times:
            res += ">>>>> {} {}\n".format(line[0], line[1])
        print(res)

        assert best_config is not None
        return best_config, res


def get_shape_str(mkn):
    return "_".join([str(v) for v in mkn])

def save_configs(
    configs: Dict[int, BenchmarkConfig],
    filename = "machete_mm_linear.json"
) -> None:

    def get_dict_to_save(configs):
        d, d_bench_info = {}, {}
        for shape_str, (config, bench_time) in configs.items():
            m, k, n = [v for v in shape_str.split("_")]
            kn = f"{k}_{n}"
            d.setdefault(kn, {})[m] = get_schedule_name(config)
            d_bench_info.setdefault(kn, {})[m] = bench_time
        #d["bench_time"] = d_bench_info
        return d

    d = get_dict_to_save(configs)
    
    print(f"Writing best config to {filename}...")
    with open(filename, "w") as f:
        json.dump(d, f, indent=4)
        f.write("\n")


def main(args: argparse.Namespace):
    print(args)

    KNs_parallelDims = [
        ([1536, 24576], 1), # self_attn.q_b_proj (col)
        ([16384, 7168], 0), # self_attn.o_proj (row)
        ([18432, 7168], 0), # mlp.down_proj (row)
        ([7168, 36864], 1), # mlp.gate_up_proj (col)
    ]
    batch_sizes = [
        1, 8, 16, 32, 64, 128
    ]
    
    def refactor_kn(kn, parallel_dim, tp_size):
        if tp_size<0: return kn
        assert kn[parallel_dim] % tp_size == 0, f"dim{parallel_dim} ({kn[parallel_dim]}) must be divisible by tp_size {tp_size}"
        kn[parallel_dim] = kn[parallel_dim] // tp_size
        return kn
    
    kns = [refactor_kn(kn, parallel_dim, args.tp_size) for kn, parallel_dim in KNs_parallelDims]
    shapes = [[bs] + kn for bs in batch_sizes for kn  in kns]
    #print("shapes", shapes)

    ray.init()
    num_gpus = int(ray.available_resources()["GPU"])
    workers = [BenchmarkWorker.remote(args.seed) for _ in range(num_gpus)]
    types = TUNE_TYPE
    group_size = 128

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
                shape,
                types,
                group_size,
                search_space,
            )
            for shape in shapes
        ],
    )
    
    best_configs = {
        get_shape_str(shape): [sort_config(config[0]), config[1]] for shape, config in zip(shapes, configs)
    }
    save_configs(best_configs)
    end = time.perf_counter()
    print(f"Tuning took {end - start:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", "--tp", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    main(args)
