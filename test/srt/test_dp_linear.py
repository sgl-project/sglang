import argparse

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.profiler as tprof
from torch.profiler import ProfilerActivity


class ReplicatedLinear(nn.Module):
    def __init__(
        self, in_features, out_features, bias=False, init_std=0.02, device=None
    ):
        super(ReplicatedLinear, self).__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, device=device) * init_std
        )

    def forward(self, x):
        return F.linear(x, self.weight)


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bs",
        type=int,
        default=128 * 1024,
        help="total input tokens",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=8,
        help="total input tokens",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="enable torch profile",
    )
    return parser.parse_args()


def kernel1(kernel_func, x, profile=False):
    if not profile:
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        q = kernel_func(x)
        end_event.record()
        torch.cuda.synchronize()
        e2e_time = start_event.elapsed_time(end_event)
        return q, e2e_time
    else:
        with tprof.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=tprof.tensorboard_trace_handler("./kernel1_log"),
            # with_stack=True,
            with_modules=True,
        ) as p:
            q = kernel_func(x)
    return q, -1


def kernel2(kernel_func, x, tp_size, profile=False):
    if not profile:
        start_event = torch.cuda.Event(enable_timing=True)
        comm_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        local_rank = dist.get_rank()  # TODO: only consider 1x node
        bs = x.shape[0]
        assert bs % tp_size == 0
        local_bs = bs // tp_size
        start_idx = local_rank * local_bs
        end_idx = start_idx + local_bs
        local_x = x[start_idx:end_idx, :]
        local_q = kernel_func(local_x)
        # print(f"kernel2 rank: {dist.get_rank()} local_q: {local_q.shape}, local_x: {local_x.shape}, w: {q_a_proj.weight.shape}")
        q = torch.zeros(
            bs, local_q.shape[1], dtype=local_q.dtype, device=local_q.device
        )
        dist.all_gather_into_tensor(q, local_q)
        comm_event.record()
        torch.cuda.synchronize()
        e2e_time = start_event.elapsed_time(comm_event)
        return q, e2e_time
    else:
        with tprof.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=tprof.tensorboard_trace_handler("./kernel2_log"),
            # with_stack=True,
            with_modules=True,
        ) as p:
            local_rank = dist.get_rank()  # TODO: only consider 1x node
            bs = x.shape[0]
            assert bs % tp_size == 0
            local_bs = bs // tp_size
            start_idx = local_rank * local_bs
            end_idx = start_idx + local_bs
            local_x = x[start_idx:end_idx, :]
            with tprof.record_function("forward"):
                local_q = kernel_func(local_x)
            q = torch.zeros(
                bs, local_q.shape[1], dtype=local_q.dtype, device=local_q.device
            )
            dist.all_gather_into_tensor(q, local_q)
        return q, -1


def test():
    args = parse_arg()
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())  # set correct gpu device for each rank
    device = torch.device(f"cuda:{dist.get_rank()}")
    hidden_size = 7168
    q_lora_rank = 1536
    q_a_proj = ReplicatedLinear(hidden_size, q_lora_rank).to(device)
    hidden_states = torch.randn(args.bs, hidden_size, device=device)
    # warmup for kernel1
    for _ in range(2):
        o1, _ = kernel1(q_a_proj, hidden_states)
    torch.cuda.synchronize()
    o1, t1 = kernel1(q_a_proj, hidden_states, args.profile)
    assert o1.shape == (
        args.bs,
        q_lora_rank,
    ), f"Kernel1 output shape mismatch: {o1.shape}"
    dist.all_reduce(torch.tensor(t1, dtype=torch.float).cuda(), op=dist.ReduceOp.MAX)
    if dist.get_rank() == 0:
        print(f"Kernel1, e2e completed in {t1:.2f} ms")
    # Warmup for kernel2
    for _ in range(2):
        o2, _ = kernel2(q_a_proj, hidden_states, args.tp)
    torch.cuda.synchronize()
    o2, t2 = kernel2(q_a_proj, hidden_states, args.tp, args.profile)
    assert o2.shape == (
        args.bs,
        q_lora_rank,
    ), f"Kernel2 output shape mismatch: {o2.shape}"
    dist.all_reduce(torch.tensor(t2, dtype=torch.float).cuda(), op=dist.ReduceOp.MAX)
    if dist.get_rank() == 0:
        print(f"Kernel2, e2e completed in {t2:.2f} ms")


if __name__ == "__main__":
    test()
    dist.barrier()
