################################################################################
#
# Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################


from functools import partial
import torch
import numpy as np
import os
import sys
import torch
import datetime

print = partial(print, file=sys.stderr)

RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
NNODES = WORLD_SIZE // LOCAL_WORLD_SIZE
NODE_ID = RANK // LOCAL_WORLD_SIZE
TP_LOCAL_GROUP = None
torch.cuda.set_device(LOCAL_RANK)
barrier_tensor = torch.tensor([1], device="cuda")
torch.distributed.init_process_group(
    backend="nccl", world_size=WORLD_SIZE, rank=RANK, timeout=datetime.timedelta(seconds=1800)
)
assert torch.distributed.is_initialized()
# use all ranks as tp group
TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")


def init_seed():
    os.environ["NCCL_DEBUG"] = os.getenv("NCCL_DEBUG", "ERROR")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_printoptions(precision=2)
    torch.manual_seed(3 + RANK)
    torch.cuda.manual_seed_all(3 + RANK)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    np.random.seed(3 + RANK)


def is_local_tp_group_initialized():
    return TP_LOCAL_GROUP is not None


def init_local_groups():
    global TP_LOCAL_GROUP
    if is_local_tp_group_initialized():
        return
    if NNODES == 1:
        TP_LOCAL_GROUP = TP_GROUP
    else:
        for n in range(NNODES):
            ranks = list(range(LOCAL_WORLD_SIZE * n, LOCAL_WORLD_SIZE * (n + 1)))
            pg = torch.distributed.new_group(
                ranks=ranks,
                backend="nccl",
            )
            if RANK in ranks:
                TP_LOCAL_GROUP = pg
    assert LOCAL_RANK == RANK % LOCAL_WORLD_SIZE
    assert TP_LOCAL_GROUP.rank() == RANK % LOCAL_WORLD_SIZE
    assert TP_LOCAL_GROUP.size() == LOCAL_WORLD_SIZE


def get_profiler(exp_name, warmups=1, iters=1):
    run_id = os.environ["TORCHELASTIC_RUN_ID"]
    import pathlib

    prof_dir = f"prof/{pathlib.Path(sys.argv[0]).stem}_{run_id}/{exp_name}"
    os.makedirs(prof_dir, exist_ok=True)
    return torch.profiler.profile(
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            prof_dir, worker_name=f"{RANK}", use_gzip=True
        ),
        schedule=torch.profiler.schedule(wait=1, warmup=warmups, active=iters, repeat=1),
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    )


def _async_barrier():
    torch.distributed.all_reduce(barrier_tensor, op=torch.distributed.ReduceOp.MAX, async_op=False)


def run_perf(expname, warmups, iters, func, sync_per_iter=False):
    profiler = get_profiler(expname, warmups, iters)
    profiler.start()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stop_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for n in range(warmups + iters):
        if n >= warmups:
            start_events[n - warmups].record()
            func(iter=n)
            stop_events[n - warmups].record()
        else:
            func(iter=n)
        if sync_per_iter:
            _async_barrier()
        profiler.step()
    profiler.stop()
    [e.synchronize() for e in stop_events]
    elapsed_time_avg = (
        sum(
            [
                start_event.elapsed_time(stop_event)
                for start_event, stop_event in zip(start_events, stop_events)
            ]
        )
        / iters
    )
    print(f"expname: {expname}, avg: {elapsed_time_avg} ms/iter")
    return elapsed_time_avg
