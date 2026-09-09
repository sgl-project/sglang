"""Exercise the real SGLang dispatcher with synthetic experts and real PyNccl.

The configuration/coordinator are minimal experiment scaffolding. Communication,
FP8 post-quantization and combine use the implementation under review unchanged.
No model weights or replacement communication kernels are involved.
"""

from dataclasses import dataclass
from datetime import timedelta
from itertools import product
from types import SimpleNamespace
from typing import Annotated

import torch
import torch.distributed as dist

from .comparison import compare
from .environment import binding_check, prepare_jit, require_pair
from .oracle import (
    dequantize_fp8,
    make_fixture,
    validate_capacity,
)


def initialize(capacity, *, graph_enabled=False):
    rank, local_rank = require_pair(ep=True)
    bindings = binding_check()
    bindings["jit"] = prepare_jit()
    from sglang.srt.arg_groups.arg_utils import NS
    from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
    from sglang.srt.layers.moe.utils import MoeA2ABackend, NcclEpMode
    from sglang.srt.runtime_context import get_context, get_flags

    @dataclass
    class LabConfig:
        enable_deterministic_inference: Annotated[bool, NS("exec.deterministic")] = (
            False
        )
        enable_nccl_ep_cuda_graph: Annotated[bool, NS("exec.moe")] = graph_enabled

    get_context().set_server_args(LabConfig())
    flags = get_flags().moe
    flags.a2a_backend = MoeA2ABackend.NCCL_EP
    flags.nccl_ep_mode = NcclEpMode.LOW_LATENCY
    flags.nccl_ep_num_max_dispatch_tokens_per_rank = capacity
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    comm = PyNcclCommunicator(
        group=dist.group.WORLD, device=local_rank, library_path=bindings["nccl_path"]
    )
    if not comm.available:
        raise RuntimeError("The real PyNccl communicator could not initialize")
    coordinator = SimpleNamespace(
        world_size=2,
        rank=rank,
        rank_in_group=rank,
        device=torch.device("cuda", local_rank),
        pynccl_comm=comm,
        barrier=dist.barrier,
    )
    return rank, coordinator, bindings


def forward_layer(dispatcher, x, ids, weights, rank, *, identity=False):
    """Only GPU work: safe to use as a runner callback once EP capture is enabled.

    Snapshot receive counters, but retain the dispatcher's actual combined
    output so multi-layer tests can detect scratch aliasing. CPU oracle checks
    run after the complete forward, outside capture and the staged transaction.
    """
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    dispatched = dispatcher.dispatch(x, StandardTopKOutput(weights, ids, None))
    received = dequantize_fp8(dispatched.hidden_states, dispatched.hidden_states_scale)
    counters = dispatched.masked_m.clone()
    factors = torch.arange(rank * 2 + 1, rank * 2 + 3, device=x.device, dtype=x.dtype)
    expert_output = received if identity else received * factors[:, None, None]
    combined = dispatcher.combine(
        DeepEPLLCombineInput(
            expert_output, dispatched.topk_ids, dispatched.topk_weights
        )
    )
    return received, counters, combined


def run_layer(dispatcher, batch, rank, *, identity=False):
    inputs = tuple(
        items[rank].cuda() for items in (batch.tokens, batch.expert_ids, batch.weights)
    )
    actual = forward_layer(dispatcher, *inputs, rank, identity=identity)
    compare(batch, rank, *actual, identity=identity)


def exercise_eager(*, buckets=(8, 16, 32), layers=2, identity=False, **unused):
    rank, coordinator, bindings = initialize(max(buckets))
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher.nccl_ep import (
        NcclEpBuffer,
        NcclEpDispatcher,
    )

    dispatchers = [
        NcclEpDispatcher(
            MoeRunnerConfig(
                num_experts=4,
                num_local_experts=2,
                hidden_size=2048,
                top_k=2,
                params_dtype=torch.bfloat16,
                layer_id=layer,
            ),
            coordinator,
        )
        for layer in range(layers)
    ]
    checked = 0
    for bucket, case, change, step in product(
        buckets,
        ("balanced", "hotspot", "padding", "empty_rank", "all_masked"),
        ("tokens", "routing", "weights", "all"),
        (0, 1, 0),
    ):
        batch = make_fixture(bucket, case=case, step=step, change=change)
        validate_capacity(batch, max(buckets))
        for dispatcher in dispatchers:
            run_layer(dispatcher, batch, rank, identity=identity)
            checked += 1
    torch.cuda.synchronize()
    NcclEpBuffer.destroy()
    # PyNccl has no destructor in this pinned revision. The experiment owns
    # its communicator and explicitly closes it through the real core binding.
    import nccl.core as core

    core.Communicator(ptr=coordinator.pynccl_comm.comm.value).destroy()
    coordinator.pynccl_comm.available = False
    coordinator.pynccl_comm.disabled = True
    dist.destroy_process_group()
    return {
        "implementation": "sglang_dispatcher_eager",
        "checked": checked,
        "bindings": bindings,
        "fp8_scales_applied": True,
        "tolerance": {"rtol": 0, "atol": 0},
    }
