"""Two-rank Marlin/DeepEP correctness and component timing on synthetic weights.

Run directly with torchrun --standalone --nproc-per-node=2. No checkpoint download.
"""

import json
import os
import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.marlin import (
    fused_experts_deepep_to_marlin,
    fused_experts_none_to_marlin,
)
from sglang.srt.layers.moe.token_dispatcher import deepep
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import DeepEPMode, MoeRunnerBackend
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.marlin_deepep_utils import (
    assert_reference_close,
    make_experts,
    reference,
    reference_tolerances,
)

register_cuda_ci(est_time=120, stage="base-b", runner_config="2-gpu-large")


def test_distributed():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(Path(__file__).resolve()),
        ],
        check=True,
        timeout=1200,
        env=os.environ | {"SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32"},
    )


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    world = dist.get_world_size()
    assert world == 2
    hidden, experts = 4096, 4
    config = MoeRunnerConfig(params_dtype=torch.bfloat16)
    dispatcher = deepep.DeepEPDispatcher(
        group=group,
        router_topk=2,
        num_experts=experts,
        num_local_experts=experts // world,
        hidden_size=hidden,
        params_dtype=torch.bfloat16,
        deepep_mode=DeepEPMode.AUTO,
        async_finish=True,
        return_recv_hook=True,
        runner_backend=MoeRunnerBackend.MARLIN,
    )
    for format in ("gptq4", "gptq8", "awq", "mxfp4", "nvfp4"):
        info, matrices = make_experts(format, experts=experts, hidden=hidden)
        local = replace(
            info,
            **{
                f.name: getattr(info, f.name).chunk(world)[rank].contiguous()
                for f in fields(info)
                if isinstance(getattr(info, f.name), torch.Tensor)
            },
        )
        config.routed_scaling_factor = None if format == "mxfp4" else 1.7
        for tokens, skew in (
            (17 + rank * 3, False),
            (9, True),
            (0 if rank else 7, True),
            (0, False),
        ):
            torch.manual_seed(100 + rank)
            x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) / 4
            ids = torch.stack(
                (
                    torch.arange(tokens, device="cuda") % experts,
                    (torch.arange(tokens, device="cuda") + 1) % experts,
                ),
                -1,
            ).long()
            if skew:
                ids[:] = torch.tensor([0, 1], device="cuda")
            weights = torch.rand(tokens, 2, device="cuda") * 1.3
            topk = StandardTopKOutput(weights, ids, None)
            expected = reference(x, ids, weights, matrices, config)
            standard = fused_experts_none_to_marlin(
                StandardDispatchOutput(x, None, topk), info, config
            ).hidden_states
            assert_reference_close(standard, expected, format)
            for normal in (True, False):

                def run():
                    dispatched = dispatcher.dispatch(x, topk)
                    result = fused_experts_deepep_to_marlin(dispatched, local, config)
                    if not normal:
                        # Communication padding is unspecified. Poison it to prove
                        # that eager and captured combine read only valid rows.
                        capacity = result.hidden_states.shape[1]
                        padding = (
                            torch.arange(capacity, device=x.device)[None, :]
                            >= dispatched.masked_m[:, None]
                        )
                        result.hidden_states.masked_fill_(
                            padding[..., None], float("nan")
                        )
                    return dispatcher.combine(result)

                with patch.object(
                    deepep, "get_is_extend_in_batch", return_value=normal
                ):
                    # Compile each local expert shape before timed collectives.
                    out = run()
                    torch.cuda.synchronize()
                    assert_reference_close(out, expected, format)
                    # Both kernels are independently checked against the reference;
                    # comparing two rounded results allows both error budgets.
                    torch.testing.assert_close(
                        out,
                        standard,
                        **{
                            key: 2 * value
                            for key, value in reference_tolerances(format).items()
                        },
                    )
                    if tokens and not skew:
                        timings = []
                        for _ in range(6):
                            dist.barrier()
                            events = [
                                torch.cuda.Event(enable_timing=True) for _ in range(4)
                            ]
                            events[0].record()
                            dispatched = dispatcher.dispatch(x, topk)
                            events[1].record()
                            result = fused_experts_deepep_to_marlin(
                                dispatched, local, config
                            )
                            events[2].record()
                            out = dispatcher.combine(result)
                            events[3].record()
                            torch.cuda.synchronize()
                            timings.append(
                                [
                                    events[i].elapsed_time(events[i + 1])
                                    for i in range(3)
                                ]
                            )
                        if rank == 0:
                            ms = torch.tensor(timings[1:]).mean(0).tolist()
                            print(
                                json.dumps(
                                    dict(
                                        format=format,
                                        mode="normal" if normal else "low_latency",
                                        tokens_per_rank=[17, 20],
                                        dispatch_ms=ms[0],
                                        experts_ms=ms[1],
                                        combine_ms=ms[2],
                                        total_ms=sum(ms),
                                    )
                                ),
                                flush=True,
                            )
                    if not normal:
                        # End-to-end captured dispatch + experts + combine.
                        graph = torch.cuda.CUDAGraph()
                        dist.barrier()
                        with torch.cuda.graph(graph):
                            captured = run()
                        for _ in range(2):
                            graph.replay()
                            torch.cuda.synchronize()
                            assert_reference_close(captured, expected, format)
        dist.barrier()
    deepep.DeepEPBuffer._state().buffer = None
    dist.destroy_process_group()
    if rank == 0:
        print(
            "Marlin + DeepEP distributed correctness and graph replay passed",
            flush=True,
        )


if __name__ == "__main__":
    if "LOCAL_RANK" not in os.environ:
        test_distributed()
    else:
        # Supply application configuration; collectives, kernels, events and
        # graph replay execute on the actual GPUs.
        with (
            patch.object(moe_utils, "get_server_args", return_value=None),
            patch.object(
                moe_utils,
                "get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(deepep_dispatcher_output_dtype="auto")
                ),
            ),
        ):
            main()
