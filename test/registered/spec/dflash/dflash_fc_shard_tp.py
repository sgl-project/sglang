# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""torchrun check: the column-sharded draft fc + multimem gather (SGLANG_OPT_DFLASH_FC_SHARD) == the replicated
fc, through DFlashWorkerV2._project_target_hidden, eager and replayed from a CUDA graph.

    torchrun --nproc-per-node 4 test/registered/spec/dflash/dflash_fc_shard_tp.py
Prints FAIL lines and one TP_RESULT line per rank; exits nonzero on any failure.
"""

import functools
import os
import sys
import types

import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.srt.runtime_context import get_parallel, publish
from sglang.srt.server_args import ServerArgs


def main() -> int:
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    publish(ServerArgs(model_path="dummy"), role="tokenizer")
    dist.init_process_group(backend="gloo")
    ps._WORLD = ps.init_world_group(ranks=list(range(world)), local_rank=local_rank, backend="nccl")
    get_parallel().override_permanently(world_group=ps._WORLD)
    # The fusion and the gatherer resolve the TP group at call time; the world group stands in for it.
    ps.get_tp_group = lambda: ps._WORLD
    if os.environ.get("L6_MUT"):
        print(f"K31_MUTATION {os.environ['L6_MUT']} applied", flush=True)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_dflash_context_fusion as T

    from sglang.srt.environ import envs
    from sglang.srt.speculative.dflash_context_fusion import DFlashContextFusion
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    gen = torch.Generator(device="cuda").manual_seed(3)  # same weights on every rank (replicated fc)
    model = T._draft_model(gen)
    with envs.SGLANG_OPT_DFLASH_FC_SHARD.override(True):
        fusion = DFlashContextFusion(model)
    w = types.SimpleNamespace(draft_model=model, _context_fusion=fusion)
    project = functools.partial(DFlashWorkerV2._project_target_hidden, w)
    stock = model.project_target_hidden
    failures, checks = 0, 0
    xgen = torch.Generator(device="cuda").manual_seed(99)
    with torch.inference_mode():
        for rows in (1, 4, 8):
            for kind in ("rand", "large", "cancel"):
                x = T._ctx_hidden(rows, xgen, kind).repeat(1, 6)
                ref, got = stock(x), project(x)
                checks += 1
                nd = int((ref != got).sum())
                if nd:
                    failures += 1
                    print(f"[rank{local_rank}] FAIL eager rows {rows} {kind}: {nd} cells differ", flush=True)
            if fusion._fc_algo.get(rows) is None:
                failures += 1
                print(f"[rank{local_rank}] FAIL rows {rows}: fc shard not engaged", flush=True)
        # Replay from a graph, as the aux graph does: static input, new values each replay.
        static = T._ctx_hidden(8, xgen, "rand").repeat(1, 6)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            project(static)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            out = project(static)
        for i, kind in enumerate(("rand", "large", "cancel", "rand")):
            static.copy_(T._ctx_hidden(8, xgen, kind).repeat(1, 6))
            g.replay()
            torch.cuda.synchronize()
            checks += 1
            nd = int((stock(static) != out).sum())
            if nd:
                failures += 1
                print(f"[rank{local_rank}] FAIL graph replay {i} {kind}: {nd} cells differ", flush=True)
    dist.barrier()
    print(f"[rank{local_rank}] TP_RESULT world={world} checks={checks} failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
