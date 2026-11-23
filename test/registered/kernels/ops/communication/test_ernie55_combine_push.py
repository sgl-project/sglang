# Copyright 2026 Baidu ERNIE Team and The SGLang Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import atexit
import logging
import os

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.communication.ernie55_combine_push import swapab_combine_push
from sglang.kernels.ops.communication.nvlink_comm import all_gather_push
from sglang.srt.distributed.comm_cleanup import register_comm_cleanup
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.runtime_context import get_parallel, get_platform
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.cuda is None
    or not get_platform().is_blackwell,
    reason="the combine-push Gluon kernel targets Blackwell",
)


@cache_once
def _comm() -> CustomAllReduceV2:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)), local_rank=local_rank, backend="nccl"
    )
    get_parallel().override_permanently(world_group=coord)
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    comm = CustomAllReduceV2(
        coord.cpu_group,
        torch.device(f"cuda:{local_rank}"),
        max_pull_size=0,
        max_push_size=1 << 20,
    )
    assert not comm.disabled
    register_comm_cleanup(comm)
    return comm


def _combine_reference(part, maxv, sumv, sink):
    """Merge per-tile (partial, max, sum) and the sink logit over all heads."""
    mx = torch.where(sumv > 0, maxv, -torch.inf).double()
    m = torch.maximum(mx.amax(1), sink.double())
    m = torch.where(m.isinf(), 0, m)
    factor = torch.exp(mx - m[:, None])
    den = (sumv.double() * factor).sum(1) + torch.exp(sink.double() - m)
    out = (part.double() * factor[..., None]).sum(1) / den[..., None]
    return torch.where(den[..., None] > 0, out, 0).bfloat16()


@pytest.mark.parametrize("num_tokens", [1, 6, 16])
@pytest.mark.parametrize("zeros", [False, True])
def test_all_gather_push_head_major(
    num_tokens, zeros, num_heads=64, head_dim=512, calls=3
):
    """The MLA low-latency layout: head-major shards gather into [heads, tokens, d]."""
    comm = _comm()
    rank, n_local = dist.get_rank(), num_heads // dist.get_world_size()
    g = torch.Generator(device="cuda").manual_seed(num_tokens)
    local = torch.empty(
        (n_local, num_tokens, head_dim), dtype=torch.bfloat16, device="cuda"
    )
    out = torch.empty(
        (num_heads, num_tokens, head_dim), dtype=torch.bfloat16, device="cuda"
    )
    for _ in range(calls):
        full = torch.randn(
            (num_tokens, num_heads, head_dim), device="cuda", generator=g
        ).bfloat16()
        if zeros:
            full.zero_()
        local.copy_(full[:, rank * n_local : (rank + 1) * n_local].transpose(0, 1))
        out.fill_(torch.nan)
        all_gather_push(comm.obj, local.view(-1, head_dim), out.view(-1, head_dim))
        assert torch.equal(out.transpose(0, 1), full)


@pytest.mark.parametrize(
    "num_tokens,nt,late",
    [(1, 3, False), (6, 18, True), (5, 7, True), (4, 1, False), (16, 10, True)],
)
def test_combine_push(
    num_tokens,
    nt,
    late,
    num_heads=64,
    head_dim=512,
    tokens_per_req=2,
    compress_ratio=128,
    calls=3,
):
    comm = _comm()
    rank, n_local = dist.get_rank(), num_heads // dist.get_world_size()
    heads = slice(rank * n_local, (rank + 1) * n_local)
    g = torch.Generator(device="cuda").manual_seed(num_tokens)
    t = torch.arange(num_tokens, device="cuda")
    # A token with pos + 1 - extra_len * compress_ratio <= t % tokens_per_req is
    # late: it waits until its sync slot reaches the launch count.
    is_late = (t % 3 != 1) & late
    extra_lengths = torch.randint(
        1, 5, (num_tokens,), dtype=torch.int32, device="cuda", generator=g
    )
    positions = extra_lengths * compress_ratio - 1 + t % tokens_per_req + ~is_late
    sync = torch.zeros(8 + num_tokens, dtype=torch.int32, device="cuda")
    for call in range(calls):
        shape = (num_tokens, nt, num_heads)
        part = torch.randn((*shape, head_dim), device="cuda", generator=g)
        maxv = torch.randn(shape, device="cuda", generator=g) * 4
        sumv = torch.rand(shape, device="cuda", generator=g) * 8
        sumv[torch.rand(shape, device="cuda", generator=g) < 0.25] = 0
        sumv[0] = 0
        sink = torch.randn(num_heads, device="cuda", generator=g)
        sync[8:][is_late] = call + 1
        out = swapab_combine_push(
            comm,
            part[:, :, heads].contiguous(),
            maxv[..., heads].contiguous(),
            sumv[..., heads].contiguous(),
            sink[heads].contiguous(),
            nt,
            (sync, extra_lengths, positions, tokens_per_req, compress_ratio),
        )
        ref = _combine_reference(part, maxv, sumv, sink)
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1.6e-2)
        assert sync[2:4].view(torch.int64).item() == (call + 1) << 20


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=[4])
