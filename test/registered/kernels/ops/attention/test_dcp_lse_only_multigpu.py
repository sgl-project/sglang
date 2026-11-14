"""Multi-GPU check of the LSE-only DCP merge (``--dcp-comm-backend ag_delay_reduce``).

A rank projects every head of its DCP group and lets the attention-TP
all-reduce sum the context partitions along with the head partitions; the
reference merges the context partials first and projects only its own head
shard. That equivalence is the premise of the backend, and it only holds once
real DCP and TP groups are nested.

Usage::

    python test/registered/kernels/ops/attention/test_dcp_lse_only_multigpu.py
"""

from __future__ import annotations

import atexit
import logging
import os

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.srt.layers.dcp import cp_lse_ag_out
from sglang.srt.runtime_context import SpawnRanks, get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.parallel_groups import publish

register_cuda_ci(est_time=90, stage="extra-b", runner_config="8-gpu-h200")

HEAD_DIM = 128
HIDDEN = 512


@cache_once
def _setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    publish(
        ServerArgs(model_path="dummy", tp_size=world_size),
        role="test",
        ranks=SpawnRanks(world_rank=local_rank),
    )
    coord = ps.init_world_group(
        ranks=list(range(world_size)), local_rank=local_rank, backend="nccl"
    )
    ps._WORLD = coord
    get_parallel().override_permanently(world_group=coord)
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    # The two routes run different-width FP32 GEMMs; TF32 rounds them apart
    # by ~1e-3 and would hide the composition this test pins down.
    torch.backends.cuda.matmul.allow_tf32 = False
    return coord


_DCP_GROUPS: dict[int, ps.GroupCoordinator] = {}


def _dcp_group(tp: ps.GroupCoordinator, dcp_size: int) -> ps.GroupCoordinator:
    """Consecutive ranks form one DCP group, nested inside tensor parallel.

    Cached per size: every GroupCoordinator holds an NCCL communicator, and
    building one per parametrization exhausts NVLS multicast memory.
    """
    if dcp_size in _DCP_GROUPS:
        return _DCP_GROUPS[dcp_size]
    local_rank = int(os.environ["LOCAL_RANK"])
    group_ranks = [
        list(range(first, first + dcp_size))
        for first in range(0, tp.world_size, dcp_size)
    ]
    group = ps.init_model_parallel_group(
        group_ranks=group_ranks,
        local_rank=local_rank,
        backend="nccl",
        use_custom_allreduce=False,
        group_name=f"dcp_test_{dcp_size}",
    )
    _DCP_GROUPS[dcp_size] = group
    return group


def _reference(dcp, partial, lse, own_heads, weight_own):
    """Merge the context partials first, then project this rank's head shard."""
    gathered_out = [torch.empty_like(partial) for _ in range(dcp.world_size)]
    gathered_lse = [torch.empty_like(lse) for _ in range(dcp.world_size)]
    dist.all_gather(gathered_out, partial, group=dcp.device_group)
    dist.all_gather(gathered_lse, lse, group=dcp.device_group)
    outs = torch.stack(gathered_out).float()
    lses = torch.stack(gathered_lse).float()
    alpha = (lses - lses.logsumexp(0)).exp()
    merged = (outs * alpha[..., None]).sum(0)[:, own_heads]
    return torch.nn.functional.linear(merged.flatten(1), weight_own.float())


@pytest.mark.parametrize("batch", [1, 8, 129])
@pytest.mark.parametrize("dcp_size", [2, 4])
@pytest.mark.parametrize("is_lse_base_on_e", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_lse_only_merge_matches_head_sharded_reference(
    batch, dcp_size, is_lse_base_on_e, dtype
):
    tp = _setup()
    if tp.world_size % dcp_size or tp.world_size == dcp_size:
        pytest.skip(f"world {tp.world_size} needs >1 DCP group of {dcp_size}")
    dcp = _dcp_group(tp, dcp_size)
    rank, world = tp.rank_in_group, tp.world_size
    # Heads are TP-sharded; a DCP group's ranks all hold the group's heads.
    heads_per_rank = 2
    group_heads = heads_per_rank * dcp_size
    total_heads = heads_per_rank * world

    # Every rank must build the same full output weight to slice from.
    generator = torch.Generator(device="cuda").manual_seed(7)
    o_weight = (
        torch.randn(HIDDEN, total_heads * HEAD_DIM, device="cuda", generator=generator)
        * 0.05
    ).bfloat16()
    group_start = (rank // dcp_size) * group_heads
    # Within the gathered partials the group's heads start at 0; the output
    # weight is sliced with the global head index instead.
    own_in_group = (rank % dcp_size) * heads_per_rank
    own_start = group_start + own_in_group
    own_heads = slice(own_in_group, own_in_group + heads_per_rank)
    weight_group = o_weight[
        :, group_start * HEAD_DIM : (group_start + group_heads) * HEAD_DIM
    ].contiguous()
    weight_own = o_weight[
        :, own_start * HEAD_DIM : (own_start + heads_per_rank) * HEAD_DIM
    ].contiguous()

    local = torch.Generator(device="cuda").manual_seed(100 + rank)
    partial = torch.randn(
        batch, group_heads, HEAD_DIM, device="cuda", generator=local
    ).to(dtype)
    lse = torch.randn(batch, group_heads, device="cuda", generator=local) * 4
    # A context shard with no keys returns -inf LSE next to a NaN partial.
    if batch > 1:
        lse[0, rank % group_heads] = -float("inf")
        partial[0, rank % group_heads] = torch.nan
    merge_lse = lse if is_lse_base_on_e else lse / torch.log(torch.tensor(2.0))

    corrected = cp_lse_ag_out(
        partial, merge_lse, dcp, is_lse_base_on_e=is_lse_base_on_e
    )
    assert corrected.shape == partial.shape
    # FP32: the claim under test is the composition, not BF16 rounding.
    actual = tp.all_reduce(
        torch.nn.functional.linear(corrected.float().flatten(1), weight_group.float())
    )

    expected = tp.all_reduce(
        _reference(dcp, partial.nan_to_num(), lse, own_heads, weight_own)
    )
    # The merge kernel stores in the partial's dtype.
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)


multigpu_pytest_main(__name__, __file__, num_gpus=[4, 8])
