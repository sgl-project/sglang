"""DCP fi_a2a: FlashInfer's fused all-to-all + LSE reduce must equal the
all-gather + Triton merge it replaces, including chunked, strided, graph-replayed
and fully-masked inputs, next to custom all-reduce v2 on the same symmetric memory.

Run with ``python test_dcp_fi_a2a_reduce.py --num-gpu 2,4,8``. Skips unless the
host can run the fused reduce (Blackwell, FlashInfer with the op).
"""

import os

import pytest
import torch
import torch.distributed as dist

from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton
from sglang.srt.distributed import parallel_state as ps
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.layers.dcp import comm
from sglang.srt.utils.common import fi_a2a_platform_blocker
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.test_utils import publish_build_topology

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

# Workspace geometry: 3 tokens x 16 heads = 48 rows, so the (heads, batch) cases
# below cover a single call, an exactly-full call and multi-chunk calls.
MAX_TOKENS, LOCAL_HEADS, HEAD_DIM = 3, 16, 512

_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
_SKIP_REASON = fi_a2a_platform_blocker(
    dcp_size=_WORLD_SIZE, tp_size=_WORLD_SIZE, pp_size=1, nnodes=1
)
pytestmark = pytest.mark.skipif(
    _SKIP_REASON is not None, reason=f"fused fi_a2a reduce {_SKIP_REASON}"
)


@pytest.fixture(scope="module")
def dcp():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    ps.init_distributed_environment(
        world_size=_WORLD_SIZE,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
    )
    publish_build_topology(tp_size=_WORLD_SIZE, world_rank=rank)
    # Builds custom all-reduce v2, whose symmetric memory is allocated before the
    # fused workspaces, as in serving.
    ps.initialize_model_parallel()
    group = ps.get_tp_group()
    capture_stream = torch.cuda.Stream()
    comm.init_fi_a2a_workspace(
        cp_group=group,
        max_tokens=MAX_TOKENS,
        local_heads=LOCAL_HEADS,
        head_dim=HEAD_DIM,
        dtype=torch.bfloat16,
        capture_stream=capture_stream,
    )
    yield group, capture_stream
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()


def _partials(group, *, batch, heads, dtype, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed + group.rank_in_group)
    total_heads = heads * group.world_size
    out = torch.randn(
        batch, total_heads, HEAD_DIM, dtype=dtype, device="cuda", generator=gen
    )
    lse = torch.randn(
        batch, total_heads, dtype=torch.float32, device="cuda", generator=gen
    )
    return out, lse


def _reference(group, out, lse, *, is_lse_base_on_e):
    """Gather every rank's partials, keep this rank's heads, merge with Triton."""
    heads = out.shape[1] // group.world_size
    own = slice(group.rank_in_group * heads, (group.rank_in_group + 1) * heads)
    all_out = [torch.empty_like(out) for _ in range(group.world_size)]
    all_lse = [torch.empty_like(lse) for _ in range(group.world_size)]
    dist.all_gather(all_out, out, group=group.device_group)
    dist.all_gather(all_lse, lse, group=group.device_group)
    merged, _ = dcp_lse_combine_triton(
        torch.stack([o[:, own] for o in all_out]).contiguous(),
        torch.stack([s[:, own] for s in all_lse]).contiguous(),
        is_lse_base_on_e=is_lse_base_on_e,
    )
    return merged


def _fused(group, out, lse, *, is_lse_base_on_e):
    return comm.dcp_a2a_lse_reduce(
        out, lse, group, is_lse_base_on_e=is_lse_base_on_e, comm_backend="fi_a2a"
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("is_lse_base_on_e", [False, True])
@pytest.mark.parametrize(
    "heads,batch", [(16, 1), (16, 3), (16, 7), (2, 1), (2, 24), (2, 49)]
)
def test_matches_all_gather_reference(dcp, dtype, is_lse_base_on_e, heads, batch):
    """Guards the head-to-peer mapping and the token chunking (B=7 at 16 heads
    and B=49 at 2 heads exceed the 48-row workspace)."""
    group, _ = dcp
    out, lse = _partials(group, batch=batch, heads=heads, dtype=dtype, seed=11)
    expected = _reference(group, out, lse, is_lse_base_on_e=is_lse_base_on_e)
    got = _fused(group, out, lse, is_lse_base_on_e=is_lse_base_on_e)
    torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-2)


def test_rows_empty_on_every_rank_are_zero(dcp):
    """A row with no KV on any rank (LSE -inf everywhere, e.g. graph padding)
    merges to 0, where the Triton combine divides 0 by 0."""
    group, _ = dcp
    out, lse = _partials(
        group, batch=2, heads=LOCAL_HEADS, dtype=torch.bfloat16, seed=23
    )
    empty = torch.arange(0, lse.shape[1], LOCAL_HEADS, device="cuda")
    lse[:, empty] = float("-inf")  # local head 0 of every destination rank
    expected = _reference(group, out, lse, is_lse_base_on_e=False)
    got = _fused(group, out, lse, is_lse_base_on_e=False)
    assert torch.equal(got[:, 0], torch.zeros_like(got[:, 0]))
    torch.testing.assert_close(got[:, 1:], expected[:, 1:], rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "layout", ["strided_head_dim", "misaligned_rows", "lse_with_unit_dim"]
)
def test_layouts_the_kernel_cannot_read_are_adapted(dcp, layout):
    """The kernel reads unit-stride rows that start on 8-byte boundaries, and
    FlashMLA returns its LSE as [B, H, 1]; such inputs are copied or reshaped,
    not rejected."""
    group, _ = dcp
    out, lse = _partials(
        group, batch=3, heads=LOCAL_HEADS, dtype=torch.bfloat16, seed=71
    )
    expected = _reference(group, out, lse, is_lse_base_on_e=True)
    if layout == "strided_head_dim":
        wide = out.new_zeros(*out.shape[:-1], 2 * HEAD_DIM)
        wide[..., ::2] = out
        out = wide[..., ::2]
    elif layout == "misaligned_rows":
        flat = out.new_empty(out.numel() + 1)
        flat[1:] = out.flatten()
        out = flat[1:].view(out.shape)  # 2 bytes past an 8-byte boundary
    else:
        lse = lse.unsqueeze(-1)
    got = _fused(group, out, lse, is_lse_base_on_e=True)
    torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-2)


def test_graph_replay_and_eager_calls_interleave(dcp):
    """The captured graph and eager decode use different workspaces on one
    ordered stream; replays and eager calls must not corrupt each other."""
    group, capture_stream = dcp
    static_out, static_lse = _partials(
        group, batch=2, heads=LOCAL_HEADS, dtype=torch.bfloat16, seed=31
    )
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        _fused(group, static_out, static_lse, is_lse_base_on_e=False)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            static_result = _fused(
                group, static_out, static_lse, is_lse_base_on_e=False
            )
    torch.cuda.current_stream().wait_stream(capture_stream)
    for step in range(3):
        new_out, new_lse = _partials(
            group, batch=2, heads=LOCAL_HEADS, dtype=torch.bfloat16, seed=40 + step
        )
        static_out.copy_(new_out)
        static_lse.copy_(new_lse)
        graph.replay()
        eager_out, eager_lse = _partials(
            group, batch=3, heads=LOCAL_HEADS, dtype=torch.bfloat16, seed=50 + step
        )
        eager_result = _fused(group, eager_out, eager_lse, is_lse_base_on_e=False)
        torch.testing.assert_close(
            static_result,
            _reference(group, new_out, new_lse, is_lse_base_on_e=False),
            rtol=1e-2,
            atol=1e-2,
        )
        torch.testing.assert_close(
            eager_result,
            _reference(group, eager_out, eager_lse, is_lse_base_on_e=False),
            rtol=1e-2,
            atol=1e-2,
        )


def test_third_stream_raises_before_the_collective(dcp):
    """Only the capture and serving streams own a workspace; a third stream
    must fail on the host on every rank, not spin in the kernel."""
    group, _ = dcp
    out, lse = _partials(
        group, batch=1, heads=LOCAL_HEADS, dtype=torch.bfloat16, seed=61
    )
    _fused(group, out, lse, is_lse_base_on_e=False)  # binds the serving stream
    with torch.cuda.stream(torch.cuda.Stream()):
        with pytest.raises(RuntimeError, match="one ordered CUDA stream"):
            _fused(group, out, lse, is_lse_base_on_e=False)


def test_custom_all_reduce_v2_stays_exact_around_fused_calls(dcp):
    """The fused workspaces share torch symmetric memory with custom all-reduce
    v2, which allocates first; neither may disturb the other."""
    group, _ = dcp
    ca_comm = group.ca_comm
    if not isinstance(ca_comm, CustomAllReduceV2) or ca_comm.disabled:
        pytest.skip("custom all-reduce v2 is not enabled on this group")
    world_size = group.world_size
    x = torch.full(
        (4096,), group.rank_in_group + 1, dtype=torch.bfloat16, device="cuda"
    )
    want_sum = torch.full_like(x, world_size * (world_size + 1) // 2)
    out, lse = _partials(
        group, batch=2, heads=LOCAL_HEADS, dtype=torch.bfloat16, seed=81
    )
    expected = _reference(group, out, lse, is_lse_base_on_e=False)
    for _ in range(2):
        assert torch.equal(ca_comm.custom_all_reduce(x), want_sum)
        got = _fused(group, out, lse, is_lse_base_on_e=False)
        torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-2)
    assert torch.equal(ca_comm.custom_all_reduce(x), want_sum)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(2, 4))
