"""Row-parallel LatentMoE tail (SGLANG_K3_TAIL_SHARD): correctness gates.

The tail after the MoE all-reduce -- the latent RMSNorm, the replicated
``routed_expert_up_proj`` GEMM and the ``_add3`` -- is token-local: no
cross-token reduction happens anywhere in it. The feature therefore replaces
"every rank computes all T rows" with "rank r computes rows [r*T/w, (r+1)*T/w)
and the ranks exchange what they uniquely computed", paying one bf16
all-gather.

The claim is exactness, not approximation. The all-reduce is untouched and the
all-gather is unquantized, so the only thing that could move a value is a GEMM
whose ``M`` changed. These tests pin both halves of that:

* ``test_norm_and_add_are_row_local`` -- RMSNorm and the adds are strictly
  row-local, so slicing before the norm must equal slicing after it, bitwise.
  This is the algebra the feature rests on, and it needs no GPU.
* ``test_gate_refuses_what_it_cannot_serve`` -- the off-by-default flag and
  the token floor. A silent True here is the failure mode the gate exists to
  prevent, and neither arm needs a process group.
* ``test_shard_rows_partition`` -- the row slices tile [0, T) exactly once, in
  rank order, so the all-gather's dim-0 concatenation reconstructs the
  original token order.
* ``test_tail_shard_matches_replicated`` -- the end-to-end comparison at the
  T=19456 reference shape: replicated vs sharded-then-gathered over a real
  ``[3584, 7168]`` bf16 weight. This is the one that exercises the GEMM, and
  it is the reason the file is multi-GPU.
"""

import os

import pytest
import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_amd_ci(est_time=60, suite="nightly-amd-8-gpu-mi35x")

# Production shapes (Kimi-K3): latent width 3584, hidden 7168, chunk 19456.
LATENT, HIDDEN, TOKENS = 3584, 7168, 19456
EPS = 1e-6

_DEVICE = None


def _device():
    return torch.device("cuda", int(os.environ["LOCAL_RANK"]))


def _init_world():
    """One process group per worker process, shared by every test here."""
    global _DEVICE
    if _DEVICE is None:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        # "nccl" resolves to RCCL on ROCm, as elsewhere in the suite.
        dist.init_process_group(
            backend="nccl", device_id=torch.device("cuda", local_rank)
        )
        _DEVICE = torch.device("cuda", local_rank)
    return dist.get_rank(), dist.get_world_size()


def _rmsnorm(x, gamma, eps=EPS):
    """The row-local form the model uses (see KimiK3MoE._latent_norm)."""
    f = x.float()
    return (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * gamma


def test_norm_and_add_are_row_local():
    """Slicing before the norm must equal slicing after it, bitwise.

    RMSNorm reduces over the row only, so a row of the full-batch
    normalization is unaffected by which other rows were present. If this ever
    fails, the feature is not exact and the rest of the file is moot.
    """
    torch.manual_seed(0)
    # CPU on purpose: this is pure tensor algebra and must be checkable
    # without a GPU or a process group.
    x = torch.randn(64, LATENT, dtype=torch.bfloat16)
    gamma = torch.randn(LATENT, dtype=torch.bfloat16).abs() + 0.5
    rows = slice(8, 24)

    assert torch.equal(_rmsnorm(x, gamma)[rows], _rmsnorm(x[rows], gamma)), (
        "RMSNorm is not row-local"
    )

    shared = torch.randn(64, HIDDEN, dtype=torch.bfloat16)
    prefix = torch.randn(64, HIDDEN, dtype=torch.bfloat16)
    a = torch.randn(64, HIDDEN, dtype=torch.bfloat16)

    # The tail add is elementwise, so it is row-local in the same way.
    assert torch.equal(
        ((a + shared) + prefix)[rows],
        (a[rows] + shared[rows]) + prefix[rows],
    )


def test_gate_refuses_what_it_cannot_serve(monkeypatch):
    """The flag defaults off, and the floor refuses a below-crossover chunk.

    ``tail_shard_eligible`` checks the flag, then the token floor, and only
    then touches the process group -- so these two arms run without one.
    """
    from sglang.srt.layers import k3_sp

    # 1. Shipping default: off. The path must not be taken at any shape.
    monkeypatch.delenv("SGLANG_K3_TAIL_SHARD", raising=False)
    assert envs.SGLANG_K3_TAIL_SHARD.get() is False
    assert k3_sp.tail_shard_enabled() is False
    assert k3_sp.tail_shard_eligible(TOKENS) is False

    # 2. Armed, but the chunk is below the crossover -- where the all-gather's
    #    fixed cost exceeds the GEMM saving (measured 0.525x at T=32).
    monkeypatch.setenv("SGLANG_K3_TAIL_SHARD", "1")
    floor = envs.SGLANG_K3_TAIL_SHARD_MIN_TOKENS.get()
    assert floor > 0, "a non-positive floor would arm the path at decode sizes"
    assert k3_sp.tail_shard_enabled() is True
    assert k3_sp.tail_shard_eligible(floor - 1) is False
    assert k3_sp.tail_shard_eligible(32) is False


@pytest.mark.skipif(
    not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", 1)) < 2,
    reason="needs a multi-GPU process group",
)
@torch.inference_mode()
def test_shard_rows_partition():
    """The per-rank row slices tile [0, T) exactly once, in rank order."""
    rank, world = _init_world()
    shard = TOKENS // world
    rows = slice(rank * shard, (rank + 1) * shard)

    seen = torch.zeros(TOKENS, dtype=torch.int32, device=_device())
    seen[rows] = 1
    dist.all_reduce(seen, op=dist.ReduceOp.SUM)
    assert int(seen.min()) == 1 and int(seen.max()) == 1, (
        "row slices do not partition the batch exactly once"
    )

    # The concatenation is the identity on token order, which is why the
    # all-gather needs no permutation on either side.
    local = torch.arange(rows.start, rows.stop, device=_device(), dtype=torch.int32)
    gathered = torch.empty(TOKENS, dtype=torch.int32, device=_device())
    dist.all_gather_into_tensor(gathered, local)
    assert torch.equal(
        gathered, torch.arange(TOKENS, device=_device(), dtype=torch.int32)
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", 1)) < 2,
    reason="needs a multi-GPU process group",
)
@torch.inference_mode()
def test_tail_shard_matches_replicated():
    """Replicated tail vs sharded tail + all-gather, at the reference shape.

    The only step that can differ is the GEMM, which now runs at M = T/w
    instead of M = T. A measured run was bit-identical (0 of 139,460,608
    elements), but that is a property of the selected BLAS kernel rather than
    of the algorithm, so it is asserted rather than assumed -- a kernel
    selection change would surface here.
    """
    rank, world = _init_world()
    torch.manual_seed(1234)  # identical inputs on every rank, like the server
    dev = _device()

    latent = torch.randn(TOKENS, LATENT, device=dev, dtype=torch.bfloat16)
    shared = torch.randn(TOKENS, HIDDEN, device=dev, dtype=torch.bfloat16)
    prefix = torch.randn(TOKENS, HIDDEN, device=dev, dtype=torch.bfloat16)
    w_up = torch.randn(LATENT, HIDDEN, device=dev, dtype=torch.bfloat16) * 0.02
    gamma = torch.randn(LATENT, device=dev, dtype=torch.bfloat16).abs() + 0.5

    ref = torch.mm(_rmsnorm(latent, gamma), w_up) + shared + prefix

    rows = slice(rank * (TOKENS // world), (rank + 1) * (TOKENS // world))
    shard = torch.mm(_rmsnorm(latent[rows], gamma), w_up) + shared[rows] + prefix[rows]
    got = torch.empty(TOKENS, HIDDEN, device=dev, dtype=torch.bfloat16)
    dist.all_gather_into_tensor(got, shard.contiguous())

    mismatched = int((ref != got).sum())
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
    if rank == 0:
        print(f"\ntail shard: {mismatched} of {TOKENS * HIDDEN} elements differ")


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(2, 8))
