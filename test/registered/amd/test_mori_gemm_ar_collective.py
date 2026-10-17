"""Cross-rank behaviour of the fused wo_b all-reduce.

Two things only a multi-rank run can show. First, that a construction failure on
one rank turns the path off on all of them: mori's phases end in device-side
barriers, so a rank that falls back while its peers fuse spins on the GPU with
no timeout and no error. Second, that the collective actually computes the
all-reduce it replaces.
"""

import os
import time
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=240, suite="stage-c-test-large-8-gpu-amd-mi35x")

N, K = 5120, 4096
PORT = 29731
#: M per rank for the end-to-end case. A multiple of world_size * _BLOCK_M at
#: every world size this runs at, so no padding is involved.
M = 8192
#: Long enough for a cold FlyDSL compile (~10s measured) with room to spare, and
#: short enough that a rank stranded in a collective fails instead of hanging CI.
SPAWN_TIMEOUT = 300
#: Relative L2 each all-gather wire may reach, keyed by whether it is fp8.
#: The bf16 leg is exact up to summation order (2.4e-3 measured); the fp8 leg
#: carries e4m3's mantissa width, which is a fixed cost rather than a margin --
#: 2.3e-2 on an MI355X TP4 microbench, 2.4e-2 end to end in review.
WIRE_REL_L2_BOUND = {False: 2e-2, True: 3e-2}


class _Group:
    def __init__(self, rank, world, cpu_group):
        self.ranks = list(range(world))
        self.world_size = world
        self.rank_in_group = rank
        self.cpu_group = cpu_group


def _setup(rank, world):
    """A fresh rank with sglang's TP group stood in for."""
    # mori's puts are compiled out without this, and then the all-reduce returns
    # mostly the local slice while every kernel still launches.
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{PORT}"
    )
    cpu_group = dist.new_group(ranks=list(range(world)), backend="gloo")

    from sglang.srt.runtime_context import get_parallel

    group = _Group(rank, world, cpu_group)
    # Permanent rather than scoped: the workers read it from several call
    # depths and there is nothing to restore in a process that exits after.
    get_parallel().override_permanently(tp_group=group, tp_size=world)
    return group


class _StubWoB:
    """``_FusedWoB``'s surface, raising at whichever stage the scenario names.

    A stub rather than the real op: the agreement has to be exercised at stages
    the real op cannot be made to fail at on demand. ``closed`` records whether
    a rank that did build released it.
    """

    fail_rank = -1
    fail_stage = ""
    closed = []

    def __init__(self, m_max, n, k):
        self.m_max, self.n, self.k = m_max, n, k
        self._maybe_fail("construct")

    def _maybe_fail(self, stage):
        if dist.get_rank() == self.fail_rank and self.fail_stage == stage:
            raise RuntimeError(f"stub failure during {stage}")

    def warmup(self):
        self._maybe_fail("warmup")

    def self_test(self):
        self._maybe_fail("self_test")

    def close(self):
        type(self).closed.append(dist.get_rank())


def _agreement_worker(rank, world, fail_rank, fail_stage):
    from unittest.mock import patch

    _setup(rank, world)
    import sglang.srt.layers.mori_gemm_ar as mg

    _StubWoB.fail_rank, _StubWoB.fail_stage, _StubWoB.closed = fail_rank, fail_stage, []
    with patch.object(mg, "_FusedWoB", _StubWoB):
        built = mg._construct_collectively(16384, N, K)

    _assert_agreed(mg, rank, built, expect=fail_rank < 0)
    if fail_rank >= 0:
        # A rank that got far enough to hold an op must release it: the window's
        # VMM reservation is a per-process cost for a path nothing will call.
        held = not (rank == fail_rank and fail_stage == "construct")
        assert _StubWoB.closed == ([rank] if held else []), (
            f"rank {rank} failing at {fail_stage}: closed={_StubWoB.closed}"
        )
    dist.destroy_process_group()


def _unique_id_worker(rank, world):
    """Rank 0 cannot produce a unique id, which its peers are waiting on.

    ``_FusedWoB`` broadcasts from inside the ``except`` for exactly this: raising
    before the broadcast leaves every peer blocked in it while this rank goes on
    to the host agreement, which is two different collectives on one gloo group.
    """
    from unittest.mock import patch

    import mori.cco

    _setup(rank, world)
    import sglang.srt.layers.mori_gemm_ar as mg

    def boom():
        raise RuntimeError("stub failure producing the unique id")

    ctx = patch.object(mori.cco.Communicator, "get_unique_id", boom)
    with ctx if rank == 0 else _nullcontext():
        built = mg._construct_collectively(16384, N, K)

    _assert_agreed(mg, rank, built, expect=False)
    dist.destroy_process_group()


def _numerics_worker(rank, world, fp8_gather):
    _setup(rank, world)
    from mori.ops.gemm_ar import preshuffle_b

    import sglang.srt.layers.mori_gemm_ar as mg
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import mxfp8_e4m3_quantize
    from sglang.srt.environ import envs

    torch.manual_seed(rank)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / 8
    w = (torch.randn(N, K, device="cuda") / 8).to(torch.float8_e4m3fn)
    w_exps = torch.randint(
        120, 135, (N // 32, K // 32), dtype=torch.uint8, device="cuda"
    )
    b_scale = w_exps.t().contiguous().to(torch.int32).reshape(-1)

    # Read once, in _FusedWoB.__init__, so the override has to wrap the build.
    with envs.SGLANG_OPT_FUSED_WO_B_AR_FP8_GATHER.override(fp8_gather):
        state = mg._FusedWoB(m_max=M, n=N, k=K)
        try:
            state.warmup()
            state.self_test()
            q, x_scale = mg.quantize_packed(x)
            out = state.run(q, x_scale, preshuffle_b(w), b_scale).clone()
        finally:
            state.close()

    # Same operands the op was given, dequantised exactly -- ue8m0 scales are
    # powers of two -- so the only difference left is summation order and the
    # bf16 output rounding. The A scales come from the canonical quantiser,
    # which the contract test pins to the packed one value for value.
    _, x_exps = mxfp8_e4m3_quantize(x)
    ref = (q.float() * _descale(x_exps)) @ (
        w.float() * _descale(w_exps).repeat_interleave(32, 0)
    ).T
    ref = ref.to(torch.bfloat16).float()
    dist.all_reduce(ref)

    rel = ((out.float() - ref).norm() / ref.norm()).item()
    bound = WIRE_REL_L2_BOUND[fp8_gather]
    assert rel < bound, (
        f"rank {rank}: relative L2 {rel:.3e} against the split path, bound {bound:g}"
    )
    dist.destroy_process_group()


def _assert_agreed(mg, rank, built, expect):
    assert built is expect, f"rank {rank}: built={built}, expected {expect}"
    assert mg._disabled is not expect, f"rank {rank}: _disabled={mg._disabled}"
    assert (mg._state is not None) is expect, f"rank {rank}: _state={mg._state}"


def _descale(exps):
    """ue8m0 ``[*, K // 32]`` -> the fp32 multiplier for each of its K columns."""
    return torch.exp2(exps.float() - 127).repeat_interleave(32, -1)


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


class TestFusedWoBCollective(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = min(8, torch.cuda.device_count())
        if cls.world < 2:
            raise unittest.SkipTest("needs at least 2 GPUs")
        try:
            import mori.ops.gemm_ar  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("this build has no mori.ops.gemm_ar")

    def _spawn(self, fn, *args):
        """Run `fn` on every rank, failing rather than hanging if one strands."""
        ctx = mp.start_processes(
            fn,
            args=(self.world, *args),
            nprocs=self.world,
            join=False,
            start_method="spawn",
        )
        deadline = time.monotonic() + SPAWN_TIMEOUT
        while not ctx.join(timeout=1):
            if time.monotonic() > deadline:
                for proc in ctx.processes:
                    proc.kill()
                self.fail(f"{fn.__name__} did not finish in {SPAWN_TIMEOUT}s")

    def test_every_rank_builds_when_none_fails(self):
        self._spawn(_agreement_worker, -1, "")

    def test_a_failing_rank_disables_every_rank(self):
        """One rank short of an op means none of them may use one.

        The stages are separate scenarios because ``_disable`` has different
        work to do at each: before the op exists there is nothing to release,
        after it there is a window reservation that nothing will read.
        """
        for stage in ("construct", "warmup", "self_test"):
            with self.subTest(stage=stage):
                self._spawn(_agreement_worker, 1, stage)

    def test_unique_id_failure_does_not_strand_the_peers(self):
        self._spawn(_unique_id_worker)

    def test_fused_result_matches_the_split_path(self):
        """What signature conformance cannot show: that the bytes move.

        A mori built with ``BUILD_CCO_SDMA=OFF`` runs every kernel, returns
        mostly the local slice, and measures faster for it.

        Both wires, because they fail differently: the bf16 leg loses bytes
        silently, while the fp8 leg is lossy by construction and so needs a
        bound of its own before a regression can be told from e4m3.
        """
        for fp8_gather in (False, True):
            with self.subTest(wire="fp8" if fp8_gather else "bf16"):
                self._spawn(_numerics_worker, fp8_gather)


if __name__ == "__main__":
    unittest.main()
