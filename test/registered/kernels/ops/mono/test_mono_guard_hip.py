"""The HIP monokernel waits under a guard (RFC #43949), on one gfx950 GPU:
a satisfied wait leaves the error words clean, a wait nobody satisfies stops
at its deadline with the ``deadline`` word set, later waits stop at once and
keep a ``peer`` word, the fault injection stalls the named site, and a failed
wait stops the launch (it never runs on)."""

import subprocess
import sys
import time

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.version.hip
        and "gfx95" in torch.cuda.get_device_properties(0).gcnArchName
    ),
    reason="needs a gfx950 GPU",
)


def _build(publish: bool, fault_site=None, stop=False):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels import buffer_ops as bo
    from flydsl.expr import const_expr
    from flydsl.expr.typing import Int64

    from sglang.kernels.ops.mono.common_hip import guard
    from sglang.kernels.ops.mono.common_hip.ops import CM_DEV, rsrc
    from sglang.kernels.ops.mono.common_hip.plan import THREADS
    from sglang.kernels.ops.mono.common_hip.sync import Mailbox, sreg

    name = f"mono_guard_test_p{int(publish)}_f{fault_site or 0}_s{int(stop)}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k(epoch: Int64, box: Int64):
        with guard.scope(guard.Guard(epoch, fault_site=fault_site, stop=stop)):
            # held in a dict, as the kernels hold it: traced branches carry IR values only
            c = {"mb": Mailbox(fx.Int32(7))}
            if const_expr(publish):
                if (fx.block_idx.x == 1) & (fx.thread_idx.x == 0):
                    c["mb"].put(sreg(box, 0, "x"), 0, fx.Int32(42))
            got = c["mb"].poll([(sreg(box, 0, "x"), 0, 1)])
            if (fx.block_idx.x == 0) & (fx.thread_idx.x == 0):
                bo.buffer_store(got[0][0], rsrc(box), 2, cache_modifier=CM_DEV)

    @flyc.jit
    def launch(epoch: Int64, box: Int64, stream: fx.Stream = fx.Stream(None)):
        k(epoch, box).launch(grid=(2,), block=(THREADS,), stream=stream)

    return launch


def _epoch(deadline_s):
    from sglang.kernels.ops.mono.common_hip import guard

    e = torch.zeros(64, dtype=torch.int32, device="cuda")
    ticks = guard.deadline_ticks(deadline_s)
    lo = ticks & 0xFFFFFFFF
    e[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    e[guard.GUARD_DEADLINE + 1] = ticks >> 32
    return e


def _words(e):
    from sglang.kernels.ops.mono.common_hip import guard

    return [int(e[i]) for i in guard.ERROR_WORDS]


def _run(launch, e, box):
    t = time.monotonic()
    launch(e.data_ptr(), box.data_ptr(), stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    return time.monotonic() - t


def test_satisfied_wait_is_clean():
    e, box = _epoch(5.0), torch.zeros(4, dtype=torch.int32, device="cuda")
    _run(_build(publish=True), e, box)
    assert _words(e) == [0, 0, 0]
    assert int(box[2]) == 42


def test_unsatisfied_wait_returns_at_its_deadline_and_later_waits_at_once():
    e, box = _epoch(0.3), torch.zeros(4, dtype=torch.int32, device="cuda")
    took = _run(_build(publish=False), e, box)
    deadline, peer, fault = _words(e)
    assert deadline == 2 and fault == 0, _words(e)  # site 1, stored as 1 + site
    assert 0.25 < took < 30, took
    # a later launch's waits see the failure and return at once
    box.zero_()
    took = _run(_build(publish=False), e, box)
    assert took < 0.25, took
    assert _words(e)[1] == 1  # the peer word: it stopped on a failure elsewhere


def test_fault_injection_stalls_the_named_site():
    e, box = _epoch(0.3), torch.zeros(4, dtype=torch.int32, device="cuda")
    _run(_build(publish=True, fault_site=1), e, box)
    assert _words(e)[0] == 2, _words(e)


def _stop_child():
    """A wait nobody satisfies, with the stop on: the launch must not return."""
    e, box = _epoch(0.2), torch.zeros(4, dtype=torch.int32, device="cuda")
    _run(_build(publish=False, stop=True), e, box)
    print("RETURNED", flush=True)


def test_a_failed_wait_stops_the_launch():
    code = (
        "import importlib.util, sys;"
        f"spec = importlib.util.spec_from_file_location('g', {__file__!r});"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
        "m._stop_child()"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert r.returncode != 0 and "RETURNED" not in r.stdout, (
        r.returncode,
        r.stdout[-500:],
    )
