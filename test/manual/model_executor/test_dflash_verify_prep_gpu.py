# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Stream protocol of the DFlash verify prep beside the draft (SGLANG_OPT_DFLASH_VERIFY_PREP_BESIDE_DRAFT).

Each round is built the way the runners use it: the main stream produces the verify inputs, marks the draft
start, runs the "draft" (a sleep kernel), the side stream stages the slots and a metadata buffer derived from
them, and a "verify" on the main stream copies slots and metadata out. Expected values are closed-form
functions of the round number (distinct in every field and round); outputs are poisoned before each round, and each
round's sources stay alive so the allocator cannot hand a later round the same bytes. Needs one CUDA GPU.
"""

import contextlib

import pytest
import torch

from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    CudaGraphBufferRegistry,
    GraphSlot,
)
from sglang.srt.model_executor.input_buffers import share_input_buffer
from sglang.srt.model_executor.runner import dflash_verify_prep as vp

# A GPU test: fail loudly instead of skipping where there is no GPU.
assert torch.cuda.is_available(), "test_dflash_verify_prep_gpu needs a CUDA GPU"

BS, TOK = 2, 16
SLOTS = (
    "positions",
    "out_cache_loc",
    "req_pool_indices",
    "seq_lens",
    "mamba_track_indices",
)


def _cycles_per_us() -> float:
    # Raise the SM clock from idle first; the ratio below still drifts with the clock, so tests compare
    # sleeps with each other, never with wall-clock constants.
    torch.cuda._sleep(50_000_000)
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    torch.cuda._sleep(2_000_000)
    end.record()
    end.synchronize()
    return 2_000_000 / (start.elapsed_time(end) * 1000.0)


CYCLES_PER_US = _cycles_per_us()


class _DelayedPrep(vp.DFlashVerifyPrep):
    """DFlashVerifyPrep plus delay injection (test only).

    ``delay_cycles``: a side-stream sleep right after the side stream's wait on the
    mark exposes a consumer that does not wait for the side stream.
    ``main_delay_cycles``: a main-stream sleep right after the mark exposes a
    side-stream read of anything the main stream makes after it.
    """

    def __init__(self, *, stream, delay_cycles: int = 0, main_delay_cycles: int = 0):
        super().__init__(stream=stream)
        self._delay_cycles = delay_cycles
        self._main_delay_cycles = main_delay_cycles

    def mark_draft_start(self) -> None:
        super().mark_draft_start()
        if self._main_delay_cycles > 0:
            torch.cuda._sleep(self._main_delay_cycles)

    @contextlib.contextmanager
    def side(self):
        first = not self._issued
        # The body runs on the side stream, after the side stream's wait on the mark.
        with super().side() as main:
            if first and self._delay_cycles > 0:
                torch.cuda._sleep(self._delay_cycles)
            yield main


def _sleep_us(us: float) -> None:
    torch.cuda._sleep(int(us * CYCLES_PER_US))


def _expected(r: int) -> dict:
    tok = torch.arange(TOK, dtype=torch.int64)
    req = torch.arange(BS, dtype=torch.int64)
    return {
        "positions": tok + 1000 * (r + 1) + 7,
        "out_cache_loc": tok * 3 + 7 * (r + 1) + 50001,
        "req_pool_indices": req + 13 * (r + 1) + 5,
        "seq_lens": req * 11 + 100 * (r + 1) + 9,
        "mamba_track_indices": req * 2 + 3 * (r + 1) + 70000,
    }


def _produce(r: int) -> dict:
    """Round r's verify inputs, written by kernels on the current stream (fresh tensors)."""
    tok = torch.arange(TOK, dtype=torch.int64, device="cuda")
    req = torch.arange(BS, dtype=torch.int64, device="cuda")
    return {
        "positions": tok + 1000 * (r + 1) + 7,
        "out_cache_loc": tok * 3 + 7 * (r + 1) + 50001,
        "req_pool_indices": req + 13 * (r + 1) + 5,
        "seq_lens": req * 11 + 100 * (r + 1) + 9,
        "mamba_track_indices": req * 2 + 3 * (r + 1) + 70000,
    }


def _registry() -> CudaGraphBufferRegistry:
    reg = CudaGraphBufferRegistry(
        device=torch.device("cuda"), max_bs=BS, max_num_tokens=TOK
    )
    for name in SLOTS:
        tokens = name in ("positions", "out_cache_loc")
        shape = (lambda b, t: (t,)) if tokens else (lambda b, t: (b,))
        reg.register_slot(
            GraphSlot(name, shape, torch.int64, axis="tokens" if tokens else "bs")
        )
    return reg


class _Round:
    """One registry, metadata buffer and output buffer reused across rounds, like the verify runner's."""

    def __init__(self, prep: vp.DFlashVerifyPrep):
        self.prep = prep
        self.reg = _registry()
        self.meta = torch.zeros(BS, dtype=torch.int64, device="cuda")
        self.side_done = torch.cuda.Event(enable_timing=True)

    def _prep_metadata(self) -> None:
        # Stands for the backend prep: reads the staged slot, writes its own buffer.
        self.meta.copy_(self.reg.get_slot("seq_lens").buffer[:BS] * 2 + 1)
        self.side_done.record()

    def stage(self, sources: dict) -> None:
        vp.stage_beside_draft(
            prep=self.prep,
            registry=self.reg,
            sources=sources,
            run_prep=self._prep_metadata,
            check=False,
        )

    def verify(self, out: torch.Tensor) -> None:
        cols = [
            self.reg.get_slot(n).buffer[
                : (TOK if n in ("positions", "out_cache_loc") else BS)
            ]
            for n in SLOTS
        ]
        out.copy_(torch.cat(cols + [self.meta]))


def _reference(r: int) -> torch.Tensor:
    e = _expected(r)
    return torch.cat([e[n] for n in SLOTS] + [e["seq_lens"] * 2 + 1])


def _poisoned_out() -> torch.Tensor:
    return torch.full((2 * TOK + 4 * BS,), -1, dtype=torch.int64, device="cuda")


def test_verify_waits_for_side_work_that_ends_after_the_draft():
    """The side stream finishes 3 ms after the 0.3 ms draft: the verify must still see every staged value."""
    prep = _DelayedPrep(
        stream=torch.cuda.Stream(), delay_cycles=int(3000 * CYCLES_PER_US)
    )
    rnd = _Round(prep)
    outs, srcs = [], []
    for r in range(3):
        out = _poisoned_out()
        srcs.append(_produce(r))
        prep.mark_draft_start()
        _sleep_us(300)
        rnd.stage(srcs[-1])
        rnd.verify(out)
        outs.append(out)
    torch.cuda.synchronize()
    for r, out in enumerate(outs):
        assert torch.equal(out.cpu(), _reference(r)), f"round {r}: {out.cpu().tolist()}"


def test_side_work_runs_beside_the_draft():
    """With no delay the staged prep completes while the draft still runs (the point of the knob)."""
    prep = vp.DFlashVerifyPrep(stream=torch.cuda.Stream())
    rnd = _Round(prep)
    t0, draft_end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    out = _poisoned_out()
    src = _produce(1)
    t0.record()
    prep.mark_draft_start()
    _sleep_us(8000)
    draft_end.record()
    rnd.stage(src)
    rnd.verify(out)
    torch.cuda.synchronize()
    assert torch.equal(out.cpu(), _reference(1))
    side_ms, draft_ms = t0.elapsed_time(rnd.side_done), t0.elapsed_time(draft_end)
    # The draft must last at least 1 ms, and the side work must end in its first half.
    assert draft_ms >= 1.0, f"draft took only {draft_ms:.3f} ms"
    assert side_ms < 0.5 * draft_ms, (
        f"side done at {side_ms:.3f} ms, draft end at {draft_ms:.3f} ms"
    )


def test_next_round_does_not_overwrite_slots_the_verify_still_reads():
    """Round r's verify starts 2 ms late; round r+1's staging (issued right after) must not touch the slots
    before it, because round r+1's side work waits on its own mark, which follows round r's verify."""
    prep = vp.DFlashVerifyPrep(stream=torch.cuda.Stream())
    rnd = _Round(prep)
    out0 = _poisoned_out()
    src0 = _produce(0)
    prep.mark_draft_start()
    _sleep_us(100)
    rnd.stage(src0)
    _sleep_us(2000)
    rnd.verify(out0)
    out1 = _poisoned_out()
    src1 = _produce(1)
    prep.mark_draft_start()
    _sleep_us(100)
    rnd.stage(src1)
    rnd.verify(out1)
    torch.cuda.synchronize()
    assert torch.equal(out0.cpu(), _reference(0)), f"round 0 read {out0.cpu().tolist()}"
    assert torch.equal(out1.cpu(), _reference(1))


def test_a_new_mark_settles_a_round_that_never_reached_its_verify():
    """Side work issued after a mark (the mamba track rebuild) is consumed on the main stream after the next
    mark when the round's verify never ran: the new mark orders the main stream after it."""
    prep = _DelayedPrep(
        stream=torch.cuda.Stream(), delay_cycles=int(3000 * CYCLES_PER_US)
    )
    out = torch.full((BS,), -1, dtype=torch.int64, device="cuda")
    prep.mark_draft_start()
    with prep.side() as main:
        made = torch.arange(BS, dtype=torch.int64, device="cuda") * 7 + 3
        made.record_stream(main)
    prep.mark_draft_start()
    out.copy_(made)
    torch.cuda.synchronize()
    assert out.cpu().tolist() == [3, 10]
    prep.settle()


def test_main_delay_exposes_a_source_made_after_the_mark():
    """A staged source the main stream makes after the mark breaks the contract: with the main-stream
    delay the side stream always copies it before it exists, and the per-round check reports it."""
    prep = _DelayedPrep(
        stream=torch.cuda.Stream(), main_delay_cycles=int(3000 * CYCLES_PER_US)
    )
    rnd = _Round(prep)
    prep.mark_draft_start()
    late = _produce(7)
    with pytest.raises(RuntimeError, match="differs"):
        vp.stage_beside_draft(
            prep=prep,
            registry=rnd.reg,
            sources=late,
            run_prep=rnd._prep_metadata,
            check=True,
        )
    torch.cuda.synchronize()


def test_check_rejects_a_pool_alias_and_a_stale_slot():
    reg = CudaGraphBufferRegistry(
        device=torch.device("cuda"), max_bs=BS, max_num_tokens=TOK
    )
    reg.register_slot(GraphSlot("seq_lens", lambda b, t: (b,), torch.int64, axis="bs"))
    aliased = share_input_buffer(
        "l42_test_alias", torch.zeros(BS, dtype=torch.int64, device="cuda")
    )
    reg.register_slot(
        GraphSlot("req_pool_indices", lambda b, t: (b,), torch.int64, axis="bs"),
        bind=aliased,
    )
    src = torch.tensor([4, 5], dtype=torch.int64, device="cuda")
    reg.get_slot("seq_lens").buffer.copy_(src)
    vp._check_staged_slots(registry=reg, sources={"seq_lens": src})
    with pytest.raises(RuntimeError, match="aliases"):
        vp._check_staged_slots(registry=reg, sources={"req_pool_indices": src})
    reg.get_slot("seq_lens").buffer.fill_(-1)
    with pytest.raises(RuntimeError, match="differs"):
        vp._check_staged_slots(registry=reg, sources={"seq_lens": src})
