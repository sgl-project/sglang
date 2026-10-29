"""SGLANG_OPT_DSV41_DSPARK_SIM_DRAW_AHEAD: the simulated acceptance draw and the sim
bonus seed move before the draft launch. They read only the host RNG, the batch size
and the request temperatures, so on fixed seeds the ahead order writes exactly what
the after-draft order wrote; with real acceptance the switch is inert."""

import random
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.dspark_components import dspark_worker_v2 as worker_mod
from sglang.srt.speculative.dspark_components.dspark_verify import (
    DsparkVerifyEpilogue,
    TargetVerifyExecutor,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

GAMMA, BS = 5, 3


def _worker():
    correct_len = torch.zeros((1,), dtype=torch.int64)
    epilogue = DsparkVerifyEpilogue(
        max_bs=BS + 1,
        verify_num_draft_tokens=GAMMA + 1,
        device="cpu",
        tp_sync=SimpleNamespace(sync=lambda site, t: None, enabled=lambda s: False),
        simulated_correct_len=correct_len,
        simulate_fold=True,
    )
    executor = TargetVerifyExecutor.__new__(TargetVerifyExecutor)
    executor.gamma = GAMMA
    executor._simulate_acc_len = 3.51
    executor._simulated_correct_len = correct_len
    executor.verify_epilogue = epilogue
    w = worker_mod.DSparkWorkerV2.__new__(worker_mod.DSparkWorkerV2)
    w._verify_executor = executor
    w.num_sim_draws_ahead = 0
    return w


def _draft_stand_in():
    # The draft's host path draws no host randomness; device work is order-free here.
    torch.ones(4, 4) @ torch.ones(4, 4)


def _snapshot(w):
    e = w._verify_executor.verify_epilogue
    return (
        int(w._verify_executor._simulated_correct_len.item()),
        int(e.sim_seed_buf.item()),
        e.sim_temperatures_buf[:BS].tolist(),
    )


def _temps(step):
    # Greedy (no bonus temperature) and sampled steps interleaved.
    return None if step % 3 == 0 else torch.tensor([0.7, 1.0, 0.3]) + step / 100


def test_ahead_matches_after_draft_on_fixed_seeds():
    steps = 40
    torch.manual_seed(1234)
    random.seed(1234)
    after, w = [], _worker()
    for step in range(steps):
        _draft_stand_in()
        w._verify_executor.begin_simulated_step()
        temps = _temps(step)
        w._verify_executor.verify_epilogue.begin_sim_bonus(BS, temps)
        after.append(_snapshot(w))

    torch.manual_seed(1234)
    random.seed(1234)
    ahead, w = [], _worker()
    for step in range(steps):
        temps = _temps(step)
        info = None if temps is None else SimpleNamespace(temperatures=temps)
        w._begin_simulated_step_ahead(bs=BS, simulate_bonus_sampling_info=info)
        _draft_stand_in()
        ahead.append(_snapshot(w))

    assert ahead == after
    assert len({s[0] for s in after}) > 1  # the draw really varies across steps
    assert w.num_sim_draws_ahead == steps


def test_inert_under_real_acceptance():
    with envs.SGLANG_OPT_DSV41_DSPARK_SIM_DRAW_AHEAD.override(True):
        assert not worker_mod.sim_draw_ahead_enabled(None)
        assert worker_mod.sim_draw_ahead_enabled(torch.zeros((1,), dtype=torch.int64))
    with envs.SGLANG_OPT_DSV41_DSPARK_SIM_DRAW_AHEAD.override(False):
        assert not worker_mod.sim_draw_ahead_enabled(
            torch.zeros((1,), dtype=torch.int64)
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
