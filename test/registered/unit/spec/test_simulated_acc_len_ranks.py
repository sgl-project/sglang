"""SGLANG_SIMULATE_ACC_LEN draws ignore the process RNG, so TP ranks draw alike."""

import random
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.speculative import dflash_worker_v2 as dflash_mod
from sglang.srt.speculative.dspark_components.dspark_verify import TargetVerifyExecutor
from sglang.srt.speculative.spec_utils import simulated_acc_len_generator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _dspark_draw(monkeypatch):
    executor = TargetVerifyExecutor.__new__(TargetVerifyExecutor)
    executor.gamma = 5
    executor._simulate_acc_len = 3.5
    executor._simulated_correct_len = torch.zeros((1,), dtype=torch.int64)
    executor._simulated_acc_len_rng = simulated_acc_len_generator(17)
    executor.verify_epilogue = None

    def draw():
        executor.begin_simulated_step()
        return executor._simulated_correct_len.item(), executor._simulated_bonus_seed

    return draw


def _dflash_draw(monkeypatch):
    monkeypatch.setattr(dflash_mod, "SIMULATE_ACC_LEN", 2.5)
    worker = SimpleNamespace(
        _selector_sample=None,
        _tp_sync=SimpleNamespace(sync=lambda site, tensor: None),
        _use_triton_accept_bonus=False,
        _simulated_acc_len_rng=simulated_acc_len_generator(17),
        block_size=4,
    )

    def draw():
        return dflash_mod.DFlashWorkerV2._accept_block(
            worker,
            candidates=torch.tensor([[9, 1, 2, 3]]),
            next_token_logits=torch.zeros(1, 4, 8),
            sampling_info=SimpleNamespace(is_all_greedy=True),
            draft_input=object(),
            prefix_lens=torch.tensor([3]),
            bs=1,
        )[1].item()

    return draw


@pytest.mark.parametrize("make_draw", [_dspark_draw, _dflash_draw])
def test_ranks_draw_alike_whatever_else_reads_the_process_rng(make_draw, monkeypatch):
    ranks = [make_draw(monkeypatch) for _ in range(2)]
    draws = [[], []]
    for step in range(40):
        for rank, draw in enumerate(ranks):
            # rank 1 consumes the process RNG more often than rank 0
            random.seed(rank)
            torch.manual_seed(rank)
            for _ in range(rank * step):
                random.random()
                torch.rand(1)
            draws[rank].append(draw())
    assert draws[0] == draws[1]
    assert len(set(draws[0])) > 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
