# SPDX-License-Identifier: Apache-2.0
"""The step CUDA graph's fixed-shape simulated acceptance equals the regular one.

``simulated_accept_index_from_device_length`` reads an already drawn acceptance
length from a device tensor so that a captured step does not freeze it. For
every length, both token modes and several batch sizes it must produce exactly
what ``spec_utils.generate_simulated_accept_index`` produces for that length:
the accept index (whose shape never depends on the length),
``num_correct_drafts`` and every element of ``predict``.
"""

import unittest

import torch

from sglang.srt.speculative.eagle_step_cuda_graph_runner import (
    simulated_accept_index_from_device_length,
)
from sglang.srt.speculative.spec_utils import generate_simulated_accept_index
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

SPEC_STEPS = 3
NUM_DRAFT_TOKENS = SPEC_STEPS + 1


def _inputs(bs: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    width = NUM_DRAFT_TOKENS
    accept_index = torch.full((bs, width), -1, dtype=torch.int32)
    accept_index[:, 0] = torch.arange(bs, dtype=torch.int32) * width
    num_accepted = torch.randint(0, width, (bs,), generator=gen)
    for row in range(bs):
        for col in range(1, int(num_accepted[row]) + 1):
            accept_index[row, col] = row * width + col
    predict = torch.randint(0, 1000, (bs * width,), dtype=torch.int32, generator=gen)
    num_correct_drafts = num_accepted.to(torch.int32)
    candidates = torch.randint(0, 1000, (bs, width), dtype=torch.int64, generator=gen)
    target_predict = torch.randint(
        0, 1000, (bs, width), dtype=torch.int64, generator=gen
    )
    return accept_index, predict, num_correct_drafts, candidates, target_predict


class TestSimulatedAcceptFixedShape(CustomTestCase):
    def test_matches_regular_path_for_every_length(self):
        for mode in ("real-draft-token", "fixed"):
            for bs in (1, 2, 3, 4):
                for length in range(1, SPEC_STEPS + 2):
                    with self.subTest(mode=mode, bs=bs, length=length):
                        ref = _inputs(bs, seed=bs * 31 + length)
                        got = _inputs(bs, seed=bs * 31 + length)
                        # match-expected with an integral mean always draws it.
                        ref_index = generate_simulated_accept_index(
                            accept_index=ref[0],
                            predict=ref[1],
                            num_correct_drafts=ref[2],
                            candidates=ref[3],
                            target_predict=ref[4],
                            bs=bs,
                            spec_steps=SPEC_STEPS,
                            simulate_acc_len=float(length),
                            simulate_acc_method="match-expected",
                            simulate_acc_token_mode=mode,
                        )
                        got_index = simulated_accept_index_from_device_length(
                            *got,
                            SPEC_STEPS,
                            torch.tensor([length], dtype=torch.int32),
                            mode,
                        )
                        self.assertEqual(got_index.dtype, ref_index.dtype)
                        self.assertTrue(torch.equal(got_index, ref_index))
                        self.assertTrue(torch.equal(got[1], ref[1]))
                        self.assertTrue(torch.equal(got[2], ref[2]))


if __name__ == "__main__":
    unittest.main()
