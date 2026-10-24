import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention.graph_variants import (
    DSV41_CANDIDATE_FILTERED,
    Dsv41CandidateGraphVariants,
)
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dspark_components.dspark_verify import (
    TargetVerifyExecutor,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_W = 6
_SPAN = 64
_COMMITTED = [_SPAN - _W, 10]
_VERIFY_LENS = [_W, 3]


def _verify(run, *, self_adds: bool):
    """Returns (lengths, sum, live lengths) at preparation and (lengths, sum) after."""
    executor = TargetVerifyExecutor.__new__(TargetVerifyExecutor)
    executor.verify_num_draft_tokens = _W
    executor._verify_backend_self_adds_seq_lens_cache = self_adds
    executor._target_is_dsv41 = False
    executor.target_worker = SimpleNamespace(
        forward_batch_generation=lambda **_: SimpleNamespace(
            logits_output=None, can_run_cuda_graph=True
        )
    )
    batch = SimpleNamespace(
        seq_lens_cpu=torch.tensor(_COMMITTED, dtype=torch.int64),
        seq_lens_sum=sum(_COMMITTED),
        out_cache_loc=None,
    )
    prepared = []

    def prepare_for_verify(verify_input, batch, worker):
        prepared.append(
            (
                batch.seq_lens_cpu.tolist(),
                batch.seq_lens_sum,
                verify_input.live_seq_lens_cpu.tolist(),
            )
        )
        return None, None

    with mock.patch.object(DFlashVerifyInput, "prepare_for_verify", prepare_for_verify):
        run(executor, batch)
    return prepared[0], (batch.seq_lens_cpu.tolist(), batch.seq_lens_sum)


def _run_static(executor, batch):
    bs = len(_COMMITTED)
    executor.run_non_compact(
        batch=batch,
        draft_input=SimpleNamespace(nxt_kv_lens_cpu=None),
        verify_ids_2d=torch.zeros((bs, _W), dtype=torch.int64),
        verify_window=SimpleNamespace(
            positions_2d=torch.zeros((bs, _W), dtype=torch.int64),
            verify_cache_loc=torch.zeros(bs * _W, dtype=torch.int64),
            kv_loc_plan=None,
        ),
        sampling_info=None,
    )


def _run_compact(executor, batch):
    num_tokens = sum(_VERIFY_LENS)
    executor._run_ragged(
        batch=batch,
        layout=RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=_VERIFY_LENS, device="cpu", grid=[6, 12]
        ),
        ragged_window=SimpleNamespace(
            verify_ids=torch.zeros(num_tokens, dtype=torch.int64),
            positions=torch.zeros(num_tokens, dtype=torch.int64),
            verify_cache_loc=torch.zeros(num_tokens, dtype=torch.int64),
            window_index=None,
        ),
        sampling_info=None,
        kv_loc_plan=None,
    )


class TestCompactVerifyCommittedSeqLens(CustomTestCase):
    def test_compact_selects_the_same_candidate_graph_as_static(self):
        variants = Dsv41CandidateGraphVariants(
            graph_limits=(("candidate_unfiltered", _SPAN),),
            capture_labels=("candidate_unfiltered", DSV41_CANDIDATE_FILTERED),
            verify_extra_tokens=_W,
        )
        committed = (_COMMITTED, sum(_COMMITTED))
        for run in (_run_static, _run_compact):
            prepared, after = _verify(run, self_adds=True)
            self.assertEqual(prepared, (*committed, _COMMITTED))
            self.assertEqual(after, committed)

        lens = torch.tensor(_COMMITTED)
        self.assertEqual(
            variants.select(SimpleNamespace(seq_lens_cpu=lens)),
            "candidate_unfiltered",
        )
        self.assertEqual(
            variants.select(SimpleNamespace(seq_lens_cpu=lens + 1)),
            DSV41_CANDIDATE_FILTERED,
        )

    def test_compact_extends_lengths_for_other_backends(self):
        prepared, after = _verify(_run_compact, self_adds=False)
        extended = [c + v for c, v in zip(_COMMITTED, _VERIFY_LENS)]
        self.assertEqual(prepared, (extended, sum(extended), _COMMITTED))
        self.assertEqual(after, (_COMMITTED, sum(_COMMITTED)))


if __name__ == "__main__":
    unittest.main()
