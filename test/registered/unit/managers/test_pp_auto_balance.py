import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.scheduler_pp_mixin import _plan_decode_transfers
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestPPAutoBalance(unittest.TestCase):
    def test_pp4_long_tail(self):
        self.assertEqual(_plan_decode_transfers([8, 0, 0, 0], 0, 8), [0, 2, 2, 2])
        self.assertEqual(_plan_decode_transfers([2, 1, 2, 1], 0, 8), [0] * 4)
        self.assertEqual(_plan_decode_transfers([8, 3, 3, 3], 0, 3), [0] * 4)

    def test_one_rotation_balances_all_live_rows(self):
        for initial in product(range(6), repeat=4):
            sizes = list(initial)
            for source in range(4):
                moves = _plan_decode_transfers(sizes, source, 5)
                self.assertEqual(moves[source], 0)
                sizes[source] -= sum(moves)
                sizes = [size + count for size, count in zip(sizes, moves)]
                self.assertGreaterEqual(min(sizes), 0)
                self.assertLessEqual(max(sizes), 5)
            self.assertEqual(sum(sizes), sum(initial))
            self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_minimum_split_size(self):
        self.assertEqual(_plan_decode_transfers([4, 0, 0, 0], 0, 4), [0, 1, 1, 1])
        self.assertEqual(_plan_decode_transfers([4, 0, 0, 0], 0, 4, 2), [0, 2, 0, 0])
        self.assertEqual(_plan_decode_transfers([3, 0, 0, 0], 0, 4, 2), [0] * 4)
        self.assertEqual(_plan_decode_transfers([3, 0, 1, 0], 0, 4, 2), [0, 0, 1, 0])
        self.assertEqual(_plan_decode_transfers([4, 0, 0, 0], 0, 1, 2), [0] * 4)

    def test_minimum_size_config(self):
        from sglang.srt.server_args import ServerArgs

        self.assertEqual(ServerArgs(model_path="dummy").pp_min_micro_batch_size, 1)
        for minimum, maximum in ((0, 4), (3, 2)):
            with self.subTest(minimum=minimum, maximum=maximum):
                args = ServerArgs(
                    model_path="dummy",
                    pp_size=4,
                    enable_pp_auto_balance=True,
                    pp_min_micro_batch_size=minimum,
                    pp_max_micro_batch_size=maximum,
                )
                with self.assertRaisesRegex(ValueError, "--pp-min-micro-batch-size"):
                    args.check_server_args()

    def test_handoff_preserves_rows_and_sampling_history(self):
        import torch

        from sglang.test.test_utils import maybe_stub_sgl_kernel

        maybe_stub_sgl_kernel()
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
        from sglang.srt.model_executor.forward_mode import (
            CaptureHiddenMode,
            ForwardMode,
        )
        from sglang.srt.sampling.penaltylib.frequency_penalty import (
            BatchedFrequencyPenalizer,
        )
        from sglang.srt.sampling.penaltylib.orchestrator import (
            BatchedPenalizerOrchestrator,
        )
        from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

        reqs = [
            SimpleNamespace(
                grammar=None,
                return_logprob=False,
                return_hidden_states_mode=CaptureHiddenMode.NULL,
                sampling_params=SimpleNamespace(top_k=1, frequency_penalty=1.0),
            )
            for _ in range(8)
        ]
        initial = torch.arange(8)
        donor = ScheduleBatch(
            reqs=reqs,
            device="cpu",
            model_config=SimpleNamespace(is_encoder_decoder=False),
            req_pool_indices=initial + 20,
            req_pool_indices_cpu=initial + 20,
            seq_lens=initial + 10,
            orig_seq_lens=initial + 10,
            input_ids=initial,
            forward_mode=ForwardMode.DECODE,
        )
        penalties = BatchedPenalizerOrchestrator(8, donor, {BatchedFrequencyPenalizer})
        donor.sampling_info = SamplingBatchInfo(
            temperatures=torch.ones(8, 1),
            top_ps=torch.ones(8),
            top_ks=torch.ones(8),
            min_ps=torch.zeros(8),
            is_all_greedy=True,
            is_any_greedy=True,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            vocab_size=8,
            device="cpu",
            penalizer_orchestrator=penalties,
        )
        penalties.cumulate_output_tokens(initial)
        scheduler = SchedulerPPMixin()
        scheduler.running_mbs = [donor] + [ScheduleBatch(reqs=[]) for _ in range(3)]
        # Slot 1 has an outstanding prefill. Reserve its rows and keep its
        # membership stable until its result arrives.
        prefill = ScheduleBatch(reqs=reqs[:2], forward_mode=ForwardMode.EXTEND)
        scheduler.mbs = [donor, prefill, None, None]
        scheduler._pp_incoming = [None] * 4
        with patch(
            "sglang.srt.managers.scheduler_pp_mixin.get_parallel",
            return_value=SimpleNamespace(
                pp_max_micro_batch_size=8, pp_min_micro_batch_size=3
            ),
        ):
            scheduler._pp_balance_decode(0, donor)
            self.assertEqual(len(donor.reqs), 4)
            self.assertIsNone(scheduler._pp_incoming[3])
            self.assertEqual(prefill.reqs, reqs[:2])
            self.assertTrue(all(b.is_empty() for b in scheduler.running_mbs[1:]))
            batches = [donor, *(b for b in scheduler._pp_incoming if b is not None)]
            observed = []
            for batch in batches:
                rows = batch.input_ids
                observed.extend(rows.tolist())
                self.assertEqual(batch.req_pool_indices.tolist(), (rows + 20).tolist())
                self.assertEqual(batch.seq_lens.tolist(), (rows + 10).tolist())
                owner = batch.sampling_info.penalizer_orchestrator
                self.assertIs(owner.batch, batch)
                penalty = owner.penalizers[BatchedFrequencyPenalizer]
                self.assertIs(penalty.orchestrator, owner)
                torch.testing.assert_close(
                    penalty.cumulated_frequency_penalties, torch.eye(8)[rows]
                )
            self.assertEqual(sorted(observed), list(range(8)))
            pending = scheduler._pp_incoming[2]
            self.assertIs(
                scheduler._pp_balance_decode(2, scheduler.running_mbs[2]), pending
            )
            self.assertIsNone(scheduler._pp_incoming[2])


if __name__ == "__main__":
    unittest.main()
