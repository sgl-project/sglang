import types
import unittest
from array import array
from unittest.mock import patch

import torch

from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from sglang.srt.sampling.penalties import (  # noqa: E402
    BatchedFrequencyPenalizer,
    BatchedMinNewTokensPenalizer,
    BatchedPenalizerOrchestrator,
)

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_req():
    return types.SimpleNamespace(
        decode_batch_idx=0,
        kv=types.SimpleNamespace(kv_committed_len=3, kv_allocated_len=3),
        beam_group=None,
    )


def _make_decode_batch():
    batch = ScheduleBatch(reqs=[_make_req(), _make_req()])
    batch.device = "cpu"
    batch.model_config = types.SimpleNamespace(is_encoder_decoder=False)
    batch.enable_overlap = False
    batch.spec_algorithm = types.SimpleNamespace(is_none=lambda: True)
    batch.sampling_info = types.SimpleNamespace(
        penalizer_orchestrator=types.SimpleNamespace(is_required=False)
    )
    batch.hisparse_coordinator = None
    batch.seq_lens = torch.tensor([3, 5], dtype=torch.int64)
    batch.seq_lens_cpu = torch.tensor([3, 5], dtype=torch.int64)
    batch.orig_seq_lens = torch.tensor([3, 5], dtype=torch.int32)
    return batch


class TestPrepareForDecodeMambaInit(CustomTestCase):
    def test_spec_decode_drops_extend_mamba_init_metadata(self):
        batch = ScheduleBatch(reqs=[])
        batch.spec_algorithm = types.SimpleNamespace(is_none=lambda: False)
        batch.sampling_info = types.SimpleNamespace(
            penalizer_orchestrator=types.SimpleNamespace(is_required=False)
        )
        batch.mamba_cow_src_indices = torch.tensor([1])
        batch.mamba_cow_dst_indices = torch.tensor([2])
        batch.mamba_clear_indices = torch.tensor([3])

        with patch(
            "sglang.srt.speculative.spec_utils.spec_prepare_for_decode"
        ) as prepare:
            batch.prepare_for_decode()

        prepare.assert_called_once_with(batch)
        self.assertIsNone(batch.mamba_cow_src_indices)
        self.assertIsNone(batch.mamba_cow_dst_indices)
        self.assertIsNone(batch.mamba_clear_indices)


class TestPrepareForDecodeSeqLensOwnership(unittest.TestCase):
    def test_decode_seq_lens_bump_is_out_of_place(self):
        """Each prepare_for_decode call rebinds seq-lens tensors to new +1 objects without mutating the old ones."""
        batch = _make_decode_batch()

        # The mamba-extra-buffer predicate reads the published bags, so the
        # fixture publishes a config with the strategy off.
        override = get_context().override_server_args(
            mamba_radix_cache_strategy="no_buffer"
        )
        override.install()
        self.addCleanup(override.restore)
        with patch(
            "sglang.srt.managers.schedule_batch.alloc_for_decode",
            return_value=torch.tensor([6, 7], dtype=torch.int64),
        ):
            for step in range(1, 3):
                prev_seq_lens = batch.seq_lens
                prev_seq_lens_cpu = batch.seq_lens_cpu
                prev_orig_seq_lens = batch.orig_seq_lens
                prev_values = (
                    prev_seq_lens.clone(),
                    prev_seq_lens_cpu.clone(),
                    prev_orig_seq_lens.clone(),
                )

                batch.prepare_for_decode()

                self.assertIsNot(batch.seq_lens, prev_seq_lens)
                self.assertIsNot(batch.seq_lens_cpu, prev_seq_lens_cpu)
                self.assertIsNot(batch.orig_seq_lens, prev_orig_seq_lens)
                expected = torch.tensor([3 + step, 5 + step], dtype=torch.int64)
                self.assertTrue(torch.equal(batch.seq_lens, expected))
                self.assertTrue(torch.equal(batch.seq_lens_cpu, expected))
                self.assertTrue(
                    torch.equal(batch.orig_seq_lens, expected.to(torch.int32))
                )
                self.assertTrue(torch.equal(prev_seq_lens, prev_values[0]))
                self.assertTrue(torch.equal(prev_seq_lens_cpu, prev_values[1]))
                self.assertTrue(torch.equal(prev_orig_seq_lens, prev_values[2]))


_VOCAB_SIZE = 16


def _make_penalty_batch(initial_output_ids):
    reqs = [
        types.SimpleNamespace(
            sampling_params=types.SimpleNamespace(
                frequency_penalty=1.0, min_new_tokens=100, stop_token_ids=None
            ),
            eos_token_ids=None,
            tokenizer=types.SimpleNamespace(
                additional_stop_token_ids=None, eos_token_id=0
            ),
            output_ids=array("q", output_ids),
            penalty_cumulated_len=0,
        )
        for output_ids in initial_output_ids
    ]
    batch = ScheduleBatch(reqs=reqs)
    batch.device = "cpu"
    orchestrator = BatchedPenalizerOrchestrator(
        _VOCAB_SIZE, batch, {BatchedFrequencyPenalizer, BatchedMinNewTokensPenalizer}
    )
    batch.sampling_info = types.SimpleNamespace(penalizer_orchestrator=orchestrator)
    return batch


class TestCumulatePenaltyOutputTokens(CustomTestCase):
    def test_feeds_every_committed_token_once(self):
        """Steps of one or several tokens per request (and none, as on the first
        decode under overlap) each reach the penalizers exactly once."""
        batch = _make_penalty_batch([[7], []])
        steps = [([7, 7, 3], [4]), ([1], [4]), ([2, 7, 7], [9]), ([5], [9, 4, 4, 4])]
        for committed in steps:
            batch.cumulate_penalty_output_tokens()
            for req, tokens in zip(batch.reqs, committed):
                req.output_ids.extend(tokens)
        batch.cumulate_penalty_output_tokens()

        penalizers = batch.sampling_info.penalizer_orchestrator.penalizers
        for i, req in enumerate(batch.reqs):
            tokens = torch.tensor(req.output_ids, dtype=torch.int64)
            torch.testing.assert_close(
                penalizers[BatchedFrequencyPenalizer].cumulated_frequency_penalties[i],
                torch.bincount(tokens, minlength=_VOCAB_SIZE).float(),
            )
            self.assertEqual(
                penalizers[BatchedMinNewTokensPenalizer].len_output_tokens[i].item(),
                len(tokens),
            )


if __name__ == "__main__":
    unittest.main()
