import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.overlap_utils import resolve_forward_inputs  # noqa: E402
from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.sampling.penaltylib.frequency_penalty import (  # noqa: E402
    BatchedFrequencyPenalizer,
)
from sglang.srt.sampling.penaltylib.orchestrator import (  # noqa: E402
    BatchedPenalizerOrchestrator,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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


VOCAB_SIZE = 16
REQ_LAST_OUTPUT_TOKEN = 4
RELAYED_TOKEN = 9


def _make_penalized_decode_batch(enable_overlap):
    batch = _make_decode_batch()
    batch.enable_overlap = enable_overlap
    batch.req_pool_indices = torch.tensor([0, 1], dtype=torch.int64)
    for req in batch.reqs:
        req.origin_input_ids = [1, 2]
        req.output_ids = [REQ_LAST_OUTPUT_TOKEN]
        req.sampling_params = types.SimpleNamespace(frequency_penalty=1.0)
    orchestrator = BatchedPenalizerOrchestrator(
        vocab_size=VOCAB_SIZE, batch=batch, penalizers={BatchedFrequencyPenalizer}
    )
    batch.sampling_info = types.SimpleNamespace(penalizer_orchestrator=orchestrator)
    return batch, orchestrator.penalizers[BatchedFrequencyPenalizer]


class _PenaltyOnlySamplingInfo:
    def __init__(self, orchestrator):
        self.penalizer_orchestrator = orchestrator

    def merge_batch(self, other):
        self.penalizer_orchestrator.merge(other.penalizer_orchestrator)


def _make_penalized_prefill_batch():
    req = types.SimpleNamespace(
        origin_input_ids=[1, 2, 3],
        output_ids=[],
        sampling_params=types.SimpleNamespace(frequency_penalty=1.0),
    )
    batch = ScheduleBatch(reqs=[req])
    batch.device = "cpu"
    batch.model_config = types.SimpleNamespace(is_encoder_decoder=False)
    batch.enable_overlap = True
    batch.spec_algorithm = types.SimpleNamespace(is_none=lambda: True)
    batch.forward_mode = ForwardMode.EXTEND
    batch.req_pool_indices = torch.tensor([3], dtype=torch.int64)
    batch.req_pool_indices_cpu = torch.tensor([3], dtype=torch.int64)
    batch.seq_lens = torch.tensor([3], dtype=torch.int64)
    batch.seq_lens_cpu = torch.tensor([3], dtype=torch.int64)
    batch.orig_seq_lens = torch.tensor([3], dtype=torch.int32)
    batch.out_cache_loc = torch.tensor([0, 1, 2], dtype=torch.int64)
    batch.prefix_lens = [0]
    batch.extend_lens = [3]
    batch.extend_num_tokens = 3
    batch.extend_logprob_start_lens = [0]
    batch.prefill_input_ids_cpu = torch.tensor([1, 2, 3], dtype=torch.int64)
    orchestrator = BatchedPenalizerOrchestrator(
        vocab_size=VOCAB_SIZE, batch=batch, penalizers={BatchedFrequencyPenalizer}
    )
    batch.sampling_info = _PenaltyOnlySamplingInfo(orchestrator)
    return batch


class TestPrepareForDecodePenaltyHistory(unittest.TestCase):
    def _prepare_for_decode(self, batch):
        override = get_context().override_server_args(
            mamba_radix_cache_strategy="no_buffer"
        )
        override.install()
        self.addCleanup(override.restore)
        with patch(
            "sglang.srt.managers.schedule_batch.alloc_for_decode",
            return_value=torch.tensor([6, 7], dtype=torch.int64),
        ):
            batch.prepare_for_decode()

    def test_overlap_penalizes_relayed_token(self):
        """Under overlap, penalties must count the token the future map relays to
        this forward, not the one-step-stale req.output_ids[-1]."""
        batch, penalizer = _make_penalized_decode_batch(enable_overlap=True)
        self._prepare_for_decode(batch)
        future_map = types.SimpleNamespace(
            output_tokens_buf=torch.full((2,), RELAYED_TOKEN, dtype=torch.int64),
            spec_algo=types.SimpleNamespace(is_none=lambda: True),
        )
        resolve_forward_inputs(batch, future_map)

        counts = penalizer.cumulated_frequency_penalties
        self.assertEqual(counts[:, RELAYED_TOKEN].tolist(), [1.0, 1.0])
        self.assertEqual(counts[:, REQ_LAST_OUTPUT_TOKEN].tolist(), [0.0, 0.0])

    def test_overlap_mixed_chunk_penalizes_relayed_token_on_decode_rows(self):
        """In a mixed step the running rows count the relayed token and the
        prefill rows, which have no output yet, count nothing."""
        running, _ = _make_penalized_decode_batch(enable_overlap=True)
        running.req_pool_indices_cpu = running.req_pool_indices.clone()
        self._prepare_for_decode(running)
        for req in running.reqs:
            req._refresh_fill_ids = lambda: None
            req.set_extend_range = lambda start, end: None
        prefill = _make_penalized_prefill_batch()

        prefill.mix_with_running(running)
        future_map = types.SimpleNamespace(
            output_tokens_buf=torch.full((4,), RELAYED_TOKEN, dtype=torch.int64),
            spec_algo=types.SimpleNamespace(is_none=lambda: True),
        )
        resolve_forward_inputs(prefill, future_map)

        penalizer = prefill.sampling_info.penalizer_orchestrator.penalizers[
            BatchedFrequencyPenalizer
        ]
        counts = penalizer.cumulated_frequency_penalties
        self.assertEqual(counts[:, RELAYED_TOKEN].tolist(), [0.0, 1.0, 1.0])
        self.assertEqual(counts.sum(dim=1).tolist(), [0.0, 1.0, 1.0])

    def test_non_overlap_penalizes_req_last_output_token(self):
        """Without overlap resolve_forward_inputs skips its gather branch, so
        prepare_for_decode must count the current req.output_ids[-1] itself."""
        batch, penalizer = _make_penalized_decode_batch(enable_overlap=False)
        self._prepare_for_decode(batch)

        counts = penalizer.cumulated_frequency_penalties
        self.assertEqual(counts[:, REQ_LAST_OUTPUT_TOKEN].tolist(), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
