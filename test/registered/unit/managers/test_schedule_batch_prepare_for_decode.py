import types
import unittest
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_req():
    return types.SimpleNamespace(
        decode_batch_idx=0,
        kv_committed_len=3,
        kv_allocated_len=3,
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


class TestPrepareForDecodeSeqLensOwnership(unittest.TestCase):
    def test_decode_seq_lens_bump_is_out_of_place(self):
        """Each prepare_for_decode call rebinds seq-lens tensors to new +1 objects without mutating the old ones."""
        batch = _make_decode_batch()

        server_args = types.SimpleNamespace(
            enable_mamba_extra_buffer=lambda: False,
        )
        with (
            patch(
                "sglang.srt.managers.schedule_batch.alloc_for_decode",
                return_value=torch.tensor([6, 7], dtype=torch.int64),
            ),
            patch(
                "sglang.srt.managers.schedule_batch.get_global_server_args",
                return_value=server_args,
            ),
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


if __name__ == "__main__":
    unittest.main()
