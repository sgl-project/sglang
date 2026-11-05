"""SGLANG_ENABLE_METADATA_GLUE_GRAPH stays off where the per-replay metadata prep
computes values on the host: DFlash-family verify plans, and an attention backend
whose prep reads seq_lens_cpu / seq_lens_sum (FlashInfer decode plans every step
from them). The glue graph replays only the prep's device ops, so those values would
replay frozen at their capture-time lengths and decode attention would silently use
stale plans."""

import unittest
from types import SimpleNamespace

from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    metadata_glue_off_reason,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _spec(dflash=False):
    return SimpleNamespace(is_dflash_family=lambda: dflash)


class TestMetadataGlueOffReason(CustomTestCase):
    def test_a_host_free_prep_may_be_glued(self):
        backend = SimpleNamespace(needs_cpu_seq_lens=False)
        self.assertEqual(metadata_glue_off_reason(_spec(), backend), "")

    def test_flashinfer_decode_prep_reads_host_lengths(self):
        backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
        self.assertTrue(backend.needs_cpu_seq_lens)
        why = metadata_glue_off_reason(_spec(), backend)
        self.assertIn("FlashInferAttnBackend", why)
        self.assertIn("seq_lens_cpu", why)

    def test_dflash_family_stays_off(self):
        backend = SimpleNamespace(needs_cpu_seq_lens=False)
        self.assertIn("DFlash", metadata_glue_off_reason(_spec(dflash=True), backend))


if __name__ == "__main__":
    unittest.main()
