import unittest
from types import SimpleNamespace

import torch

from sglang.srt.beam_search.batch_tail import BeamTail, BeamTailEntry
from sglang.srt.beam_search.logits_capture import (
    capture_pre_sample_logits,
    is_beam_only_batch,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req(beam: bool):
    return SimpleNamespace(beam_group=SimpleNamespace() if beam else None)


def _decode_batch(reqs, members=2):
    entries, start = [], 0
    for i, req in enumerate(reqs):
        if req.beam_group is not None:
            entries.append(BeamTailEntry(req.beam_group, i, start, start + members))
            start += members
    tail = BeamTail(num_base_rows=len(reqs), entries=entries) if entries else None
    batch = SimpleNamespace(beam_tail=tail, forward_mode=ForwardMode.DECODE, reqs=reqs)
    return batch, start


def _extend_batch(reqs, mode=ForwardMode.EXTEND):
    return SimpleNamespace(beam_tail=None, forward_mode=mode, reqs=reqs)


class TestBeamOnlyBatch(CustomTestCase):
    def test_any_regular_row_keeps_the_sampler(self):
        """Only a batch whose every reqs-aligned row is a beam leader may skip the
        sampler; a regular request in it still needs its sampled token."""
        beam, regular = _req(True), _req(False)
        self.assertTrue(is_beam_only_batch(_decode_batch([beam, _req(True)])[0]))
        self.assertFalse(is_beam_only_batch(_decode_batch([regular, beam])[0]))
        self.assertFalse(is_beam_only_batch(_decode_batch([regular])[0]))
        self.assertTrue(is_beam_only_batch(_extend_batch([beam])))
        self.assertFalse(is_beam_only_batch(_extend_batch([beam, regular])))
        self.assertFalse(is_beam_only_batch(_extend_batch([beam], ForwardMode.MIXED)))

    def test_mixed_decode_capture_survives_in_place_sampling(self):
        """The sampler rewrites the reqs-aligned logits in place, so a batch that
        still samples must hand the coordinator a copy of its leader rows."""
        batch, num_tail = _decode_batch([_req(False), _req(True)])
        logits = torch.randn((2 + num_tail, 4))
        leader_row = logits[1].clone()
        output = LogitsProcessorOutput(next_token_logits=logits)
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE, positions=torch.zeros(2 + num_tail)
        )
        capture_pre_sample_logits(batch, forward_batch, output)
        output.next_token_logits.mul_(0)
        self.assertTrue(torch.equal(output.beam.leader_logits[0], leader_row))


if __name__ == "__main__":
    unittest.main()
