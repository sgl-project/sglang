"""A late overlap readback must see this step's folded accept outputs, not the
next replay's. The interleaving is forced: a spin kernel on copy_stream stands in
for a busy copy engine, so the readback always runs after the second rewrite on
forward_stream. Views must show the hazard, the snapshot must read step N."""

import unittest

import torch

from sglang.srt.speculative.dspark_components.dspark_verify import (
    DsparkVerifyEpilogue,
)
from sglang.srt.utils.common import async_d2h
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

BS, STRIDE = 4, 5
FIELDS = (
    "correct_len",
    "bonus",
    "cap_trim_lens",
    "commit_lens",
    "new_seq_lens",
    "out_tokens",
)
# Long enough that the second rewrite on forward_stream always lands first.
SPIN_CYCLES = 500_000_000


class _StaticTpSync:
    def sync(self, site, values):
        return values


def _epilogue(device) -> DsparkVerifyEpilogue:
    return DsparkVerifyEpilogue(
        max_bs=BS,
        verify_num_draft_tokens=STRIDE,
        device=device,
        tp_sync=_StaticTpSync(),
    )


def _rewrite(epilogue: DsparkVerifyEpilogue, step: int) -> None:
    """What a target-verify replay does to every accept buffer."""
    for field in FIELDS:
        getattr(epilogue, f"{field}_buf").fill_(step)


def _readback_after_next_replay(handover: str) -> dict:
    """forward_stream: replay N, (later) replay N+1; copy_stream: delayed readback."""
    forward_stream = torch.cuda.Stream()
    copy_stream = torch.cuda.Stream()
    epilogue = _epilogue("cuda")
    with torch.cuda.stream(forward_stream):
        _rewrite(epilogue, 1)
        if handover == "view":
            outs = epilogue.read_accept(BS)
        else:
            outs = epilogue.snapshot_accept(BS)
    # Scheduler._copy_result_to_cpu: the readback waits only for forward N.
    copy_stream.wait_stream(forward_stream)
    with torch.cuda.stream(copy_stream):
        torch.cuda._sleep(SPIN_CYCLES)  # the copy engine is busy
        host = {field: async_d2h(getattr(outs, field)) for field in FIELDS}
    # The overlap loop launches forward N+1 without waiting for the copy.
    with torch.cuda.stream(forward_stream):
        _rewrite(epilogue, 2)
    torch.cuda.synchronize()
    return {field: sorted(set(host[field].flatten().tolist())) for field in FIELDS}


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class TestDsparkOverlapReadbackStreams(CustomTestCase):
    def test_views_read_the_next_replay(self):
        # The hazard itself; if this stops reproducing, the test no longer
        # exercises the interleaving the snapshot guards against.
        seen = _readback_after_next_replay("view")
        for field in FIELDS:
            self.assertEqual(seen[field], [2], field)

    def test_snapshot_reads_this_step(self):
        seen = _readback_after_next_replay("snapshot")
        for field in FIELDS:
            self.assertEqual(seen[field], [1], field)


if __name__ == "__main__":
    unittest.main(verbosity=2)
