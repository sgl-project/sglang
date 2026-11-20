import types
import unittest
from array import array

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import (  # noqa: E402
    ScheduleBatch,
    compute_extend_logprob_start_len,
)
from sglang.srt.model_executor.forward_mode import ForwardMode  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ENCODER_LEN = 4
EXTEND_LEN = 6  # 4 encoder (image) tokens + 2 decoder text tokens


def _make_req(logprob_start_len):
    return types.SimpleNamespace(
        multimodal_inputs=types.SimpleNamespace(num_image_tokens=ENCODER_LEN),
        prefix_len=0,
        extend_len=EXTEND_LEN,
        logprob_start_len=logprob_start_len,
    )


def _make_extend_batch(reqs):
    batch = ScheduleBatch(reqs=reqs)
    batch.device = "cpu"
    batch.forward_mode = ForwardMode.EXTEND
    batch.prefix_lens = [0] * len(reqs)
    batch.extend_lens = [EXTEND_LEN] * len(reqs)
    batch.extend_num_tokens = EXTEND_LEN * len(reqs)
    batch.out_cache_loc = torch.arange(EXTEND_LEN * len(reqs), dtype=torch.int64)
    return batch


def _run_prepare(batch):
    input_ids = [array("q", range(EXTEND_LEN)) for _ in batch.reqs]
    seq_lens = [EXTEND_LEN for _ in batch.reqs]
    batch.prepare_encoder_info_extend(input_ids, seq_lens)


class TestEncoderLogprobStartSentinel(CustomTestCase):
    def test_minus_one_skips_input_logprobs_on_later_chunk(self):
        """A later chunk must not collect input logprobs for the -1 sentinel."""
        req = _make_req(logprob_start_len=-1)
        batch = _make_extend_batch([req])
        _run_prepare(batch)

        # Chunk 1 is complete: the next extend begins at prefix 6 in a
        # 10-token prompt, with the remaining 4 tokens in this chunk.
        next_chunk_start = compute_extend_logprob_start_len(
            logprob_start_len=req.logprob_start_len,
            prefix_len=6,
            extend_len=4,
            full_untruncated_fill_len=10,
        )
        self.assertEqual(4 - next_chunk_start, 0)

    def test_explicit_start_len_is_still_clamped_to_encoder_len(self):
        req = _make_req(logprob_start_len=2)
        batch = _make_extend_batch([req])
        _run_prepare(batch)
        self.assertEqual(req.logprob_start_len, ENCODER_LEN)


if __name__ == "__main__":
    unittest.main()
