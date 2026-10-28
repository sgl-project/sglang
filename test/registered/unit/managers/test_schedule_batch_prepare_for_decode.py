import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_req(
    req_pool_idx, kv_allocated_len, num_image_tokens=None, kv_committed_len=None
):
    return types.SimpleNamespace(
        kv=types.SimpleNamespace(
            req_pool_idx=req_pool_idx,
            kv_committed_len=(
                kv_allocated_len if kv_committed_len is None else kv_committed_len
            ),
            kv_allocated_len=kv_allocated_len,
        ),
        multimodal_inputs=(
            types.SimpleNamespace(num_image_tokens=num_image_tokens)
            if num_image_tokens is not None
            else None
        ),
        beam_group=None,
    )


def _make_step(reqs, is_encoder_decoder=False, speculative=False):
    step = ScheduleBatch(reqs=reqs)
    step.device = "cpu"
    step.model_config = types.SimpleNamespace(
        is_encoder_decoder=is_encoder_decoder, vocab_size=8
    )
    step.enable_overlap = False
    step.spec_algorithm = types.SimpleNamespace(is_none=lambda: not speculative)
    return step


class TestPrepareForDecodeFromRequests(unittest.TestCase):
    def setUp(self):
        # The mamba-extra-buffer predicate reads the published bags, so the
        # fixture publishes a config with the strategy off.
        override = get_context().override_server_args(
            mamba_radix_cache_strategy="no_buffer"
        )
        override.install()
        self.addCleanup(override.restore)

    def _prepare(self, step):
        with (
            patch(
                "sglang.srt.managers.schedule_batch.alloc_for_decode",
                return_value=torch.tensor([6, 7], dtype=torch.int64),
            ) as alloc,
        ):
            step.prepare_for_decode()
        return alloc

    def test_lengths_come_from_the_allocated_kv(self):
        """A decode step's row length is the request's allocated KV length plus
        this step's token, whatever the requests' output lists hold."""
        step = _make_step([_make_req(4, 3), _make_req(9, 5)])
        self._prepare(step)

        expected = torch.tensor([4, 6], dtype=torch.int64)
        self.assertTrue(torch.equal(step.seq_lens, expected))
        self.assertTrue(torch.equal(step.seq_lens_cpu, expected))
        self.assertTrue(torch.equal(step.orig_seq_lens, expected.to(torch.int32)))
        self.assertEqual(step.req_pool_indices_cpu.tolist(), [4, 9])

    def test_speculative_rows_take_their_committed_lengths(self):
        """A speculative row allocates ahead of its committed KV; the step's
        length is the committed one, and spec_prepare_for_decode allocates."""
        step = _make_step(
            [_make_req(4, 9, kv_committed_len=3), _make_req(9, 12, kv_committed_len=5)],
            speculative=True,
        )
        alloc = self._prepare(step)

        self.assertEqual(step.seq_lens_cpu.tolist(), [3, 5])
        self.assertIsNone(step.out_cache_loc)
        alloc.assert_not_called()

    def test_encoder_decoder_lengths_exclude_the_encoder_tokens(self):
        step = _make_step(
            [_make_req(0, 10, num_image_tokens=4), _make_req(1, 5)],
            is_encoder_decoder=True,
        )
        self._prepare(step)

        self.assertEqual(step.encoder_lens_cpu, [4, 0])
        self.assertEqual(step.seq_lens_cpu.tolist(), [7, 6])


if __name__ == "__main__":
    unittest.main()
