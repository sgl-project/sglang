import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch

register_cpu_ci(est_time=6, suite="stage-a-test-cpu")


class _DummySamplingInfo:
    def merge_batch(self, other):
        del other


class _DummyModelConfig:
    is_encoder_decoder = False


def _make_batch(output_ids: torch.Tensor | None) -> SimpleNamespace:
    return SimpleNamespace(
        maybe_wait_verify_done=lambda: None,
        sampling_info=_DummySamplingInfo(),
        model_config=_DummyModelConfig(),
        req_pool_indices=torch.tensor([0], dtype=torch.int64),
        seq_lens=torch.tensor([1], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([1], dtype=torch.int64),
        orig_seq_lens=torch.tensor([1], dtype=torch.int64),
        out_cache_loc=None,
        seq_lens_sum=1,
        output_ids=output_ids,
        mamba_track_indices=None,
        mamba_track_mask=None,
        mamba_track_seqlens=None,
        return_logprob=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        reqs=[object()],
        multimodal_inputs=None,
        has_stream=False,
        has_grammar=False,
        return_hidden_states=False,
        is_prefill_only=True,
        spec_info=None,
    )


class TestScheduleBatchMerge(unittest.TestCase):
    def test_merge_keeps_existing_output_ids_when_other_side_is_missing(self):
        extend_batch = _make_batch(torch.tensor([5], dtype=torch.int64))
        decode_batch = _make_batch(None)

        ScheduleBatch.merge_batch(extend_batch, decode_batch)

        self.assertTrue(torch.equal(extend_batch.output_ids, torch.tensor([5])))
        self.assertEqual(len(extend_batch.reqs), 2)
        self.assertEqual(extend_batch.seq_lens.shape[0], 2)

    def test_merge_adopts_other_output_ids_when_self_side_is_missing(self):
        decode_batch = _make_batch(None)
        prebuilt_batch = _make_batch(torch.tensor([7], dtype=torch.int64))

        ScheduleBatch.merge_batch(decode_batch, prebuilt_batch)

        self.assertTrue(torch.equal(decode_batch.output_ids, torch.tensor([7])))
        self.assertEqual(len(decode_batch.reqs), 2)
        self.assertEqual(decode_batch.seq_lens.shape[0], 2)

    def test_merge_keeps_existing_output_id_concat_behavior(self):
        first = _make_batch(torch.tensor([5], dtype=torch.int64))
        second = _make_batch(torch.tensor([7], dtype=torch.int64))

        ScheduleBatch.merge_batch(first, second)

        self.assertTrue(torch.equal(first.output_ids, torch.tensor([5, 7])))
        self.assertEqual(len(first.reqs), 2)


if __name__ == "__main__":
    unittest.main()
