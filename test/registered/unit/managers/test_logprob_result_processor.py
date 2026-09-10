import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components.logprob_result_processor import (
    SchedulerLogprobResultProcessor,
)
from sglang.srt.server_args import MIS_DELIMITER_TOKEN_ID

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _processor(enable_mis=False, vocab_size=100):
    # Only `enable_mis` and `vocab_size` were used.
    return SchedulerLogprobResultProcessor(
        server_args=SimpleNamespace(enable_mis=enable_mis),
        model_config=SimpleNamespace(vocab_size=vocab_size),
    )


def _logprob(top_logprobs_num=0, token_ids_logprob=None):
    return SimpleNamespace(
        top_logprobs_num=top_logprobs_num,
        token_ids_logprob=token_ids_logprob,
        input_token_logprobs_val=None,
        input_token_logprobs_idx=None,
        input_top_logprobs_val=None,
        input_top_logprobs_idx=None,
        input_token_ids_logprobs_val=None,
        input_token_ids_logprobs_idx=None,
        output_token_logprobs_val=[],
        output_token_logprobs_idx=[],
        output_top_logprobs_val=[],
        output_top_logprobs_idx=[],
        output_token_ids_logprobs_val=[],
        output_token_ids_logprobs_idx=[],
    )


def _req(
    *,
    origin_input_ids=(10, 11, 12, 13),
    logprob_start_len=1,
    top_logprobs_num=0,
    token_ids_logprob=None,
    is_prefill_only=False,
    multi_item_delimiter_indices=None,
):
    return SimpleNamespace(
        logprob=_logprob(top_logprobs_num, token_ids_logprob),
        origin_input_ids=list(origin_input_ids),
        logprob_start_len=logprob_start_len,
        return_logprob=True,
        is_prefill_only=is_prefill_only,
        multi_item_delimiter_indices=multi_item_delimiter_indices,
        input_token_logprobs=None,
        temp_input_top_logprobs_val=None,
        temp_input_top_logprobs_idx=None,
        temp_input_token_ids_logprobs_val=None,
        temp_input_token_ids_logprobs_idx=None,
    )


class TestSchedulerLogprobResultProcessor(CustomTestCase):
    def test_regular_input_logprobs_align_and_clip_vocab_boundary(self):
        processor = _processor(vocab_size=100)
        # The ids 99 and 100 hope to be clipped.
        req = _req(origin_input_ids=[10, 11, 99, 100], logprob_start_len=1)
        output = SimpleNamespace(input_token_logprobs=(-0.1, -0.2, -0.3))

        processor.add_input_logprob_return_values(
            0, req, output, logprob_pt=0, num_input_logprobs=3, last_prefill_chunk=True
        )

        self.assertEqual(req.logprob.input_token_logprobs_val, [None, -0.1, -0.2])
        self.assertEqual(req.logprob.input_token_logprobs_idx, [11, 0, 0])
        # Temporary buffer should be cleaned.
        self.assertIsNone(req.input_token_logprobs)

    def test_top_and_token_ids_logprobs_drop_sampling_token_and_clear_temp(self):
        processor = _processor()
        # Cover optional input logprob payloads for top-k and requested token ids.
        req = _req(top_logprobs_num=2, token_ids_logprob=[7, 8, 9])
        output = SimpleNamespace(
            input_token_logprobs=(-0.1, -0.2, -0.3),
            input_top_logprobs_val=[["top-a", "top-b", "top-sampling"]],
            input_top_logprobs_idx=[["idx-a", "idx-b", "idx-sampling"]],
            input_token_ids_logprobs_val=[torch.tensor([-7.0, -8.0, -9.0])],
            input_token_ids_logprobs_idx=[[7, 8, 9]],
        )

        processor.add_input_logprob_return_values(
            0, req, output, logprob_pt=0, num_input_logprobs=3, last_prefill_chunk=True
        )

        self.assertEqual(req.logprob.input_top_logprobs_val, [None, "top-a", "top-b"])
        self.assertEqual(req.logprob.input_token_ids_logprobs_val, [None, -7.0, -8.0])
        self.assertIsNone(req.temp_input_top_logprobs_val)
        self.assertIsNone(req.temp_input_token_ids_logprobs_val)

        # Re-entering after input logprobs are finalized should be a no-op
        # so retract or reschedule paths do not overwrite existing results.
        processor.add_input_logprob_return_values(
            0,
            req,
            SimpleNamespace(input_token_logprobs=(-9.0, -8.0)),
            logprob_pt=0,
            num_input_logprobs=2,
            last_prefill_chunk=True,
        )

        self.assertEqual(req.logprob.input_token_logprobs_val, [None, -0.1, -0.2])
        self.assertEqual(req.logprob.input_token_logprobs_idx, [11, 12, 13])

    def test_multi_item_scoring_keeps_delimiter_logprobs_without_leading_none(self):
        # Test MIS path.
        processor = _processor(
            enable_mis=True,
            vocab_size=MIS_DELIMITER_TOKEN_ID + 2,
        )
        req = _req(
            top_logprobs_num=1,
            token_ids_logprob=[42],
            is_prefill_only=True,
            multi_item_delimiter_indices=[1, 3],
        )
        output = SimpleNamespace(
            input_token_logprobs=(-1.0, -2.0),
            input_top_logprobs_val=[["top-delim-1", "top-delim-2"]],
            input_top_logprobs_idx=[["idx-delim-1", "idx-delim-2"]],
            input_token_ids_logprobs_val=[torch.tensor([-4.0, -5.0])],
            input_token_ids_logprobs_idx=[[42, 43]],
        )

        processor.add_input_logprob_return_values(
            0, req, output, logprob_pt=0, num_input_logprobs=2, last_prefill_chunk=True
        )

        # Should not be shifted.
        self.assertEqual(req.logprob.input_token_logprobs_val, [-1.0, -2.0])
        self.assertEqual(
            req.logprob.input_token_logprobs_idx,
            [MIS_DELIMITER_TOKEN_ID, MIS_DELIMITER_TOKEN_ID],  # Placeholder token ids.
        )
        self.assertEqual(
            req.logprob.input_top_logprobs_val, ["top-delim-1", "top-delim-2"]
        )

    def test_add_logprob_return_values_initializes_empty_input_containers(self):
        processor = _processor()
        req = _req(top_logprobs_num=1, token_ids_logprob=[42])
        output = SimpleNamespace(
            next_token_logprobs=(-0.4,),
            next_token_top_logprobs_val=[["next-top"]],
            next_token_top_logprobs_idx=[["next-idx"]],
            next_token_token_ids_logprobs_val=[torch.tensor([-4.0, -5.0])],
            next_token_token_ids_logprobs_idx=[[42, 43]],
        )

        processor.add_logprob_return_values(
            0, req, pt=0, next_token_ids=[123], num_input_logprobs=0, output=output
        )

        self.assertEqual(req.logprob.input_token_logprobs_val, [])  # Not None.
        self.assertEqual(req.logprob.output_token_logprobs_val, [-0.4])
        self.assertEqual(req.logprob.output_token_logprobs_idx, [123])
        self.assertEqual(req.logprob.output_token_ids_logprobs_val, [[-4.0, -5.0]])

        req = _req()
        output = SimpleNamespace(
            next_token_logprobs=(-0.5,),
            input_token_logprobs=(-0.1, -0.2, -0.3),
        )

        ret = processor.add_logprob_return_values(
            0, req, pt=0, next_token_ids=[124], num_input_logprobs=3, output=output
        )

        self.assertEqual(ret, 3)
        self.assertEqual(req.logprob.input_token_logprobs_val, [None, -0.1, -0.2])
        self.assertEqual(req.logprob.input_token_logprobs_idx, [11, 12, 13])
        self.assertEqual(req.logprob.output_token_logprobs_val, [-0.5])
        self.assertEqual(req.logprob.output_token_logprobs_idx, [124])

    def test_calculate_num_input_logprobs_regular_and_mis(self):
        # Regular requests count every input position in the extend range.
        self.assertEqual(
            _processor().calculate_num_input_logprobs(
                _req(), extend_input_len=5, extend_logprob_start_len=2
            ),
            3,
        )

        # MIS requests count only delimiter positions inside that range.
        req = _req(is_prefill_only=True, multi_item_delimiter_indices=[1, 2, 4, 6])
        self.assertEqual(
            _processor(enable_mis=True).calculate_num_input_logprobs(
                req, extend_input_len=5, extend_logprob_start_len=2
            ),
            2,
        )


if __name__ == "__main__":
    unittest.main()
