"""The dense LoRA kernels report activations whose rows are not the batch's tokens.

The kernels size their grid from the batch metadata and never look at
``x.shape[0]``, so a mis-sized input (a cached prefix projected through a
LoRA-wrapped kv_b_proj on a path the chunked-prefix flag does not reroute)
misplaces the delta. ``check_lora_token_rows`` warns once per process ahead of
the eager sgemm launches and never raises.

    python -m pytest test/registered/unit/lora/test_lora_token_rows_check.py -v
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.lora import utils as lora_utils
from sglang.srt.lora.utils import LoRABatchInfo, check_lora_token_rows
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _batch_info(expected_tokens, use_cuda_graph=False):
    return LoRABatchInfo(
        use_cuda_graph=use_cuda_graph,
        bs=2,
        num_segments=2,
        seg_indptr=torch.tensor([0, 3, 5], dtype=torch.int32),
        weight_indices=torch.zeros(2, dtype=torch.int32),
        lora_ranks=torch.tensor([16], dtype=torch.int64),
        scalings=torch.ones(1),
        max_len=3,
        seg_lens=torch.tensor([3, 2], dtype=torch.int32),
        expected_tokens=expected_tokens,
    )


class TestCheckLoRATokenRows(CustomTestCase):
    def setUp(self):
        super().setUp()
        patcher = patch.object(lora_utils, "_warned_lora_token_rows", False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_matching_rows_are_silent(self):
        with patch.object(lora_utils.logger, "warning") as warning:
            check_lora_token_rows(torch.empty(5, 8), _batch_info(5))
        warning.assert_not_called()

    def test_mismatch_warns_once_and_never_raises(self):
        with patch.object(lora_utils.logger, "warning") as warning:
            check_lora_token_rows(torch.empty(9, 8), _batch_info(5))
            check_lora_token_rows(torch.empty(2, 8), _batch_info(5))
        warning.assert_called_once()
        self.assertEqual(warning.call_args.args[1:], (9, 5))

    def test_graph_and_unsized_infos_are_not_checked(self):
        # Graph infos carry the captured width; decode routing infos carry none.
        with patch.object(lora_utils.logger, "warning") as warning:
            check_lora_token_rows(
                torch.empty(2, 8), _batch_info(5, use_cuda_graph=True)
            )
            check_lora_token_rows(torch.empty(2, 8), _batch_info(None))
        warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
