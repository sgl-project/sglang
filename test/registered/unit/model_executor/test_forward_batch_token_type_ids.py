"""Regression tests for cross-encoder token type metadata assembly."""

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.forward_batch_info import (  # noqa: E402
    ForwardBatch,
    ForwardMode,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_forward_batch():
    return ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=2,
        input_ids=torch.zeros(5, dtype=torch.long),
        req_pool_indices=torch.zeros(2, dtype=torch.long),
        seq_lens=torch.ones(2, dtype=torch.long),
        out_cache_loc=torch.zeros(2, dtype=torch.long),
        seq_lens_sum=2,
    )


class TestForwardBatchTokenTypeIds(unittest.TestCase):
    def test_flattens_request_values_in_batch_order(self):
        forward_batch = _make_forward_batch()
        batch = SimpleNamespace(
            device=torch.device("cpu"),
            reqs=[
                SimpleNamespace(token_type_ids=[0, 1]),
                SimpleNamespace(token_type_ids=[1, 0, 1]),
            ],
        )

        forward_batch._maybe_init_non_generation_fields(batch)

        torch.testing.assert_close(
            forward_batch.token_type_ids,
            torch.tensor([0, 1, 1, 0, 1], dtype=torch.int64),
        )

    def test_omits_metadata_when_no_request_provides_it(self):
        forward_batch = _make_forward_batch()
        batch = SimpleNamespace(
            device=torch.device("cpu"),
            reqs=[
                SimpleNamespace(token_type_ids=None),
                SimpleNamespace(token_type_ids=None),
            ],
        )

        forward_batch._maybe_init_non_generation_fields(batch)

        self.assertIsNone(forward_batch.token_type_ids)

    def test_fills_missing_request_metadata_with_segment_zero(self):
        forward_batch = _make_forward_batch()
        batch = SimpleNamespace(
            device=torch.device("cpu"),
            reqs=[
                SimpleNamespace(
                    origin_input_ids=array("q", [101, 42]),
                    token_type_ids=None,
                ),
                SimpleNamespace(
                    origin_input_ids=array("q", [101, 43, 102]),
                    token_type_ids=[0, 1, 1],
                ),
            ],
        )

        forward_batch._maybe_init_non_generation_fields(batch)

        torch.testing.assert_close(
            forward_batch.token_type_ids,
            torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64),
        )


if __name__ == "__main__":
    unittest.main()
