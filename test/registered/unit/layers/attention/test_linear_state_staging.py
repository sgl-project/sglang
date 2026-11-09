import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
    FlashInferGDNKernel,
)
from sglang.srt.layers.attention.linear.kernels.state_pool_io import (
    stage_pool_states,
    unstage_pool_states,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _kernel():
    kernel = FlashInferGDNKernel.__new__(FlashInferGDNKernel)
    kernel.use_state_pool = False
    kernel._prefill_needs_fp32_state = False
    kernel._aligned_input_buffers = {}
    kernel._aligned_parameter_cache = {}
    kernel._alignment_fallback_warned = False

    def prefill(**kw):
        kw["output_state"].copy_(kw["initial_state"] + 2)
        return torch.zeros_like(kw["v"]), kw["output_state"]

    kernel._prefill_fn = prefill
    return kernel


def _strided_pool(rows=8):
    # A state pool whose slots are strided inside wider entries, as in a unified
    # arena; slot 0 is the padding sink.
    envelope = torch.full((rows, 48), 19.0)
    states = torch.as_strided(envelope, (rows, 1, 4, 4), (48, 16, 4, 1))
    states[0].fill_(0)
    for slot in range(1, rows):
        states[slot].fill_(slot)
    return states


def _extend(kernel, states, indices, num_seqs):
    qkv = torch.ones((1, num_seqs, 1, 4))
    gate = torch.zeros((1, num_seqs, 1))
    with patch(
        "sglang.kernels.ops.attention.fla.l2norm.gdn_prefill_qkv_prepare_fwd",
        side_effect=lambda q, k, v: (q, k, v),
    ):
        kernel.extend(
            qkv,
            qkv,
            qkv,
            gate,
            gate,
            ssm_states=states,
            cache_indices=indices,
            query_start_loc=torch.arange(num_seqs + 1, dtype=torch.int32),
        )


class TestLinearStateStaging(CustomTestCase):
    def test_staged_prefill_writes_every_live_request_back(self):
        """A prefill on contiguous staged copies updates each live request's
        pool slot, including the request staged first, and leaves padding,
        the sink and other slots alone."""
        pool = _strided_pool()
        conv = _strided_pool()
        before = pool.clone()
        indices = torch.tensor([5, 2, -1, 0], dtype=torch.int32)

        (conv_copy, ssm_copy), rows, pool_rows = stage_pool_states(
            (conv, pool), indices
        )
        self.assertTrue(ssm_copy.is_contiguous())
        _extend(_kernel(), ssm_copy, rows, num_seqs=4)
        unstage_pool_states((conv, pool), (conv_copy, ssm_copy), pool_rows)

        torch.testing.assert_close(pool[5], before[5] + 2)
        torch.testing.assert_close(pool[2], before[2] + 2)
        for slot in (0, 1, 3, 4, 6, 7):
            torch.testing.assert_close(pool[slot], before[slot])
        torch.testing.assert_close(conv, _strided_pool())

    def test_zero_based_staging_loses_the_first_request(self):
        """Why staging keeps the sink at row 0: the kernel treats state index 0
        as padding, so a request staged at row 0 would get no state."""
        pool = _strided_pool()
        copy = pool[torch.tensor([5])].contiguous()
        _extend(_kernel(), copy, torch.tensor([0], dtype=torch.int32), num_seqs=1)
        self.assertEqual(torch.count_nonzero(copy), 0)


if __name__ == "__main__":
    unittest.main()
