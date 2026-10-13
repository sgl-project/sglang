"""Unit tests for the PD decode-side Mamba speculative-state sizing.

Covers HybridMambaDecodeReqToTokenPool: the speculative intermediate scratch
(intermediate_ssm / intermediate_conv_window) is indexed positionally by
running-batch slot, so it is sized by `size` (the running capacity), not by
`size + pre_alloc_size` (which reserves req_to_token slots for in-transfer
requests that never enter target_verify). The GDN ReplaySSM-spec path keys that
scratch by req_pool_idx, so it keeps the padded `size + pre_alloc_size`.
"""

import unittest
from unittest.mock import patch

from sglang.srt.disaggregation.decode import HybridMambaDecodeReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDecodeMambaSpecStateSize(CustomTestCase):
    SIZE = 8
    PRE_ALLOC = 16

    def _spec_state_size(self, *, enable_linear_replayssm_spec: bool) -> int:
        """Build the pool on CPU and return the mamba_spec_state_size it would
        allocate, capturing it at the _init_mamba_pool boundary so no GPU state
        is created."""
        captured = {}

        def fake_init_mamba_pool(self, *, mamba_spec_state_size, **kwargs):
            captured["mamba_spec_state_size"] = mamba_spec_state_size

        with patch.object(
            HybridMambaDecodeReqToTokenPool,
            "_init_mamba_pool",
            fake_init_mamba_pool,
        ):
            HybridMambaDecodeReqToTokenPool(
                size=self.SIZE,
                max_context_len=4,
                device="cpu",
                enable_memory_saver=False,
                cache_params=None,
                mamba_layer_ids=[0],
                speculative_num_draft_tokens=6,
                enable_mamba_extra_buffer=False,
                pre_alloc_size=self.PRE_ALLOC,
                enable_overlap_schedule=True,
                mamba_size=self.SIZE + self.PRE_ALLOC,
                enable_linear_replayssm_spec=enable_linear_replayssm_spec,
            )
        self.assertIn("mamba_spec_state_size", captured)
        return captured["mamba_spec_state_size"]

    def test_default_sizes_spec_state_by_running_capacity(self):
        # The pre_alloc_size in-transfer slots never reach target_verify, so the
        # positionally-indexed spec scratch only needs `size` rows.
        self.assertEqual(
            self._spec_state_size(enable_linear_replayssm_spec=False),
            self.SIZE,
        )

    def test_replayssm_spec_keeps_padded_size(self):
        # GDN ReplaySSM-spec ring/cursors are keyed by req_pool_idx, which spans
        # the full size + pre_alloc_size range; the padded size must be kept.
        self.assertEqual(
            self._spec_state_size(enable_linear_replayssm_spec=True),
            self.SIZE + self.PRE_ALLOC,
        )


if __name__ == "__main__":
    unittest.main()
