import unittest

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Above the fp16 midpoint 1 + 2^-11 (single-rounds up to 1 + 2^-10) but below
# the bf16 midpoint 1 + 2^-8 (rounds to 1.0, which then stays 1.0 in fp16):
# any fp32 -> bf16 -> fp16 double rounding loses the increment. The 2^-23 tail
# is the last fp32 mantissa bit at 1.x, so the probe is fp32-exact (a 2^-24
# tail would round back to the midpoint itself).
DOUBLE_ROUND_PROBE = 1.0 + 2.0**-11 + 2.0**-23


class TestTrackMambaStateDtype(CustomTestCase):
    """The fp32 track snapshot is cast to the pool dtype exactly once.

    ``_track_mamba_state_extend`` reads the in-kernel fp32 snapshot
    (``h_track_buf``) and casts it to ``ssm_states.dtype`` in a single ``.to``.
    This must hold for every ``--mamba-ssm-dtype``: fp32 keeps full precision,
    bf16 matches the (already correct) legacy path, and fp16 must not inherit
    the old double rounding through the bf16 per-chunk states ``h``.
    """

    @staticmethod
    def _run_track_copy(pool_dtype, h_track_buf, dst_slots, batch_rows):
        metadata = ForwardMetadata(
            has_mamba_track_mask=True,
            # Only numel() gates the copy on this path; the h-row values
            # themselves are unused when h_track_buf is given.
            track_ssm_h_src=torch.zeros(len(dst_slots), dtype=torch.long),
            track_ssm_h_dst=torch.tensor(dst_slots),
            track_ssm_h_batch_src=torch.tensor(batch_rows),
            track_ssm_final_src=torch.empty(0, dtype=torch.long),
            track_ssm_final_dst=torch.empty(0, dtype=torch.long),
            # Required by the dataclass; unused on this path.
            query_start_loc=torch.zeros(1, dtype=torch.int32),
            mamba_cache_indices=torch.zeros(1, dtype=torch.long),
        )
        ssm_states = torch.zeros(8, *h_track_buf.shape[1:], dtype=pool_dtype)
        # The method touches no `self` state; call it unbound so this stays a
        # pure bookkeeping test.
        MambaAttnBackendBase._track_mamba_state_extend(
            None, None, None, ssm_states, metadata, h_track_buf=h_track_buf
        )
        return ssm_states

    def test_snapshot_cast_once_to_pool_dtype(self):
        torch.manual_seed(0)
        h_track_buf = torch.randn(3, 2, 4, 4, dtype=torch.float32)
        h_track_buf[0, 0, 0, 0] = DOUBLE_ROUND_PROBE
        for pool_dtype in (torch.float32, torch.bfloat16, torch.float16):
            with self.subTest(pool_dtype=pool_dtype):
                ssm_states = self._run_track_copy(
                    pool_dtype, h_track_buf, dst_slots=[5, 2], batch_rows=[0, 2]
                )
                # Single rounding of the fp32 snapshot, in batch-row order.
                self.assertTrue(
                    torch.equal(ssm_states[5], h_track_buf[0].to(pool_dtype))
                )
                self.assertTrue(
                    torch.equal(ssm_states[2], h_track_buf[2].to(pool_dtype))
                )
                untouched = torch.ones(8, dtype=torch.bool)
                untouched[[5, 2]] = False
                self.assertTrue(torch.all(ssm_states[untouched] == 0))

    def test_fp16_pool_is_not_double_rounded_through_bf16(self):
        h_track_buf = torch.full((1, 1, 1, 1), DOUBLE_ROUND_PROBE)
        ssm_states = self._run_track_copy(
            torch.float16, h_track_buf, dst_slots=[3], batch_rows=[0]
        )
        # fp32 -> fp16 rounds the probe UP to 1 + 2^-10; the legacy path
        # (fp32 -> bf16 h -> fp16) collapsed it to exactly 1.0.
        self.assertEqual(ssm_states[3, 0, 0, 0].item(), 1.0 + 2.0**-10)

    def test_no_unaligned_rows_leaves_pool_untouched(self):
        # Aligned-only tracking: the h branch is gated off entirely.
        h_track_buf = torch.randn(2, 1, 1, 1, dtype=torch.float32)
        ssm_states = self._run_track_copy(
            torch.float16, h_track_buf, dst_slots=[], batch_rows=[]
        )
        self.assertTrue(torch.all(ssm_states == 0))


if __name__ == "__main__":
    unittest.main()
