import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=13, suite="base-a-test-cpu")


def _pp_mgr(start, end, ratios):
    mgr = object.__new__(CommonKVManager)
    mgr.kv_args = SimpleNamespace(
        prefill_start_layer=start,
        prefill_end_layer=end,
        mla_compression_ratios=ratios,
    )
    return mgr


class TestDSV4UnifiedFp8PpSlice(CustomTestCase):
    RATIOS = [4, 4, 128, 4]
    # kv_data groups: 2*c4 + c128 bf16, 3*c4 + 2*c128 under the fp8 two-pool
    BF16_KV_LEN = 7
    FP8_KV_LEN = 11

    def _slice(self, dst, start, end, state_type=None, src=None):
        mgr = _pp_mgr(start, end, self.RATIOS)
        if src is None:
            src = list(range(100, 100 + 8))
        return mgr._mla_slice_ptrs_for_pp(src, dst, self.RATIOS, state_type)

    def test_mixed_fp8_and_bf16_peers_are_rejected(self):
        cases = (
            (None, self.BF16_KV_LEN, self.FP8_KV_LEN),
            (None, self.FP8_KV_LEN, self.BF16_KV_LEN),
            (StateType.SWA_RING, len(self.RATIOS), 2 * len(self.RATIOS)),
            (StateType.SWA_RING, 2 * len(self.RATIOS), len(self.RATIOS)),
        )
        for state_type, src_len, dst_len in cases:
            with self.subTest(state_type=state_type, src_len=src_len):
                with self.assertRaisesRegex(ValueError, "SGLANG_DSV4_UNIFIED_KV_FP8"):
                    self._slice(
                        list(range(dst_len)),
                        1,
                        3,
                        state_type,
                        src=list(range(100, 100 + src_len)),
                    )


if __name__ == "__main__":
    unittest.main()
