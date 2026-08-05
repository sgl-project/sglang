import unittest

import torch

from sglang.srt.layers.attention.dsa_backend import (
    _should_all_gather_dsa_trtllm_fp8_kv,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSATRTLLMFP8CP(unittest.TestCase):
    def test_nope_kv_is_not_gathered_again(self):
        self.assertFalse(
            _should_all_gather_dsa_trtllm_fp8_kv(
                save_kv_cache=True,
                cos_sin_cache=None,
                dsa_prefill_cp=True,
            )
        )

    def test_fused_rope_kv_is_gathered_after_rope(self):
        self.assertTrue(
            _should_all_gather_dsa_trtllm_fp8_kv(
                save_kv_cache=True,
                cos_sin_cache=torch.empty(0),
                dsa_prefill_cp=True,
            )
        )

    def test_non_cp_and_no_cache_paths_do_not_gather(self):
        cos_sin_cache = torch.empty(0)
        self.assertFalse(
            _should_all_gather_dsa_trtllm_fp8_kv(
                save_kv_cache=True,
                cos_sin_cache=cos_sin_cache,
                dsa_prefill_cp=False,
            )
        )
        self.assertFalse(
            _should_all_gather_dsa_trtllm_fp8_kv(
                save_kv_cache=False,
                cos_sin_cache=cos_sin_cache,
                dsa_prefill_cp=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
