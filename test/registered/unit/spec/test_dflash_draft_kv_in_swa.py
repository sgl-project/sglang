"""Which DFLASH drafts may keep their KV in the target's SWA sub-pool
(SGLANG_DFLASH_DRAFT_KV_IN_TARGET_SWA): every draft layer must slide within the
target's window, since the pool keeps only what SWA eviction retains."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.speculative.dflash_utils import dflash_draft_kv_fits_swa
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _draft(layer_types, sliding_window=None, **dflash_config):
    return {
        "num_hidden_layers": len(layer_types),
        "layer_types": list(layer_types),
        "sliding_window": sliding_window,
        "dflash_config": dflash_config,
    }


class TestDflashDraftKvFitsSwa(CustomTestCase):
    def fits(
        self,
        draft,
        target_window,
        *,
        draft_window_size=None,
        disagg="null",
        cuda=True,
        backend="fa4",
        kv_dtype="auto",
    ):
        with (
            patch(
                "sglang.srt.speculative.dflash_utils.get_spec",
                return_value=SimpleNamespace(
                    speculative_draft_window_size=draft_window_size,
                    speculative_draft_attention_backend=backend,
                ),
            ),
            patch(
                "sglang.srt.runtime_context.get_disagg",
                return_value=SimpleNamespace(disaggregation_mode=disagg),
            ),
            patch(
                "sglang.srt.platforms.current_platform",
                SimpleNamespace(is_cuda=lambda: cuda),
            ),
        ):
            return dflash_draft_kv_fits_swa(
                draft_hf_config=draft,
                target_window=target_window,
                draft_kv_cache_dtype=kv_dtype,
            )

    def test_all_sliding_draft_within_the_target_window(self):
        sliding = ["sliding_attention"] * 3
        # HF windows include the current token: 2048 -> window_left 2047
        self.assertTrue(self.fits(_draft(sliding, 2048), 2047))
        self.assertTrue(self.fits(_draft(sliding, 1024), 2047))
        self.assertFalse(self.fits(_draft(sliding, 2049), 2047))
        # window_left 0 is read as full attention (-1), so it must not qualify.
        self.assertTrue(self.fits(_draft(sliding, 2), 2047))
        self.assertFalse(self.fits(_draft(sliding, 1), 2047))

    def test_a_full_attention_layer_or_no_target_window_disqualifies(self):
        mixed = ["sliding_attention", "full_attention"]
        self.assertFalse(self.fits(_draft(mixed, 1024), 2047))
        self.assertFalse(self.fits(_draft(["full_attention"] * 2), 2047))
        self.assertFalse(self.fits(_draft(["sliding_attention"] * 2, 1024), None))

    def test_mla_draft_compact_window_and_pd_disaggregation_disqualify(self):
        sliding = ["sliding_attention"] * 2
        self.assertFalse(self.fits(_draft(sliding, 1024, attention_mode="mla"), 2047))
        self.assertFalse(self.fits(_draft(sliding, 1024), 2047, draft_window_size=512))
        self.assertFalse(self.fits(_draft(sliding, 1024), 2047, disagg="prefill"))

    def test_only_cuda_backends_that_map_into_the_swa_sub_pool_qualify(self):
        """Ascend builds a plain NPU pool with no register_mapping, which would
        fail the draft's startup; other platforms and backends never take it."""
        sliding = _draft(["sliding_attention"] * 2, 1024)
        self.assertTrue(self.fits(sliding, 2047, backend="trtllm_mha"))
        self.assertFalse(self.fits(sliding, 2047, cuda=False, backend="ascend"))
        self.assertFalse(self.fits(sliding, 2047, cuda=False, backend="triton"))
        self.assertFalse(self.fits(sliding, 2047, backend="torch_native"))
        self.assertFalse(self.fits(sliding, 2047, backend=None))

    def test_only_kv_dtypes_the_plain_swa_sub_pool_stores_qualify(self):
        """The draft's SWA sub-pool is a plain MHA pool: an mxfp8 or FP4 draft KV
        would lose its scale buffers / quant method there."""
        sliding = _draft(["sliding_attention"] * 2, 1024)
        self.assertTrue(self.fits(sliding, 2047, kv_dtype="fp8_e4m3"))
        for kv_dtype in ("mxfp8", "nvfp4", "fp4_mx_block16"):
            self.assertFalse(self.fits(sliding, 2047, kv_dtype=kv_dtype), kv_dtype)


if __name__ == "__main__":
    unittest.main()
