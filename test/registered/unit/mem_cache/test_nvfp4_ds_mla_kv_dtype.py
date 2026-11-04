"""--kv-cache-dtype nvfp4_ds_mla resolution.

nvfp4_ds_mla stores the DSA MLA cache as 352 B NVFP4 records read only by
Prims-TS, but its element and query dtype stay E4M3, so it must not reach the
float4_e2m1fn_x2 ("nvfp4") pool recipe. The pool row width, the pool writer and
the DSA backend all follow one resolved flag; a draft that inherits the string
keeps raw FP8 unless it is itself a DSA model.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
    resolve_kv_cache_quant,
)
from sglang.srt.mem_cache import kv_cache_configurator, kv_cache_dtype
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_CUDA = SimpleNamespace(is_cpu=lambda: False)


def _resolve(**kwargs):
    args = dict(
        server_args_kv_cache_dtype="nvfp4_ds_mla",
        quant_config=None,
        model_dtype=torch.bfloat16,
        is_draft_worker=False,
        is_dflash=False,
        speculative_draft_attention_backend=None,
    )
    args.update(kwargs)
    with (
        patch.object(kv_cache_dtype, "current_platform", _CUDA),
        patch(
            "sglang.srt.configs.model_config.is_deepseek_dsa",
            side_effect=lambda cfg: cfg.dsa,
        ),
    ):
        return kv_cache_dtype.configure_kv_cache_dtype(**args)


class TestNvfp4DsMlaDtype(CustomTestCase):
    def test_target_resolves_to_e4m3_not_the_fp4_recipe(self):
        self.assertEqual(_resolve(), ("nvfp4_ds_mla", torch.float8_e4m3fn))
        # The FP4 pool branches key off this registry; nvfp4_ds_mla must miss it.
        self.assertIsNone(resolve_kv_cache_quant("nvfp4_ds_mla"))
        self.assertTrue(kv_cache_dtype.is_fp8_family_kv_cache_dtype("nvfp4_ds_mla"))
        self.assertFalse(kv_cache_dtype.is_fp8_family_kv_cache_dtype("nvfp4"))

    def test_draft_inherits_raw_fp8_unless_dsa(self):
        """A non-DSA draft (DFLASH cutedsl_mla) has no NVFP4 reader; its runner
        string must read fp8_e4m3 so FP8 consumers keep treating it as such."""
        non_dsa = SimpleNamespace(dsa=False)
        dsa = SimpleNamespace(dsa=True)
        for hf_config, expected in (
            (None, "fp8_e4m3"),
            (non_dsa, "fp8_e4m3"),
            (dsa, "nvfp4_ds_mla"),
        ):
            with self.subTest(hf_config=hf_config):
                self.assertEqual(
                    _resolve(is_draft_worker=True, hf_config=hf_config),
                    (expected, torch.float8_e4m3fn),
                )


class TestNvfp4DsMlaRowWidth(CustomTestCase):
    """calculate_mla_kv_cache_dim sizes the pool from the local role only."""

    @staticmethod
    def _dim(*, kv_dtype, mode, prefill, decode):
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(), kv_lora_rank=512, qk_rope_head_dim=64
        )
        exec_ = SimpleNamespace(
            kernel=SimpleNamespace(
                dsa_prefill_backend=prefill, dsa_decode_backend=decode
            )
        )
        with (
            patch.object(kv_cache_configurator, "is_deepseek_dsa", return_value=True),
            patch("sglang.srt.configs.model_config.is_deepseek_dsa", return_value=True),
            patch(
                "sglang.srt.runtime_context.get_model",
                return_value=SimpleNamespace(kv_cache_dtype=kv_dtype),
            ),
            patch.object(kv_cache_configurator, "get_exec", return_value=exec_),
            patch.object(
                kv_cache_configurator,
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_mode=mode),
            ),
        ):
            return kv_cache_configurator.calculate_mla_kv_cache_dim(
                model_config=model_config, kv_cache_dtype=torch.float8_e4m3fn
            )

    def test_local_role_layout(self):
        self.assertEqual(
            self._dim(
                kv_dtype="nvfp4_ds_mla", mode="decode", prefill=None, decode="prims_ts"
            ),
            352,
        )
        self.assertEqual(
            self._dim(
                kv_dtype="fp8_e4m3", mode="null", prefill="prims_ts", decode="prims_ts"
            ),
            576,
        )
        with self.assertRaisesRegex(ValueError, "prims_ts for every local role"):
            self._dim(
                kv_dtype="nvfp4_ds_mla",
                mode="null",
                prefill="prims_ts",
                decode="trtllm",
            )


if __name__ == "__main__":
    unittest.main()
