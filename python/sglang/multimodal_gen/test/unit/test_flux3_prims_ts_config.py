# SPDX-License-Identifier: Apache-2.0
"""CPU-safe tests for the FLUX.3 PrimsTS tactic table and its TMEM overlap upgrade."""

import unittest
from unittest import mock

import pytest

prims_ts_gemm = pytest.importorskip(
    "flashinfer.prims_ts.gemm", reason="PrimsTS ships with flashinfer (CUDA-only)"
)

from sglang.kernels.ops.diffusion.gemm import flux3_prims_ts_config as config_module

_OVERLAP_EPILOGUES = frozenset({"swiglu", "qkv_qknorm_rope"})


def _tuned(support, *, operand_format, epilogue, k, n, bucket, output_format="bf16"):
    # Configs are cached per shape; the overlap support changes between calls.
    config_module._tuned_config.cache_clear()
    with mock.patch.object(
        config_module, "_tmem_overlap_epilogues", return_value=support
    ):
        return config_module._tuned_config(
            arch=103,
            operand_format=operand_format,
            output_format=output_format,
            epilogue=epilogue,
            k=k,
            n=n,
            bucket=bucket,
            head_dim=128 if epilogue == "qkv_qknorm_rope" else None,
            qkv_scale=False,
        )


def _table_keys():
    for (fmt, epilogue, k, n), rows in config_module._TACTICS.items():
        # MLP-up feeds MLP-down a packed FP4 activation in the NVFP4 fast path.
        packed = (fmt, epilogue) == ("nvfp4_e2m1", "swiglu")
        for bucket in rows:
            for output_format in ("bf16", "nvfp4_e2m1") if packed else ("bf16",):
                yield dict(
                    operand_format=fmt,
                    epilogue=epilogue,
                    k=k,
                    n=n,
                    bucket=bucket,
                    output_format=output_format,
                )


class TestFlux3PrimsTsTmemOverlap(unittest.TestCase):
    def setUp(self):
        self.addCleanup(config_module._tuned_config.cache_clear)
        config_module._tmem_overlap_epilogues.cache_clear()
        self.addCleanup(config_module._tmem_overlap_epilogues.cache_clear)

    def test_overlap_upgrade_sweeps_the_tactic_table(self):
        for key in _table_keys():
            with self.subTest(**key):
                off = _tuned(frozenset(), **key)
                on = _tuned(_OVERLAP_EPILOGUES, **key)
                # Overlap and the TMA output store cannot share tensor memory.
                for config in (off, on):
                    self.assertFalse(config.tmem_overlap and config.use_tma_store)
                if (
                    key["operand_format"] == "nvfp4_e2m1"
                    and key["epilogue"] in _OVERLAP_EPILOGUES
                ):
                    # FlashInfer without the overlap PRs rejects it for these epilogues.
                    self.assertFalse(off.tmem_overlap)
                    # A re-tuned row outside the overlap constraints would
                    # silently lose the FlashInfer speed-up.
                    self.assertTrue(on.tmem_overlap)
                    self.assertFalse(on.use_tma_store)
                else:
                    self.assertEqual(on, off)

        # NVFP4 TP=2 shards miss the table; FlashInfer's default tile_k=128 is
        # below the FP4 TMA K-box and fails at startup.
        for epilogue, n in (("swiglu", 9216), ("qkv_qknorm_rope", 4608)):
            key = dict(operand_format="nvfp4_e2m1", epilogue=epilogue, k=3072, n=n)
            with self.subTest(fallback=epilogue):
                off = _tuned(frozenset(), bucket=8192, **key)
                self.assertEqual(_tuned(_OVERLAP_EPILOGUES, bucket=8192, **key), off)
                self.assertEqual(off.tile_k, 256)

    def test_support_is_read_from_the_prepare_function_signatures(self):
        def with_overlap(weight, *, tmem_overlap=False):
            pass

        def without_overlap(weight):
            pass

        with mock.patch.multiple(
            prims_ts_gemm,
            prepare_fp4_linear_swiglu=with_overlap,
            prepare_fp4_qkv_qknorm_rope=without_overlap,
        ):
            self.assertEqual(
                config_module._tmem_overlap_epilogues(), frozenset({"swiglu"})
            )


if __name__ == "__main__":
    unittest.main()
