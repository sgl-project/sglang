"""--triton-attention-num-kv-splits is a cap the operator owns.

Left unset, the Triton backend picks the platform default (16 on ROCm, 8
elsewhere) and, for MLA, raises it with the CU count and context length. Set,
the value is used as given on every platform and every attention type. The one
exception is gfx942 MLA, where anything above 256 faults in ROCm graph replay.
"""

import unittest
from unittest.mock import patch

from sglang.srt.layers.attention import triton_backend
from sglang.srt.layers.attention.triton_backend import _resolve_max_kv_splits
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

# A full MI355X (256 CUs) at 32k context: the MLA heuristic lands on 256.
SM_COUNT = 256
CONTEXT_LEN = 32768


def _platform(default: int, gfx942: bool = False):
    return patch.multiple(
        triton_backend,
        DEFAULT_TRITON_NUM_KV_SPLITS=default,
        _is_xpu=False,
        _is_gfx942=gfx942,
    )


def _resolve(configured, use_mla, sm_count=SM_COUNT):
    return _resolve_max_kv_splits(configured, use_mla, sm_count, CONTEXT_LEN)


class TestTritonMaxKvSplits(CustomTestCase):
    def test_unset_takes_the_platform_default(self):
        for default in (16, 8):
            with self.subTest(default=default), _platform(default):
                self.assertEqual(_resolve(None, use_mla=False), default)

    def test_an_explicit_value_survives_off_mla(self):
        with _platform(16):
            self.assertEqual(_resolve(32, use_mla=False), 32)
            self.assertEqual(_resolve(8, use_mla=False), 8)

    def test_unset_mla_scales_with_the_device(self):
        with _platform(16):
            self.assertEqual(_resolve(None, use_mla=True), 256)

    def test_an_explicit_value_survives_on_mla(self):
        with _platform(16):
            self.assertEqual(_resolve(8, use_mla=True), 8)
            self.assertEqual(_resolve(64, use_mla=True), 64)

    def test_gfx942_mla_stays_at_or_below_256(self):
        with _platform(16, gfx942=True):
            self.assertEqual(_resolve(None, use_mla=True, sm_count=304), 256)
            self.assertEqual(_resolve(512, use_mla=True, sm_count=304), 256)
            self.assertEqual(_resolve(8, use_mla=True, sm_count=304), 8)


class TestNumKvSplitsValidation(CustomTestCase):
    def test_counts_below_one_are_rejected(self):
        for value in (0, -1):
            with self.subTest(value=value):
                server_args = ServerArgs(
                    model_path="dummy", triton_attention_num_kv_splits=value
                )
                with self.assertRaisesRegex(
                    ValueError, "--triton-attention-num-kv-splits must be >= 1"
                ):
                    server_args.resolve_once()

    def test_unset_and_positive_counts_resolve(self):
        for value in (None, 1, 32):
            with self.subTest(value=value):
                server_args = ServerArgs(
                    model_path="dummy", triton_attention_num_kv_splits=value
                )
                server_args.resolve_once()
                self.assertEqual(server_args.triton_attention_num_kv_splits, value)


if __name__ == "__main__":
    unittest.main()
