"""Server-args validation + dispatch fallback tests for Double Sparsity (M5).

Pins all the startup error paths from the plan's non-goals list:
- Missing --double-sparsity-config when DS is enabled.
- DS coexistence with --enable-hisparse.
- FP8 KV cache.
- --page-size != 1.
- Speculative decoding.
- recent_tokens < 1.
- min_seq_len > max_selected_per_request.

Plus the runtime-config validate path and the algorithm's heavy-channel
mismatch error.
"""

import unittest
from pathlib import Path

import torch

from sglang.srt.mem_cache.sparsity.algorithms.double_sparsity import (
    DoubleSparsityAlgorithm,
)
from sglang.srt.mem_cache.sparsity.algorithms.double_sparsity_config import (
    DoubleSparsityRuntimeConfig,
    parse_calibration_file,
)
from sglang.srt.mem_cache.sparsity.core.sparse_coordinator import SparseConfig
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


FIXTURE_PATH = Path(__file__).parent / "_fixtures" / "tiny_ds_calibration.json"


def _make_args(**over):
    sa = ServerArgs.__new__(ServerArgs)
    sa.enable_double_sparsity = over.get("enable_double_sparsity", True)
    sa.double_sparsity_config = over.get("double_sparsity_config", str(FIXTURE_PATH))
    sa.enable_hisparse = over.get("enable_hisparse", False)
    sa.kv_cache_dtype = over.get("kv_cache_dtype", "bfloat16")
    sa.page_size = over.get("page_size", 1)
    sa.speculative_algorithm = over.get("speculative_algorithm", None)
    sa.double_sparsity_recent_tokens = over.get("double_sparsity_recent_tokens", 4)
    sa.double_sparsity_min_seq_len = over.get("double_sparsity_min_seq_len", 128)
    sa.double_sparsity_max_selected_per_request = over.get(
        "double_sparsity_max_selected_per_request", 256
    )
    return sa


class TestServerArgsValidation(CustomTestCase):
    def test_disabled_is_noop(self):
        sa = _make_args(enable_double_sparsity=False, double_sparsity_config=None)
        sa._handle_double_sparsity()  # must not raise

    def test_missing_config_path(self):
        sa = _make_args(double_sparsity_config=None)
        with self.assertRaisesRegex(ValueError, "--double-sparsity-config"):
            sa._handle_double_sparsity()

    def test_hisparse_coexistence(self):
        sa = _make_args(enable_hisparse=True)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            sa._handle_double_sparsity()

    def test_fp8_kv_cache(self):
        sa = _make_args(kv_cache_dtype="fp8_e4m3")
        with self.assertRaisesRegex(ValueError, "kv_cache_dtype"):
            sa._handle_double_sparsity()

    def test_page_size_not_one(self):
        sa = _make_args(page_size=64)
        with self.assertRaisesRegex(ValueError, "page-size 1"):
            sa._handle_double_sparsity()

    def test_speculative_decoding(self):
        sa = _make_args(speculative_algorithm="EAGLE")
        with self.assertRaisesRegex(ValueError, "speculative"):
            sa._handle_double_sparsity()

    def test_recent_tokens_zero(self):
        sa = _make_args(double_sparsity_recent_tokens=0)
        with self.assertRaisesRegex(ValueError, "recent-tokens"):
            sa._handle_double_sparsity()

    def test_min_seq_len_exceeds_max_selected(self):
        sa = _make_args(
            double_sparsity_min_seq_len=512,
            double_sparsity_max_selected_per_request=256,
        )
        with self.assertRaisesRegex(ValueError, "max-selected-per-request"):
            sa._handle_double_sparsity()


class TestRuntimeConfigInvariants(CustomTestCase):
    def _rt(self, **over):
        d = dict(
            heavy_channels=8,
            token_budget=64,
            recent_tokens=4,
            sink_tokens=4,
            min_seq_len=128,
            max_selected_per_request=256,
            gqa_reduction="max_abs",
            klabel_dtype="bf16",
        )
        d.update(over)
        return DoubleSparsityRuntimeConfig(**d)

    def test_recent_tokens_must_be_at_least_one(self):
        with self.assertRaisesRegex(ValueError, "recent_tokens"):
            self._rt(recent_tokens=0).validate()

    def test_min_seq_len_at_most_max_selected(self):
        with self.assertRaisesRegex(ValueError, "max_selected_per_request"):
            self._rt(min_seq_len=512, max_selected_per_request=256).validate()


class TestAlgorithmHeavyChannelMismatch(CustomTestCase):
    def test_runtime_vs_calibration_mismatch_raises(self):
        calib = parse_calibration_file(FIXTURE_PATH)
        rt = DoubleSparsityRuntimeConfig(
            heavy_channels=4,  # calibration has 8 → mismatch
            token_budget=16,
            recent_tokens=2,
            sink_tokens=1,
            min_seq_len=8,
            max_selected_per_request=64,
            gqa_reduction="max_abs",
            klabel_dtype="bf16",
        )
        sc = SparseConfig(algorithm="double_sparsity", backend="fa3", page_size=1)
        with self.assertRaisesRegex(ValueError, "heavy_channels"):
            DoubleSparsityAlgorithm(
                sc,
                torch.device("cpu"),
                runtime_config=rt,
                calibration=calib,
                tp_size=1,
                tp_rank=0,
                num_kv_heads_local=4,
                num_q_heads_local=8,
                head_dim=16,
            )


class TestRadixAttentionDsEnabledDefault(CustomTestCase):
    """RadixAttention.__init__ default — DS-off path is the common path
    and must remain byte-for-byte unchanged from main."""

    def test_ds_enabled_default_false(self):
        from sglang.srt.layers.radix_attention import RadixAttention

        attn = RadixAttention(
            num_heads=4,
            head_dim=16,
            scaling=0.25,
            num_kv_heads=2,
            layer_id=0,
        )
        self.assertFalse(attn.ds_enabled)

    def test_attribute_is_settable(self):
        from sglang.srt.layers.radix_attention import RadixAttention

        attn = RadixAttention(
            num_heads=4,
            head_dim=16,
            scaling=0.25,
            num_kv_heads=2,
            layer_id=0,
        )
        attn.ds_enabled = True
        self.assertTrue(attn.ds_enabled)


if __name__ == "__main__":
    unittest.main()
