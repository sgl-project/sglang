"""CPU-only HCCL sizing policy coverage; importing the runtime is unnecessary."""

import importlib.util
import os
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "npu_autotune_policy_test",
    ROOT / "python/sglang/srt/hardware_backend/npu/autotune.py",
)
auto = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = auto
SPEC.loader.exec_module(auto)


def context():
    return auto.NpuTuningContext(
        hidden_size=6144,
        num_experts_per_tok=8,
        tp_size=4,
        ep_size=1,
        chunked_prefill_size=16384,
        max_prefill_tokens=32768,
        dtype="torch.bfloat16",
    )


class TestNpuAutotune(unittest.TestCase):
    def test_hccl_current_shape(self):
        self.assertEqual(auto.hccl_buffer_mb(16384, 6144, 2), 256)

    def test_representative_prefill_rows(self):
        for chunk in (16384, -1, 0):
            c = replace(context(), chunked_prefill_size=chunk, max_prefill_tokens=16384)
            self.assertEqual(auto.representative_prefill_rows(c), 16384)
        self.assertEqual(
            auto.representative_prefill_rows(
                replace(context(), chunked_prefill_size=8192)
            ),
            8192,
        )
        for chunk, maximum in ((16384, 0), (0, 0), (-1, -1)):
            with (
                self.subTest(chunk=chunk, maximum=maximum),
                self.assertRaises(ValueError),
            ):
                auto.representative_prefill_rows(
                    replace(
                        context(),
                        chunked_prefill_size=chunk,
                        max_prefill_tokens=maximum,
                    )
                )

    def test_hccl_rounding_and_clamping(self):
        self.assertEqual(auto.hccl_buffer_mb(1, 1, 2), 64)
        self.assertEqual(auto.hccl_buffer_mb(1 << 20, 6144, 2), 1024)
        self.assertEqual(auto.hccl_buffer_mb(16384, 6144, 2, headroom=1), 192)
        self.assertEqual(auto.hccl_buffer_mb(16385, 6144, 2, headroom=1), 224)

    def test_hccl_rejects_bad_parameters(self):
        for kw in (
            {"headroom": 0.5},
            {"headroom": float("nan")},
            {"quantum_mb": 0},
            {"max_mb": 32},
        ):
            with self.assertRaises(ValueError):
                auto.hccl_buffer_mb(16384, 6144, 2, **kw)

    def test_deepep_normal_formula(self):
        # Combine streaming off (default): the full sequence must fit with two
        # tensors per expert: 2 * (106 + 16384*2*6144*8 / 2^20) + 20 = 3304.
        self.assertEqual(auto.deepep_normal_buffer_mb(16384, 6144, 8), 3328)
        # 2 * (106 + 96) + 20 = 424.
        self.assertEqual(auto.deepep_normal_buffer_mb(1024, 6144, 8), 448)
        # Combine streaming on: one 8192-token round, three tensors per expert:
        # 2 * (106 + 1152) + 20 = 2536.
        self.assertEqual(
            auto.deepep_normal_buffer_mb(16384, 6144, 8, combine_long_seq=True), 2560
        )
        # A user-tuned round size is respected: 2 * (106 + 504) + 20 = 1240.
        self.assertEqual(
            auto.deepep_normal_buffer_mb(16384, 6144, 8, 3584, combine_long_seq=True),
            1248,
        )
        # Ant-moving with combine streaming off takes the larger term.
        self.assertEqual(
            auto.deepep_normal_buffer_mb(16384, 6144, 8, 3584, ant_moving=True), 3328
        )
        self.assertEqual(
            auto.deepep_normal_buffer_mb(16384, 6144, 8, ant_moving=True), 3328
        )
        # The documented floor dominates for tiny models.
        self.assertEqual(auto.deepep_normal_buffer_mb(1, 512, 1), 256)
        self.assertEqual(auto.deepep_normal_buffer_mb(16384, 6144, 8, max_mb=256), 256)

    def test_deepep_rejects_bad_parameters(self):
        for args, kw in (
            ((16384, 6144, 0), {}),
            ((16384, 6144, 8), {"quantum_mb": 0}),
            ((16384, 6144, 8), {"max_mb": 100}),
        ):
            with self.assertRaises(ValueError):
                auto.deepep_normal_buffer_mb(*args, **kw)

    def test_disabled_initialization(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(
                auto,
                "context_from_server_args",
                side_effect=AssertionError("must not inspect model"),
            ),
        ):
            auto.initialize(None)

    def test_initialization_sets_both_buffers(self):
        with (
            patch.dict(
                os.environ,
                {
                    "SGLANG_NPU_AUTO_HCCL_BUFFSIZE": "1",
                    "SGLANG_NPU_AUTO_DEEPEP_HCCL_BUFFSIZE": "1",
                },
                clear=True,
            ),
            patch.object(auto, "context_from_server_args", return_value=context()),
        ):
            auto.initialize(SimpleNamespace())
            self.assertEqual(os.environ["HCCL_BUFFSIZE"], "256")
            self.assertEqual(os.environ["DEEPEP_HCCL_BUFFSIZE"], "3328")

    def test_initialization_dry_run_only_logs(self):
        with (
            patch.dict(
                os.environ,
                {
                    "SGLANG_NPU_AUTO_HCCL_BUFFSIZE": "1",
                    "SGLANG_NPU_AUTO_DEEPEP_HCCL_BUFFSIZE": "1",
                    "SGLANG_NPU_AUTOTUNE_DRY_RUN": "1",
                },
                clear=True,
            ),
            patch.object(auto, "context_from_server_args", return_value=context()),
            self.assertLogs(auto.logger, level="INFO") as logs,
        ):
            auto.initialize(SimpleNamespace())
            self.assertNotIn("HCCL_BUFFSIZE", os.environ)
            self.assertNotIn("DEEPEP_HCCL_BUFFSIZE", os.environ)
            output = "\n".join(logs.output)
            self.assertIn("HCCL_BUFFSIZE=256", output)
            self.assertIn("DEEPEP_HCCL_BUFFSIZE=3328", output)
            self.assertIn("source=deterministic-auto", output)
            self.assertIn("collective_bytes=201326592", output)

    def test_initialization_respects_deepep_envs(self):
        for extra, expected in (
            ({"DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1"}, "2560"),
            (
                {
                    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "3584",
                    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "32",
                    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
                },
                "1248",
            ),
        ):
            with (
                self.subTest(expected=expected),
                patch.dict(
                    os.environ,
                    {"SGLANG_NPU_AUTO_DEEPEP_HCCL_BUFFSIZE": "1", **extra},
                    clear=True,
                ),
                patch.object(auto, "context_from_server_args", return_value=context()),
            ):
                auto.initialize(SimpleNamespace())
                self.assertEqual(os.environ["DEEPEP_HCCL_BUFFSIZE"], expected)

    def test_initialization_explicit_values_win(self):
        with (
            patch.dict(
                os.environ,
                {
                    "SGLANG_NPU_AUTO_HCCL_BUFFSIZE": "1",
                    "SGLANG_NPU_AUTO_DEEPEP_HCCL_BUFFSIZE": "1",
                    "HCCL_BUFFSIZE": "1000",
                    "DEEPEP_HCCL_BUFFSIZE": "1200",
                },
                clear=True,
            ),
            patch.object(auto, "context_from_server_args", return_value=context()),
            self.assertLogs(auto.logger, level="INFO") as logs,
        ):
            auto.initialize(SimpleNamespace())
            self.assertEqual(os.environ["HCCL_BUFFSIZE"], "1000")
            self.assertEqual(os.environ["DEEPEP_HCCL_BUFFSIZE"], "1200")
            output = "\n".join(logs.output)
            self.assertIn("HCCL_BUFFSIZE=1000 source=explicit", output)
            self.assertIn("DEEPEP_HCCL_BUFFSIZE=1200 source=explicit", output)

    def test_initialization_skips_deepep_for_dense_models(self):
        with (
            patch.dict(
                os.environ,
                {"SGLANG_NPU_AUTO_DEEPEP_HCCL_BUFFSIZE": "1"},
                clear=True,
            ),
            patch.object(
                auto,
                "context_from_server_args",
                return_value=replace(context(), num_experts_per_tok=None),
            ),
            self.assertLogs(auto.logger, level="INFO") as logs,
        ):
            auto.initialize(SimpleNamespace())
            self.assertNotIn("DEEPEP_HCCL_BUFFSIZE", os.environ)
            self.assertIn("not applicable", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
