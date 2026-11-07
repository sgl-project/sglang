"""CPU-only HCCL sizing policy coverage; importing the runtime is unnecessary."""

import ast
import importlib.util
import os
import subprocess
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
        ep_size=4,
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
        with self.assertRaisesRegex(ValueError, "required_MiB=3328 cap_MiB=256"):
            auto.deepep_normal_buffer_mb(16384, 6144, 8, max_mb=256)

    def test_deepep_rejects_bad_parameters(self):
        for args, kw in (
            ((16384, 6144, 0), {}),
            ((16384, 6144, 8), {"quantum_mb": 0}),
            ((16384, 6144, 8), {"max_mb": 100}),
        ):
            with self.assertRaises(ValueError):
                auto.deepep_normal_buffer_mb(*args, **kw)

    def initialize(self, ctx=None, **args):
        config = SimpleNamespace(moe_a2a_backend="deepep", deepep_mode="auto")
        config.__dict__.update(args)
        with patch.object(
            auto, "context_from_server_args", return_value=ctx or context()
        ):
            auto.initialize(config)

    def test_default_initialization_and_worker_inheritance(self):
        with patch.dict(os.environ, {}, clear=True):
            self.initialize()
            self.assertEqual(os.environ["HCCL_BUFFSIZE"], "256")
            self.assertEqual(os.environ["DEEPEP_HCCL_BUFFSIZE"], "3328")
            output = subprocess.check_output(
                [
                    sys.executable,
                    "-c",
                    "import os; print(os.environ['HCCL_BUFFSIZE'], os.environ['DEEPEP_HCCL_BUFFSIZE'])",
                ],
                text=True,
            )
            self.assertEqual(output.strip(), "256 3328")
            # Inherited settings need no model inspection or recalculation.
            with patch.object(
                auto, "context_from_server_args", side_effect=AssertionError
            ):
                auto.initialize(None)

    def test_explicit_hccl_skips_all_calculation(self):
        for overrides in (
            {"HCCL_BUFFSIZE": "1000"},
            {"HCCL_BUFFSIZE": "1000", "DEEPEP_HCCL_BUFFSIZE": "1200"},
        ):
            with (
                self.subTest(overrides=overrides),
                patch.dict(os.environ, overrides, clear=True),
                patch.object(
                    auto, "context_from_server_args", side_effect=AssertionError
                ),
                self.assertLogs(auto.logger, level="INFO") as logs,
            ):
                auto.initialize(None)
                self.assertEqual(dict(os.environ), overrides)
                self.assertIn("source=explicit", "\n".join(logs.output))
                if "DEEPEP_HCCL_BUFFSIZE" not in overrides:
                    self.assertIn("source=inherited", "\n".join(logs.output))

    def test_explicit_deepep_bypasses_overflow_and_transport_parsing(self):
        with patch.dict(
            os.environ,
            {
                "DEEPEP_HCCL_BUFFSIZE": "8000",
                "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "invalid",
            },
            clear=True,
        ):
            self.initialize(
                ctx=replace(context(), hidden_size=7168, chunked_prefill_size=32768)
            )
            self.assertEqual(os.environ["HCCL_BUFFSIZE"], "576")
            self.assertEqual(os.environ["DEEPEP_HCCL_BUFFSIZE"], "8000")

    def test_deepep_ceiling_and_rounding(self):
        self.assertEqual(auto.deepep_normal_buffer_mb(1, 512, 1), 256)
        # raw = 4096 MiB; one extra row crosses the rounding boundary.
        self.assertEqual(auto.deepep_normal_buffer_mb(1932, 524288, 1), 4096)
        for rows, hidden, topk, expected in (
            (1933, 524288, 1, "required_MiB=4128"),
            (32768, 7168, 8, "required_MiB=7424"),
        ):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, expected):
                auto.deepep_normal_buffer_mb(rows, hidden, topk)
        with self.assertRaisesRegex(ValueError, "required_MiB=3328 cap_MiB=3327"):
            auto.deepep_normal_buffer_mb(16384, 6144, 8, max_mb=3327)

    def test_failure_leaves_environment_unchanged(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "required_MiB=7424") as error:
                self.initialize(
                    ctx=replace(context(), hidden_size=7168, chunked_prefill_size=32768)
                )
            self.assertEqual(dict(os.environ), {})
            for field in (
                "rows=32768",
                "hidden=7168",
                "topk=8",
                "per_round_tokens=8192",
                "combine_long_seq=False",
                "cap_MiB=4096",
                "DEEPEP_HCCL_BUFFSIZE",
                "LONG_SEQ_ROUND=1",
            ):
                self.assertIn(field, str(error.exception))

    def test_deepep_applicability(self):
        for ctx, args in (
            (replace(context(), num_experts_per_tok=None), {}),
            (replace(context(), ep_size=1), {}),
            (context(), {"moe_a2a_backend": "none"}),
            (context(), {"moe_a2a_backend": "ascend_tp"}),
            (context(), {"moe_a2a_backend": "deepep_v2"}),
            (context(), {"deepep_mode": "low_latency"}),
        ):
            with (
                self.subTest(ctx=ctx, args=args),
                patch.dict(os.environ, {}, clear=True),
                self.assertLogs(auto.logger, level="INFO") as logs,
            ):
                self.initialize(ctx=ctx, **args)
                self.assertEqual(os.environ["HCCL_BUFFSIZE"], "256")
                self.assertNotIn("DEEPEP_HCCL_BUFFSIZE", os.environ)
                self.assertIn("not applicable", "\n".join(logs.output))
        for mode in ("auto", "normal"):
            with patch.dict(os.environ, {}, clear=True):
                self.initialize(deepep_mode=mode)
                self.assertEqual(os.environ["DEEPEP_HCCL_BUFFSIZE"], "3328")

    def test_dtype_payload(self):
        for dtype, expected in (
            ("torch.bfloat16", "256"),
            ("float16", "256"),
            ("torch.float16", "256"),
            ("float32", "480"),
            ("torch.float32", "480"),
        ):
            with self.subTest(dtype=dtype), patch.dict(os.environ, {}, clear=True):
                self.initialize(
                    ctx=replace(context(), dtype=dtype), moe_a2a_backend="none"
                )
                self.assertEqual(os.environ["HCCL_BUFFSIZE"], expected)

    def test_transport_settings(self):
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
            (
                {
                    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "3584",
                    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "32",
                },
                "3328",
            ),
        ):
            with self.subTest(extra=extra), patch.dict(os.environ, extra, clear=True):
                self.initialize()
                self.assertEqual(os.environ["DEEPEP_HCCL_BUFFSIZE"], expected)
                for name, value in extra.items():
                    self.assertEqual(os.environ[name], value)

    def test_invalid_transport_settings(self):
        tokens = "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS"
        rounds = "DEEPEP_NORMAL_LONG_SEQ_ROUND"
        for extra in (
            {tokens: "8192"},
            {rounds: "1"},
            *(
                {tokens: t, rounds: r}
                for t, r in (
                    ("bad", "1"),
                    ("31", "1"),
                    ("8193", "1"),
                    ("8192", "0"),
                    ("32", "257"),
                    ("8192", "17"),
                    ("8192 ", "1"),
                    ("8_192", "1"),
                    ("", "1"),
                )
            ),
            {"DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "true"},
        ):
            with self.subTest(extra=extra), patch.dict(os.environ, extra, clear=True):
                with self.assertRaises(ValueError):
                    self.initialize()
                self.assertEqual(dict(os.environ), extra)

    def test_startup_hook_precedes_workers_and_is_npu_only(self):
        tree = ast.parse((ROOT / "python/sglang/srt/entrypoints/engine.py").read_text())
        functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        launch = functions["_launch_subprocesses"]
        calls = {
            node.func.id: node.lineno
            for node in ast.walk(launch)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        worker_lines = [
            node.lineno
            for node in ast.walk(launch)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_launch_scheduler_processes"
        ]
        self.assertLess(calls["_set_envs_and_config"], min(worker_lines))
        hook = functions["_set_envs_and_config"].body[1]
        # Execute the real guarded hook without importing the engine runtime.
        module = ast.Module(body=[hook], type_ignores=[])
        with patch.dict(
            sys.modules, {"sglang.srt.hardware_backend.npu.autotune": auto}
        ):
            for device in ("cpu", "cuda", "npu"):
                with patch.object(auto, "initialize") as initialize:
                    cfg = SimpleNamespace(device=device)
                    exec(compile(module, "engine_npu_hook", "exec"), {"cfg": cfg})  # noqa: S102
                    self.assertEqual(initialize.call_count, int(device == "npu"))


if __name__ == "__main__":
    unittest.main()
